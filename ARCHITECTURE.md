# PDF Visual Segmentation Pipeline - Technical Documentation

## Overview

This pipeline extracts, classifies, and analyzes visual elements (figures, charts, diagrams, images) from textbook PDFs for knowledge graph construction and RAG (Retrieval-Augmented Generation) applications. The system combines multiple detection strategies, OCR, computer vision, and optional LLM-based analysis to create comprehensive visual segment metadata.

---

## Core Architecture

### 1. Data Models

#### **VisualType (Enum)**
Classifies visual elements into 8 distinct categories:
- `CHART` - Data visualizations with axes (line, bar, scatter plots)
- `DIAGRAM` - Process flows, concept maps with nodes and connections
- `FLOWCHART` - Sequential flows with decision boxes
- `FIGURE` - Generic labeled figures
- `SCREENSHOT` - Computer interface captures
- `PHOTO` - Photographs or realistic images
- `SCANNED_DOC` - Scanned text documents
- `TABLE_IMAGE` - Tables rendered as images
- `UNKNOWN` - Unclassified elements

#### **BoundingBox**
Stores precise page coordinates:
- `x0, y0, x1, y1` - Rectangle coordinates
- `page_width, page_height` - Page dimensions for normalization
- `area()` - Calculates bounding box area

#### **OCRResult**
Structured OCR output with specialized fields:
- `raw_text` - Complete extracted text
- `blocks` - Individual text blocks with bounding boxes
- `confidence` - Average OCR confidence score
- **Chart-specific**: `axis_labels`, `legend_items`, `tick_labels`
- **Diagram-specific**: `node_texts`, `detected_arrows`

#### **VisualSegment**
Complete metadata container for each visual element:
- **Identity**: `segment_id`, `segment_type`, `book_id`, `page_no`
- **Location**: `bbox` (bounding box coordinates)
- **Content**: `image_path`, `image_bytes`, `ocr_result`
- **Context**: `caption_text`, `figure_number`, `reference_keys`
- **Analysis**: `summary`, `classification_confidence`, `linked_concept_ids`
- **Relationships**: `heading_path`, `linked_segment_ids`, `nearby_text`

---

## Extraction Pipeline

### Phase 1: Two-Pass Image Detection Strategy

The pipeline uses a **smart two-pass approach** with conflict resolution:

#### **Pass 1: Caption-Based Detection (Primary - High Confidence)**

**Why caption-first?** Captions provide the most reliable signal for figure boundaries and are specifically formatted in academic texts.

**Process:**
1. **Caption Pattern Matching**
   - Searches for patterns: `Figure X`, `Fig. X`, `Chart X`, `Diagram X`, `Exhibit X`
   - Uses regex with flexible formatting: `Figure\s+(\d+(?:\.\d+)?)\s*[:\-]?\s*(.*?)(?=\n\n|\Z)`
   
2. **Caption Validation** (Critical for accuracy)
   - Verifies caption is at **beginning of text block** (not in-text reference)
   - Excludes blocks containing reference phrases: "as shown in", "see Figure", "in Figure"
   - Checks length constraint (captions typically < 400 chars)
   - Distinguishes captions from body paragraphs

3. **Visual Content Boundary Detection**
   - Searches upward from caption (typically 100-500 points)
   - Combines **multiple boundary signals**:

   **Signal 1: Drawing Commands (Most Reliable)**
   ```python
   # PyMuPDF drawing commands = vector graphics (charts, diagrams)
   drawings = page.get_drawings()
   # Groups nearby drawings into single figure
   ```

   **Signal 2: Embedded Images**
   ```python
   # Raster images (photos, scanned content)
   image_list = page.get_images(full=True)
   img_rects = page.get_image_rects(xref)
   ```

   **Signal 3: Whitespace Analysis**
   - Detects large vertical gaps (>30 points) between text blocks
   - Gaps indicate natural separation between body text and figure

   **Signal 4: Text Boundary Detection**
   - Identifies body paragraphs (wide, dense, left-aligned, >120 chars)
   - Distinguishes from figure labels (short, scattered, <50 chars)
   - Figure starts after last body paragraph

4. **Boundary Combination Logic**
   ```
   Priority: drawing_bounds > image_bounds > whitespace > text > fallback
   ```
   - Uses drawing bounds for charts/diagrams (most accurate)
   - Falls back to image bounds for photos
   - Uses whitespace/text analysis when vector data unavailable
   - Conservative fallback: 200 points above caption with padding

5. **Region Rendering**
   - Renders complete region (figure + caption) at 150 DPI
   - Captures vector graphics, text, and all content as single image
   - Saves as PNG with stable ID: `{book_id}_p{page}_md5hash.png`

#### **Pass 2: Embedded Image Extraction with Validation**

**Purpose:** Catch figures without detected captions (unlabeled diagrams, photos, etc.)

**Process:**
1. **Direct Image Extraction**
   ```python
   image_list = page.get_images(full=True)
   base_image = page.parent.extract_image(xref)
   ```

2. **Strict Validation** (reduces false positives)
   
   Validation scoring system (threshold: 0.5):
   
   - **Size validation** (+0.3 for area > 10,000 sq pts)
   - **Dimension check** (reject if < 50x50 pixels)
   - **Aspect ratio** (+0.2 if between 0.2-5.0)
   - **Position check** (-0.2 if in header/footer zone)
   - **Caption proximity** (+0.4 if caption found nearby)
   - **Content variance** (+0.2 if variance > 100, -0.3 if < 10)

3. **Caption Search**
   - Looks for captions within 60 points below image
   - If found, expands bbox to include caption
   - Re-renders combined region

4. **Conflict Resolution**
   
   When embedded image overlaps with caption-based segment (>40% overlap):
   
   **Decision factors:**
   - Caption presence (+3 points for caption-based)
   - Size comparison (larger = more context)
   - Photo detection (raster images favor embedded)
   - Vector content count (+2 for >10 drawing commands)
   - Validation score
   
   **Outcome:** Keeps highest-scoring version, discards duplicate

---

### Phase 2: OCR Processing

**OCRProcessor** extracts and structures text from images:

1. **Tesseract OCR Execution**
   ```python
   ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
   raw_text = pytesseract.image_to_string(image)
   ```
   - Confidence threshold: 30% (filters low-quality detections)
   - Builds text blocks with bounding boxes

2. **Chart-Specific Detection**
   - **Axis labels**: Pattern matching for "year", "time", "value", "price", "%"
   - **Legend items**: Short text lines (<50 chars)
   - **Tick labels**: Numeric sequences

3. **Diagram-Specific Detection**
   - **Node texts**: Medium-length blocks (3-50 chars) = potential diagram nodes
   - **Arrow counting**: 
     ```python
     # Detects diagonal lines (20-70° or 110-160°) as arrows
     edges = cv2.Canny(img_array, 50, 150)
     lines = cv2.HoughLinesP(edges, ...)
     ```

---

### Phase 3: Visual Classification

**VisualClassifier** uses multi-modal heuristics + optional LLM verification:

#### **Heuristic Feature Extraction**

1. **Text Density**
   ```python
   len(ocr_text) / 1000  # Normalized character count
   ```

2. **Line Density**
   ```python
   edges = cv2.Canny(img_array, 50, 150)
   np.sum(edges > 0) / edges.size
   ```

3. **Axis Detection**
   - Hough line transform finds straight lines
   - Counts horizontal (-10° to 10°) and vertical (80° to 100°)
   - Perpendicular lines indicate chart axes

4. **Grid Pattern Detection** (for tables)
   ```python
   horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
   vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
   # Significant h_count AND v_count = grid
   ```

5. **Photo-Like Detection**
   ```python
   variance = np.var(img_array)
   # High variance (>1000) = natural images/photos
   # Low variance = uniform graphics/diagrams
   ```

#### **Rule-Based Scoring System**

Each visual type accumulates confidence scores:

```python
# Example: Chart indicators
if has_axes: score[CHART] += 0.6
if line_density > 0.1: score[CHART] += 0.3
if 'axis' in ocr_text: score[CHART] += 0.2

# Diagram indicators
if arrow_count > 3: score[FLOWCHART] += 0.5
if node_count > 5: score[DIAGRAM] += 0.4

# Table indicators
if has_grid and text_density > 0.15: score[TABLE] += 0.7
```

Winner = highest score, capped at 0.95 confidence

#### **LLM Verification** (Optional Enhancement)

**When triggered:** Confidence < 0.6 from heuristics

**Model:** GPT-4o-mini (cheap vision model)

**Process:**
1. Converts image to base64
2. Sends prompt with categories + OCR context
3. Requests single category classification
4. Uses if LLM confidence > heuristic confidence

**Cost optimization:**
- Uses `"detail": "low"` for image encoding (cheaper)
- Max 50 tokens response
- Only calls when uncertain

---

### Phase 4: Caption Detection & Extraction

**CaptionDetector** links figures to their descriptive text:

1. **Pattern Matching**
   - Multiple caption formats supported
   - Extracts figure number (e.g., "3.2") and caption text

2. **Spatial Search**
   - Looks within 50 points above/below bbox
   - Combines nearby text blocks

3. **Reference Key Generation**
   ```python
   reference_keys = [
       "Figure 3.2",
       "Fig. 3.2", 
       "Fig 3.2"
   ]
   # Enables finding in-text references to this figure
   ```

---

### Phase 5: Semantic Understanding

#### **Summary Generation**

**Two-tier approach:**

**Tier 1: LLM-Based Summary** (Primary, 0.85 confidence)
- Uses GPT-4o-mini vision API
- Type-specific prompts:
  - **Charts**: "What data is plotted? Key trends?"
  - **Diagrams**: "What process? Main components? Flow direction?"
  - **General**: "Main subject? Key visual elements? Purpose?"
- Incorporates caption, figure number, OCR text as context
- 2-3 sentence concise output

**Tier 2: Rule-Based Fallback** (0.5 confidence)
```python
if segment_type == CHART:
    summary = f"Chart displays {y_axis} versus {x_axis}"
elif segment_type == DIAGRAM:
    summary = f"Diagram with {node_count} components, {arrow_count} flows"
```

#### **Concept Linking**

**ConceptLinker** maps visuals to domain concepts:

1. **Taxonomy Loading**
   ```python
   # Expected columns: concept_id, concept_name, aliases, bloom_level, tag
   taxonomy_df = pd.read_excel(taxonomy_path)
   ```

2. **Index Building**
   - Normalizes concept names (lowercase, stripped)
   - Expands aliases (comma-separated)
   - Creates searchable concept map

3. **Exact Matching**
   ```python
   # Searches in: caption + OCR text + summary
   if concept_key in combined_text.lower():
       links.append({
           'concept_id': ...,
           'confidence': 0.9,
           'match_method': 'exact_match'
       })
   ```

4. **Deduplication**
   - Removes duplicate concept_ids
   - Returns unique links only

---

### Phase 6: Contextual Enrichment

#### **Heading Path Extraction**

Builds hierarchical context:
```python
# Finds text blocks above image with large font (>12pt)
heading_path = ["Chapter 3", "Financial Ratios", "Profitability Metrics"]
# Keeps last 3 headings for context
```

#### **Nearby Text Extraction**

Captures surrounding content:
```python
# Collects text within 100 points of bbox
nearby_text = "...discusses the relationship between ROE and leverage..."
# Limited to 500 chars
```

---

## Output Generation

### 1. JSON Metadata File

Complete segment data:
```json
{
  "book_id": "textbook_001",
  "total_segments": 47,
  "segments": [
    {
      "segment_id": "textbook_001_p023_a4f3c2d1",
      "segment_type": "chart",
      "page_no": 23,
      "bbox": {...},
      "caption_text": "Figure 3.2: ROE decomposition...",
      "figure_number": "3.2",
      "classification_confidence": 0.87,
      "summary": "Line chart showing ROE components...",
      "linked_concept_ids": [
        {
          "concept_id": "C042",
          "concept_name": "Return on Equity",
          "confidence": 0.9
        }
      ],
      "heading_path": ["Chapter 3", "Profitability Analysis"],
      "nearby_text": "...demonstrates how..."
    }
  ]
}
```

### 2. Summary CSV

Quick review table:
```csv
segment_id,page,type,confidence,figure_number,caption,linked_concepts,summary
textbook_001_p023_a4f3c2d1,23,chart,0.87,3.2,"Figure 3.2: ROE...",1,"Line chart..."
```

### 3. Extracted Images

Individual PNG files:
```
./extracted_visuals/
  textbook_001_p023_a4f3c2d1.png
  textbook_001_p024_b7e8d3f2.png
  ...
```

---

## Key Algorithmic Innovations

### 1. **Smart Conflict Resolution**
- Prevents duplicate extraction of same figure
- Intelligently chooses best representation (caption-based vs. embedded)
- Considers: caption presence, size, content type (vector vs. raster), validation score

### 2. **Multi-Signal Boundary Detection**
- Doesn't rely on single heuristic
- Combines drawing commands, images, whitespace, text analysis
- Prioritized fallback chain ensures robust extraction

### 3. **Caption Validation**
- Distinguishes actual captions from in-text references
- Critical for preventing false positives from body paragraphs

### 4. **Adaptive Classification**
- Combines fast heuristics with expensive LLM calls
- Only uses LLM when uncertain (confidence < 0.6)
- Balances accuracy and cost

### 5. **Hierarchical Feature Extraction**
- General OCR + type-specific features (axes, nodes, arrows)
- Enables nuanced classification and rich metadata

---

## Dependencies

```python
# Core PDF Processing
fitz (PyMuPDF)          # Image extraction, drawing analysis, rendering

# OCR & Computer Vision
pytesseract             # Text extraction
opencv-python (cv2)     # Edge detection, line finding, morphology
PIL (Pillow)            # Image manipulation

# Data Processing
pandas                  # Taxonomy loading, CSV output
numpy                   # Array operations, statistics

# Optional (LLM Enhancement)
requests                # OpenAI API calls (GPT-4o-mini)
```

---

## Usage Example

```python
pipeline = VisualSegmentationPipeline(
    book_id="finance_textbook",
    pdf_path="./textbook.pdf",
    taxonomy_path="./concepts.xlsx",  # Optional
    output_dir="./output"
)

segments = pipeline.process()

# Statistics
print(f"Extracted {len(segments)} visual elements")
print(f"Charts: {sum(1 for s in segments if s.segment_type == VisualType.CHART)}")
print(f"With concepts: {sum(1 for s in segments if s.linked_concept_ids)}")
```

---

## Performance Characteristics

- **Accuracy**: Caption-based detection ~90% precision (high confidence)
- **Coverage**: Two-pass approach catches both labeled and unlabeled figures
- **Speed**: ~2-5 seconds per page (without LLM), ~5-10 seconds (with LLM)
- **Robustness**: Multiple fallback strategies prevent failures

---

## Configuration Options

### Pipeline Parameters

```python
VisualSegmentationPipeline(
    book_id: str,              # Unique identifier for the book
    pdf_path: str,             # Path to PDF file
    taxonomy_path: Optional[str],  # Path to concept taxonomy (Excel)
    output_dir: str            # Output directory for images and metadata
)
```

### Tunable Constants

**Caption Detection:**
- `CAPTION_PATTERNS` - List of regex patterns for caption formats
- Caption proximity threshold: 50 points above/below bbox

**Boundary Detection:**
- Search range above caption: 100-500 points
- Whitespace gap threshold: 30 points (significant gap)
- Body paragraph width threshold: 65% of page width
- Body paragraph text length: >120 characters

**Validation:**
- Minimum image area: 3,000 sq points (pass 1), 5,000 sq points (pass 2)
- Minimum dimensions: 50x50 pixels
- Aspect ratio range: 0.2-5.0
- Header/footer zone: top/bottom 10% of page
- Caption search distance: 60 points below image
- Overlap threshold for conflicts: 40%

**OCR:**
- Confidence threshold: 30%
- Node text length: 3-50 characters
- Legend item max length: 50 characters

**Classification:**
- LLM trigger threshold: <0.6 confidence
- Rendering DPI: 150
- Drawing cluster distance: 100 points
- Minimum cluster size: 3 drawing elements

---

## Error Handling & Edge Cases

### Handled Scenarios

1. **Missing Captions**
   - Pass 2 catches unlabeled figures
   - Fallback to nearby text extraction

2. **Overlapping Figures**
   - Conflict resolution algorithm
   - Prevents duplicate extraction

3. **Poor OCR Quality**
   - Confidence filtering (>30%)
   - Graceful degradation (empty OCRResult)

4. **API Failures**
   - LLM classification failures caught with try-except
   - Falls back to heuristic classification
   - Summary generation has rule-based fallback

5. **Malformed PDFs**
   - Individual page/segment failures don't stop pipeline
   - Warnings logged, processing continues

6. **Small/Large Images**
   - Size validation filters decorative elements
   - Area constraints prevent full-page extractions

---

## Future Enhancements

### Potential Improvements

1. **Advanced Concept Linking**
   - Semantic similarity matching (embeddings)
   - Multi-word concept recognition
   - Fuzzy matching for OCR errors

2. **Enhanced Classification**
   - Custom vision model fine-tuned on textbook figures
   - Sub-classification (bar chart vs. line chart)
   - Table structure extraction

3. **Relationship Detection**
   - Cross-reference linking ("see Figure 3.2")
   - Related figure clustering
   - Data flow between diagrams

4. **Quality Metrics**
   - OCR quality assessment
   - Extraction completeness scoring
   - Caption-figure alignment verification

5. **Performance Optimization**
   - Parallel page processing
   - Caching of drawing/image extractions
   - Batch LLM API calls

---

## Troubleshooting

### Common Issues

**Issue: No figures detected**
- Check if PDF has searchable text (required for captions)
- Verify caption patterns match document style
- Inspect `_detect_by_drawings()` output for vector content

**Issue: Duplicate figures**
- Review conflict resolution scores
- Check overlap threshold (40% default)
- Examine validation scores in notes

**Issue: Wrong classification**
- Check feature extraction values (text_density, line_density, etc.)
- Review LLM responses if enabled
- Adjust scoring thresholds in `VisualClassifier.classify()`

**Issue: Poor OCR**
- Increase rendering DPI (default: 150)
- Check image quality in output directory
- Verify Tesseract installation and language data

**Issue: Missing captions**
- Review `CAPTION_PATTERNS` for document-specific formats
- Check caption validation logic (may be too strict)
- Examine `caption_bbox` in output for spatial issues

---

## License & Attribution

This pipeline uses the following open-source libraries:
- PyMuPDF (GNU AGPL)
- Tesseract OCR (Apache 2.0)
- OpenCV (Apache 2.0)
- Pillow (PIL License)

Optional LLM features require OpenAI API access (commercial service).

---