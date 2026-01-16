"""
Textbook Visual Segmentation Pipeline - Mistral API Integration
Uses Mistral's vision models for classification and summary generation
"""

import json
import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from enum import Enum
import io
import os

# Core dependencies
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import requests
import base64


class VisualType(str, Enum):
    """Classification of visual elements"""
    CHART = "chart"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    FIGURE = "figure"
    SCREENSHOT = "screenshot"
    PHOTO = "photo"
    SCANNED_DOC = "scanned_document"
    TABLE_IMAGE = "table_image"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Page coordinates for visual elements"""
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "x0": self.x0, "y0": self.y0,
            "x1": self.x1, "y1": self.y1,
            "width": self.x1 - self.x0,
            "height": self.y1 - self.y0,
            "page_width": self.page_width,
            "page_height": self.page_height
        }
    
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class OCRResult:
    """Structured OCR output"""
    raw_text: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    # Chart-specific fields
    axis_labels: Dict[str, str] = field(default_factory=dict)
    legend_items: List[str] = field(default_factory=list)
    tick_labels: Dict[str, List[str]] = field(default_factory=dict)
    
    # Diagram-specific fields
    node_texts: List[str] = field(default_factory=list)
    detected_arrows: int = 0


@dataclass
class MermaidRepresentation:
    """Mermaid diagram representation for better LLM understanding"""
    mermaid_code: Optional[str] = None
    diagram_type: Optional[str] = None  # flowchart, sequence, class, etc.
    extraction_confidence: float = 0.0
    extraction_notes: str = ""


@dataclass
class VisualSegment:
    """Complete visual segment with all metadata"""
    segment_id: str
    segment_type: VisualType
    
    # Source tracking
    book_id: str
    page_no: int
    bbox: BoundingBox
    
    # Extracted content
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    
    # Caption and reference
    caption_text: Optional[str] = None
    figure_number: Optional[str] = None
    reference_keys: List[str] = field(default_factory=list)
    
    # OCR and analysis
    ocr_result: Optional[OCRResult] = None
    
    # Mermaid representation (NEW)
    mermaid_repr: Optional[MermaidRepresentation] = None
    
    # Classification
    classification_confidence: float = 0.0
    classification_method: str = "heuristic"
    
    # Semantic understanding
    summary: Optional[str] = None
    summary_confidence: float = 0.0
    
    # Concept linking
    linked_concept_ids: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context linking
    heading_path: List[str] = field(default_factory=list)
    linked_segment_ids: List[str] = field(default_factory=list)
    nearby_text: Optional[str] = None
    
    # Metadata
    extraction_method: str = "native"
    confidence: float = 1.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        result = asdict(self)
        result['segment_type'] = self.segment_type.value
        result['bbox'] = self.bbox.to_dict() if self.bbox else None
        result.pop('image_bytes', None)
        return result


class MistralVisionAPI:
    """Handler for Mistral Vision API calls"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        # Pixtral is Mistral's vision model
        self.vision_model = "pixtral-12b-2409"
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def classify_visual(self, image: Image.Image, ocr_result: OCRResult) -> Tuple[VisualType, float, str]:
        """
        Classify visual type using Mistral's vision model
        Returns (visual_type, confidence, method)
        """
        if not self.api_key:
            print("    WARNING: MISTRAL_API_KEY not found, falling back to heuristics")
            return VisualType.FIGURE, 0.3, "fallback_heuristic"
        
        try:
            img_base64 = self._encode_image(image)
            
            # Build context from OCR
            ocr_context = ""
            if ocr_result and ocr_result.raw_text:
                ocr_context = f"\n\nText detected in image:\n{ocr_result.raw_text[:300]}"
            
            prompt = f"""Analyze this image and classify it into ONE of these categories:

**Categories:**
- CHART: Data visualization with axes (line, bar, scatter, pie charts, histograms)
- DIAGRAM: Process diagrams, system architecture, concept maps with nodes/connections
- FLOWCHART: Sequential flow with decision boxes, process steps, and arrows
- TABLE: Grid structure with rows and columns (may be image of a table)
- PHOTO: Photograph or realistic image of real-world objects/scenes
- SCREENSHOT: Computer interface, application, or software screenshot
- SCANNED_DOC: Scanned text document or book page
- FIGURE: Generic labeled figure that doesn't fit the above categories

**Instructions:**
1. Carefully examine the visual content
2. Consider the structure, purpose, and typical use case
3. Choose the SINGLE most appropriate category
4. Provide your confidence level (0.0 to 1.0)

{ocr_context}

**Response format (JSON only):**
{{
  "category": "CHART|DIAGRAM|FLOWCHART|TABLE|PHOTO|SCREENSHOT|SCANNED_DOC|FIGURE",
  "confidence": 0.85,
  "reasoning": "Brief explanation of why this category fits best"
}}"""

            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_base64}"
                                }
                            ]
                        }
                    ],
                    "max_tokens": 300,
                    "temperature": 0.1  # Low temperature for consistent classification
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse JSON response
                try:
                    # Try to extract JSON from markdown code blocks
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                    elif '```' in content:
                        # Remove any code block markers
                        content = re.sub(r'```\w*\s*', '', content).strip()
                    
                    classification = json.loads(content)
                    
                    category = classification.get('category', 'FIGURE').upper()
                    confidence = float(classification.get('confidence', 0.7))
                    
                    # Map to VisualType
                    type_mapping = {
                        'CHART': VisualType.CHART,
                        'DIAGRAM': VisualType.DIAGRAM,
                        'FLOWCHART': VisualType.FLOWCHART,
                        'TABLE': VisualType.TABLE_IMAGE,
                        'PHOTO': VisualType.PHOTO,
                        'SCREENSHOT': VisualType.SCREENSHOT,
                        'SCANNED_DOC': VisualType.SCANNED_DOC,
                        'FIGURE': VisualType.FIGURE
                    }
                    
                    visual_type = type_mapping.get(category, VisualType.FIGURE)
                    return visual_type, min(confidence, 0.95), "mistral_vision"
                    
                except json.JSONDecodeError:
                    print(f"    Failed to parse Mistral response as JSON: {content[:200]}")
            else:
                print(f"    Mistral API error: {response.status_code} - {response.text[:200]}")
                
        except Exception as e:
            print(f"    Mistral classification failed: {e}")
        
        return VisualType.FIGURE, 0.3, "fallback_heuristic"
    
    def generate_summary(self, segment: 'VisualSegment') -> Tuple[Optional[str], float]:
        """
        Generate comprehensive summary using Mistral vision model
        Returns (summary, confidence)
        """
        if not self.api_key or not segment.image_path:
            return None, 0.0
        
        try:
            image = Image.open(segment.image_path)
            img_base64 = self._encode_image(image)
            
            # Build rich context
            context_parts = []
            if segment.caption_text:
                context_parts.append(f"**Caption:** {segment.caption_text}")
            if segment.figure_number:
                context_parts.append(f"**Figure Number:** {segment.figure_number}")
            if segment.segment_type:
                context_parts.append(f"**Visual Type:** {segment.segment_type.value}")
            if segment.ocr_result and segment.ocr_result.raw_text:
                context_parts.append(f"**Text in Image:** {segment.ocr_result.raw_text[:400]}")
            if segment.heading_path:
                context_parts.append(f"**Section Context:** {' > '.join(segment.heading_path)}")
            if segment.mermaid_repr and segment.mermaid_repr.mermaid_code:
                context_parts.append(f"**Mermaid Representation:**\n```mermaid\n{segment.mermaid_repr.mermaid_code}\n```")
            
            context = "\n".join(context_parts) if context_parts else "No additional context available"
            
            # Create type-specific prompts
            if segment.segment_type == VisualType.CHART:
                prompt = f"""Analyze this chart image and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Chart Type:** Identify the specific chart type (line, bar, scatter, pie, histogram, etc.)
2. **Variables:** What data/variables are being plotted? (X-axis, Y-axis, categories)
3. **Key Findings:** What are the main trends, patterns, or insights visible?
4. **Data Range:** Approximate ranges or scale of the data if visible
5. **Notable Features:** Any outliers, intersections, or significant data points

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this chart without seeing it."""

            elif segment.segment_type in [VisualType.DIAGRAM, VisualType.FLOWCHART]:
                prompt = f"""Analyze this {"flowchart" if segment.segment_type == VisualType.FLOWCHART else "diagram"} and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Purpose:** What process, system, or concept does this illustrate?
2. **Components:** List the main components, stages, or nodes
3. **Relationships:** Describe the connections, flow direction, or relationships
4. **Key Insights:** What is the main takeaway or learning objective?
5. **Structure:** How is the information organized? (hierarchical, sequential, cyclic, etc.)

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this {"flowchart" if segment.segment_type == VisualType.FLOWCHART else "diagram"} without seeing it."""

            elif segment.segment_type == VisualType.TABLE_IMAGE:
                prompt = f"""Analyze this table and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Table Purpose:** What data or information is being presented?
2. **Structure:** Describe rows, columns, and headers
3. **Key Values:** Highlight important numbers, patterns, or comparisons
4. **Insights:** What conclusions or comparisons can be drawn?

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this table without seeing it."""

            else:
                prompt = f"""Analyze this image and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Main Subject:** What is the primary focus or content?
2. **Key Elements:** Describe the important visual components
3. **Purpose:** What is this image meant to illustrate or teach?
4. **Educational Value:** What should a student learn from this?

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this image without seeing it."""
            
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_base64}"
                                }
                            ]
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result['choices'][0]['message']['content'].strip()
                
                # Remove markdown formatting if present
                summary = re.sub(r'\*\*.*?\*\*:?\s*', '', summary)
                summary = summary.strip()
                
                return summary, 0.85
            else:
                print(f"    Mistral summary generation error: {response.status_code}")
                
        except Exception as e:
            print(f"    Mistral summary generation failed: {e}")
        
        return None, 0.0
    
    def extract_mermaid_representation(self, image: Image.Image, segment: 'VisualSegment') -> Optional[MermaidRepresentation]:
        """
        Extract Mermaid diagram representation for flowcharts/diagrams
        This helps LLMs better understand the structure
        """
        if not self.api_key:
            return None
        
        # Only attempt for diagrams and flowcharts
        if segment.segment_type not in [VisualType.DIAGRAM, VisualType.FLOWCHART]:
            return None
        
        try:
            img_base64 = self._encode_image(image)
            
            ocr_context = ""
            if segment.ocr_result and segment.ocr_result.raw_text:
                ocr_context = f"\n\n**Text detected in diagram:**\n{segment.ocr_result.raw_text[:500]}"
            
            prompt = f"""Convert this {"flowchart" if segment.segment_type == VisualType.FLOWCHART else "diagram"} into Mermaid syntax.

**Instructions:**
1. Carefully identify all nodes/components and their text labels
2. Identify all connections/arrows and their directions
3. Choose appropriate Mermaid diagram type:
   - `graph TD` or `graph LR` for flowcharts (Top-Down or Left-Right)
   - `flowchart TD` or `flowchart LR` for detailed flowcharts with decision nodes
   - `graph` for simple diagrams
4. Use the detected text for node labels
5. Maintain the visual hierarchy and flow direction

{ocr_context}

**Example output format:**
```mermaid
flowchart TD
    A[Start Process] --> B{{Decision Point}}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```

**Response format:**
Provide ONLY the Mermaid code block, no additional explanation."""

            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_base64}"
                                }
                            ]
                        }
                    ],
                    "max_tokens": 800,
                    "temperature": 0.2
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Extract Mermaid code from response
                mermaid_match = re.search(r'```mermaid\s*(.*?)\s*```', content, re.DOTALL)
                if mermaid_match:
                    mermaid_code = mermaid_match.group(1).strip()
                    
                    # Detect diagram type from mermaid code
                    diagram_type = "graph"
                    if "flowchart" in mermaid_code[:50]:
                        diagram_type = "flowchart"
                    elif "sequenceDiagram" in mermaid_code[:50]:
                        diagram_type = "sequence"
                    
                    return MermaidRepresentation(
                        mermaid_code=mermaid_code,
                        diagram_type=diagram_type,
                        extraction_confidence=0.75,
                        extraction_notes="Extracted via Mistral vision model"
                    )
                else:
                    print(f"    No Mermaid code block found in response")
                    
        except Exception as e:
            print(f"    Mermaid extraction failed: {e}")
        
        return None


class CaptionDetector:
    """Detects figure captions and numbering"""
    
    CAPTION_PATTERNS = [
        r'Figure\s+(\d+(?:\.\d+)?)\s*[:\-]?\s*(.*?)(?=\n\n|\Z)',
        r'Fig\.\s+(\d+(?:\.\d+)?)\s*[:\-]?\s*(.*?)(?=\n\n|\Z)',
        r'Exhibit\s+(\d+(?:\.\d+)?)\s*[:\-]?\s*(.*?)(?=\n\n|\Z)',
        r'Chart\s+(\d+(?:\.\d+)?)\s*[:\-]?\s*(.*?)(?=\n\n|\Z)',
        r'Diagram\s+(\d+(?:\.\d+)?)\s*[:\-]?\s*(.*?)(?=\n\n|\Z)',
    ]
    
    @staticmethod
    def detect_caption(text_blocks: List[Dict], bbox: BoundingBox, 
                      page_height: float) -> Tuple[Optional[str], Optional[str]]:
        """Detect caption near the visual element"""
        caption_candidates = []
        
        for block in text_blocks:
            block_bbox = block.get('bbox', [0, 0, 0, 0])
            block_y = block_bbox[1]
            block_text = block.get('text', '').strip()
            
            if abs(block_y - bbox.y1) < 50 or abs(bbox.y0 - block_bbox[3]) < 50:
                caption_candidates.append(block_text)
        
        combined_text = ' '.join(caption_candidates)
        
        for pattern in CaptionDetector.CAPTION_PATTERNS:
            match = re.search(pattern, combined_text, re.IGNORECASE | re.DOTALL)
            if match:
                figure_number = match.group(1)
                caption_text = match.group(2).strip() if len(match.groups()) > 1 else ""
                return figure_number, caption_text
        
        if combined_text:
            return None, combined_text[:200]
        
        return None, None

class OCRProcessor:
    """Handles OCR extraction and structured field parsing"""
    
    @staticmethod
    def process_image(image: Image.Image) -> OCRResult:
        """Run OCR and extract structured information"""
        
        # Run Tesseract OCR with detailed output
        try:
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            raw_text = pytesseract.image_to_string(image)
        except Exception as e:
            return OCRResult(raw_text="", confidence=0.0)
        
        # Build blocks with bounding boxes
        blocks = []
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                blocks.append({
                    'text': ocr_data['text'][i],
                    'bbox': [
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    ],
                    'confidence': ocr_data['conf'][i]
                })
        
        avg_confidence = np.mean([b['confidence'] for b in blocks]) if blocks else 0.0
        
        # Detect chart-specific elements
        axis_labels = OCRProcessor._detect_axis_labels(raw_text, blocks)
        legend_items = OCRProcessor._detect_legend(raw_text)
        
        # Detect diagram elements
        node_texts = OCRProcessor._detect_nodes(blocks)
        arrow_count = OCRProcessor._count_arrows(image)
        
        return OCRResult(
            raw_text=raw_text,
            blocks=blocks,
            confidence=avg_confidence / 100.0,
            axis_labels=axis_labels,
            legend_items=legend_items,
            node_texts=node_texts,
            detected_arrows=arrow_count
        )
    
    @staticmethod
    def _detect_axis_labels(text: str, blocks: List[Dict]) -> Dict[str, str]:
        """Attempt to identify axis labels"""
        labels = {}
        # Simple heuristic: look for common patterns
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['year', 'time', 'date']):
                labels['x'] = line.strip()
            elif any(keyword in line.lower() for keyword in ['value', 'price', 'amount', '%']):
                labels['y'] = line.strip()
        return labels
    
    @staticmethod
    def _detect_legend(text: str) -> List[str]:
        """Detect legend entries"""
        # Look for series/legend patterns
        legend_items = []
        lines = text.split('\n')
        for line in lines:
            if len(line.strip()) > 0 and len(line.strip()) < 50:
                legend_items.append(line.strip())
        return legend_items[:10]  # Limit to reasonable number
    
    @staticmethod
    def _detect_nodes(blocks: List[Dict]) -> List[str]:
        """Extract text blocks that could be diagram nodes"""
        nodes = []
        for block in blocks:
            text = block['text'].strip()
            if 3 < len(text) < 50:  # Reasonable node text length
                nodes.append(text)
        return nodes
    
    @staticmethod
    def _count_arrows(image: Image.Image) -> int:
        """Count arrow-like shapes (simplified detection)"""
        img_array = np.array(image.convert('L'))
        edges = cv2.Canny(img_array, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=30, maxLineGap=10)
        
        # Rough estimate: count diagonal lines as potential arrows
        if lines is None:
            return 0
        
        arrow_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 20 < angle < 70 or 110 < angle < 160:
                arrow_count += 1
        
        return min(arrow_count // 3, 20)  # Normalize

class ConceptLinker:
    """Links visual segments to concepts from taxonomy"""
    
    def __init__(self, taxonomy_df: pd.DataFrame):
        """
        Initialize with taxonomy DataFrame.
        Expected columns: Level, Concept, Tag(s), Rationale, Page(s)
        """
        self.taxonomy_df = taxonomy_df
        self._build_concept_index()
    
    def _build_concept_index(self):
        """Build search index for concepts with correct column names"""
        self.concept_map = {}
        
        # Print available columns for debugging
        print(f"Available columns in taxonomy: {list(self.taxonomy_df.columns)}")
        
        for idx, row in self.taxonomy_df.iterrows():
            # Use the correct column names: Level, Concept, Tag(s), Rationale, Page(s)
            concept_name = row.get('Concept', '')
            
            if not concept_name:
                continue
            
            # Generate concept_id dynamically from concept name
            concept_id = self._generate_concept_id(concept_name, idx)
            
            # Store normalized concept name
            norm_name = concept_name.lower().strip()
            self.concept_map[norm_name] = {
                'concept_id': concept_id,
                'concept_name': concept_name,
                'bloom_level': row.get('Level', ''),
                'tag': row.get('Tag(s)', ''),
                'pages': row.get('Page(s)', '')
            }
            
            # Create n-grams for multi-word concepts (helps with partial matching)
            words = concept_name.split()
            if len(words) > 1:
                # Store each significant word as a key (helps catch partial matches)
                for word in words:
                    word_norm = word.lower().strip()
                    # Only store words longer than 3 chars to avoid noise
                    if len(word_norm) > 3 and word_norm not in ['and', 'the', 'with', 'from']:
                        if word_norm not in self.concept_map:
                            self.concept_map[word_norm] = self.concept_map[norm_name]
            
            # Handle tags as potential aliases
            tags = row.get('Tag(s)', '')
            if pd.notna(tags) and tags:
                for tag in str(tags).split(','):
                    tag_norm = tag.lower().strip()
                    if tag_norm and len(tag_norm) > 2:
                        self.concept_map[tag_norm] = self.concept_map[norm_name]
        
        print(f"Built concept index with {len(self.concept_map)} entries")
    
    def _generate_concept_id(self, concept_name: str, index: int) -> str:
        """
        Generate a unique concept_id from concept name.
        Format: concept_<normalized_name>_<index>
        Example: 'Investment Banker' -> 'concept_investment_banker_042'
        """
        # Normalize the concept name
        normalized = concept_name.lower().strip()
        
        # Replace spaces and special chars with underscores
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'[-\s]+', '_', normalized)
        
        # Limit length to 50 chars
        if len(normalized) > 50:
            normalized = normalized[:50]
        
        # Create ID with index for uniqueness
        concept_id = f"concept_{normalized}_{index:03d}"
        
        return concept_id
    
    def link_concepts(self, segment: VisualSegment) -> List[Dict[str, Any]]:
        """
        Link segment to relevant concepts with enhanced matching.
        Returns list of {concept_id, confidence, match_method}
        """
        links = []
        
        # Collect searchable text
        search_texts = []
        if segment.caption_text:
            search_texts.append(segment.caption_text)
        if segment.ocr_result and segment.ocr_result.raw_text:
            search_texts.append(segment.ocr_result.raw_text)
        if segment.summary:
            search_texts.append(segment.summary)
        
        combined_text = ' '.join(search_texts).lower()
        
        # Clean and normalize the text
        combined_text = self._clean_text(combined_text)
        
        # Strategy 1: Exact phrase matching (highest confidence)
        for concept_key, concept_data in self.concept_map.items():
            if len(concept_key) > 3 and concept_key in combined_text:
                # Check if this is the full concept name or just a word fragment
                is_full_concept = (concept_key == concept_data['concept_name'].lower().strip())
                
                confidence = 0.95 if is_full_concept else 0.75
                match_method = 'exact_match' if is_full_concept else 'partial_match'
                
                links.append({
                    'concept_id': concept_data['concept_id'],
                    'concept_name': concept_data['concept_name'],
                    'bloom_level': concept_data['bloom_level'],
                    'tag': concept_data['tag'],
                    'pages': concept_data.get('pages', ''),
                    'confidence': confidence,
                    'match_method': match_method
                })
        
        # Strategy 2: Word-based fuzzy matching for multi-word concepts
        combined_words = set(combined_text.split())
        
        for concept_key, concept_data in self.concept_map.items():
            concept_words = set(concept_key.split())
            
            # Skip single-word concepts (already caught by exact matching)
            if len(concept_words) <= 1:
                continue
            
            # Calculate word overlap
            matching_words = concept_words.intersection(combined_words)
            overlap_ratio = len(matching_words) / len(concept_words)
            
            # If significant overlap and not already matched
            if overlap_ratio >= 0.6:  # At least 60% of concept words found
                # Check if already added
                already_added = any(
                    link['concept_id'] == concept_data['concept_id'] 
                    for link in links
                )
                
                if not already_added:
                    confidence = min(0.7, overlap_ratio)
                    links.append({
                        'concept_id': concept_data['concept_id'],
                        'concept_name': concept_data['concept_name'],
                        'bloom_level': concept_data['bloom_level'],
                        'tag': concept_data['tag'],
                        'pages': concept_data.get('pages', ''),
                        'confidence': confidence,
                        'match_method': 'fuzzy_match'
                    })
        
        # Strategy 3: Stem-based matching (for variations like "invest" vs "investment")
        if len(links) < 3:  # Only if we haven't found many matches
            links.extend(self._stem_based_matching(combined_text, links))
        
        # Remove duplicates and sort by confidence
        unique_links = self._deduplicate_links(links)
        unique_links.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Found {len(unique_links)} concept links")
        for link in unique_links[:5]:  # Print top 5
            print(f"  - {link['concept_name']} ({link['confidence']:.2f}, {link['match_method']})")
        
        return unique_links
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better matching"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Remove standalone single characters (OCR noise)
        text = re.sub(r'\b\w\b', '', text)
        
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _stem_based_matching(self, text: str, existing_links: List[Dict]) -> List[Dict]:
        """Match based on word stems (e.g., 'invest' matches 'investment')"""
        additional_links = []
        
        # Simple stemming: remove common suffixes
        def simple_stem(word: str) -> str:
            suffixes = ['ing', 'ed', 'er', 's', 'ment', 'tion', 'ness', 'ity']
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 3:
                    return word[:-len(suffix)]
            return word
        
        text_stems = {simple_stem(word) for word in text.split() if len(word) > 3}
        
        for concept_key, concept_data in self.concept_map.items():
            # Skip if already matched
            if any(link['concept_id'] == concept_data['concept_id'] for link in existing_links):
                continue
            
            concept_stems = {simple_stem(word) for word in concept_key.split() if len(word) > 3}
            
            # Check stem overlap
            matching_stems = text_stems.intersection(concept_stems)
            if matching_stems and len(matching_stems) >= len(concept_stems) * 0.5:
                additional_links.append({
                    'concept_id': concept_data['concept_id'],
                    'concept_name': concept_data['concept_name'],
                    'bloom_level': concept_data['bloom_level'],
                    'tag': concept_data['tag'],
                    'pages': concept_data.get('pages', ''),
                    'confidence': 0.6,
                    'match_method': 'stem_match'
                })
        
        return additional_links
    
    def _deduplicate_links(self, links: List[Dict]) -> List[Dict]:
        """Remove duplicate concepts, keeping highest confidence"""
        seen = {}
        
        for link in links:
            concept_id = link['concept_id']
            if concept_id not in seen or link['confidence'] > seen[concept_id]['confidence']:
                seen[concept_id] = link
        
        return list(seen.values())

class VisualSegmentationPipeline:
    """Main pipeline with Mistral API integration"""
    
    def __init__(self, book_id: str, pdf_path: str, 
                 taxonomy_path: Optional[str] = None,
                 output_dir: str = "./output",
                 use_mermaid: bool = True):
        self.book_id = book_id
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mermaid = use_mermaid
        
        # Initialize Mistral API
        self.mistral_api = MistralVisionAPI()
        # api_key="aE9nzpmp8JWHHguJMTurCrDrJCTfodaP"
        
        # Load taxonomy if provided
        self.concept_linker = None
        if taxonomy_path and Path(taxonomy_path).exists():
            taxonomy_df = pd.read_excel(taxonomy_path)
            from __main__ import ConceptLinker  # Import from main module
            self.concept_linker = ConceptLinker(taxonomy_df)
        
        self.segments: List[VisualSegment] = []
    
    def process(self) -> List[VisualSegment]:
        """
        Run complete pipeline.
        Returns list of visual segments.
        """
        print(f"Processing PDF: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(len(doc)):
            print(f"Processing page {page_num + 1}/{len(doc)}...")
            page = doc[page_num]
            
            # Extract images using PyMuPDF
            page_segments = self._extract_images_from_page(page, page_num)
            
            # Process each segment
            for segment in page_segments:
                self._process_segment(segment, page, doc)
            
            self.segments.extend(page_segments)
        
        doc.close()
        
        # Save results
        self._save_results()
        
        print(f"\nExtraction complete! Found {len(self.segments)} visual elements.")
        return self.segments

    def _extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[VisualSegment]:
        """
        Two-pass extraction with smart conflict resolution:
        Pass 1: Caption-based detection (high confidence)
        Pass 2: Direct image extraction with validation
        """
        segments = []
        
        # PASS 1: Caption-based detection (PRIMARY - high confidence)
        caption_regions = self._detect_by_captions(page, page.rect)
        
        print(f"  Pass 1: Found {len(caption_regions)} caption-based regions")
        
        for idx, region in enumerate(caption_regions):
            try:
                image, image_bytes = self._render_region(page, region['bbox'])
                if image is None:
                    continue
                
                segment_id = self._generate_segment_id(page_num, region['bbox'], image_bytes)
                image_filename = f"{segment_id}.png"
                image_path = self.output_dir / image_filename
                image.save(image_path)
                
                segment = VisualSegment(
                    segment_id=segment_id,
                    segment_type=VisualType.UNKNOWN,
                    book_id=self.book_id,
                    page_no=page_num + 1,
                    bbox=region['bbox'],
                    image_path=str(image_path),
                    image_bytes=image_bytes,
                    extraction_method='caption_based',
                    caption_text=region.get('caption'),
                    notes=region.get('notes', ''),
                    confidence=0.9  # High confidence from captions
                )
                
                # Extract figure number
                if segment.caption_text:
                    for pattern in CaptionDetector.CAPTION_PATTERNS:
                        match = re.search(pattern, segment.caption_text, re.IGNORECASE)
                        if match:
                            segment.figure_number = match.group(1)
                            segment.reference_keys = [
                                f"Figure {match.group(1)}",
                                f"Fig. {match.group(1)}",
                                f"Fig {match.group(1)}"
                            ]
                            break
                
                segments.append(segment)
                
            except Exception as e:
                print(f"    Warning: Could not render caption region {idx}: {e}")
                continue
        
        # PASS 2: Direct embedded image extraction with validation
        print(f"  Pass 2: Checking for embedded images...")
        embedded_candidates = self._extract_embedded_images_validated(page, page_num)
        
        print(f"  Pass 2: Found {len(embedded_candidates)} embedded image candidates")
        
        # For each embedded candidate, decide: keep it, or use caption-based version?
        for candidate in embedded_candidates:
            # Check for conflicts with caption-based segments
            conflict = self._find_conflicting_segment(candidate, segments)
            
            if conflict:
                # There's overlap - decide which to keep
                decision, reason = self._resolve_conflict(candidate, conflict, page)
                
                if decision == "keep_embedded":
                    print(f"    Replacing caption-based with embedded image: {reason}")
                    segments.remove(conflict)
                    segments.append(candidate)
                else:
                    print(f"    Keeping caption-based, discarding embedded: {reason}")
                    # Don't add candidate
            else:
                # No conflict - this is a figure without a detected caption
                # Could be a photo, unlabeled diagram, etc.
                print(f"    Adding embedded image (no caption detected)")
                segments.append(candidate)
        
        print(f"  Final: {len(segments)} visual segments on page {page_num + 1}")
        return segments
    
    def _extract_embedded_images_validated(self, page: fitz.Page, page_num: int) -> List[VisualSegment]:
        """
        Extract embedded images with strict validation to reduce false positives.
        Only keeps images that are likely to be meaningful figures.
        """
        candidates = []
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                continue
            
            # Get image position
            img_rects = page.get_image_rects(xref)
            if not img_rects:
                continue
            
            rect = img_rects[0]
            bbox = BoundingBox(
                x0=rect.x0, y0=rect.y0,
                x1=rect.x1, y1=rect.y1,
                page_width=page.rect.width,
                page_height=page.rect.height
            )
            
            # VALIDATION: Check if this is a meaningful figure
            validation_score, validation_notes = self._validate_embedded_image(image, bbox, page)
            
            if validation_score < 0.5:  # Threshold for keeping
                continue
            
            # Try to find caption nearby (even if caption-based detection missed it)
            caption_text = self._find_caption_near_bbox(page, bbox)
            
            # If caption found, expand bbox to include it
            if caption_text:
                text_blocks = self._extract_text_blocks(page)
                for block in text_blocks:
                    if caption_text[:30] in block['text']:
                        caption_bbox = block['bbox']
                        bbox = BoundingBox(
                            x0=min(bbox.x0, caption_bbox[0]),
                            y0=bbox.y0,
                            x1=max(bbox.x1, caption_bbox[2]),
                            y1=max(bbox.y1, caption_bbox[3]),
                            page_width=bbox.page_width,
                            page_height=bbox.page_height
                        )
                        # Re-render with caption
                        image, image_bytes = self._render_region(page, bbox)
                        break
            
            # Generate segment
            segment_id = self._generate_segment_id(page_num, bbox, image_bytes)
            image_filename = f"{segment_id}.png"
            image_path = self.output_dir / image_filename
            image.save(image_path)
            
            segment = VisualSegment(
                segment_id=segment_id,
                segment_type=VisualType.UNKNOWN,
                book_id=self.book_id,
                page_no=page_num + 1,
                bbox=bbox,
                image_path=str(image_path),
                image_bytes=image_bytes,
                extraction_method='embedded_image',
                caption_text=caption_text,
                confidence=validation_score,
                notes=f"Validation: {validation_notes}"
            )
            
            candidates.append(segment)
        
        return candidates
    
    def _validate_embedded_image(self, image: Image.Image, bbox: BoundingBox, 
                                 page: fitz.Page) -> Tuple[float, str]:
        """
        Validate if embedded image is a meaningful figure (not logo, icon, decoration).
        Returns (confidence_score, notes)
        """
        score = 0.0
        notes = []
        
        # Check 1: Size validation
        area = bbox.area()
        if area < 3000:  # Very small
            return 0.0, "too_small"
        elif area > 10000:
            score += 0.3
            notes.append("good_size")
        else:
            score += 0.1
            notes.append("moderate_size")
        
        # Check 2: Dimension validation
        if image.width < 50 or image.height < 50:
            return 0.0, "tiny_dimensions"
        
        if image.width > 200 and image.height > 200:
            score += 0.2
            notes.append("substantial_dimensions")
        
        # Check 3: Aspect ratio (very wide/tall images might be decorations)
        aspect_ratio = image.width / image.height if image.height > 0 else 1.0
        if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
            score += 0.2
            notes.append("good_aspect_ratio")
        else:
            score -= 0.1
            notes.append("unusual_aspect_ratio")
        
        # Check 4: Position on page (avoid headers/footers)
        page_height = page.rect.height
        y_position = bbox.y0 / page_height
        
        if y_position < 0.1 or y_position > 0.9:  # Top/bottom 10%
            score -= 0.2
            notes.append("likely_header_footer")
        else:
            score += 0.1
            notes.append("good_position")
        
        # Check 5: Check if near a caption (strong signal)
        nearby_caption = self._find_caption_near_bbox(page, bbox)
        if nearby_caption:
            score += 0.4
            notes.append("has_caption")
        
        # Check 6: Image content analysis (simple)
        img_array = np.array(image.convert('L'))
        variance = np.var(img_array)
        
        if variance < 10:  # Nearly uniform color (likely decoration)
            score -= 0.3
            notes.append("low_variance")
        elif variance > 100:  # Good content variance
            score += 0.2
            notes.append("good_content_variance")
        
        return min(score, 1.0), ", ".join(notes)
    
    def _find_caption_near_bbox(self, page: fitz.Page, bbox: BoundingBox) -> Optional[str]:
        """Find caption text near a bbox (within 60 points below)"""
        text_blocks = self._extract_text_blocks(page)
        
        for block in text_blocks:
            block_bbox = block['bbox']
            
            # Check if block is below the image
            vertical_distance = block_bbox[1] - bbox.y1
            horizontal_overlap = (min(bbox.x1, block_bbox[2]) - max(bbox.x0, block_bbox[0]))
            
            if 0 <= vertical_distance <= 60 and horizontal_overlap > 0:
                text = block['text']
                # Check for caption patterns
                for pattern in CaptionDetector.CAPTION_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        return text
        
        return None
    
    def _find_conflicting_segment(self, candidate: VisualSegment, 
                                  existing_segments: List[VisualSegment]) -> Optional[VisualSegment]:
        """Find if candidate overlaps significantly with existing segments"""
        for segment in existing_segments:
            overlap_ratio = self._calculate_overlap_ratio(candidate.bbox, segment.bbox)
            if overlap_ratio > 0.4:  # Significant overlap
                return segment
        return None
    
    def _calculate_overlap_ratio(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap as ratio of smaller bbox area"""
        x_overlap = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0))
        y_overlap = max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))
        overlap_area = x_overlap * y_overlap
        
        area1 = bbox1.area()
        area2 = bbox2.area()
        smaller_area = min(area1, area2)
        
        return overlap_area / smaller_area if smaller_area > 0 else 0
    
    def _resolve_conflict(self, embedded: VisualSegment, caption_based: VisualSegment,
                         page: fitz.Page) -> Tuple[str, str]:
        """
        Decide which segment to keep when there's overlap.
        Returns ("keep_embedded" or "keep_caption", reason)
        """
        reasons = []
        embedded_score = 0
        caption_score = 0
        
        # Factor 1: Caption presence (strong signal for caption-based)
        if caption_based.caption_text:
            caption_score += 3
            reasons.append("caption_based has caption")
        
        # Factor 2: Check if embedded image is JUST the image without caption
        # while caption-based includes both image + caption (preferred)
        embedded_area = embedded.bbox.area()
        caption_area = caption_based.bbox.area()
        
        if caption_area > embedded_area * 1.2:  # Caption-based is notably larger
            caption_score += 2
            reasons.append("caption_based includes more context")
        elif embedded_area > caption_area * 1.2:
            embedded_score += 1
            reasons.append("embedded is larger")
        
        # Factor 3: Check if embedded is actual raster image (photos prefer embedded)
        if embedded.extraction_method == 'embedded_image':
            # Check if it's photo-like
            img = Image.open(embedded.image_path)
            img_array = np.array(img.convert('L'))
            variance = np.var(img_array)
            
            if variance > 1000:  # Photo-like
                embedded_score += 2
                reasons.append("embedded is photo-like (raster)")
        
        # Factor 4: Drawing commands suggest vector graphics (prefer caption-based rendering)
        drawings = page.get_drawings()
        drawings_in_caption_region = 0
        
        for drawing in drawings:
            draw_rect = drawing.get('rect', fitz.Rect(0, 0, 0, 0))
            # Check if drawing overlaps with caption-based bbox
            if (caption_based.bbox.x0 <= draw_rect.x0 <= caption_based.bbox.x1 and
                caption_based.bbox.y0 <= draw_rect.y0 <= caption_based.bbox.y1):
                drawings_in_caption_region += 1
        
        if drawings_in_caption_region > 10:  # Many drawing commands = chart/diagram
            caption_score += 2
            reasons.append("many vector drawings (chart/diagram)")
        
        # Factor 5: Validation score for embedded
        if embedded.confidence > 0.7:
            embedded_score += 1
            reasons.append(f"embedded has high validation ({embedded.confidence:.2f})")
        
        # Decision
        if caption_score > embedded_score:
            return "keep_caption", "; ".join(reasons)
        else:
            return "keep_embedded", "; ".join(reasons)
    
    def _detect_visual_regions(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """
        Detect visual element regions using multiple strategies:
        1. Caption-based detection (look for "Figure X", "Chart X", etc.)
        2. Non-text block detection (areas with drawings/vectors)
        3. Hybrid approach combining both
        """
        visual_regions = []
        page_rect = page.rect
        
        # Strategy 1: Caption-based detection
        caption_regions = self._detect_by_captions(page, page_rect)
        visual_regions.extend(caption_regions)
        
        # Strategy 2: Non-text block detection (for visuals without clear captions)
        drawing_regions = self._detect_by_drawings(page, page_rect)
        
        # Merge regions that don't overlap with caption-detected regions
        # Also check if drawing region contains a caption to avoid duplicates
        for draw_region in drawing_regions:
            # Check overlap with caption-detected regions
            if self._overlaps_with_existing(draw_region['bbox'], visual_regions):
                continue
            
            # Check if this drawing region contains any of the detected captions
            # This prevents detecting the same figure twice (once with caption, once without)
            contains_caption = False
            for cap_region in caption_regions:
                if 'caption_bbox' in cap_region:
                    cap_bbox = cap_region['caption_bbox']
                    # Check if caption is within or near this drawing region
                    draw_bbox = draw_region['bbox']
                    # Caption is "near" if it's within 50 points below the drawing region
                    if (draw_bbox.x0 <= cap_bbox[0] <= draw_bbox.x1 and
                        draw_bbox.y1 - 50 <= cap_bbox[1] <= draw_bbox.y1 + 50):
                        contains_caption = True
                        break
            
            if not contains_caption:
                visual_regions.append(draw_region)
        
        return visual_regions
    
    def _detect_by_captions(self, page: fitz.Page, page_rect: fitz.Rect) -> List[Dict]:
        """
        Detect visual regions by finding caption patterns and determining bbox.
        Only detects actual captions, not in-text references.
        """
        regions = []
        text_dict = page.get_text("dict")
        
        # Find all caption text blocks
        caption_blocks = []
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                
                block_text = block_text.strip()
                
                # Check if this block contains a caption pattern
                caption_match = None
                for pattern in CaptionDetector.CAPTION_PATTERNS:
                    match = re.search(pattern, block_text, re.IGNORECASE)
                    if match:
                        caption_match = match
                        break
                
                if caption_match:
                    # CRITICAL: Verify this is an actual caption, not an in-text reference
                    # Real captions typically:
                    # 1. Start with "Figure X" (at beginning of block)
                    # 2. Are relatively short (not full paragraphs)
                    # 3. Don't contain phrases like "as shown in", "see Figure", "in Figure"
                    
                    # Check if pattern match is at the start of the block
                    match_position = caption_match.start()
                    is_at_start = match_position < 20  # Allow some leading whitespace/punctuation
                    
                    # Check for reference phrases (indicates this is body text, not a caption)
                    reference_phrases = [
                        r'\bas shown in\b',
                        r'\bsee figure\b',
                        r'\bin figure\b',
                        r'\brefer to\b',
                        r'\baccording to\b',
                        r'\bas illustrated in\b',
                        r'\bas depicted in\b'
                    ]
                    
                    has_reference_phrase = any(
                        re.search(phrase, block_text.lower()) 
                        for phrase in reference_phrases
                    )
                    
                    # Check block length (captions are typically < 300 chars, paragraphs are longer)
                    is_short_enough = len(block_text) < 400
                    
                    # Calculate block width (captions often narrower than full paragraphs)
                    block_width = block_bbox[2] - block_bbox[0]
                    page_width = page_rect.width
                    
                    # Only consider it a caption if:
                    # - Starts near beginning of block
                    # - No reference phrases
                    # - Reasonably short
                    if is_at_start and not has_reference_phrase and is_short_enough:
                        caption_blocks.append({
                            'text': block_text,
                            'bbox': block_bbox
                        })
        
        # For each caption, find the visual region (typically above the caption)
        for caption_block in caption_blocks:
            caption_bbox = caption_block['bbox']
            
            # Define search region (area above caption)
            # Typical figures are 100-400 points tall
            search_y_start = max(0, caption_bbox[1] - 500)
            search_y_end = caption_bbox[1]
            
            # Find the visual content region (INCLUDING the caption)
            visual_bbox = self._find_visual_content_above(
                page, search_y_start, search_y_end, caption_bbox
            )
            
            if visual_bbox:
                # Extend bbox to include the caption
                visual_bbox_with_caption = BoundingBox(
                    x0=min(visual_bbox.x0, caption_bbox[0]),
                    y0=visual_bbox.y0,
                    x1=max(visual_bbox.x1, caption_bbox[2]),
                    y1=caption_bbox[3] + 5,  # Include caption bottom with small padding
                    page_width=visual_bbox.page_width,
                    page_height=visual_bbox.page_height
                )
                
                regions.append({
                    'bbox': visual_bbox_with_caption,
                    'caption': caption_block['text'],
                    'detection_method': 'caption_based',
                    'notes': f'Detected via caption: {caption_block["text"][:50]}',
                    'caption_bbox': caption_bbox  # Store for deduplication
                })
        
        return regions
    
    def _find_visual_content_above(self, page: fitz.Page, 
                                   y_start: float, y_end: float,
                                   caption_bbox: List[float]) -> Optional[BoundingBox]:
        """
        Find the visual content region above a caption with smart boundary detection.
        Uses multiple signals to find the true visual edges.
        """
        page_rect = page.rect
        
        # Collect all potential boundary signals
        boundaries = {
            'drawing_bounds': None,
            'image_bounds': None,
            'whitespace_boundary': None,
            'text_boundary': None
        }
        
        # SIGNAL 1: Drawing commands (most reliable for vector graphics)
        drawings = page.get_drawings()
        drawings_in_region = []
        
        for drawing in drawings:
            draw_rect = drawing.get('rect', fitz.Rect(0, 0, 0, 0))
            if y_start <= draw_rect.y0 < y_end:
                drawings_in_region.append(draw_rect)
        
        if drawings_in_region:
            x0 = min(r.x0 for r in drawings_in_region)
            y0 = min(r.y0 for r in drawings_in_region)
            x1 = max(r.x1 for r in drawings_in_region)
            y1 = max(r.y1 for r in drawings_in_region)
            boundaries['drawing_bounds'] = (x0, y0, x1, y1)
        
        # SIGNAL 2: Embedded images in region
        image_list = page.get_images(full=True)
        images_in_region = []
        
        for img_info in image_list:
            xref = img_info[0]
            img_rects = page.get_image_rects(xref)
            for rect in img_rects:
                if y_start <= rect.y0 < y_end:
                    images_in_region.append(rect)
        
        if images_in_region:
            x0 = min(r.x0 for r in images_in_region)
            y0 = min(r.y0 for r in images_in_region)
            x1 = max(r.x1 for r in images_in_region)
            y1 = max(r.y1 for r in images_in_region)
            boundaries['image_bounds'] = (x0, y0, x1, y1)
        
        # SIGNAL 3: Whitespace analysis - find largest vertical gap
        whitespace_boundary = self._find_whitespace_boundary(page, y_start, y_end, caption_bbox)
        if whitespace_boundary:
            boundaries['whitespace_boundary'] = whitespace_boundary
        
        # SIGNAL 4: Text analysis - distinguish body paragraphs from figure labels
        text_boundary = self._find_text_boundary(page, y_start, y_end, caption_bbox, page_rect)
        if text_boundary:
            boundaries['text_boundary'] = text_boundary
        
        # DECISION LOGIC: Combine signals to find best bbox
        final_bbox = self._combine_boundary_signals(boundaries, caption_bbox, page_rect, y_start, y_end)
        
        return final_bbox
    
    def _find_whitespace_boundary(self, page: fitz.Page, y_start: float, 
                                  y_end: float, caption_bbox: List[float]) -> Optional[Tuple]:
        """
        Analyze vertical whitespace to find natural boundary between content.
        Large gaps indicate separation between body text and figure.
        """
        text_dict = page.get_text("dict")
        
        # Get all text block positions in the region
        block_positions = []
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                if y_start <= block_bbox[1] < y_end:
                    block_positions.append({
                        'y_top': block_bbox[1],
                        'y_bottom': block_bbox[3],
                        'x0': block_bbox[0],
                        'x1': block_bbox[2]
                    })
        
        if not block_positions:
            return None
        
        # Sort by vertical position
        block_positions.sort(key=lambda b: b['y_bottom'])
        
        # Find the largest vertical gap
        largest_gap = 0
        gap_position = None
        
        for i in range(len(block_positions) - 1):
            gap = block_positions[i+1]['y_top'] - block_positions[i]['y_bottom']
            if gap > largest_gap and gap > 20:  # Minimum gap threshold
                largest_gap = gap
                gap_position = block_positions[i]['y_bottom']
        
        # If we found a significant gap, the figure starts after it
        if gap_position and largest_gap > 30:  # Significant whitespace
            return (None, gap_position + 5, None, None)  # Only y_start boundary
        
        return None
    
    def _find_text_boundary(self, page: fitz.Page, y_start: float, y_end: float,
                           caption_bbox: List[float], page_rect: fitz.Rect) -> Optional[Tuple]:
        """
        Analyze text blocks to distinguish body paragraphs from figure labels.
        Body paragraphs are wide, dense, and left-aligned.
        Figure labels are short, scattered, and can be anywhere.
        """
        text_dict = page.get_text("dict")
        body_paragraphs = []
        figure_text = []
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                
                if not (y_start <= block_bbox[1] < y_end):
                    continue
                
                # Extract text
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                
                block_text = block_text.strip()
                block_width = block_bbox[2] - block_bbox[0]
                block_height = block_bbox[3] - block_bbox[1]
                text_length = len(block_text)
                
                # STRICT body paragraph criteria
                is_body_paragraph = (
                    block_width > page_rect.width * 0.65 and  # Very wide (65%+)
                    text_length > 120 and  # Substantial text
                    block_height > 35 and  # Multi-line
                    block_bbox[0] < page_rect.width * 0.15  # Left-aligned
                )
                
                if is_body_paragraph:
                    body_paragraphs.append(block_bbox)
                else:
                    # Could be figure label, axis label, etc.
                    figure_text.append(block_bbox)
        
        # If we found body paragraphs, figure starts after the last one
        if body_paragraphs:
            body_paragraphs.sort(key=lambda b: b[3], reverse=True)
            last_para_bottom = body_paragraphs[0][3]
            
            # Add buffer to ensure we don't clip figure
            y_top = last_para_bottom + 20
            
            # Get horizontal bounds from figure text if available
            if figure_text:
                x0 = min(b[0] for b in figure_text)
                x1 = max(b[2] for b in figure_text)
                return (x0, y_top, x1, None)
            else:
                return (None, y_top, None, None)
        
        return None
    
    def _combine_boundary_signals(self, boundaries: Dict, caption_bbox: List[float],
                                  page_rect: fitz.Rect, y_start: float, 
                                  y_end: float) -> Optional[BoundingBox]:
        """
        Intelligently combine all boundary signals to find optimal bbox.
        Priority: drawing_bounds > image_bounds > whitespace > text > fallback
        """
        
        # STRATEGY 1: If we have drawing bounds, use them (most accurate for charts)
        if boundaries['drawing_bounds']:
            x0, y0, x1, y1 = boundaries['drawing_bounds']
            
            # Apply whitespace boundary if it makes sense
            if boundaries['whitespace_boundary'] and boundaries['whitespace_boundary'][1]:
                ws_y_start = boundaries['whitespace_boundary'][1]
                if ws_y_start > y0:  # Whitespace suggests later start
                    y0 = ws_y_start
            
            # Minimal padding
            x0 = max(0, x0 - 10)
            y0 = max(y_start, y0 - 10)
            x1 = min(page_rect.width, x1 + 10)
            y1 = min(caption_bbox[1] - 5, y1 + 10)
            
            if x1 - x0 > 50 and y1 - y0 > 50:
                return BoundingBox(x0, y0, x1, y1, page_rect.width, page_rect.height)
        
        # STRATEGY 2: If we have embedded images, use them
        if boundaries['image_bounds']:
            x0, y0, x1, y1 = boundaries['image_bounds']
            
            # Check whitespace boundary
            if boundaries['whitespace_boundary'] and boundaries['whitespace_boundary'][1]:
                ws_y_start = boundaries['whitespace_boundary'][1]
                if ws_y_start > y0:
                    y0 = ws_y_start
            
            # Minimal padding
            x0 = max(0, x0 - 5)
            y0 = max(y_start, y0 - 5)
            x1 = min(page_rect.width, x1 + 5)
            y1 = min(caption_bbox[1] - 5, y1 + 5)
            
            if x1 - x0 > 50 and y1 - y0 > 50:
                return BoundingBox(x0, y0, x1, y1, page_rect.width, page_rect.height)
        
        # STRATEGY 3: Use whitespace boundary if available
        if boundaries['whitespace_boundary']:
            ws = boundaries['whitespace_boundary']
            
            x0 = ws[0] if ws[0] is not None else max(0, caption_bbox[0] - 30)
            y0 = ws[1] if ws[1] is not None else y_start
            x1 = ws[2] if ws[2] is not None else min(page_rect.width, caption_bbox[2] + 30)
            y1 = caption_bbox[1] - 10
            
            if x1 - x0 > 80 and y1 - y0 > 60:
                return BoundingBox(x0, y0, x1, y1, page_rect.width, page_rect.height)
        
        # STRATEGY 4: Use text boundary analysis
        if boundaries['text_boundary']:
            tb = boundaries['text_boundary']
            
            x0 = tb[0] if tb[0] is not None else max(0, caption_bbox[0] - 30)
            y0 = tb[1] if tb[1] is not None else y_start
            x1 = tb[2] if tb[2] is not None else min(page_rect.width, caption_bbox[2] + 30)
            y1 = caption_bbox[1] - 10
            
            if x1 - x0 > 80 and y1 - y0 > 60:
                return BoundingBox(x0, y0, x1, y1, page_rect.width, page_rect.height)
        
        # FALLBACK: Conservative approach
        # Use caption width as guide, but be more conservative with height
        x0 = max(0, caption_bbox[0] - 20)
        x1 = min(page_rect.width, caption_bbox[2] + 20)
        
        # For y, if no other signals, use a reasonable default (200 points above caption)
        y0 = max(y_start, caption_bbox[1] - 250)
        y1 = caption_bbox[1] - 10
        
        # Only return if dimensions are reasonable
        if x1 - x0 > 100 and y1 - y0 > 80 and (y1 - y0) < 500:
            return BoundingBox(x0, y0, x1, y1, page_rect.width, page_rect.height)
        
        return None
    
    def _detect_by_drawings(self, page: fitz.Page, page_rect: fitz.Rect) -> List[Dict]:
        """
        Detect visual regions by analyzing drawing commands and vector content.
        This catches figures/charts without clear captions.
        """
        regions = []
        
        # Get drawing commands
        drawings = page.get_drawings()
        
        if not drawings:
            return regions
        
        # Cluster drawings into regions
        drawing_clusters = self._cluster_drawings(drawings, page_rect)
        
        for cluster in drawing_clusters:
            # Create bbox from cluster
            x0 = min(d['rect'][0] for d in cluster)
            y0 = min(d['rect'][1] for d in cluster)
            x1 = max(d['rect'][2] for d in cluster)
            y1 = max(d['rect'][3] for d in cluster)
            
            # Add padding
            padding = 10
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(page_rect.width, x1 + padding)
            y1 = min(page_rect.height, y1 + padding)
            
            bbox = BoundingBox(
                x0=x0, y0=y0, x1=x1, y1=y1,
                page_width=page_rect.width,
                page_height=page_rect.height
            )
            
            # Filter by size (skip very small or very large regions)
            area = bbox.area()
            if 5000 < area < page_rect.width * page_rect.height * 0.8:
                regions.append({
                    'bbox': bbox,
                    'caption': None,
                    'detection_method': 'drawing_based',
                    'notes': f'Detected from {len(cluster)} drawing commands'
                })
        
        return regions
    
    def _cluster_drawings(self, drawings: List[Dict], page_rect: fitz.Rect,
                         distance_threshold: float = 100) -> List[List[Dict]]:
        """
        Cluster nearby drawings into groups (likely parts of same figure).
        """
        if not drawings:
            return []
        
        # Simple clustering: group drawings that are close to each other
        clusters = []
        used = set()
        
        for i, draw1 in enumerate(drawings):
            if i in used:
                continue
            
            cluster = [draw1]
            used.add(i)
            
            # Find nearby drawings
            for j, draw2 in enumerate(drawings):
                if j in used or j == i:
                    continue
                
                # Calculate distance between drawings
                dist = self._drawing_distance(draw1['rect'], draw2['rect'])
                
                if dist < distance_threshold:
                    cluster.append(draw2)
                    used.add(j)
            
            # Only keep clusters with substantial content
            if len(cluster) >= 3:  # At least 3 drawing elements
                clusters.append(cluster)
        
        return clusters
    
    def _drawing_distance(self, rect1: fitz.Rect, rect2: fitz.Rect) -> float:
        """Calculate minimum distance between two rectangles"""
        # Convert to list if Rect object
        if hasattr(rect1, 'x0'):
            r1 = [rect1.x0, rect1.y0, rect1.x1, rect1.y1]
        else:
            r1 = rect1
        
        if hasattr(rect2, 'x0'):
            r2 = [rect2.x0, rect2.y0, rect2.x1, rect2.y1]
        else:
            r2 = rect2
        
        # Check if rectangles overlap
        if (r1[0] <= r2[2] and r1[2] >= r2[0] and 
            r1[1] <= r2[3] and r1[3] >= r2[1]):
            return 0
        
        # Calculate minimum distance
        dx = max(0, max(r1[0] - r2[2], r2[0] - r1[2]))
        dy = max(0, max(r1[1] - r2[3], r2[1] - r1[3]))
        
        return (dx**2 + dy**2)**0.5
    
    def _overlaps_with_existing(self, bbox: BoundingBox, 
                                existing_regions: List[Dict]) -> bool:
        """Check if bbox significantly overlaps with existing regions"""
        for region in existing_regions:
            existing_bbox = region['bbox']
            
            # Calculate overlap
            x_overlap = max(0, min(bbox.x1, existing_bbox.x1) - max(bbox.x0, existing_bbox.x0))
            y_overlap = max(0, min(bbox.y1, existing_bbox.y1) - max(bbox.y0, existing_bbox.y0))
            overlap_area = x_overlap * y_overlap
            
            # If more than 50% overlap, consider it the same region
            bbox_area = bbox.area()
            if overlap_area > bbox_area * 0.5:
                return True
        
        return False
    
    def _render_region(self, page: fitz.Page, bbox: BoundingBox, 
                      dpi: int = 150) -> Tuple[Optional[Image.Image], Optional[bytes]]:
        """
        Render a specific region of the page to an image.
        This captures vector graphics, text, and everything in that region.
        """
        # Create a clip rectangle for the region
        clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        
        # Render the page region to a pixmap
        # Higher DPI = better quality but larger files
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default PDF DPI
        
        pixmap = page.get_pixmap(matrix=mat, clip=clip_rect)
        
        # Convert pixmap to PIL Image
        img_data = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        return image, img_data
    

    def _process_segment(self, segment: VisualSegment, page: fitz.Page, doc: fitz.Document):
        """Process a single visual segment with Mistral API"""
        
        image = Image.open(segment.image_path)
        
        # 1. Run OCR (using existing OCRProcessor)
        from __main__ import OCRProcessor
        segment.ocr_result = OCRProcessor.process_image(image)
        
        # 2. Classify using Mistral Vision API
        print(f"    Classifying with Mistral API...")
        visual_type, confidence, method = self.mistral_api.classify_visual(image, segment.ocr_result)
        segment.segment_type = visual_type
        segment.classification_confidence = confidence
        segment.classification_method = method
        print(f"    → Classified as {visual_type.value} (confidence: {confidence:.2f})")
        
        # 3. Extract Mermaid representation for diagrams/flowcharts
        if self.use_mermaid and segment.segment_type in [VisualType.DIAGRAM, VisualType.FLOWCHART]:
            print(f"    Extracting Mermaid representation...")
            segment.mermaid_repr = self.mistral_api.extract_mermaid_representation(image, segment)
            if segment.mermaid_repr and segment.mermaid_repr.mermaid_code:
                print(f"    → Mermaid extraction successful ({segment.mermaid_repr.diagram_type})")
        
        # 4. Detect caption
        from __main__ import CaptionDetector
        text_blocks = self._extract_text_blocks(page)
        figure_num, caption = CaptionDetector.detect_caption(
            text_blocks, segment.bbox, page.rect.height
        )
        segment.figure_number = figure_num
        segment.caption_text = caption
        
        if figure_num:
            segment.reference_keys = [
                f"Figure {figure_num}",
                f"Fig. {figure_num}",
                f"Fig {figure_num}"
            ]
        
        # 5. Generate summary using Mistral API
        print(f"    Generating summary with Mistral API...")
        summary, summary_conf = self.mistral_api.generate_summary(segment)
        if summary:
            segment.summary = summary
            segment.summary_confidence = summary_conf
            print(f"    → Summary generated (confidence: {summary_conf:.2f})")
        else:
            # Fallback to rule-based summary
            print(f"    → Using fallback rule-based summary")
            segment.summary = self._generate_fallback_summary(segment)
            segment.summary_confidence = 0.5
        
        # 6. Link to concepts
        if self.concept_linker:
            segment.linked_concept_ids = self.concept_linker.link_concepts(segment)
        
        # 7. Extract context
        segment.heading_path = self._extract_heading_path(page, segment.bbox)
        segment.nearby_text = self._extract_nearby_text(page, segment.bbox)
    
    def _generate_fallback_summary(self, segment: VisualSegment) -> str:
        """Rule-based fallback summary"""
        parts = []
        
        if segment.segment_type == VisualType.CHART:
            parts.append("This chart displays")
            if segment.ocr_result and segment.ocr_result.axis_labels:
                axes = segment.ocr_result.axis_labels
                if 'x' in axes and 'y' in axes:
                    parts.append(f"{axes['y']} versus {axes['x']}")
        elif segment.segment_type == VisualType.DIAGRAM:
            parts.append("This diagram illustrates a system or process")
        elif segment.segment_type == VisualType.FLOWCHART:
            parts.append("This flowchart shows a sequential process")
        else:
            parts.append(f"This {segment.segment_type.value}")
        
        if segment.caption_text:
            parts.append(f"Caption: {segment.caption_text[:100]}")
        
        return ". ".join(parts)
    
    def _generate_segment_id(self, page_num: int, bbox: BoundingBox, 
                        image_bytes: bytes) -> str:
        """Generate stable segment ID"""
        # Create hash from page, bbox, and image content
        hash_input = f"{self.book_id}_{page_num}_{bbox.x0}_{bbox.y0}_{bbox.x1}_{bbox.y1}"
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        return f"{self.book_id}_p{page_num:03d}_{content_hash}"
    
    def _extract_text_blocks(self, page: fitz.Page) -> List[Dict]:
        """Extract text blocks with bounding boxes"""
        blocks = []
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                
                blocks.append({
                    'text': block_text.strip(),
                    'bbox': block.get("bbox", [0, 0, 0, 0])
                })
        
        return blocks
    
    def _generate_summary(self, segment: VisualSegment) -> str:
        """Generate semantic summary using LLM for robustness"""
        
        # Try LLM-based summary first (more robust)
        llm_summary = self._generate_llm_summary(segment)
        if llm_summary and len(llm_summary) > 20:
            segment.summary_confidence = 0.85
            return llm_summary
        
        # Fallback to rule-based summary
        parts = []
        
        if segment.segment_type == VisualType.CHART:
            parts.append("This chart displays")
            if segment.ocr_result and segment.ocr_result.axis_labels:
                axes = segment.ocr_result.axis_labels
                if 'x' in axes and 'y' in axes:
                    parts.append(f"{axes['y']} versus {axes['x']}")
                elif 'y' in axes:
                    parts.append(f"{axes['y']}")
        
        elif segment.segment_type == VisualType.DIAGRAM:
            node_count = len(segment.ocr_result.node_texts) if segment.ocr_result else 0
            parts.append(f"This diagram illustrates a process or system with {node_count} components")
            if segment.ocr_result and segment.ocr_result.detected_arrows > 0:
                parts.append(f"showing {segment.ocr_result.detected_arrows} relationships or flows")
        
        elif segment.segment_type == VisualType.TABLE_IMAGE:
            parts.append("This table presents structured data")
        
        elif segment.segment_type == VisualType.PHOTO:
            parts.append("This photograph shows")
        
        else:
            parts.append(f"This {segment.segment_type.value}")
        
        if segment.caption_text:
            caption_snippet = segment.caption_text[:150]
            parts.append(f"Caption: {caption_snippet}")
        
        summary = ". ".join(parts)
        segment.summary_confidence = 0.5
        
        return summary
    
    def _generate_llm_summary(self, segment: VisualSegment) -> Optional[str]:
        """
        Generate summary using LLM vision API (GPT-4o-mini)
        Returns concise 1-3 sentence description
        """
        try:
            import base64
            import os
            import requests
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key or not segment.image_path:
                return None
            
            # Load and encode image
            image = Image.open(segment.image_path)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Build context
            context_parts = []
            if segment.caption_text:
                context_parts.append(f"Caption: {segment.caption_text}")
            if segment.figure_number:
                context_parts.append(f"Figure number: {segment.figure_number}")
            if segment.ocr_result and segment.ocr_result.raw_text:
                context_parts.append(f"Text in image: {segment.ocr_result.raw_text[:200]}")
            
            context = "\n".join(context_parts) if context_parts else "No additional context"
            
            # Prepare prompt based on visual type
            if segment.segment_type == VisualType.CHART:
                prompt = f"""Analyze this chart and provide a concise 2-3 sentence summary covering:
1. What type of chart it is (line, bar, scatter, etc.)
2. What data/variables are being plotted
3. Key trend or finding visible

Context:
{context}

Be specific and factual."""

            elif segment.segment_type == VisualType.DIAGRAM:
                prompt = f"""Analyze this diagram and provide a concise 2-3 sentence summary covering:
1. What process or system it represents
2. Main components or stages
3. Key relationships or flow direction

Context:
{context}

Be specific and factual."""

            else:
                prompt = f"""Describe what this image shows in 2-3 concise sentences. Focus on:
1. Main subject or content
2. Key visual elements
3. Purpose or what it illustrates

Context:
{context}

Be specific and factual."""
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 150
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result['choices'][0]['message']['content'].strip()
                return summary
            
        except Exception as e:
            print(f"    LLM summary generation failed: {e}")
        
        return None
    
    def _extract_heading_path(self, page: fitz.Page, bbox: BoundingBox) -> List[str]:
        """Extract heading hierarchy for context"""
        # Simple implementation: look for larger font sizes above the image
        headings = []
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                
                # Only consider blocks above the image
                if block_bbox[3] < bbox.y0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_size = span.get("size", 10)
                            text = span.get("text", "").strip()
                            
                            # Heuristic: headings are typically larger font
                            if font_size > 12 and len(text) > 3:
                                headings.append(text)
        
        return headings[-3:] if headings else []  # Keep last 3 headings
    
    def _extract_nearby_text(self, page: fitz.Page, bbox: BoundingBox) -> str:
        """Extract text near the visual for context"""
        nearby_text = []
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                
                # Check if block is near the visual (within 100 points)
                vertical_distance = min(
                    abs(block_bbox[1] - bbox.y1),
                    abs(bbox.y0 - block_bbox[3])
                )
                
                if vertical_distance < 100:
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "") + " "
                    
                    nearby_text.append(block_text.strip())
        
        return " ".join(nearby_text)[:500]  # Limit length
    
    def _save_results(self):
        """Save segmentation results to JSON"""
        output_json = self.output_dir / f"{self.book_id}_visual_segments.json"
        
        results = {
            'book_id': self.book_id,
            'pdf_path': self.pdf_path,
            'total_segments': len(self.segments),
            'segments': [seg.to_dict() for seg in self.segments]
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_json}")
        
        # Also create a summary CSV
        self._save_summary_csv()
    
    def _save_summary_csv(self):
        """Create summary CSV for easy review"""
        summary_data = []
        
        for seg in self.segments:
            summary_data.append({
                'segment_id': seg.segment_id,
                'page': seg.page_no,
                'type': seg.segment_type.value,
                'confidence': f"{seg.classification_confidence:.2f}",
                'figure_number': seg.figure_number or '',
                'caption': seg.caption_text[:100] if seg.caption_text else '',
                'ocr_text': seg.ocr_result.raw_text[:100] if seg.ocr_result else '',
                'linked_concepts': len(seg.linked_concept_ids),
                'summary': seg.summary[:100] if seg.summary else ''
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / f"{self.book_id}_visual_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")

    # [Include all other helper methods from original class]
    # _extract_text_blocks, _extract_heading_path, _extract_nearby_text, etc.


# Example usage
if __name__ == "__main__":
    # Set your Mistral API key
    # os.environ['MISTRAL_API_KEY'] = 'your_mistral_api_key_here'
    
    pipeline = VisualSegmentationPipeline(
        book_id="textbook_001",
        pdf_path="D:\\D-Downloads\\chapter1.pdf",
        taxonomy_path="D:\\D-Downloads\\Segmentation_Zvi Bodie, Alex Kane, Alan J. Marcus - Investments (2023, McGraw Hill).xlsx",
        output_dir="./extracted_visuals",
        use_mermaid=False  # Enable Mermaid extraction
    )
    
    segments = pipeline.process()
    
    print("\n=== Extraction Summary ===")
    print(f"Total visual elements: {len(segments)}")
    print(f"Segments with Mermaid representations: {sum(1 for s in segments if s.mermaid_repr)}")