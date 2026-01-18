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
from difflib import SequenceMatcher
import io
import os

# Core dependencies
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
from collections import defaultdict
import pandas as pd
import numpy as np
import cv2
import requests
import base64

try:
    from sklearn.cluster import KMeans, DBSCAN
except ImportError:
    print("Warning: scikit-learn not installed. Some advanced features will be disabled.")
    KMeans = None
    DBSCAN = None

class VisualType(str, Enum):
    """Classification of visual elements"""
    FIGURE = "figure"
    CHART = "chart"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    IMAGE = "image"
    UNKNOWN = "unknown"

@dataclass
class ChartSpecificData:
    """Enhanced data extraction for charts/graphs"""
    chart_subtype: Optional[str] = None  # line, bar, scatter, pie, histogram, yield_curve
    axes_info: Dict[str, Any] = field(default_factory=dict)
    value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    legend_items: List[str] = field(default_factory=list)
    series_count: int = 0
    grid_detected: bool = False
    color_scheme: List[str] = field(default_factory=list)
    estimated_data_points: int = 0
    tick_labels: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DiagramSpecificData:
    """Enhanced data extraction for diagrams/flowcharts"""
    diagram_subtype: Optional[str] = None  # process_flow, decision_tree, hierarchy, cycle, causal, system
    node_count: int = 0
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    arrow_count: int = 0
    hierarchy_detected: bool = False
    layout_type: Optional[str] = None  # hierarchical_vertical, hierarchical_horizontal, circular, free_form
    shapes_detected: Dict[str, int] = field(default_factory=dict)
    has_decision_points: bool = False


@dataclass
class ImageSpecificData:
    """Enhanced data extraction for images (photos, screenshots, illustrations)"""
    image_subtype: Optional[str] = None  # screenshot, photo, illustration, scanned_page, embedded_graphic
    contains_text: bool = False
    text_density: str = "none"  # none, sparse, moderate, dense
    is_embedded_table: bool = False
    dominant_colors: List[str] = field(default_factory=list)
    estimated_content_type: Optional[str] = None  # interface, document, scene, object, mixed


@dataclass
class FigureSpecificData:
    """Enhanced data for labeled figures"""
    is_composite: bool = False  # Contains sub-figures like (a), (b), (c)
    sub_figure_count: int = 0
    contains_chart: bool = False
    contains_diagram: bool = False
    contains_image: bool = False

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

    # Type-specific rich data
    chart_data: Optional[ChartSpecificData] = None
    diagram_data: Optional[DiagramSpecificData] = None
    image_data: Optional[ImageSpecificData] = None
    figure_data: Optional[FigureSpecificData] = None
    
    # Text extracted from inside the image (for search + linking)
    extracted_text_structured: Dict[str, List[str]] = field(default_factory=dict)

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

    @staticmethod
    def _convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: VisualSegment._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [VisualSegment._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict with type-specific data"""
        result = asdict(self)
        result['segment_type'] = self.segment_type.value
        result['bbox'] = self.bbox.to_dict() if self.bbox else None
        result.pop('image_bytes', None)
        
        # ADD TYPE-SPECIFIC DATA TO OUTPUT:
        if self.chart_data:
            result['chart_details'] = {
                'subtype': self.chart_data.chart_subtype,
                'axes': self.chart_data.axes_info,
                'legend': self.chart_data.legend_items,
                'series_count': self.chart_data.series_count,
                'data_points': self.chart_data.estimated_data_points,
                'has_grid': self.chart_data.grid_detected,
                'colors': self.chart_data.color_scheme,
                'value_ranges': self.chart_data.value_ranges,
                'tick_labels': self.chart_data.tick_labels
            }
        
        if self.diagram_data:
            result['diagram_details'] = {
                'subtype': self.diagram_data.diagram_subtype,
                'node_count': self.diagram_data.node_count,
                'nodes': self.diagram_data.nodes[:15],  # Limit for JSON size
                'connection_count': len(self.diagram_data.connections),
                'arrow_count': self.diagram_data.arrow_count,
                'layout_type': self.diagram_data.layout_type,
                'has_hierarchy': self.diagram_data.hierarchy_detected,
                'has_decision_points': self.diagram_data.has_decision_points,
                'shapes': self.diagram_data.shapes_detected
            }
        
        if self.image_data:
            result['image_details'] = {
                'subtype': self.image_data.image_subtype,
                'contains_text': self.image_data.contains_text,
                'text_density': self.image_data.text_density,
                'is_embedded_table': self.image_data.is_embedded_table,
                'content_type': self.image_data.estimated_content_type,
                'dominant_colors': self.image_data.dominant_colors[:5]
            }
        
        if self.figure_data:
            result['figure_details'] = {
                'is_composite': self.figure_data.is_composite,
                'sub_figure_count': self.figure_data.sub_figure_count,
                'contains_chart': self.figure_data.contains_chart,
                'contains_diagram': self.figure_data.contains_diagram,
                'contains_image': self.figure_data.contains_image
            }
        
        # Include structured text extraction
        if self.extracted_text_structured:
            result['extracted_text_structured'] = self.extracted_text_structured
        
        result = self._convert_to_json_serializable(result)

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
        """Classify visual type using Mistral's vision model"""
        if not self.api_key:
            print("    WARNING: MISTRAL_API_KEY not found, falling back to heuristics")
            return VisualType.FIGURE, 0.3, "fallback_heuristic"
        
        try:
            img_base64 = self._encode_image(image)
            
            # Build context from OCR
            ocr_context = ""
            if ocr_result and ocr_result.raw_text:
                ocr_context = f"\n\nText detected in image:\n{ocr_result.raw_text[:300]}"
            
            # CORRECTED PROMPT: Prioritize specific types over generic FIGURE
            prompt = f"""Analyze this image and classify it into ONE of these categories, prioritizing specific types over generic ones:

**Categories (in priority order):**
1. **CHART**: Data visualization with numerical axes, tick labels, and data series (line chart, bar chart, scatter plot, histogram, pie chart, yield curve). Must have measurable data plotted on axes.

2. **FLOWCHART**: Sequential decision flow with specific flowchart shapes (rectangles for processes, diamonds for decisions, directed arrows showing flow path). Shows step-by-step logic or workflow.

3. **DIAGRAM**: Process flow, system architecture, concept map, organizational chart, or causal diagram with labeled nodes/boxes and connecting arrows/lines showing relationships. Does NOT have numerical data axes.

4. **IMAGE**: Photograph, screenshot, illustration, scanned page, or embedded graphic. Can contain text but does NOT have data axes or structured flow diagrams.

5. **FIGURE**: Generic labeled visual element. Use ONLY when the content doesn't clearly fit into the above specific categories, OR when it's a composite containing multiple types (e.g., Figure 2.1 showing both a chart and a photo side-by-side).

**Classification Priority Rules:**
1. **Always prioritize specific types**: If it's clearly a chart with axes → CHART (not FIGURE), even if it has a caption "Figure 2.1"
2. **CHART criteria**: 
   - Has X and Y axes with numerical scales/tick marks
   - Plots quantitative data (bars, lines, points, pie slices)
   - Example: "Figure 2.3: Revenue Growth" showing a line graph → Classify as CHART
3. **FLOWCHART vs DIAGRAM**:
   - FLOWCHART: Shows sequential steps with decision points (diamonds), clear start/end
   - DIAGRAM: Shows relationships, hierarchies, or systems without sequential flow
4. **IMAGE criteria**:
   - Photos, screenshots, illustrations
   - Can have text overlays but no structured axes or flow
   - Embedded tables/scanned documents count as IMAGE
5. **FIGURE - last resort**:
   - Only use when genuinely unclear or composite
   - Mixed content (chart + diagram together)
   - Highly abstract or unclear visual

{ocr_context}

**Response format (JSON only):**
{{
  "category": "CHART|FLOWCHART|DIAGRAM|IMAGE|FIGURE",
  "confidence": "score from 0-1"
}}

**Examples:**
- "Figure 2.1: GDP Growth 2020-2024" with line graph → CHART (not FIGURE)
- Boxes connected with arrows showing a process → DIAGRAM
- Decision tree with diamond shapes → FLOWCHART  
- Screenshot of a software interface → IMAGE
- Composite showing chart + photo side-by-side → FIGURE"""

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
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse JSON response
                try:
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                    elif '```' in content:
                        content = re.sub(r'```\w*\s*', '', content).strip()
                    
                    classification = json.loads(content)
                    
                    category = classification.get('category', 'FIGURE').upper()
                    confidence = float(classification.get('confidence', 0.7))
                    
                    # UPDATED MAPPING:
                    type_mapping = {
                        'CHART': VisualType.CHART,
                        'DIAGRAM': VisualType.DIAGRAM,
                        'FLOWCHART': VisualType.FLOWCHART,
                        'IMAGE': VisualType.IMAGE,
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
            
            # Generate type-specific prompt
            prompt = self._generate_type_specific_prompt(segment)
            
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
    
    def _generate_type_specific_prompt(self, segment: 'VisualSegment') -> str:
        """Generate specialized prompt based on visual type"""
        
        # Build context
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
        
        # Type-specific prompts
        if segment.segment_type == VisualType.CHART:
            return f"""Analyze this chart image and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Chart Type:** Identify the specific chart type (line, bar, scatter, pie, histogram, etc.)
2. **Variables:** What data/variables are being plotted? (X-axis, Y-axis, categories)
3. **Key Findings:** What are the main trends, patterns, or insights visible?
4. **Data Range:** Approximate ranges or scale of the data if visible
5. **Notable Features:** Any outliers, intersections, or significant data points

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this chart without seeing it."""

        elif segment.segment_type == VisualType.DIAGRAM:
            return f"""Analyze this diagram and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Purpose:** What process, system, or concept does this illustrate?
2. **Components:** List the main components, stages, or nodes
3. **Relationships:** Describe the connections, flow direction, or relationships
4. **Key Insights:** What is the main takeaway or learning objective?
5. **Structure:** How is the information organized? (hierarchical, sequential, cyclic, etc.)

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this diagram without seeing it."""

        elif segment.segment_type == VisualType.FLOWCHART:
            return f"""Analyze this flowchart and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Purpose:** What decision process or workflow does this represent?
2. **Stages:** List the main stages or decision points
3. **Flow:** Describe the flow direction and logic
4. **Decision Points:** What are the key decisions or branches?
5. **Outcome:** What are the possible end states or outcomes?

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this flowchart without seeing it."""

        elif segment.segment_type == VisualType.IMAGE:
            return f"""Analyze this image and provide a clear, educational summary suitable for a student.

        **Context:**
        {context}

        **Required Analysis:**
        1. **Main Subject:** Identify the primary topic or concept illustrated.
        2. **Key Visual Elements:** Describe important components (diagrams, labels, axes, annotations, icons).
        3. **Definitions:** Explicitly extract and restate any definitions shown in the image (including boxed text, callouts, or captions).
        4. **Formulas & Expressions:** List any formulas, equations, or mathematical expressions visible (including those inside table cells), preserving their original structure.
        5. **Variables & Meanings:** For each variable or symbol shown, explain what it represents if indicated in the image.
        6. **Tables or Structured Layouts:** If the image contains a table, matrix, or grid-like structure, summarize its rows, columns, and what relationships or comparisons it conveys.
        7. **Purpose & Learning Outcome:** Explain what the image is intended to teach and what a student should understand after studying it.

        **Output Rules:**
        - Do **not** invent information that is not visible in the image.
        - Prefer concise, factual language.

        **Format:**
        Provide a compact, well-structured explanation (3-6 concise sentences or short bullet points) that allows a student to understand the content without seeing the image.
        """

        elif segment.segment_type == VisualType.FIGURE:
            return f"""Analyze this figure and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Type:** What type of content does this figure contain? (chart, diagram, image, or composite)
2. **Main Content:** Describe the primary visual elements
3. **Purpose:** What concept or information is being illustrated?
4. **Key Takeaway:** What is the main learning objective?

**Format:** Provide 3-4 concise, factual sentences that would help a student understand this figure without seeing it."""

        else:  # UNKNOWN
            return f"""Analyze this visual element and provide a detailed, educational summary.

**Context:**
{context}

**Required Analysis:**
1. **Content:** Describe what you see
2. **Purpose:** What might this be illustrating?
3. **Educational Value:** What could a student learn from this?

**Format:** Provide 2-3 concise, factual sentences."""

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
    
    # Initialize PaddleOCR once (class-level)
    _paddle_ocr = None
    
    @classmethod
    def get_paddle_ocr(cls):
        """Lazy initialization of PaddleOCR for version 3.x"""
        if cls._paddle_ocr is None:
            cls._paddle_ocr = PaddleOCR(
                use_textline_orientation=True, 
                lang='en'            # Language: 'en' for English, 'ch' for Chinese
            )
        return cls._paddle_ocr

    @staticmethod
    def process_image(image: Image.Image) -> OCRResult:
        """Run OCR and extract structured information using PaddleOCR 3.3.2"""
        
        try:
            # Get PaddleOCR instance
            ocr = OCRProcessor.get_paddle_ocr()
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)

            # Validate image
            if img_array is None or img_array.size == 0:
                raise ValueError("Invalid or empty image")
            
            # Ensure image has correct dimensions
            if len(img_array.shape) == 2:
                # Grayscale to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                # Single channel to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # RGBA to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
            # Run PaddleOCR using predict() method
            # Returns a generator of Result objects
            result_generator = ocr.predict(img_array)
            
            # Parse PaddleOCR 3.3.2 output
            blocks = []
            raw_text_parts = []
            confidences = []
            
            # Iterate through results (generator yields Result objects)
            for result_obj in result_generator:
                # Access the json attribute to get the actual data
                result_data = result_obj.json.get('res')
                
                # Extract dt_polys (detection polygons) and rec_texts (recognized texts)
                dt_polys = result_data.get('dt_polys', [])
                rec_texts = result_data.get('rec_texts', [])
                rec_scores = result_data.get('rec_scores', [])
                
                # Process each detected text region
                for i, (bbox_points, text, score) in enumerate(zip(dt_polys, rec_texts, rec_scores)):
                    # bbox_points is a numpy array with shape (4, 2) - 4 corner points
                    # Convert to [x0, y0, x1, y1] format
                    if len(bbox_points) >= 4:
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        
                        bbox = [
                            min(x_coords),  # x0 (left)
                            min(y_coords),  # y0 (top)
                            max(x_coords),  # x1 (right)
                            max(y_coords)   # y1 (bottom)
                        ]
                        
                        blocks.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': score * 100  # Convert to percentage (0-100)
                        })
                        
                        raw_text_parts.append(text)
                        confidences.append(score * 100)
            
            # Combine all text with newlines
            raw_text = '\n'.join(raw_text_parts)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
        except Exception as e:
            print(f"PaddleOCR processing failed: {e}")
            import traceback
            traceback.print_exc()
            return OCRResult(raw_text="", confidence=0.0)
        
        # Detect chart-specific elements (using existing helper methods)
        axis_labels = OCRProcessor._detect_axis_labels(raw_text, blocks)
        legend_items = OCRProcessor._detect_legend(raw_text)
        
        # Detect diagram elements (using existing helper methods)
        node_texts = OCRProcessor._detect_nodes(blocks)
        arrow_count = OCRProcessor._count_arrows(image)
        
        return OCRResult(
            raw_text=raw_text,
            blocks=blocks,
            confidence=avg_confidence / 100.0,  # Normalize to 0-1 range
            axis_labels=axis_labels,
            legend_items=legend_items,
            node_texts=node_texts,
            detected_arrows=arrow_count
        )
    
    @staticmethod
    def extract_structured_text(ocr_result: OCRResult, segment_type: VisualType) -> Dict[str, List[str]]:
        """Extract structured text fields for search and linking"""
        structured = {
            'labels': [],
            'values': [],
            'annotations': []
        }
        
        if not ocr_result or not ocr_result.raw_text:
            return structured
        
        text = ocr_result.raw_text
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect numeric values
            if re.search(r'\d+\.?\d*', line):
                structured['values'].append(line)
            
            # Detect short labels (likely axis labels, node names)
            elif len(line) < 30:
                structured['labels'].append(line)
            
            # Everything else is annotation
            else:
                structured['annotations'].append(line)
        
        return structured
    
    @staticmethod
    def process_chart_specific(image: Image.Image, ocr_result: OCRResult) -> ChartSpecificData:
        """Extract chart-specific features"""
        chart_data = ChartSpecificData()
        
        # Detect chart subtype
        chart_data.chart_subtype = OCRProcessor._detect_chart_subtype(image, ocr_result)
        
        # Extract axes information
        chart_data.axes_info = OCRProcessor._extract_axes_detailed(ocr_result)
        
        # Extract value ranges
        chart_data.value_ranges = OCRProcessor._extract_value_ranges(ocr_result)
        
        # Extract legend
        chart_data.legend_items = OCRProcessor._detect_legend_advanced(ocr_result, (image.width, image.height))
        
        # Estimate series count (from legend or visual analysis)
        chart_data.series_count = len(chart_data.legend_items) if chart_data.legend_items else 1
        
        # Detect grid
        chart_data.grid_detected = OCRProcessor._detect_grid(image)
        
        # Extract dominant colors
        chart_data.color_scheme = OCRProcessor._extract_dominant_colors(image)
        
        # Estimate data points
        chart_data.estimated_data_points = OCRProcessor._estimate_data_points(image)
        
        # Extract tick labels
        chart_data.tick_labels = OCRProcessor._extract_tick_labels(ocr_result)
        
        return chart_data
    
    @staticmethod
    def process_diagram_specific(image: Image.Image, ocr_result: OCRResult) -> DiagramSpecificData:
        """Extract diagram-specific features"""
        diagram_data = DiagramSpecificData()
        
        # Detect diagram subtype
        diagram_data.diagram_subtype = OCRProcessor._detect_diagram_subtype(image, ocr_result)
        
        # Extract nodes
        diagram_data.nodes = OCRProcessor._extract_nodes(image, ocr_result)
        diagram_data.node_count = len(diagram_data.nodes)
        
        # Extract connections
        diagram_data.connections = OCRProcessor._extract_connections(image)
        
        # Count arrows
        diagram_data.arrow_count = ocr_result.detected_arrows if ocr_result else 0
        
        # Detect hierarchy
        diagram_data.hierarchy_detected = OCRProcessor._detect_hierarchy(diagram_data.nodes)
        
        # Detect layout type
        diagram_data.layout_type = OCRProcessor._detect_layout_type(diagram_data.nodes)
        
        # Detect shapes
        diagram_data.shapes_detected = OCRProcessor._detect_shapes(image)
        
        # Detect decision points
        diagram_data.has_decision_points = OCRProcessor._detect_decision_points(image, ocr_result)
        
        return diagram_data
    
    @staticmethod
    def process_image_specific(image: Image.Image, ocr_result: OCRResult) -> ImageSpecificData:
        """Extract image-specific features"""
        image_data = ImageSpecificData()
        
        # Detect image subtype
        image_data.image_subtype = OCRProcessor._detect_image_subtype(image, ocr_result)
        
        # Check for text
        if ocr_result and ocr_result.raw_text:
            image_data.contains_text = len(ocr_result.raw_text.strip()) > 10
            
            # Estimate text density
            char_count = len(ocr_result.raw_text)
            if char_count > 500:
                image_data.text_density = "dense"
            elif char_count > 100:
                image_data.text_density = "moderate"
            elif char_count > 0:
                image_data.text_density = "sparse"
        
        # Detect embedded tables
        image_data.is_embedded_table = OCRProcessor._detect_embedded_table(image, ocr_result)
        
        # Extract dominant colors
        image_data.dominant_colors = OCRProcessor._extract_dominant_colors(image)
        
        # Estimate content type
        image_data.estimated_content_type = OCRProcessor._estimate_content_type(image, ocr_result)
        
        return image_data
    
    @staticmethod
    def process_figure_specific(image: Image.Image, ocr_result: OCRResult) -> FigureSpecificData:
        """Analyze figure for composite structure"""
        figure_data = FigureSpecificData()
        
        # Detect sub-figures (a), (b), (c)
        if ocr_result and ocr_result.raw_text:
            subfig_pattern = r'\([a-z]\)|\b[a-z]\)'
            matches = re.findall(subfig_pattern, ocr_result.raw_text.lower())
            if len(matches) >= 2:
                figure_data.is_composite = True
                figure_data.sub_figure_count = len(matches)
        
        # Detect if contains chart (look for axes, grid)
        figure_data.contains_chart = OCRProcessor._detect_grid(image)
        
        # Detect if contains diagram (look for boxes, arrows)
        arrow_count = ocr_result.detected_arrows if ocr_result else 0
        figure_data.contains_diagram = arrow_count > 3
        
        # Detect if contains image (high variance, photo-like)
        img_array = np.array(image.convert('L'))
        variance = np.var(img_array)
        figure_data.contains_image = variance > 1000
        
        return figure_data

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
        """Detect legend entries - placeholder, use _detect_legend_advanced below"""
        lines = text.split('\n')
        legend_items = []
        for line in lines:
            clean = line.strip()
            if 3 < len(clean) < 40 and not re.match(r'^[\d\-/.,\s%$]+$', clean):
                legend_items.append(clean)
        return legend_items[:10]

    @staticmethod
    def _detect_legend_advanced(ocr_result: OCRResult, image_size: Tuple[int, int]) -> List[str]:
        """Advanced legend detection using spatial clustering"""
        if not ocr_result or not ocr_result.blocks:
            return []
        
        width, height = image_size
        candidates = []
        
        for block in ocr_result.blocks:
            text = block['text'].strip()
            bbox = block['bbox']
            
            if not text or len(text) < 3 or len(text) > 30:
                continue
            
            # Skip pure numbers
            if re.match(r'^[\d\-/.,\s%$€£¥]+$', text):
                continue
            
            x_mid = (bbox[0] + bbox[2]) / 2
            
            # Legend typically on right side (x > 60% of width)
            if x_mid > 0.6 * width:
                candidates.append({
                    'text': text,
                    'y': (bbox[1] + bbox[3]) / 2
                })
        
        if len(candidates) < 2:
            return [c['text'] for c in candidates]
        
        # Group by vertical proximity (within 50 pixels)
        candidates.sort(key=lambda x: x['y'])
        groups = []
        current_group = [candidates[0]]
        
        for i in range(1, len(candidates)):
            if candidates[i]['y'] - candidates[i-1]['y'] < 50:
                current_group.append(candidates[i])
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [candidates[i]]
        
        if len(current_group) >= 2:
            groups.append(current_group)
        
        # Return largest group
        if groups:
            largest = max(groups, key=len)
            return [item['text'] for item in largest]
        
        return []

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

    @staticmethod
    def _detect_chart_subtype(image: Image.Image, ocr_result: OCRResult) -> Optional[str]:
        """Multi-signal chart type detection with strict thresholds"""
        text = ocr_result.raw_text.lower() if ocr_result else ""
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        scores = defaultdict(float)
        
        # SIGNAL 1: Text-based detection (most reliable when present)
        if re.search(r'\bpie\b', text) and 'chart' in text:
            scores['pie'] += 3.0
        if 'scatter' in text or 'correlation' in text:
            scores['scatter'] += 3.0
        if 'candlestick' in text or all(w in text for w in ['open', 'close']):
            scores['candlestick'] += 3.0
        if re.search(r'\bbar\b.*\bchart\b|\bbar\b.*\bgraph\b', text):
            scores['bar'] += 3.0
        if re.search(r'\bline\b.*\bchart\b|\bline\b.*\bgraph\b', text):
            scores['line'] += 3.0
        
        # SIGNAL 2: Visual features
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect VERTICAL bars (BAR CHARTS) - check this FIRST
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 20)))
        v_detect = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=2)
        v_pixels = np.sum(v_detect > 0)
        
        # Detect HORIZONTAL continuous lines (LINE CHARTS)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 20), 1))
        h_detect = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=2)
        h_pixels = np.sum(h_detect > 0)
        
        # DEBUG
        print(f"DEBUG - V pixels: {v_pixels}, H pixels: {h_pixels}, Ratio H/V: {h_pixels/max(v_pixels, 1):.2f}")
        
        # LINE CHART DETECTION (priority check)
        # Line charts have much more horizontal than vertical structure
        if h_pixels > height * 8 and h_pixels > v_pixels * 1.5:
            scores['line'] += 2.5
            
            # Check for continuous lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=width//4, maxLineGap=20)
            if lines is not None:
                horizontal_long_lines = sum(1 for line in lines 
                                        if abs(line[0][3] - line[0][1]) < 10 
                                        and abs(line[0][2] - line[0][0]) > width * 0.2)
                if horizontal_long_lines >= 1:
                    scores['line'] += 1.5
            
            print(f"DEBUG - Line chart detected with score: {scores['line']}")
        
        # BAR CHART DETECTION
        elif v_pixels > width * 10:
            scores['bar'] += 2.0
            
            # Check for multiple separate bars
            contours, _ = cv2.findContours(v_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vertical_bars = sum(1 for c in contours if cv2.boundingRect(c)[3] > height * 0.2)
            
            if vertical_bars >= 3:
                scores['bar'] += 1.5
            
            print(f"DEBUG - Bar chart detected with score: {scores['bar']}")
        
        # PIE CHART DETECTION (only if NOT line/bar chart)
        # Only check for pie if we don't have strong line/bar signals
        if scores.get('line', 0) < 2.0 and scores.get('bar', 0) < 2.0:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=int(min(width, height) * 0.3),  # Circles must be very far apart
                param1=50, 
                param2=50,    # Even higher threshold
                minRadius=int(min(width, height) * 0.2),  # Large circles only
                maxRadius=int(min(width, height) * 0.45)
            )
            
            print(f"DEBUG - Circles detected: {len(circles[0]) if circles is not None else 0}")
            
            if circles is not None:
                # STRICT: Only accept if there's exactly 1 large circle
                large_circles = [c for c in circles[0] if c[2] > min(width, height) * 0.2]
                
                if len(large_circles) == 1:  # Exactly ONE large circle
                    circle_center = large_circles[0][:2].astype(int)
                    radius = int(large_circles[0][2])
                    
                    # Additional validation: check for radial structure (pie slices)
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, tuple(circle_center), radius, 255, -1)
                    
                    circle_edges = cv2.bitwise_and(edges, edges, mask=mask)
                    edge_density = np.sum(circle_edges > 0) / (np.pi * radius * radius)
                    
                    # Pie charts have high edge density (slice boundaries)
                    if edge_density > 0.015:
                        scores['pie'] += 2.5
                        print(f"DEBUG - Pie chart validated with edge density: {edge_density:.4f}")
                    else:
                        print(f"DEBUG - Circle rejected, low edge density: {edge_density:.4f}")
                else:
                    print(f"DEBUG - Circles rejected, found {len(large_circles)} large circles (need exactly 1)")
        
        # DECISION LOGIC
        print(f"DEBUG - Final scores: {dict(scores)}")
        
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            # Require minimum confidence
            if best_score >= 2.0:
                return best_type
        
        return 'unknown'

    @staticmethod
    def _extract_axes_detailed(ocr_result: OCRResult) -> Dict[str, Any]:
        """Spatial-aware axis extraction using PaddleOCR bbox positions"""
        axes = {'x_axis': {}, 'y_axis': {}}
        
        if not ocr_result or not ocr_result.blocks:
            return axes
        
        # Calculate image bounds from all blocks
        all_bboxes = [b['bbox'] for b in ocr_result.blocks]
        if not all_bboxes:
            return axes
        
        max_x = max(b[2] for b in all_bboxes)
        max_y = max(b[3] for b in all_bboxes)
        
        bottom_zone = []  # Likely x-axis labels
        left_zone = []    # Likely y-axis labels
        
        for block in ocr_result.blocks:
            bbox = block['bbox']
            text = block['text'].strip()
            if not text or len(text) < 2:
                continue
            
            x_mid = (bbox[0] + bbox[2]) / 2
            y_mid = (bbox[1] + bbox[3]) / 2
            
            # Bottom zone (y > 85% of height)
            if y_mid > 0.85 * max_y:
                bottom_zone.append((text, len(text)))
            
            # Left zone (x < 15% of width)
            if x_mid < 0.15 * max_x:
                left_zone.append((text, len(text)))
        
        # Find longest non-numeric text in each zone
        for text, length in bottom_zone:
            if not re.match(r'^[\d\-/.,\s%$€£¥]+$', text) and len(text) > 3:
                if 'label' not in axes['x_axis'] or len(text) > len(axes['x_axis']['label']):
                    axes['x_axis']['label'] = text
        
        for text, length in left_zone:
            if not re.match(r'^[\d\-/.,\s%$€£¥]+$', text) and len(text) > 3:
                if 'label' not in axes['y_axis'] or len(text) > len(axes['y_axis']['label']):
                    axes['y_axis']['label'] = text
        
        return axes

    @staticmethod
    def _extract_value_ranges(ocr_result: OCRResult) -> Dict[str, Tuple[float, float]]:
        """Enhanced numeric parsing: handles $1.5M, 23%, -45.2K, etc."""
        ranges = {}
        numbers = []
        
        if not ocr_result or not ocr_result.raw_text:
            return ranges
        
        # Enhanced pattern for numbers with units and multipliers
        pattern = r'([€£¥$]?\s*-?\d+(?:[.,]\d+)?(?:[KMBkmb])?)\s*(%|€|£|¥|\$)?'
        
        for block in ocr_result.blocks:
            for match in re.finditer(pattern, block['text']):
                try:
                    num_str = match.group(1).replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', '').strip()
                    
                    # Handle K, M, B multipliers
                    mult = 1
                    if num_str and num_str[-1] in 'KkMmBb':
                        mult = {'K': 1000, 'k': 1000, 'M': 1000000, 'm': 1000000, 'B': 1000000000, 'b': 1000000000}[num_str[-1]]
                        num_str = num_str[:-1]
                    
                    value = float(num_str) * mult
                    numbers.append(value)
                except (ValueError, IndexError):
                    continue
        
        if numbers:
            ranges['detected'] = (min(numbers), max(numbers))
            ranges['count'] = len(numbers)
        
        return ranges
    
    @staticmethod
    def _detect_grid(image: Image.Image) -> bool:
        """Morphological grid detection"""
        img_array = np.array(image.convert('L'))
        edges = cv2.Canny(img_array, 50, 150)
        
        # Detect horizontal and vertical lines separately
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=2)
        
        # Count line pixels
        h_count = np.sum(h_lines > 0)
        v_count = np.sum(v_lines > 0)
        
        # Grid has substantial lines in both directions
        return h_count > 300 and v_count > 300

    @staticmethod
    def _extract_dominant_colors(image: Image.Image, n_colors: int = 5) -> List[str]:
        """K-means clustering on non-background pixels"""
        img = image.convert('RGB')
        img_array = np.array(img)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Mask: exclude whites, blacks, grays
        mask = (hsv[:,:,1] > 30) & (hsv[:,:,2] > 40) & (hsv[:,:,2] < 240)
        pixels = img_array[mask].reshape(-1, 3)
        
        if len(pixels) < 100:
            return []
        
        # Downsample for performance
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            return ['#%02x%02x%02x' % tuple(c) for c in colors]
        except ImportError:
            # Fallback if sklearn not available
            return []
    
    @staticmethod
    def _estimate_data_points(image: Image.Image) -> int:
        """Type-aware data point counting"""
        img_array = np.array(image.convert('L'))
        edges = cv2.Canny(img_array, 50, 150)
        
        # Try blob detection for scatter-like patterns
        try:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 10
            params.maxArea = 150
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(img_array)
            if len(keypoints) > 5:
                return len(keypoints)
        except:
            pass
        
        # Fallback: edge density
        edge_pixels = np.sum(edges > 0)
        return min(edge_pixels // 150, 500)

    @staticmethod
    def _extract_tick_labels(ocr_result: OCRResult) -> Dict[str, List[str]]:
        """Position-based tick label separation"""
        tick_labels = {'x_axis': [], 'y_axis': []}
        
        if not ocr_result or not ocr_result.blocks:
            return tick_labels
        
        # Find image bounds
        all_bboxes = [b['bbox'] for b in ocr_result.blocks]
        if not all_bboxes:
            return tick_labels
        
        max_x = max(b[2] for b in all_bboxes)
        max_y = max(b[3] for b in all_bboxes)
        
        for block in ocr_result.blocks:
            text = block['text'].strip()
            bbox = block['bbox']
            
            if not text or len(text) > 20:
                continue
            
            x_mid = (bbox[0] + bbox[2]) / 2
            y_mid = (bbox[1] + bbox[3]) / 2
            
            # X-axis ticks: bottom zone
            if y_mid > 0.8 * max_y and 0.1 < x_mid / max_x < 0.9:
                tick_labels['x_axis'].append(text)
            
            # Y-axis ticks: left/right zone, numeric
            elif (x_mid < 0.15 * max_x or x_mid > 0.85 * max_x) and 0.1 < y_mid / max_y < 0.9:
                if re.match(r'^[\d\-/.,\s%$€£¥KMB]+$', text):
                    tick_labels['y_axis'].append(text)
        
        return tick_labels
    
    @staticmethod
    def _detect_diagram_subtype(image: Image.Image, ocr_result: OCRResult) -> Optional[str]:
        """Detect diagram type"""
        text = ocr_result.raw_text.lower() if ocr_result else ""
        
        if 'process' in text or 'flow' in text:
            return 'process_flow'
        elif 'decision' in text:
            return 'decision_tree'
        elif 'hierarchy' in text or 'organization' in text:
            return 'hierarchy'
        elif 'cycle' in text or 'circular' in text:
            return 'cycle'
        elif 'cause' in text or 'effect' in text:
            return 'causal'
        elif 'system' in text:
            return 'system'
        
        return 'unknown'
    
    @staticmethod
    def _extract_nodes(image: Image.Image, ocr_result: OCRResult) -> List[Dict[str, Any]]:
        """Extract diagram nodes"""
        nodes = []
        
        if not ocr_result or not ocr_result.blocks:
            return nodes
        
        for i, block in enumerate(ocr_result.blocks):
            text = block.get('text', '').strip()
            if 3 < len(text) < 100:  # Reasonable node text length
                nodes.append({
                    'id': f'node_{i}',
                    'text': text,
                    'bbox': block.get('bbox', [])
                })
        
        return nodes[:50]  # Limit
    
    @staticmethod
    def _extract_connections(image: Image.Image) -> List[Dict[str, Any]]:
        """Extract connections between nodes"""
        # Simplified: just count lines
        img_array = np.array(image.convert('L'))
        edges = cv2.Canny(img_array, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        connections = []
        if lines is not None:
            for i, line in enumerate(lines[:20]):  # Limit
                connections.append({
                    'id': f'conn_{i}',
                    'type': 'arrow'
                })
        
        return connections
    
    @staticmethod
    def _detect_hierarchy(nodes: List[Dict[str, Any]]) -> bool:
        """Detect if nodes form hierarchy"""
        if len(nodes) < 3:
            return False
        
        # Simple heuristic: check if nodes have different vertical levels
        y_positions = [node['bbox'][1] for node in nodes if 'bbox' in node]
        if not y_positions:
            return False
        
        # If nodes span multiple vertical levels, likely hierarchical
        y_range = max(y_positions) - min(y_positions)
        return y_range > 100
    
    @staticmethod
    def _detect_layout_type(nodes: List[Dict[str, Any]]) -> Optional[str]:
        """Detect diagram layout type"""
        if len(nodes) < 2:
            return None
        
        # Analyze node positions
        positions = [(node['bbox'][0], node['bbox'][1]) for node in nodes if 'bbox' in node]
        if not positions:
            return None
        
        # Simple heuristics
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        
        if y_variance > x_variance * 2:
            return 'hierarchical_vertical'
        elif x_variance > y_variance * 2:
            return 'hierarchical_horizontal'
        else:
            return 'free_form'
    
    @staticmethod
    def _detect_shapes(image: Image.Image) -> Dict[str, int]:
        """Detect common shapes in diagram"""
        shapes = {'rectangles': 0, 'circles': 0, 'diamonds': 0}
        
        img_array = np.array(image.convert('L'))
        edges = cv2.Canny(img_array, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 4:
                shapes['rectangles'] += 1
            elif len(approx) > 8:
                shapes['circles'] += 1
            elif len(approx) == 4:
                # Check if diamond (rotated square)
                shapes['diamonds'] += 1
        
        return shapes
    
    @staticmethod
    def _detect_decision_points(image: Image.Image, ocr_result: OCRResult) -> bool:
        """Detect decision points in flowchart"""
        text = ocr_result.raw_text.lower() if ocr_result else ""
        
        decision_keywords = ['if', 'yes', 'no', 'decision', 'choose', 'select']
        has_keywords = any(kw in text for kw in decision_keywords)
        
        # Also check for diamond shapes
        shapes = OCRProcessor._detect_shapes(image)
        has_diamonds = shapes.get('diamonds', 0) > 0
        
        return has_keywords or has_diamonds
    
    @staticmethod
    def _detect_image_subtype(image: Image.Image, ocr_result: OCRResult) -> Optional[str]:
        """Detect image subtype"""
        # Check text density
        text_length = len(ocr_result.raw_text) if ocr_result else 0
        
        if text_length > 500:
            return 'scanned_page'
        elif text_length > 100:
            # Could be screenshot with UI elements
            return 'screenshot'
        else:
            # Likely photo or illustration
            img_array = np.array(image.convert('L'))
            variance = np.var(img_array)
            
            if variance > 1500:
                return 'photo'
            else:
                return 'illustration'
    
    @staticmethod
    def _detect_embedded_table(image: Image.Image, ocr_result: OCRResult) -> bool:
        """Detect if image contains an embedded table"""
        if not ocr_result or not ocr_result.raw_text:
            return False
        
        # Look for table patterns
        text = ocr_result.raw_text
        
        # Check for grid-like structure in text
        lines = text.split('\n')
        numeric_lines = sum(1 for line in lines if re.search(r'\d+', line))
        
        # If many lines contain numbers, might be table
        return numeric_lines > len(lines) * 0.5 and len(lines) > 3
    
    @staticmethod
    def _estimate_content_type(image: Image.Image, ocr_result: OCRResult) -> Optional[str]:
        """Estimate overall content type"""
        text = ocr_result.raw_text.lower() if ocr_result else ""
        
        if 'window' in text or 'button' in text or 'menu' in text:
            return 'interface'
        elif len(text) > 300:
            return 'document'
        else:
            return 'mixed'
    
class ConceptLinker:
    """
    Links visual segments to concepts using multi-signal matching.
    
    Research-backed approach:
    1. TF-IDF for term importance weighting
    2. Contextual expansion (nearby text, captions)
    3. Fuzzy matching with Levenshtein distance
    4. Calibrated confidence scores based on match quality
    """
    
    def __init__(self, taxonomy_df: pd.DataFrame):
        """
        Initialize with taxonomy DataFrame.
        Expected columns: Level, Concept, Tag(s), Rationale, Page(s)
        """
        self.taxonomy_df = taxonomy_df
        self.concept_map = {}
        self.term_frequencies = defaultdict(int)  # For TF-IDF
        self.document_count = 0
        
        self._build_concept_index()
        self._compute_term_statistics()
    
    def _build_concept_index(self):
        """Build search index with normalized terms and aliases"""
        print(f"Available columns in taxonomy: {list(self.taxonomy_df.columns)}")
        
        for idx, row in self.taxonomy_df.iterrows():
            concept_name = row.get('Concept', '')
            if not concept_name:
                continue
            
            concept_id = self._generate_concept_id(concept_name, idx)
            
            # Store full concept data
            concept_entry = {
                'concept_id': concept_id,
                'concept_name': concept_name,
                'bloom_level': row.get('Level', ''),
                'tag': row.get('Tag(s)', ''),
                'pages': row.get('Page(s)', ''),
                'normalized_terms': set(),  # All searchable terms
                'primary_terms': set(),     # Core concept words
                'context_terms': set()      # Related/synonym terms
            }
            
            # Extract and normalize primary terms
            primary_terms = self._extract_terms(concept_name)
            concept_entry['primary_terms'] = primary_terms
            concept_entry['normalized_terms'].update(primary_terms)
            
            # Extract context terms from tags
            tags = row.get('Tag(s)', '')
            if pd.notna(tags) and tags:
                tag_terms = self._extract_terms(str(tags))
                concept_entry['context_terms'] = tag_terms
                concept_entry['normalized_terms'].update(tag_terms)
            
            # Store in map
            self.concept_map[concept_id] = concept_entry
        
        print(f"Built concept index with {len(self.concept_map)} concepts")
    
    def _compute_term_statistics(self):
        """Compute term frequencies across all concepts for TF-IDF"""
        all_documents = []
        
        for concept_data in self.concept_map.values():
            doc_terms = list(concept_data['normalized_terms'])
            all_documents.append(doc_terms)
            
            # Count term frequencies
            for term in doc_terms:
                self.term_frequencies[term] += 1
        
        self.document_count = len(all_documents)
        print(f"Computed term statistics for {self.document_count} concepts")
    
    def _extract_terms(self, text: str) -> set:
        """
        Extract and normalize terms from text.
        
        Normalization:
        - Lowercase
        - Remove punctuation
        - Split on whitespace and common separators
        - Filter stop words
        - Keep terms >= 3 characters
        """
        if not text:
            return set()
        
        # Normalize
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Split and filter
        terms = set()
        for word in text.split():
            word = word.strip('-_')
            
            # Filter criteria
            if len(word) >= 3 and word not in self._get_stop_words():
                terms.add(word)
        
        return terms
    
    def _get_stop_words(self) -> set:
        """Common stop words to filter out"""
        return {
            'the', 'and', 'for', 'with', 'from', 'this', 'that',
            'are', 'was', 'were', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'about', 'into', 'through', 'over', 'under'
        }
    
    def _generate_concept_id(self, concept_name: str, index: int) -> str:
        """Generate unique concept ID"""
        normalized = concept_name.lower().strip()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'[-\s]+', '_', normalized)
        
        if len(normalized) > 50:
            normalized = normalized[:50]
        
        return f"concept_{normalized}_{index:03d}"
    
    def link_concepts(self, segment: VisualSegment) -> List[Dict[str, Any]]:
        """
        Link segment to concepts using multi-signal matching.
        
        Signals:
        1. Exact phrase matching (highest confidence)
        2. Term overlap with TF-IDF weighting
        3. Fuzzy matching for near-matches
        4. Contextual expansion (nearby text)
        
        Returns ranked list of concept matches with calibrated confidence.
        """
        # Collect all searchable text with context weighting
        search_context = self._build_search_context(segment)
        
        # Extract terms from search context
        search_terms = self._extract_terms(search_context['combined_text'])
        
        # Score all concepts
        scored_matches = []
        
        for concept_id, concept_data in self.concept_map.items():
            match_score = self._score_concept_match(
                search_terms=search_terms,
                search_context=search_context,
                concept_data=concept_data
            )
            
            if match_score['total_score'] > 0.3:  # Minimum threshold
                scored_matches.append({
                    'concept_id': concept_data['concept_id'],
                    'concept_name': concept_data['concept_name'],
                    'bloom_level': concept_data['bloom_level'],
                    'tag': concept_data['tag'],
                    'pages': concept_data.get('pages', ''),
                    'confidence': match_score['total_score'],
                    'match_method': match_score['method'],
                    'match_details': match_score['details']
                })
        
        # Sort by confidence
        scored_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Log results
        print(f"Found {len(scored_matches)} concept links")
        for match in scored_matches[:5]:
            print(f"  - {match['concept_name']} ({match['confidence']:.3f}, {match['match_method']})")
        
        return scored_matches
    
    def _build_search_context(self, segment: VisualSegment) -> Dict[str, Any]:
        """
        Build weighted search context from segment.
        
        Weighting (based on information value):
        - Caption: 1.0 (most reliable)
        - Summary: 0.9 (generated, may have errors)
        - OCR text: 0.7 (may have OCR errors)
        - Nearby text: 0.5 (contextual, may be noisy)
        """
        context = {
            'caption': segment.caption_text or '',
            'summary': segment.summary or '',
            'ocr': segment.ocr_result.raw_text if segment.ocr_result else '',
            'nearby': segment.nearby_text or '',
            'weights': {
                'caption': 1.0,
                'summary': 0.9,
                'ocr': 0.7,
                'nearby': 0.5
            }
        }
        
        # Combine all text (for term extraction)
        combined_parts = []
        if context['caption']:
            combined_parts.append(context['caption'])
        if context['summary']:
            combined_parts.append(context['summary'])
        if context['ocr']:
            combined_parts.append(context['ocr'])
        if context['nearby']:
            combined_parts.append(context['nearby'])
        
        context['combined_text'] = ' '.join(combined_parts)
        
        return context
    
    def _score_concept_match(self, search_terms: set, search_context: Dict,
                        concept_data: Dict) -> Dict[str, Any]:
        """
        Multi-signal scoring for concept matching.
        
        Signals (updated with cosine similarity):
        1. Exact phrase match (0-30 points)
        2. Cosine similarity (0-30 points) ← NEW
        3. Weighted term overlap (0-25 points)
        4. Fuzzy matching (0-10 points)
        5. Context bonus (0-5 points)
        
        Total: 0-100 points, normalized to 0-1
        """
        score_breakdown = {
            'exact_phrase': 0.0,
            'cosine_similarity': 0.0,  # NEW
            'term_overlap': 0.0,
            'fuzzy_match': 0.0,
            'context_bonus': 0.0
        }
        
        concept_name = concept_data['concept_name']
        primary_terms = concept_data['primary_terms']
        all_terms = concept_data['normalized_terms']
        
        # SIGNAL 1: Exact phrase matching (30 points max)
        exact_score = self._score_exact_match(
            concept_name, 
            search_context['combined_text']
        )
        score_breakdown['exact_phrase'] = exact_score * 30
        
        # SIGNAL 2: Cosine similarity (30 points max) ← NEW
        cosine_score = self._score_cosine_similarity(
            search_terms,
            concept_data,
            search_context
        )
        score_breakdown['cosine_similarity'] = cosine_score * 30
        
        # SIGNAL 3: Weighted term overlap (25 points max)
        overlap_score = self._score_term_overlap(
            search_terms, 
            primary_terms,
            all_terms
        )
        score_breakdown['term_overlap'] = overlap_score * 25
        
        # SIGNAL 4: Fuzzy matching (10 points max)
        fuzzy_score = self._score_fuzzy_match(
            concept_name,
            search_context['combined_text']
        )
        score_breakdown['fuzzy_match'] = fuzzy_score * 10
        
        # SIGNAL 5: Context bonus (5 points max)
        context_score = self._score_context_match(
            search_context,
            concept_data
        )
        score_breakdown['context_bonus'] = context_score * 5
        
        # Compute total
        total_score = sum(score_breakdown.values()) / 100.0  # Normalize to 0-1
        
        # Determine match method (highest contributing signal)
        max_signal = max(score_breakdown, key=score_breakdown.get)
        method_map = {
            'exact_phrase': 'exact_phrase_match',
            'cosine_similarity': 'cosine_similarity',  # NEW
            'term_overlap': 'term_overlap',
            'fuzzy_match': 'fuzzy_match',
            'context_bonus': 'context_match'
        }
        
        return {
            'total_score': min(total_score, 1.0),
            'method': method_map[max_signal],
            'details': score_breakdown
        }

    def _score_exact_match(self, concept_name: str, text: str) -> float:
        """
        Exact phrase matching with position weighting.
        
        Returns 0-1:
        - 1.0: Exact match in high-value context (caption)
        - 0.8: Exact match in medium-value context (summary)
        - 0.6: Exact match in low-value context (OCR/nearby)
        """
        concept_lower = concept_name.lower().strip()
        text_lower = text.lower()
        
        if concept_lower not in text_lower:
            return 0.0
        
        # Check which context it appears in (if we had separate fields)
        # For now, return high score for exact match
        return 1.0
    
    def _score_term_overlap(self, search_terms: set, primary_terms: set,
                           all_terms: set) -> float:
        """
        TF-IDF weighted term overlap.
        
        Formula:
        score = Σ(term_weight * idf_weight) / max_possible_score
        
        Where:
        - term_weight: 1.0 for primary terms, 0.5 for context terms
        - idf_weight: log(N / df), where N = total concepts, df = term frequency
        """
        if not search_terms or not all_terms:
            return 0.0
        
        # Calculate overlap
        primary_overlap = search_terms.intersection(primary_terms)
        context_overlap = search_terms.intersection(all_terms - primary_terms)
        
        score = 0.0
        max_score = 0.0
        
        # Score primary term matches (higher weight)
        for term in primary_overlap:
            idf = self._compute_idf(term)
            score += 1.0 * idf
        
        # Score context term matches (lower weight)
        for term in context_overlap:
            idf = self._compute_idf(term)
            score += 0.5 * idf
        
        # Compute max possible score (if all primary terms matched)
        for term in primary_terms:
            idf = self._compute_idf(term)
            max_score += 1.0 * idf
        
        # Normalize
        if max_score > 0:
            return min(score / max_score, 1.0)
        
        return 0.0
    
    def _score_cosine_similarity(self, search_terms: set, concept_data: Dict,
                             search_context: Dict) -> float:
        """
        Compute cosine similarity between search context and concept.
        
        Formula:
        cos(θ) = (A · B) / (||A|| × ||B||)
        
        Where:
        - A = TF-IDF vector of search context
        - B = TF-IDF vector of concept
        - · = dot product
        - ||·|| = L2 norm
        
        Returns 0-1 similarity score.
        """
        # Build TF-IDF vectors
        search_vector = self._build_tfidf_vector(search_context)
        concept_vector = self._build_concept_tfidf_vector(concept_data)
        
        # Get all unique terms
        all_terms = set(search_vector.keys()) | set(concept_vector.keys())
        
        if not all_terms:
            return 0.0
        
        # Compute dot product
        dot_product = sum(
            search_vector.get(term, 0) * concept_vector.get(term, 0)
            for term in all_terms
        )
        
        # Compute norms
        search_norm = np.sqrt(sum(v**2 for v in search_vector.values()))
        concept_norm = np.sqrt(sum(v**2 for v in concept_vector.values()))
        
        # Avoid division by zero
        if search_norm == 0 or concept_norm == 0:
            return 0.0
        
        # Cosine similarity
        cosine_sim = dot_product / (search_norm * concept_norm)
        
        return max(0.0, min(1.0, cosine_sim))  # Clamp to [0, 1]

    def _build_tfidf_vector(self, search_context: Dict) -> Dict[str, float]:
        """
        Build TF-IDF vector for search context with context weighting.
        
        TF-IDF = TF(term) × IDF(term)
        
        Where:
        - TF = term frequency (with context weights)
        - IDF = inverse document frequency
        """
        # Count term frequencies with context weighting
        term_counts = defaultdict(float)
        
        # Caption (highest weight)
        if search_context['caption']:
            caption_terms = self._extract_terms(search_context['caption'])
            for term in caption_terms:
                term_counts[term] += search_context['weights']['caption']
        
        # Summary
        if search_context['summary']:
            summary_terms = self._extract_terms(search_context['summary'])
            for term in summary_terms:
                term_counts[term] += search_context['weights']['summary']
        
        # OCR
        if search_context['ocr']:
            ocr_terms = self._extract_terms(search_context['ocr'])
            for term in ocr_terms:
                term_counts[term] += search_context['weights']['ocr']
        
        # Nearby text
        if search_context['nearby']:
            nearby_terms = self._extract_terms(search_context['nearby'])
            for term in nearby_terms:
                term_counts[term] += search_context['weights']['nearby']
        
        # Compute TF-IDF
        tfidf_vector = {}
        total_terms = sum(term_counts.values())
        
        for term, count in term_counts.items():
            # TF (normalized by total term count)
            tf = count / total_terms if total_terms > 0 else 0
            
            # IDF
            idf = self._compute_idf(term)
            
            # TF-IDF
            tfidf_vector[term] = tf * idf
        
        return tfidf_vector

    def _build_concept_tfidf_vector(self, concept_data: Dict) -> Dict[str, float]:
        """
        Build TF-IDF vector for a concept.
        
        Weights:
        - Primary terms (concept name): 2.0
        - Context terms (tags): 1.0
        """
        term_counts = defaultdict(float)
        
        # Primary terms (higher weight)
        for term in concept_data['primary_terms']:
            term_counts[term] += 2.0
        
        # Context terms (lower weight)
        for term in concept_data['context_terms']:
            term_counts[term] += 1.0
        
        # Compute TF-IDF
        tfidf_vector = {}
        total_terms = sum(term_counts.values())
        
        for term, count in term_counts.items():
            tf = count / total_terms if total_terms > 0 else 0
            idf = self._compute_idf(term)
            tfidf_vector[term] = tf * idf
        
        return tfidf_vector

    def _compute_idf(self, term: str) -> float:
        """
        Compute IDF (Inverse Document Frequency) for term.
        
        IDF = log(N / df)
        where N = total concepts, df = number of concepts containing term
        
        Higher IDF = rarer term = more discriminative
        """
        df = self.term_frequencies.get(term, 1)
        idf = np.log((self.document_count + 1) / (df + 1)) + 1  # Smoothed
        return idf
    
    def _score_fuzzy_match(self, concept_name: str, text: str) -> float:
        """
        Fuzzy string matching for near-matches.
        
        Uses Levenshtein distance to catch:
        - Typos (OCR errors)
        - Plural variations
        - Minor spelling differences
        
        Returns 0-1 based on best match similarity.
        """
        concept_lower = concept_name.lower().strip()
        
        # Extract candidate phrases from text (same length as concept)
        words = text.lower().split()
        concept_word_count = len(concept_lower.split())
        
        best_similarity = 0.0
        
        # Check n-grams of same length as concept
        for i in range(len(words) - concept_word_count + 1):
            phrase = ' '.join(words[i:i + concept_word_count])
            
            # Compute similarity (simple character-based)
            similarity = self._string_similarity(concept_lower, phrase)
            
            if similarity > best_similarity:
                best_similarity = similarity
        
        # Only return meaningful fuzzy matches (>= 80% similar)
        return best_similarity if best_similarity >= 0.8 else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Compute string similarity (0-1) using character overlap.
        
        Simple implementation without external dependencies.
        For production, use: python-Levenshtein or difflib.SequenceMatcher
        """
        if s1 == s2:
            return 1.0
        
        # Use difflib for simplicity (built-in)
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    def _score_context_match(self, search_context: Dict, 
                            concept_data: Dict) -> float:
        """
        Context-based bonus scoring.
        
        Checks:
        - Caption mentions concept? +50%
        - Summary mentions concept? +30%
        - Nearby text mentions concept? +20%
        
        Returns 0-1
        """
        concept_name = concept_data['concept_name'].lower()
        score = 0.0
        
        if concept_name in search_context['caption'].lower():
            score += 0.5
        
        if concept_name in search_context['summary'].lower():
            score += 0.3
        
        if concept_name in search_context['nearby'].lower():
            score += 0.2
        
        return min(score, 1.0)


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
        api_key="aE9nzpmp8JWHHguJMTurCrDrJCTfodaP"
        self.mistral_api = MistralVisionAPI(api_key=api_key)
        
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
        """Process a single visual segment with type-aware extraction"""
        
        image = Image.open(segment.image_path)
        
        # STEP 1: Run OCR
        print(f"    Running OCR...")
        segment.ocr_result = OCRProcessor.process_image(image)
        
        # STEP 2: Classify using Mistral Vision API
        print(f"    Classifying with Mistral API...")
        visual_type, confidence, method = self.mistral_api.classify_visual(image, segment.ocr_result)
        segment.segment_type = visual_type
        segment.classification_confidence = confidence
        segment.classification_method = method
        print(f"    → Classified as {visual_type.value} (confidence: {confidence:.2f})")
        
        # STEP 3: Type-specific feature extraction
        print(f"    Extracting type-specific features...")
        if segment.segment_type == VisualType.CHART:
            segment.chart_data = OCRProcessor.process_chart_specific(image, segment.ocr_result)
            print(f"    → Chart subtype: {segment.chart_data.chart_subtype}")
        
        elif segment.segment_type == VisualType.DIAGRAM:
            segment.diagram_data = OCRProcessor.process_diagram_specific(image, segment.ocr_result)
            print(f"    → Diagram nodes: {segment.diagram_data.node_count}")
        
        elif segment.segment_type == VisualType.FLOWCHART:
            segment.diagram_data = OCRProcessor.process_diagram_specific(image, segment.ocr_result)
            segment.diagram_data.diagram_subtype = 'flowchart'
            print(f"    → Flowchart nodes: {segment.diagram_data.node_count}")
        
        elif segment.segment_type == VisualType.IMAGE:
            segment.image_data = OCRProcessor.process_image_specific(image, segment.ocr_result)
            print(f"    → Image subtype: {segment.image_data.image_subtype}")
        
        elif segment.segment_type == VisualType.FIGURE:
            segment.figure_data = OCRProcessor.process_figure_specific(image, segment.ocr_result)
            print(f"    → Composite figure: {segment.figure_data.is_composite}")
        
        # STEP 4: Extract structured text for search/linking
        segment.extracted_text_structured = OCRProcessor.extract_structured_text(
            segment.ocr_result, 
            segment.segment_type
        )
        
        # STEP 5: Extract Mermaid representation for diagrams/flowcharts
        if self.use_mermaid and segment.segment_type in [VisualType.DIAGRAM, VisualType.FLOWCHART]:
            print(f"    Extracting Mermaid representation...")
            segment.mermaid_repr = self.mistral_api.extract_mermaid_representation(image, segment)
            if segment.mermaid_repr and segment.mermaid_repr.mermaid_code:
                print(f"    → Mermaid extraction successful ({segment.mermaid_repr.diagram_type})")
        
        # STEP 6: Detect caption
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
        
        # STEP 7: Generate type-aware summary using Mistral API
        print(f"    Generating type-aware summary with Mistral API...")
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
        
        # STEP 8: Link to concepts
        if self.concept_linker:
            segment.linked_concept_ids = self.concept_linker.link_concepts(segment)
        
        # STEP 9: Extract context
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
        book_id="textbook_002",
        pdf_path="D:\\D-Downloads\\complex.pdf",
        taxonomy_path="D:\\D-Downloads\\Don M. Chance, Roberts Brooks - An Introduction to Derivatives and Risk Management (2015, South-Western College Pub).xlsx",
        output_dir="./extracted_visuals",
        use_mermaid=False  # Enable Mermaid extraction
    )
    
    segments = pipeline.process()
    
    print("\n=== Extraction Summary ===")
    print(f"Total visual elements: {len(segments)}")
    print(f"Segments with Mermaid representations: {sum(1 for s in segments if s.mermaid_repr)}")
