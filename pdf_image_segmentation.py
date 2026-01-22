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
    
    # NEW FIELDS for detailed content extraction
    definitions: List[Dict[str, str]] = field(default_factory=list)  # [{"term": "...", "definition": "..."}]
    formulas: List[Dict[str, str]] = field(default_factory=list)  # [{"formula": "...", "description": "...", "location": "..."}]
    variables: List[Dict[str, str]] = field(default_factory=list)  # [{"variable": "...", "meaning": "..."}]
    tables: List[Dict[str, Any]] = field(default_factory=list)  # [{"description": "...", "rows": N, "columns": N, "content_summary": "..."}]
        # NEW FIELDS for calculation extraction and verification
    input_variables: List[Dict[str, Any]] = field(default_factory=list)  # [{"variable": "...", "value": "...", "unit": "..."}]
    output_values: List[Dict[str, Any]] = field(default_factory=list)  # [{"output_name": "...", "value": "...", "location": "..."}]
    calculation_verification: Optional[Dict[str, Any]] = None  # {"verified": bool, "matches": bool, "differences": [...]}

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
                'dominant_colors': self.image_data.dominant_colors[:5],
                # NEW FIELDS
                'definitions': self.image_data.definitions,
                'formulas': self.image_data.formulas,
                'variables': self.image_data.variables,
                'tables': self.image_data.tables,
                # NEW FIELDS for calculation extraction
                'input_variables': self.image_data.input_variables,
                'output_values': self.image_data.output_values,
                'calculation_verification': self.image_data.calculation_verification
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

    def analyze_visual_comprehensive(self, image: Image.Image, ocr_result: OCRResult) -> Dict[str, Any]:
        """
        Single API call to classify, extract metadata, and generate summary.
        Returns dict with: {
            'visual_type': VisualType,
            'confidence': float,
            'metadata': dict (type-specific),
            'summary': str,
            'summary_confidence': float
        }
        """
        if not self.api_key:
            print("    WARNING: MISTRAL_API_KEY not found, using fallback")
            return self._fallback_analysis(ocr_result)
        
        try:
            img_base64 = self._encode_image(image)
            
            # Build OCR context
            ocr_context = ""
            if ocr_result and ocr_result.raw_text:
                ocr_context = f"\n\n**Text detected in image (OCR):**\n{ocr_result.raw_text[:500]}"
            
            # COMPREHENSIVE PROMPT - does everything in one call
            prompt = f"""Analyze this visual element comprehensively and provide a structured response.

{ocr_context}

**Your task has 3 parts:**

## PART 1: CLASSIFICATION
Classify this visual into ONE category (prioritize specific over generic):

1. **CHART**: Data visualization with numerical axes and plotted data (line, bar, scatter, pie, histogram)
2. **FLOWCHART**: Sequential decision flow with flowchart shapes (rectangles, diamonds, arrows)
3. **DIAGRAM**: Process flow, system architecture, concept map with labeled nodes and connections (NO numerical axes)
4. **IMAGE**: Photograph, screenshot, illustration, scanned page, embedded table
5. **FIGURE**: Generic/composite element (only if doesn't fit above categories)

**Classification Rules:**
- CHART requires numerical axes with data plotted
- FLOWCHART requires decision points (diamonds) and sequential flow
- DIAGRAM shows relationships but NO data axes
- IMAGE is photographic/illustrative content including screenshots and tables
- FIGURE is last resort or composite

## PART 2: METADATA EXTRACTION
Based on the classification, extract type-specific metadata:

**For CHART:**
- chart_subtype: (line|bar|scatter|pie|histogram|candlestick|unknown)
- x_axis_label: string or null
- y_axis_label: string or null
- legend_items: array of strings
- value_range: {{"min": number, "max": number}} or null
- data_series_count: integer
- has_grid: boolean

**For FLOWCHART:**
- node_count: integer (estimated)
- decision_points: integer (diamond shapes)
- has_start_end: boolean
- flow_direction: (top_down|left_right|mixed)

**For DIAGRAM:**
- diagram_subtype: (process_flow|decision_tree|hierarchy|cycle|system|network|unknown)
- node_count: integer (estimated)
- has_hierarchy: boolean
- layout_type: (hierarchical_vertical|hierarchical_horizontal|circular|free_form)

**For IMAGE:**
- image_subtype: (screenshot|photo|illustration|scanned_page|embedded_table|unknown)
- contains_text: boolean
- text_density: (none|sparse|moderate|dense)
- is_embedded_table: boolean
- definitions: array of {{"term": "string", "definition": "string"}}
- formulas: array of {{"formula": "string", "description": "string", "location": "string"}}
- variables: array of {{"variable": "string", "meaning": "string"}}
- tables: array of {{"description": "string", "rows": integer, "columns": integer, "headers": array, "content_summary": "string"}}
- input_variables: array of {{"variable": "string", "value": "string|number", "unit": "string"}} - Extract input variables and their values shown in the image
- output_values: array of {{"output_name": "string", "value": "string|number", "location": "string"}} - Extract calculated output values shown in the image

**CRITICAL RULES for IMAGE metadata extraction:**

**DEFINITIONS:**
- ONLY extract if you can SEE explicit definition text in the image
- Look for: boxed definitions, callouts with "Definition:", highlighted terms with explanations, glossary entries
- DO NOT infer or create definitions - they must be literally visible in the image
- Format: {{"term": "exact term shown", "definition": "exact definition text shown"}}
- If NO definitions are visible, return empty array: []

**FORMULAS:**
- Extract mathematical expressions/equations/formulas that are visible in the image OR can be inferred from context
- Look for: equals signs (=), mathematical operators (+, -, *, /, ^), mathematical notation
- If formulas are NOT explicitly visible but you can infer them from:
  * Input variables and output values shown in the image
  * Context from nearby text (OCR text provided)
  * Standard formulas for the domain (e.g., Black-Scholes for option pricing, present value formulas for finance)
  * Then INFER and include the formula with description indicating it was inferred
- IMPORTANT: You may INFER formulas based on context, but DO NOT infer or create new variables or values - only use variables and values that are explicitly shown in the image

**VARIABLES:**
- ONLY extract if the image explicitly shows variable definitions/meanings
- Look for: "where x = ...", variable legend, notation key, "let r denote..."
- Must show BOTH the variable symbol AND its meaning in the image
- DO NOT extract variables from formulas unless their meanings are also shown
- Format: {{"variable": "symbol exactly as shown", "meaning": "meaning exactly as shown"}}
- If NO variable definitions are visible, return empty array: []

**TABLES:**
- ONLY extract if you can see an actual table structure (grid with rows/columns)
- Count VISIBLE rows and columns - don't estimate if unclear
- Extract VISIBLE column headers exactly as shown
- If headers are not visible, use empty array for headers: []
- Describe what data the table contains based on what you can actually see
- For rows/columns, if you cannot count exactly (e.g., table is cut off), use your best visible count
- If NO table is visible, return empty array: []

**INPUT VARIABLES:**
- Extract input variables and their values that are explicitly shown in the image
- Look for: labeled input fields, parameter lists, "Inputs:" sections, variable names with values
- Format: {{"variable": "variable name/symbol", "value": "numerical or text value", "unit": "unit if shown (e.g., %, $, years)"}}
- Examples: {{"variable": "Asset price (S₀)", "value": "125.94", "unit": ""}}, {{"variable": "Risk-free rate (r)", "value": "4.56", "unit": "%"}}
- DO NOT infer or create variables/values - only extract what is explicitly visible
- If NO input variables are visible, return empty array: []

**OUTPUT VALUES:**
- Extract calculated output values that are explicitly shown in the image
- Look for: result sections, calculated fields, output tables, "Results:" sections
- Format: {{"output_name": "name of output (e.g., 'Call Price', 'Delta')", "value": "numerical or text value", "location": "where in image (e.g., 'Call column, Price row')"}}
- Examples: {{"output_name": "Call Price", "value": "13.5589", "location": "Black-Scholes-Merton Model, Call column"}}
- DO NOT infer or create outputs - only extract what is explicitly visible
- If NO output values are visible, return empty array: []

**GENERAL RULES:**
- When in doubt, use EMPTY ARRAY [] rather than guessing
- For variables and values: Only extract information that is LITERALLY VISIBLE in the image - DO NOT infer or create new variables or values
- For formulas: You MAY infer formulas based on context (nearby text, input/output relationships, domain knowledge) if they are not explicitly visible, but clearly mark them as inferred
- If OCR text is provided but you cannot verify it in the image, be cautious
- Preserve exact text/notation as shown - don't paraphrase or rewrite

**For FIGURE:**
- is_composite: boolean (contains multiple sub-figures like (a), (b), (c))
- sub_figure_count: integer
- contains_chart: boolean
- contains_diagram: boolean
- contains_image: boolean

## PART 3: EDUCATIONAL SUMMARY
Provide a 3-5 sentence educational summary that would help a student understand this visual without seeing it.

**For CHART:** Describe chart type, variables plotted, key trends, data range, notable features
**For FLOWCHART:** Describe the decision process, main stages, flow logic, decision points, outcomes
**For DIAGRAM:** Describe the purpose, main components, relationships, structure, key insights
**For IMAGE:** Describe main subject, key visual elements. If present: mention definitions shown, formulas visible (with their exact notation), variables explained, table structures. If calculation extraction was performed: explain the input variables extracted, formulas identified (including inferred ones), output values found, and verification results. Focus on factual description of visible content. DO NOT describe definitions/variables/tables if they are not actually visible.
**For FIGURE:** Describe the content type, main elements, purpose, key takeaway

**Summary Rules:**
- Only mention definitions/variables/tables if they are ACTUALLY visible in the image
- If the image has no formulas, you may mention formulas what you inferred based on earlier context in the summary
- Be specific and factual - describe what you see

---

**RESPONSE FORMAT (JSON only, no markdown):**
{{
  "classification": {{
    "category": "CHART|FLOWCHART|DIAGRAM|IMAGE|FIGURE",
    "confidence": 0.0-1.0
  }},
  "metadata": {{
    // Include ALL relevant fields from Part 2 based on classification
    // For IMAGE type:
    //   - formulas: [] if no formulas visible/inferrable, otherwise array of {{formula (also include inferred ones), description, location}}
    //   - variables: [] if no variable meanings shown, otherwise array of {{variable, meaning}}
    //   - tables: [] if no table visible, otherwise array of table objects
    //   - input_variables: [] if no inputs visible, otherwise array of {{variable, value, unit}}
    //   - output_values: [] if no outputs visible, otherwise array of {{output_name, value, location}}
    // CRITICAL: For variables/values - only include what is LITERALLY VISIBLE. For formulas - may infer from context.
  }},
  "summary": {{
    "text": "3-5 sentence educational summary describing only what is visible",
    "confidence": 0.0-1.0
  }}
}}

**EXAMPLES:**

Example 1 - Image with table containing formulas:
{{
  "metadata": {{
    "definitions": [],
    "formulas": [
      {{"formula": "=B2/(1+C2)^D2", "description": "Present value calculation", "location": "cell E2"}},
      {{"formula": "=SUM(E2:E10)", "description": "Total present value", "location": "cell E11"}}
    ],
    "variables": [],
    "tables": [{{
      "description": "Present value calculations for cash flows",
      "rows": 10,
      "columns": 5,
      "headers": ["Year", "Cash Flow", "Rate", "Period", "PV"],
      "content_summary": "Shows cash flows from year 1-9 with corresponding present value calculations"
    }}]
  }}
}}

Example 2 - Image with definition box but no formulas:
{{
  "metadata": {{
    "definitions": [
      {{"term": "Present Value", "definition": "The current worth of a future sum of money given a specified rate of return"}}
    ],
    "formulas": [],
    "variables": [],
    "tables": []
  }}
}}

Example 3 - Image with formula and variable legend:
{{
  "metadata": {{
    "definitions": [],
    "formulas": [
      {{"formula": "PV = FV / (1 + r)^n", "description": "Present value formula", "location": "equation box at top"}}
    ],
    "variables": [
      {{"variable": "PV", "meaning": "Present Value"}},
      {{"variable": "FV", "meaning": "Future Value"}},
      {{"variable": "r", "meaning": "interest rate per period"}},
      {{"variable": "n", "meaning": "number of periods"}}
    ],
    "tables": []
  }}
}}

Example 4 - Plain screenshot with no special content:
{{
  "metadata": {{
    "definitions": [],
    "formulas": [],
    "variables": [],
    "tables": []
  }}
}}
"""
            
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
                    "max_tokens": 1500,
                    "temperature": 0.2
                },
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse JSON response
                try:
                    # Remove markdown code blocks if present
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                    elif '```' in content:
                        content = re.sub(r'```\w*\s*', '', content).replace('```', '').strip()
                    
                    data = json.loads(content)
                    
                    # Extract and validate classification
                    classification = data.get('classification', {})
                    category = classification.get('category', 'FIGURE').upper()
                    
                    type_mapping = {
                        'CHART': VisualType.CHART,
                        'DIAGRAM': VisualType.DIAGRAM,
                        'FLOWCHART': VisualType.FLOWCHART,
                        'IMAGE': VisualType.IMAGE,
                        'FIGURE': VisualType.FIGURE
                    }
                    
                    visual_type = type_mapping.get(category, VisualType.FIGURE)
                    confidence = float(classification.get('confidence', 0.7))
                    
                    # Extract metadata
                    metadata = data.get('metadata', {})
                    
                    # Extract summary
                    summary_data = data.get('summary', {})
                    summary_text = summary_data.get('text', '')
                    summary_conf = float(summary_data.get('confidence', 0.8))
                    
                    return {
                        'visual_type': visual_type,
                        'confidence': min(confidence, 0.95),
                        'metadata': metadata,
                        'summary': summary_text,
                        'summary_confidence': summary_conf,
                        'method': 'mistral_vision_comprehensive'
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"    Failed to parse Mistral response: {e}")
                    print(f"    Response content: {content[:300]}")
                    
            else:
                print(f"    Mistral API error: {response.status_code}")
                
        except Exception as e:
            print(f"    Mistral comprehensive analysis failed: {e}")
        
        # Fallback
        return self._fallback_analysis(ocr_result)

    def _fallback_analysis(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Fallback when API fails"""
        return {
            'visual_type': VisualType.FIGURE,
            'confidence': 0.3,
            'metadata': {
                'definitions': [],
                'formulas': [],
                'variables': [],
                'tables': []
            },
            'summary': 'Visual element detected (classification unavailable)',
            'summary_confidence': 0.3,
            'method': 'fallback_heuristic'
        }

    def _convert_metadata_to_dataclasses(self, visual_type: VisualType, 
                                        metadata: Dict) -> Tuple:
        """
        Convert API metadata dict to appropriate dataclass objects.
        Returns: (chart_data, diagram_data, image_data, figure_data)
        """
        chart_data = None
        diagram_data = None
        image_data = None
        figure_data = None
        
        if visual_type == VisualType.CHART:
            chart_data = ChartSpecificData(
                chart_subtype=metadata.get('chart_subtype'),
                axes_info={
                    'x_axis': {'label': metadata.get('x_axis_label')},
                    'y_axis': {'label': metadata.get('y_axis_label')}
                },
                legend_items=metadata.get('legend_items', []),
                series_count=metadata.get('data_series_count', 0),
                grid_detected=metadata.get('has_grid', False),
                value_ranges={'detected': (
                    metadata.get('value_range', {}).get('min'),
                    metadata.get('value_range', {}).get('max')
                )} if metadata.get('value_range') else {}
            )
        
        elif visual_type in [VisualType.FLOWCHART, VisualType.DIAGRAM]:
            if visual_type == VisualType.FLOWCHART:
                subtype = 'flowchart'
            else:
                subtype = metadata.get('diagram_subtype')
            
            diagram_data = DiagramSpecificData(
                diagram_subtype=subtype,
                node_count=metadata.get('node_count', 0),
                has_decision_points=metadata.get('decision_points', 0) > 0,
                hierarchy_detected=metadata.get('has_hierarchy', False),
                layout_type=metadata.get('layout_type')
            )
        
        elif visual_type == VisualType.IMAGE:
            # Extract and validate new fields
            definitions = metadata.get('definitions', [])
            formulas = metadata.get('formulas', [])
            variables = metadata.get('variables', [])
            tables = metadata.get('tables', [])
            input_variables = metadata.get('input_variables', [])
            output_values = metadata.get('output_values', [])
            calculation_verification = metadata.get('calculation_verification')
            
            # Ensure they're lists (API might return null sometimes)
            if not isinstance(definitions, list):
                definitions = []
            if not isinstance(formulas, list):
                formulas = []
            if not isinstance(variables, list):
                variables = []
            if not isinstance(tables, list):
                tables = []
            if not isinstance(input_variables, list):
                input_variables = []
            if not isinstance(output_values, list):
                output_values = []
            
            image_data = ImageSpecificData(
                image_subtype=metadata.get('image_subtype'),
                contains_text=metadata.get('contains_text', False),
                text_density=metadata.get('text_density', 'none'),
                is_embedded_table=metadata.get('is_embedded_table', False),
                definitions=definitions,
                formulas=formulas,
                variables=variables,
                tables=tables,
                input_variables=input_variables,
                output_values=output_values,
                calculation_verification=calculation_verification
            )
        
        elif visual_type == VisualType.FIGURE:
            figure_data = FigureSpecificData(
                is_composite=metadata.get('is_composite', False),
                sub_figure_count=metadata.get('sub_figure_count', 0),
                contains_chart=metadata.get('contains_chart', False),
                contains_diagram=metadata.get('contains_diagram', False),
                contains_image=metadata.get('contains_image', False)
            )
        
        return chart_data, diagram_data, image_data, figure_data

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

    def extract_calculations_for_image(self, image: Image.Image, ocr_result: OCRResult, 
                                      nearby_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract input variables, formulas, and output values from IMAGE segments.
        Verifies outputs by cross-validating with formulas and inputs.
        """
        if not self.api_key:
            return {
                'input_variables': [],
                'output_values': [],
                'calculation_verification': None
            }
        
        try:
            img_base64 = self._encode_image(image)
            
            # Build context
            ocr_context = ""
            if ocr_result and ocr_result.raw_text:
                ocr_context = f"\n\n**Text detected in image (OCR):**\n{ocr_result.raw_text[:1000]}"
            
            nearby_context = ""
            if nearby_text:
                nearby_context = f"\n\n**Nearby text context:**\n{nearby_text[:500]}"
            
            prompt = f"""Analyze this image to extract calculation-related information.

{ocr_context}
{nearby_context}

**Your task:**
1. Extract all INPUT VARIABLES and their values shown in the image
2. Extract all OUTPUT VALUES (calculated results) shown in the image
3. Identify FORMULAS used (either visible or inferrable from context)
4. Verify outputs by checking if they match expected calculations

**INPUT VARIABLES:**
- Extract variables and their values from input sections, parameter lists, labeled fields
- Format: {{"variable": "name", "value": "value", "unit": "unit if shown"}}
- Only extract what is EXPLICITLY VISIBLE in the image

**OUTPUT VALUES:**
- Extract calculated results from output sections, result tables, calculated fields
- Format: {{"output_name": "name", "value": "value", "location": "where in image"}}
- Only extract what is EXPLICITLY VISIBLE in the image

**FORMULAS:**
- Extract formulas that are visible OR can be inferred from:
  * Input/output relationships
  * Context from nearby text
  * Domain knowledge (e.g., Black-Scholes for option pricing)
- Format: {{"formula": "formula", "description": "what it calculates", "location": "where found or 'inferred'"}}

**VERIFICATION:**
- Compare output values with expected calculations using inputs and formulas
- Note any discrepancies or matches
- Format: {{"verified": true/false, "matches": true/false, "differences": ["list of any differences found"]}}

**RESPONSE FORMAT (JSON only):**
{{
  "input_variables": [{{"variable": "...", "value": "...", "unit": "..."}}],
  "output_values": [{{"output_name": "...", "value": "...", "location": "..."}}],
  "formulas": [{{"formula": "...", "description": "...", "location": "..."}}],
  "verification": {{
    "verified": true/false,
    "matches": true/false,
    "differences": ["any differences found"]
  }}
}}
"""
            
            messages = [
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
            ]
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.vision_model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    calc_data = json.loads(json_match.group())
                    return {
                        'input_variables': calc_data.get('input_variables', []),
                        'output_values': calc_data.get('output_values', []),
                        'calculation_verification': calc_data.get('verification')
                    }
            
            return {
                'input_variables': [],
                'output_values': [],
                'calculation_verification': None
            }
            
        except Exception as e:
            print(f"    Warning: Calculation extraction failed: {e}")
            return {
                'input_variables': [],
                'output_values': [],
                'calculation_verification': None
            }


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
        self.term_in_multiword_concepts = defaultdict(int)  # token -> count of multi-word concepts containing it
        # Config for gating overly-generic single-word concepts, derived from taxonomy stats (no hardcoded list).
        # Rationale: If a single token appears across many concepts, linking a 1-word concept on that token
        # tends to create systematic false positives.
        self._single_term_generic_df_ratio_threshold = 0.08  # 8%+ of concepts contain the token => generic
        self._single_term_generic_df_min = 3                 # require at least this many concepts to call it generic
        
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
                'context_terms': set(),     # Related/synonym terms
                'aliases': set(),           # Multi-word aliases + acronyms
                'acronyms': set()           # Acronyms extracted from concept string
            }
            
            # Extract acronyms/aliases from concept string (e.g., "LIBOR (London Interbank Offer Rate)")
            parsed = self._parse_concept_name(concept_name)

            # Extract and normalize primary terms
            primary_terms = self._extract_terms(parsed['main'])
            concept_entry['primary_terms'] = primary_terms
            concept_entry['normalized_terms'].update(primary_terms)

            # Add acronyms + alias phrases as searchable signals
            concept_entry['acronyms'].update(parsed['acronyms'])
            concept_entry['aliases'].update(parsed['aliases'])
            concept_entry['normalized_terms'].update(parsed['acronyms'])
            # Also add alias terms (tokenized) to normalized_terms
            for alias in parsed['aliases']:
                concept_entry['normalized_terms'].update(self._extract_terms(alias))
            
            # Extract context terms from tags
            tags = row.get('Tag(s)', '')
            if pd.notna(tags) and tags:
                tag_terms = self._extract_terms(str(tags))
                concept_entry['context_terms'] = tag_terms
                concept_entry['normalized_terms'].update(tag_terms)
            
            # Store in map
            self.concept_map[concept_id] = concept_entry
        
        print(f"Built concept index with {len(self.concept_map)} concepts")

    def _parse_concept_name(self, concept_name: str) -> Dict[str, Any]:
        """
        Parse concept strings like:
          - "LIBOR (London Interbank Offer Rate)"
          - "Treasury Bills (T-bills)"
          - "TED Spread"
        Returns:
          { main: str, acronyms: set[str], aliases: set[str] }
        """
        if not concept_name:
            return {"main": "", "acronyms": set(), "aliases": set()}

        text = str(concept_name).strip()
        acronyms = set()
        aliases = set()

        # Pull any parenthetical content as an alias
        paren_matches = re.findall(r'\(([^)]+)\)', text)
        for p in paren_matches:
            p_clean = p.strip()
            if p_clean:
                aliases.add(p_clean)
                # If parenthetical looks like acronym ("T-bills", "LIBOR", "ETF"), store it too
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{1,15}s?", p_clean):
                    acronyms.add(p_clean.lower())

        # Remove parentheses from the main label
        main = re.sub(r'\s*\([^)]*\)\s*', ' ', text).strip()

        # If the main itself is an acronym-like token, store it
        main_token = main.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{1,15}s?", main_token):
            acronyms.add(main_token.lower())

        # Add a few normalization-friendly variants for aliases
        alias_variants = set()
        for a in list(aliases) + [main]:
            a = (a or "").strip()
            if not a:
                continue
            alias_variants.add(a)
            alias_variants.add(a.replace("-", " "))
            alias_variants.add(re.sub(r"\s+", " ", a))
        aliases |= alias_variants

        # Add special-case finance aliases that commonly appear hyphenated in OCR/LLM summaries
        # (Keeps this practical without requiring an external synonym model.)
        if "t-bill" in " ".join([main.lower()] + [x.lower() for x in aliases]):
            aliases |= {"treasury bill", "treasury bills", "treasury-bill", "treasury-bills", "t bill", "t bills"}
            acronyms |= {"t-bill", "t-bills"}
        if "libor" in " ".join([main.lower()] + [x.lower() for x in aliases]):
            aliases |= {"london interbank offer rate", "london interbank offered rate"}
            acronyms |= {"libor"}

        # Normalize acronyms
        acronyms = {self._normalize_text(a) for a in acronyms if a}

        return {"main": main, "acronyms": acronyms, "aliases": aliases}
    
    def _compute_term_statistics(self):
        """Compute term frequencies across all concepts for TF-IDF"""
        all_documents = []
        
        for concept_data in self.concept_map.values():
            doc_terms = list(concept_data['normalized_terms'])
            all_documents.append(doc_terms)
            
            # Count term frequencies
            for term in doc_terms:
                self.term_frequencies[term] += 1

            # Track which terms occur in multi-word *primary* concepts (helps gate generic 1-word concepts)
            primary_terms = concept_data.get('primary_terms', set()) or set()
            if len(primary_terms) >= 2:
                for t in primary_terms:
                    self.term_in_multiword_concepts[t] += 1
        
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
        # Keep hyphens but also treat them as separators (we add both forms)
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Split and filter
        terms = set()
        raw_tokens = text.split()
        for word in raw_tokens:
            word = word.strip('-_')
            
            # Filter criteria
            if len(word) >= 3 and word not in self._get_stop_words():
                terms.add(word)

            # Also split hyphenated tokens into parts, and add parts
            if '-' in word:
                for part in word.split('-'):
                    part = part.strip('-_')
                    if len(part) >= 3 and part not in self._get_stop_words():
                        terms.add(part)
        
        return terms

    def _normalize_text(self, text: str) -> str:
        """Lowercase, normalize whitespace, normalize hyphens for matching."""
        if not text:
            return ""
        t = str(text).lower().strip()
        t = t.replace("–", "-").replace("—", "-")
        t = re.sub(r"\s+", " ", t)
        return t

    def _is_generic_single_term(self, term: str) -> bool:
        """
        Determine whether a single-word concept term is "generic" based on taxonomy statistics.
        Generic terms appear across many concepts (high document frequency), so linking them alone is noisy.
        """
        term = self._normalize_text(term)
        if not term:
            return False

        n = max(int(self.document_count or 0), 0)
        if n <= 0:
            return False

        df = int(self.term_frequencies.get(term, 0))
        # If the token participates in any multi-word primary concept (e.g., "TED Spread"),
        # then a 1-word concept with the same token (e.g., "Spread") is usually too generic to link.
        if int(self.term_in_multiword_concepts.get(term, 0)) >= 1 and df >= 2:
            return True

        if df < self._single_term_generic_df_min:
            return False

        return (df / n) >= self._single_term_generic_df_ratio_threshold
    
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

        print('Search Terms:', search_terms)
        
        # Score all concepts
        scored_matches = []
        
        for concept_id, concept_data in self.concept_map.items():
            match_score = self._score_concept_match(
                search_terms=search_terms,
                search_context=search_context,
                concept_data=concept_data
            )
            
            if match_score['total_score'] > 0.5:  # Minimum threshold
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

        # DEDUPLICATION: Keep only lowest Bloom level for duplicate concepts
        deduplicated_matches = {}
        for match in scored_matches:
            concept_name = match['concept_name']
            
            if concept_name not in deduplicated_matches:
                # First occurrence - add it
                deduplicated_matches[concept_name] = match
            else:
                # Duplicate found - compare Bloom levels
                existing_match = deduplicated_matches[concept_name]
                existing_level = existing_match['bloom_level']
                new_level = match['bloom_level']
                
                # Keep the one with LOWER Bloom level (1 = Remember is lowest)
                if new_level < existing_level:
                    deduplicated_matches[concept_name] = match
                elif new_level == existing_level:
                    # Same level - keep the one with higher confidence
                    if match['confidence'] > existing_match['confidence']:
                        deduplicated_matches[concept_name] = match

        # Convert back to list and sort by confidence
        scored_matches = list(deduplicated_matches.values())
        scored_matches.sort(key=lambda x: x['confidence'], reverse=True)

        # Log results
        print(f"Found {len(scored_matches)} concept links (after deduplication)")

        for match in scored_matches:
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
        
        # Gate: avoid linking extremely generic one-word concepts unless there is strong evidence
        if self._should_gate_generic_single_term(concept_data, search_context):
            return {
                'total_score': 0.0,
                'method': 'gated_generic_single_term',
                'details': {k: 0.0 for k in score_breakdown}
            }

        # SIGNAL 1: Exact phrase / alias match (30 points max)
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

    def _should_gate_generic_single_term(self, concept_data: Dict, search_context: Dict) -> bool:
        """
        If the taxonomy contains concepts like "Spread" or "Rate", don't link them
        unless we have strong evidence (caption/summary exact phrase, or acronym/alias match).
        """
        primary_terms = concept_data.get("primary_terms", set()) or set()
        acronyms = concept_data.get("acronyms", set()) or set()
        aliases = concept_data.get("aliases", set()) or set()

        # Not a single-term concept => don't gate
        if len(primary_terms) >= 2:
            return False

        # Single-term concept; if it's not in our generic list, allow normal scoring
        only_term = next(iter(primary_terms), "")
        if not only_term or not self._is_generic_single_term(only_term):
            return False

        # Allow ONLY if it appears as a standalone caption/title.
        # We intentionally do NOT allow summary/OCR matches for generic single-word concepts,
        # because they appear frequently in explanatory prose (e.g., "TED spread") and cause over-linking.
        caption = self._normalize_text(search_context.get("caption", ""))

        # Standalone caption heuristics: caption is short and begins with the term
        # Examples allowed:
        #   "Spread"
        #   "Spread: definition ..."
        #   "Spread - ..."
        if caption:
            if len(caption) <= 80:
                if re.match(rf"^{re.escape(only_term)}(\b|[\s:\-–—])", caption, flags=re.IGNORECASE):
                    return False

        # Allow if any acronym/alias phrase matches strongly anywhere
        combined = self._normalize_text(search_context.get("combined_text", ""))
        for a in acronyms:
            if self._normalize_text(a) == only_term:
                continue
            if a and self._contains_whole_phrase(combined, a):
                return False
        for alias in aliases:
            alias_n = self._normalize_text(alias)
            if alias_n == only_term:
                continue
            if alias_n and self._contains_whole_phrase(combined, alias_n):
                return False

        # Otherwise: gate it out
        return True

    def _score_exact_match(self, concept_name: str, text: str) -> float:
        """
        Exact phrase matching with position weighting.
        
        Returns 0-1:
        - 1.0: Exact match in high-value context (caption)
        - 0.8: Exact match in medium-value context (summary)
        - 0.6: Exact match in low-value context (OCR/nearby)
        """
        """
        Exact match now supports:
        - whole-phrase word-boundary matching
        - hyphen/space variants ("t-bill" vs "t bill")
        - aliases/acronyms extracted from the taxonomy concept name
        """
        text_norm = self._normalize_text(text)
        if not text_norm:
            return 0.0

        parsed = self._parse_concept_name(concept_name)
        candidates = set()
        candidates.add(concept_name)
        candidates.add(parsed.get("main", ""))
        candidates |= set(parsed.get("aliases", set()))
        candidates |= set(parsed.get("acronyms", set()))

        # Score: prefer main phrase, but allow aliases
        best = 0.0
        for c in candidates:
            c_norm = self._normalize_text(c)
            if not c_norm:
                continue
            if self._contains_whole_phrase(text_norm, c_norm):
                # Boost if multi-word or acronym-like (LIBOR, TED, etc.)
                if len(c_norm.split()) >= 2 or re.fullmatch(r"[a-z]{2,10}(-[a-z]{1,10})?s?", c_norm):
                    best = max(best, 1.0)
                else:
                    best = max(best, 0.7)
        return best

    def _contains_whole_phrase(self, haystack: str, needle: str) -> bool:
        """Word-boundary phrase match; treats hyphens/spaces as equivalent-ish."""
        if not haystack or not needle:
            return False
        # Allow hyphen <-> space flexibility inside the needle
        escaped = re.escape(needle)
        escaped = escaped.replace(r"\-", r"[-\s]")
        # word boundaries around the whole phrase
        pattern = rf"(?<!\w){escaped}(?!\w)"
        return re.search(pattern, haystack, flags=re.IGNORECASE) is not None
    
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
        """
        Improved fuzzy matching:
        - operates on *tokens* and short phrases, not whole-text scanning only
        - boosts acronym/alias near-misses (OCR errors)
        - avoids matching generic single words too easily
        """
        text_norm = self._normalize_text(text)
        if not text_norm:
            return 0.0

        parsed = self._parse_concept_name(concept_name)
        main_terms = list(self._extract_terms(parsed.get("main", concept_name)))

        # If single generic token (as per taxonomy stats), don't fuzzy-match it
        # (prevents concepts like "Spread" from matching everywhere).
        if len(main_terms) == 1 and self._is_generic_single_term(main_terms[0]):
            return 0.0

        # Build candidate tokens from text
        words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text_norm)
        if not words:
            return 0.0

        best = 0.0

        # 1) Acronym fuzzy (high value): compare against tokens directly
        for ac in parsed.get("acronyms", set()):
            ac_n = self._normalize_text(ac)
            if not ac_n:
                continue
            for w in words:
                sim = self._string_similarity(ac_n, w)
                if sim >= 0.88:
                    best = max(best, sim)

        # 2) Term-level fuzzy: require at least 2 term hits for multi-term concepts
        term_hits = 0
        for t in main_terms:
            t_n = self._normalize_text(t)
            if not t_n:
                continue
            local_best = 0.0
            for w in words:
                # also allow hyphen/space variants by removing hyphens for similarity
                sim = self._string_similarity(t_n.replace("-", ""), w.replace("-", ""))
                local_best = max(local_best, sim)
            if local_best >= 0.88:
                term_hits += 1

        if len(main_terms) >= 2 and term_hits >= 2:
            best = max(best, 0.9)
        elif len(main_terms) == 1 and term_hits == 1:
            best = max(best, 0.82)

        return best if best >= 0.8 else 0.0
    
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
        """Process a single visual segment with ONE API call"""
        
        image = Image.open(segment.image_path)
        
        # STEP 1: Run OCR
        print(f"    Running OCR...")
        segment.ocr_result = OCRProcessor.process_image(image)
        
        # STEP 2: Single comprehensive API call (classification + metadata + summary)
        print(f"    Analyzing with Mistral API (comprehensive)...")
        analysis_result = self.mistral_api.analyze_visual_comprehensive(
            image, 
            segment.ocr_result
        )
        
        # STEP 3: Apply results to segment
        segment.segment_type = analysis_result['visual_type']
        segment.classification_confidence = analysis_result['confidence']
        segment.classification_method = analysis_result['method']
        segment.summary = analysis_result['summary']
        segment.summary_confidence = analysis_result['summary_confidence']
        
        print(f"    → Classified as {segment.segment_type.value} (confidence: {segment.classification_confidence:.2f})")
        
        # STEP 4: Convert API metadata to dataclass objects
        chart_data, diagram_data, image_data, figure_data = \
            self.mistral_api._convert_metadata_to_dataclasses(
                segment.segment_type,
                analysis_result['metadata']
            )
        
        segment.chart_data = chart_data
        segment.diagram_data = diagram_data
        segment.image_data = image_data
        segment.figure_data = figure_data
        
        # STEP 4.5: Extract calculations for IMAGE segments
        if segment.segment_type == VisualType.IMAGE and image_data:
            print(f"    Extracting calculations (inputs, formulas, outputs)...")
            calc_data = self.mistral_api.extract_calculations_for_image(
                image,
                segment.ocr_result,
                segment.nearby_text
            )
            
            # Merge calculation data into image_data
            if calc_data.get('input_variables'):
                image_data.input_variables = calc_data['input_variables']
            if calc_data.get('output_values'):
                image_data.output_values = calc_data['output_values']
            if calc_data.get('calculation_verification'):
                image_data.calculation_verification = calc_data['calculation_verification']
            
            # Update segment reference
            segment.image_data = image_data
            
            if calc_data.get('input_variables') or calc_data.get('output_values'):
                print(f"    → Extracted {len(calc_data.get('input_variables', []))} inputs, {len(calc_data.get('output_values', []))} outputs")
        
        # STEP 5: Extract structured text for search/linking
        segment.extracted_text_structured = OCRProcessor.extract_structured_text(
            segment.ocr_result, 
            segment.segment_type
        )
        
        # STEP 6: Mermaid extraction (optional - still separate call if needed)
        if self.use_mermaid and segment.segment_type in [VisualType.DIAGRAM, VisualType.FLOWCHART]:
            print(f"    Extracting Mermaid representation...")
            segment.mermaid_repr = self.mistral_api.extract_mermaid_representation(image, segment)
            if segment.mermaid_repr and segment.mermaid_repr.mermaid_code:
                print(f"    → Mermaid extraction successful")
        
        # STEP 7: Detect caption (keep existing logic)
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
        
        # STEP 8: Link to concepts (keep existing)
        if self.concept_linker:
            segment.linked_concept_ids = self.concept_linker.link_concepts(segment)
        
        # STEP 9: Extract context (keep existing)
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
        book_id="textbook_001",
        pdf_path="D:\\D-Downloads\\complex.pdf",
        taxonomy_path="D:\\D-Downloads\\Don M. Chance, Roberts Brooks - An Introduction to Derivatives and Risk Management (2015, South-Western College Pub).xlsx",
        output_dir="./extracted_visuals",
        use_mermaid=False  # Enable Mermaid extraction
    )
    
    segments = pipeline.process()
    
    print("\n=== Extraction Summary ===")
    print(f"Total visual elements: {len(segments)}")
    print(f"Segments with Mermaid representations: {sum(1 for s in segments if s.mermaid_repr)}")

