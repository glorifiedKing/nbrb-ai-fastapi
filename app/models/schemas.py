from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid

class DisciplineEnum(str, Enum):
    ARCHITECTURAL = "architectural"
    STRUCTURAL = "structural"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"

class OverallStatusEnum(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    REJECTED_PREFLIGHT = "REJECTED_PREFLIGHT"

class AnalyzeDrawingRequest(BaseModel):
    drawing_version_id: uuid.UUID
    drawing_id: Optional[uuid.UUID] = None
    discipline: DisciplineEnum
    file_path: str
    occupancy_class: Optional[str] = "Class A (Residential)"

class PreflightResult(BaseModel):
    is_valid: bool
    scale_detected: Optional[str] = None
    scale_factor_m_per_pt: Optional[float] = None
    dpi: int = 300
    rejection_reason: Optional[str] = None

class DiscrepancyItem(BaseModel):
    element_id: str
    room_or_grid: str
    clause_reference: str
    required_value: str
    observed_value: str
    severity: str  # ERROR, WARNING, INFO
    recommendation: str

class RoomAreaItem(BaseModel):
    name: str
    area_sqm: float

class FloorAreaMetrics(BaseModel):
    is_floor_plan: bool = False
    gross_floor_area_sqm: Optional[float] = None
    net_internal_area_sqm: Optional[float] = None
    confidence_score: float = 0.90
    source: Optional[str] = "title_block_schedule"  # title_block_schedule, vector_polygon_estimation, gemini_multimodal
    room_breakdown: List[RoomAreaItem] = []

class ComplianceResponse(BaseModel):
    drawing_version_id: uuid.UUID
    overall_status: OverallStatusEnum
    preflight_metrics: Dict[str, Any]
    summary_metrics: Dict[str, int]
    discrepancies: List[DiscrepancyItem]
    extracted_geometry: Optional[Dict[str, Any]] = None
    floor_area_metrics: Optional[FloorAreaMetrics] = None

