from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
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
    discipline: Union[DisciplineEnum, str] = DisciplineEnum.ARCHITECTURAL
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
    area_sqm: Optional[float] = None

class FloorItem(BaseModel):
    page: int
    floor_name: str
    gross_floor_area_sqm: Optional[float] = None
    net_internal_area_sqm: Optional[float] = None
    room_breakdown: List[RoomAreaItem] = []

class FloorAreaMetrics(BaseModel):
    is_floor_plan: bool = False
    gross_floor_area_sqm: Optional[float] = None  # Cumulated total gross floor area across all floors
    net_internal_area_sqm: Optional[float] = None # Cumulated net internal area
    cumulated_floors_count: int = 1
    confidence_score: float = 0.90
    source: Optional[str] = "title_block_schedule"  # title_block_schedule, vector_polygon_estimation, gemini_multimodal
    floors: List[FloorItem] = []
    room_breakdown: List[RoomAreaItem] = []

class SummaryMetricsSchema(BaseModel):
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0

class GeminiAuditSchema(BaseModel):
    """
    Developer API strict schema: has NO Dict[str, Any], NO Dict[str, int], NO UUIDs,
    and NO additionalProperties. Completely compatible with Gemini Developer API mode.
    """
    overall_status: OverallStatusEnum
    summary_metrics: SummaryMetricsSchema
    discrepancies: List[DiscrepancyItem] = []
    floor_area_metrics: Optional[FloorAreaMetrics] = None

class ComplianceResponse(BaseModel):
    drawing_version_id: uuid.UUID
    overall_status: OverallStatusEnum
    preflight_metrics: Dict[str, Any]
    summary_metrics: Dict[str, int]
    discrepancies: List[DiscrepancyItem]
    extracted_geometry: Optional[Dict[str, Any]] = None
    floor_area_metrics: Optional[FloorAreaMetrics] = None
