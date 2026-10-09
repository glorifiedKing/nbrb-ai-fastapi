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
    drawing_id: uuid.UUID
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

class ComplianceResponse(BaseModel):
    drawing_version_id: uuid.UUID
    overall_status: OverallStatusEnum
    preflight_metrics: Dict[str, Any]
    summary_metrics: Dict[str, int]
    discrepancies: List[DiscrepancyItem]
    extracted_geometry: Optional[Dict[str, Any]] = None
