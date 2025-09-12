from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ImageStats(BaseModel):
    width: int
    height: int
    channels: int
    total_pixels: int
    file_size_mb: float
    mean_brightness: float
    std_brightness: float


class WeedDetection(BaseModel):
    areas_detected: int
    total_weed_area_pixels: int
    weed_coverage_percentage: float
    detection_sensitivity: float


class ProcessingSummary(BaseModel):
    processing_time_seconds: float
    image_stats: ImageStats
    weed_detection: WeedDetection
    analysis_status: str
    detected_issues: List[str]
    scale_factor: Optional[float] = 1.0


class ProcessingParameters(BaseModel):
    sensitivity: float
    algorithm: str
    version: str


class ProcessResponse(BaseModel):
    success: bool
    result_image_url: str
    summary: ProcessingSummary
    weed_polygons: List[List[List[int]]]
    analysis_notes: str
    processing_parameters: ProcessingParameters


class ProcessStatusResponse(BaseModel):
    status: str
    algorithms_available: List[str]
    supported_formats: List[str]
    max_file_size_mb: int
    max_image_dimension: int
    version: str


# Legacy schema for backward compatibility
class LegacyProcessResponse(BaseModel):
    result_image_url: str
    notes: Optional[str] = None
