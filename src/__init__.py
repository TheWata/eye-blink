"""Eye Blink Analytics - pipeline de visao computacional e engenharia de dados.

Modulos:
    vision_tracker: ingestao de video, Face Mesh (MediaPipe) e deteccao de piscadas.
    metrics_exporter: calculo de EAR, agregacao de telemetria e exportacao para CSV.
"""

from src.metrics_exporter import (
    BlinkDetector,
    BlinkDetectorConfig,
    FrameMetrics,
    MetricsExporter,
    compute_ear,
    format_timestamp,
)
from src.vision_tracker import (
    EYE_LANDMARKS,
    TrackerConfig,
    VisionTracker,
    VisionTrackerError,
)

__all__ = [
    "BlinkDetector",
    "BlinkDetectorConfig",
    "EYE_LANDMARKS",
    "FrameMetrics",
    "MetricsExporter",
    "TrackerConfig",
    "VisionTracker",
    "VisionTrackerError",
    "compute_ear",
    "format_timestamp",
]

__version__ = "1.0.0"
