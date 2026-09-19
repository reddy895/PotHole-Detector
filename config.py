"""Configuration module for the Terminal-Based AI Pothole Detection System."""
from pathlib import Path
import os


class Config:
    """System configuration parameters."""

    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"

    # Model configuration
    DEFAULT_MODEL_PATH: Path = MODELS_DIR / "pothole.pt"
    MODEL_PATH: Path = Path(os.getenv("POTHOLE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

    # Detection thresholds
    CONFIDENCE_THRESHOLD: float = float(os.getenv("POTHOLE_CONFIDENCE_THRESHOLD", "0.35"))
    IOU_THRESHOLD: float = float(os.getenv("POTHOLE_IOU_THRESHOLD", "0.45"))
    IMAGE_SIZE: int = int(os.getenv("POTHOLE_IMG_SIZE", "640"))

    # Hardware device selection (GPU if CUDA is available, otherwise fallback to CPU)
    FORCE_DEVICE: str = os.getenv("POTHOLE_DEVICE", "auto")

    @classmethod
    def _cuda_available(cls) -> bool:
        """Safely check if CUDA is available (lazy torch import)."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def get_device(cls) -> str:
        """Automatically resolve the compute device (CUDA GPU or CPU)."""
        if cls.FORCE_DEVICE.lower() == "cpu":
            return "cpu"
        if cls.FORCE_DEVICE.lower() in ("cuda", "gpu"):
            return "cuda" if cls._cuda_available() else "cpu"
        return "cuda" if cls._cuda_available() else "cpu"

    @classmethod
    def get_device_name(cls) -> str:
        """Get human-readable device name."""
        device = cls.get_device()
        if device == "cuda":
            try:
                import torch
                return f"CUDA ({torch.cuda.get_device_name(0)})"
            except Exception:
                return "CUDA GPU"
        return "CPU"

    # Visual bounding box styling (OpenCV BGR format)
    # Bright Tangerine / Safety Orange: (20, 120, 255) for high contrast against dark asphalt
    BOX_COLOR: tuple = (20, 120, 255)
    BOX_THICKNESS: int = 3
    LABEL_BG_COLOR: tuple = (20, 120, 255)
    LABEL_TEXT_COLOR: tuple = (255, 255, 255)
    CORNER_ACCENT_COLOR: tuple = (0, 240, 255)  # Bright cyan/yellow accent for corners

    # Video & Stream defaults
    DEFAULT_FPS: int = 30
    DEFAULT_CAMERA_INDEX: int = 0
    WINDOW_TITLE: str = "AI Pothole Detection System (Press Q to quit)"

    def __init__(self):
        """Ensure runtime directories exist."""
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
