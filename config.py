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
    # Image size used for YOLO inference. Override via: POTHOLE_IMG_SIZE=640
    IMAGE_SIZE: int = int(os.getenv("POTHOLE_IMG_SIZE", "0"))  # 0 = auto-select by device
    CPU_IMAGE_SIZE: int = int(os.getenv("POTHOLE_CPU_IMG_SIZE", "416"))
    GPU_IMAGE_SIZE: int = int(os.getenv("POTHOLE_GPU_IMG_SIZE", "640"))


    # Target processing & playback FPS for video mode (0 = unthrottled maximum speed).
    TARGET_VIDEO_FPS: float = float(os.getenv("POTHOLE_TARGET_FPS", "0"))
    DEFAULT_SKIP_FRAMES: int = int(os.getenv("POTHOLE_SKIP_FRAMES", "1"))


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

    @classmethod
    def get_image_size(cls) -> int:
        """Return the effective YOLO inference image size.

        If IMAGE_SIZE is explicitly set (non-zero), that value is used.
        Otherwise auto-selects CPU_IMAGE_SIZE (320) on CPU or GPU_IMAGE_SIZE (640) on GPU.
        """
        if cls.IMAGE_SIZE != 0:
            return cls.IMAGE_SIZE
        return cls.GPU_IMAGE_SIZE if cls.get_device() == "cuda" else cls.CPU_IMAGE_SIZE

    # Visual bounding box styling (OpenCV BGR format)
    # Bright Tangerine / Safety Orange: (20, 120, 255) for high contrast against dark asphalt
    BOX_COLOR: tuple = (20, 120, 255)
    BOX_THICKNESS: int = 3
    LABEL_BG_COLOR: tuple = (20, 120, 255)
    LABEL_TEXT_COLOR: tuple = (255, 255, 255)
    CORNER_ACCENT_COLOR: tuple = (0, 240, 255)  # Bright cyan/yellow accent for corners

    # Perspective/depth compensation for severity grading.
    # Vertical Y-position in the frame is used as a depth proxy (dashcam footage):
    #   Top of frame  → far away  → use DEPTH_FAR_SCALE  (objects look small but may be large)
    #   Bottom of frame → close   → use DEPTH_NEAR_SCALE (objects look large due to perspective)
    # The raw area_ratio is divided by lerp(FAR, NEAR, y_norm) before thresholding.
    # Lower DEPTH_FAR_SCALE = more aggressive correction at the horizon.
    DEPTH_FAR_SCALE: float = float(os.getenv("POTHOLE_DEPTH_FAR_SCALE", "0.25"))
    DEPTH_NEAR_SCALE: float = float(os.getenv("POTHOLE_DEPTH_NEAR_SCALE", "1.0"))

    # Video & Stream defaults
    DEFAULT_FPS: int = int(os.getenv("POTHOLE_DEFAULT_FPS", "13"))
    DEFAULT_CAMERA_INDEX: int = 0
    WINDOW_TITLE: str = "AI Pothole Detection System (Press Q to quit)"

    # WhatsApp Automated Municipal Alert System
    WHATSAPP_ENABLED: bool = os.getenv("POTHOLE_WHATSAPP_ENABLED", "1").lower() in ("1", "true", "yes")
    WHATSAPP_AUTHORITY_PHONE: str = os.getenv("POTHOLE_AUTHORITY_PHONE", "+919591152862")
    WHATSAPP_PORT: int = int(os.getenv("POTHOLE_WHATSAPP_PORT", "5005"))
    WHATSAPP_MIN_SEVERITY: str = os.getenv("POTHOLE_WHATSAPP_MIN_SEVERITY", "Low")
    WHATSAPP_COOLDOWN_SECONDS: float = float(os.getenv("POTHOLE_WHATSAPP_COOLDOWN", "15.0"))


    def __init__(self):
        """Ensure runtime directories exist."""
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
