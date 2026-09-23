"""Helper functions for hardware telemetry, image transformations, and time formatting."""
import base64
from datetime import datetime
from typing import Any, Dict

import cv2
import numpy as np


def get_device_info(preferred_device: str = "auto") -> Dict[str, Any]:
    """Retrieve detailed hardware acceleration information.

    Args:
        preferred_device: 'auto', 'cuda', or 'cpu'.

    Returns:
        Dictionary with hardware specifications and selected runtime device.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    device_name = "CPU"
    total_memory_mb = 0
    cuda_device_name = ""

    if cuda_available and preferred_device in ("auto", "cuda"):
        active_device = "cuda:0"
        try:
            cuda_device_name = torch.cuda.get_device_name(0)
            device_name = cuda_device_name
            total_memory_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            device_name = "CUDA (Device 0)"
    else:
        active_device = "cpu"
        device_name = "CPU (Software Inference)"

    return {
        "device": active_device,
        "is_cuda": "cuda" in active_device,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "vram_mb": total_memory_mb,
    }


def encode_image_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode OpenCV BGR image into a Base64 JPEG string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", image, encode_param)
    if not success:
        raise ValueError("Failed to encode image to JPEG buffer.")
    return base64.b64encode(buffer).decode("utf-8")


def decode_image_base64(base64_string: str) -> np.ndarray:
    """Decode a Base64 image string into an OpenCV BGR image array."""
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode base64 image data.")
    return image


def get_current_timestamp() -> str:
    """Return formatted current ISO timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
