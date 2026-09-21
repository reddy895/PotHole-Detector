from config import config


def test_config_defaults():
    assert config.CONFIDENCE_THRESHOLD == 0.35
    assert config.IOU_THRESHOLD == 0.45
    assert config.TARGET_VIDEO_FPS in (0.0, 12.5, 15.0)


def test_get_image_size():
    # Should resolve to CPU size when on CPU
    size = config.get_image_size()
    assert size in (320, 416, 640)


def test_depth_scale_constants():
    assert 0.0 < config.DEPTH_FAR_SCALE <= 1.0
    assert config.DEPTH_NEAR_SCALE == 1.0
    assert config.DEPTH_FAR_SCALE < config.DEPTH_NEAR_SCALE


def test_whatsapp_config_defaults():
    assert isinstance(config.WHATSAPP_ENABLED, bool)
    assert config.WHATSAPP_PORT == 5005
    assert config.WHATSAPP_MIN_SEVERITY in ("Low", "Medium", "High")
    assert config.WHATSAPP_COOLDOWN_SECONDS > 0
