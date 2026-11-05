"""Config-driven camera image-effect pipeline."""

from sensors.image_effects.solar_glare import SolarGlare


class CameraImageEffectPipeline:
    def __init__(self, config=None, device=None):
        config = config or {}
        self.solar_glare = SolarGlare(config.get("solar_glare"), device=device)

    @property
    def enabled(self) -> bool:
        return self.solar_glare.enabled

    def apply(self, rgb, **context):
        if not self.enabled:
            return rgb
        return self.solar_glare.apply(rgb, **context)
