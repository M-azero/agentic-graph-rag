from graphrag.config.settings import OCRCfg, Secrets
from graphrag.core.errors import ConfigError
from graphrag.core.logging import get_logger
from graphrag.ocr.base import OCREngine

log = get_logger(__name__)


def _build_engine(engine: str, cfg: OCRCfg, secrets: Secrets) -> OCREngine:
    if engine == "vision_llm":
        from graphrag.ocr.vision_llm import VisionLLMOCR

        return VisionLLMOCR(cfg, secrets)
    if engine == "tesseract":
        from graphrag.ocr.tesseract import TesseractOCR

        return TesseractOCR(cfg)
    raise ConfigError(f"Unknown OCR engine: {engine}")


class FallbackOCR(OCREngine):
    """Try one engine, then another.

    This exists on top of the vision model's own failover chain because the two
    fail differently: that chain covers one API being down, this covers *every*
    vision API being down. Tesseract is local, ships in the image, and needs no
    key — so a scanned page still becomes text when nothing else answers, just
    less accurately.
    """

    def __init__(self, primary: OCREngine, backup: OCREngine, backup_name: str) -> None:
        self._primary = primary
        self._backup = backup
        self._backup_name = backup_name

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        try:
            return self._primary.extract_text(image_bytes, mime_type)
        except Exception as exc:
            log.warning("ocr_fallback", to=self._backup_name, error=str(exc))
        return self._backup.extract_text(image_bytes, mime_type)


def build_ocr(cfg: OCRCfg, secrets: Secrets) -> OCREngine:
    primary = _build_engine(cfg.engine, cfg, secrets)
    backup_name = cfg.fallback_engine
    if not backup_name or backup_name == cfg.engine:
        return primary
    try:
        backup = _build_engine(backup_name, cfg, secrets)
    except Exception as exc:
        # No backup is a worse deployment, not a broken one.
        log.warning("ocr_fallback_unavailable", engine=backup_name, error=str(exc))
        return primary
    return FallbackOCR(primary, backup, backup_name)


__all__ = ["FallbackOCR", "OCREngine", "build_ocr"]
