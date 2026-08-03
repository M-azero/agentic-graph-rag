"""Vision-LLM OCR. Default: Gemma 4 (small, local via Ollama). Swappable to
Gemini by changing one config line. Both take an image + a 'transcribe this'
prompt through the same chat interface."""

from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage

from graphrag.config.settings import OCRCfg, Secrets
from graphrag.core.messages import content_to_text
from graphrag.llm.factory import build_chat_chain
from graphrag.ocr.base import OCREngine


class VisionLLMOCR(OCREngine):
    def __init__(self, cfg: OCRCfg, secrets: Secrets) -> None:
        v = cfg.vision_llm
        self._prompt = v.prompt
        # A vision model is just a chat model that accepts image content, so
        # the ordinary chat failover chain covers OCR too.
        self._model = build_chat_chain(
            provider=v.provider,
            model=v.model,
            secrets=secrets,
            fallbacks=v.fallbacks,
            temperature=0.0,
            max_tokens=4096,
        )

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": self._prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ]
        )
        result = self._model.invoke([message])
        return content_to_text(result.content)
