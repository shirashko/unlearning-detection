from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _finish_reason_label(response: Any) -> Optional[str]:
    """Extract a clean finish reason (e.g. ``MAX_TOKENS``)."""
    try:
        candidate = response.candidates[0]
        fr = getattr(candidate, "finish_reason", None)
        if fr is None:
            return None
        fr_str = str(fr.name if getattr(fr, "name", None) else fr).strip()
        if not fr_str:
            return None
        return fr_str.split(".")[-1].upper()
    except (AttributeError, IndexError, TypeError):
        return None


class GeminiClient:
    """
    Domain-agnostic wrapper for the Gemini API.

    Handles authentication, generation parameters, and raw network I/O.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        api_key_env: str = "GOOGLE_API_KEY",
    ) -> None:
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"{api_key_env} is not set or empty, gemini API client cannot be constructed."
            )

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key_env = api_key_env
        self._api_key = api_key

    def generate_text(self, prompt: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Execute a raw text generation request against the Gemini endpoint.

        Returns:
            (text, error, finish_reason) text is empty when the call
            fails, finish_reason is set when the SDK returns candidates.
        """
        errors: List[str] = []

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            client = genai.Client(api_key=self._api_key)
            gen_cfg = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
            )
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_cfg,
            )
            text = getattr(resp, "text", "") or ""
            return text, None, _finish_reason_label(resp)
        except ImportError:
            errors.append("google-genai not installed")
        except Exception as e:
            errors.append(f"google-genai call failed: {e}")
            logger.warning(
                "google-genai call failed, falling back to legacy SDK: %s", e,
            )

        try:
            import google.generativeai as legacy  # type: ignore

            legacy.configure(api_key=self._api_key)
            gm = legacy.GenerativeModel(self.model)
            resp = gm.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "response_mime_type": "application/json",
                },
            )
            text = getattr(resp, "text", "") or ""
            return text, None, _finish_reason_label(resp)
        except ImportError:
            errors.append(
                "google-generativeai not installed, install one of "
                "'google-genai' or 'google-generativeai' to enable the judge step "
                "(e.g. `pip install google-genai`)."
            )
        except Exception as e:
            errors.append(f"google-generativeai call failed: {e}")

        error_msg = " | ".join(errors)
        logger.error("Gemini API invocation failed: %s", error_msg)
        return "", error_msg, None
