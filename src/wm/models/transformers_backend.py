from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .base import GenerationConfig, Message


class TransformersBackend:
    """Local Hugging Face Transformers backend for Qwen3.5-style models."""

    def __init__(
        self,
        model_id: str,
        cache_dir: str | Path | None = None,
        *,
        device_map: str | dict[str, Any] = "auto",
        local_files_only: bool = False,
        torch_dtype: str | torch.dtype = "auto",
    ) -> None:
        self.model_id = model_id
        self.cache_dir = str(cache_dir) if cache_dir else None

        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "local_files_only": local_files_only,
        }
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = "auto"

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    def generate(
        self,
        messages: Sequence[Message],
        generation: GenerationConfig,
    ) -> str:
        # Qwen3.5 accepts text-only multimodal messages. Passing the switch
        # here keeps the backend usable without an OpenAI-compatible server.
        inputs = self.processor.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=generation.enable_thinking,
        )
        inputs = inputs.to(self.device)
        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=generation.max_new_tokens,
                do_sample=generation.do_sample,
                temperature=(
                    generation.temperature if generation.do_sample else None
                ),
                top_p=(generation.top_p if generation.do_sample else None),
                top_k=(generation.top_k if generation.do_sample else None),
            )

        generated = output[0][input_length:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
