from dataclasses import dataclass
from typing import Any, Protocol, Sequence


Message = dict[str, Any]


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 20
    do_sample: bool = False
    enable_thinking: bool = False


class ModelBackend(Protocol):
    def generate(
        self,
        messages: Sequence[Message],
        generation: GenerationConfig,
    ) -> str:
        """Generate one assistant response for a chat prompt."""
