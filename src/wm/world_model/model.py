from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import copy
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .data import TransitionDataset, TransitionRecord, WorldModelVocab
from .state import parse_observation


DEFAULT_QLORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "out_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
)


def _hidden_size(config: Any) -> int:
    for candidate in (
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
    ):
        if candidate is not None:
            return int(candidate)
    raise ValueError("Could not determine the Qwen hidden size")


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Qwen model has no parameters") from exc


def bernoulli_kl(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    """Return KL(q || p) for factorized Bernoulli state variables."""

    q = torch.sigmoid(q_logits).clamp(1e-6, 1 - 1e-6)
    p = torch.sigmoid(p_logits).clamp(1e-6, 1 - 1e-6)
    return q * (q.log() - p.log()) + (1 - q) * (
        (1 - q).log() - (1 - p).log()
    )


class QwenTextEncoder(nn.Module):
    """Qwen text encoder with optional 4-bit QLoRA adaptation."""

    def __init__(
        self,
        model_id: str,
        cache_dir: str | Path | None = None,
        *,
        local_files_only: bool = True,
        device_map: str | dict[str, Any] = "auto",
        torch_dtype: str | torch.dtype = "auto",
        max_length: int = 1024,
        freeze: bool = True,
        qlora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if qlora and not freeze:
            raise ValueError("QLoRA requires freeze=True for the base model")
        self.qlora = qlora
        self.freeze = freeze
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        )
        load_kwargs: dict[str, Any] = {
            "cache_dir": str(cache_dir) if cache_dir else None,
            "local_files_only": local_files_only,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }
        if qlora:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "QLoRA requires a CUDA device, but torch.cuda.is_available() is false"
                )
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "QLoRA requires bitsandbytes and a Transformers build with "
                    "BitsAndBytesConfig. Run `uv sync` after installing the "
                    "project dependencies."
                ) from exc
            compute_dtype = torch.bfloat16
            if not torch.cuda.is_bf16_supported():
                compute_dtype = torch.float16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["torch_dtype"] = compute_dtype
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
        base_output_size = _hidden_size(self.model.config)
        if qlora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
                from peft import prepare_model_for_kbit_training
            except ImportError as exc:
                raise RuntimeError(
                    "QLoRA requires the `peft` package. Run `uv sync` to install "
                    "the project dependencies."
                ) from exc
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=False,
            )
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=list(
                        lora_target_modules or DEFAULT_QLORA_TARGET_MODULES
                    ),
                ),
            )
            self.model.print_trainable_parameters()
        else:
            self.model.eval()
        self.max_length = max_length
        self.output_size = base_output_size
        if freeze:
            for name, parameter in self.model.named_parameters():
                if qlora and "lora_" in name:
                    parameter.requires_grad_(True)
                else:
                    parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return _module_device(self.model)

    def train(self, mode: bool = True) -> "QwenTextEncoder":
        super().train(mode)
        if self.freeze and not self.qlora:
            # The frozen backbone must remain an inference encoder even when
            # the variational heads are put into training mode.
            self.model.eval()
        return self

    def _prepare_inputs(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        try:
            batch = self.processor(
                text=list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        except (TypeError, ValueError):
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            batch = tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self._prepare_inputs(texts)
        # LoRA adapters need gradients during training, but inference-time
        # planning/evaluation must stay under no_grad.  Keeping gradients on
        # here makes every candidate-action rollout retain a large Qwen
        # computation graph and can exhaust an otherwise free GPU.
        context = (
            torch.enable_grad()
            if self.training and (self.qlora or not self.freeze)
            else torch.no_grad()
        )
        with context:
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            hidden = hidden_states[-1]
        else:
            hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("Qwen forward pass did not return hidden states")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return hidden[:, -1, :].float()
        last_indices = attention_mask.sum(dim=1).long().clamp_min(1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_indices].float()

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        if not self.qlora:
            return {}
        from peft import get_peft_model_state_dict

        return {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(self.model).items()
        }

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not self.qlora:
            if state:
                raise ValueError("Checkpoint contains LoRA weights but QLoRA is disabled")
            return
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state)


@dataclass
class WorldModelPrediction:
    posterior_state_probabilities: torch.Tensor
    predicted_next_state_probabilities: torch.Tensor
    predicted_facts: list[list[str]]
    reconstructed_words: list[list[str]]


class QwenWorldModel(nn.Module):
    """Qwen-backed structured ALFWorld world model.

    The Qwen backbone produces an observation embedding. The posterior,
    transition, and observation heads use factorized distributions so the
    variational free-energy terms remain tractable.
    """

    def __init__(
        self,
        model_id: str,
        vocab: WorldModelVocab,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        device_map: str | dict[str, Any] = "auto",
        torch_dtype: str | torch.dtype = "auto",
        state_hidden_size: int = 512,
        action_embedding_size: int = 128,
        max_length: int = 1024,
        freeze_backbone: bool = True,
        qlora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.architecture = {
            "model_id": model_id,
            "state_hidden_size": state_hidden_size,
            "action_embedding_size": action_embedding_size,
            "max_length": max_length,
            "freeze_backbone": freeze_backbone,
            "qlora": qlora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": list(lora_target_modules or DEFAULT_QLORA_TARGET_MODULES),
        }
        self.encoder = QwenTextEncoder(
            model_id,
            cache_dir,
            local_files_only=local_files_only,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_length=max_length,
            freeze=freeze_backbone,
            qlora=qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        model_device = self.encoder.device
        self.posterior = nn.Sequential(
            nn.Linear(self.encoder.output_size, state_hidden_size),
            nn.GELU(),
            nn.LayerNorm(state_hidden_size),
            nn.Linear(state_hidden_size, len(vocab.facts)),
        ).to(model_device)
        self.action_embedding = nn.Embedding(
            max(1, len(vocab.actions)), action_embedding_size
        ).to(model_device)
        self.transition = nn.Sequential(
            nn.Linear(len(vocab.facts) + action_embedding_size, state_hidden_size),
            nn.GELU(),
            nn.LayerNorm(state_hidden_size),
            nn.Linear(state_hidden_size, len(vocab.facts)),
        ).to(model_device)
        self.observation_decoder = nn.Sequential(
            nn.Linear(len(vocab.facts), state_hidden_size),
            nn.GELU(),
            nn.LayerNorm(state_hidden_size),
            nn.Linear(state_hidden_size, len(vocab.observation_words)),
        ).to(model_device)
        self._online_vfe_initial_state: dict[str, Any] | None = None
        self._online_vfe_optimizer: torch.optim.Optimizer | None = None
        self._online_vfe_lr: float | None = None
        self._online_vfe_parameter_scope: str = "all"

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def posterior_logits(self, observations: Sequence[str]) -> torch.Tensor:
        embeddings = self.encoder(observations)
        return self.posterior(embeddings)

    def transition_logits(
        self,
        posterior_state_logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        state_probabilities = torch.sigmoid(posterior_state_logits)
        action_embeddings = self.action_embedding(actions.to(self.device))
        return self.transition(
            torch.cat([state_probabilities, action_embeddings], dim=-1)
        )

    def forward(
        self,
        observations: Sequence[str],
        next_observations: Sequence[str],
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Encode both sides in one padded batch.  Calling the Qwen backbone
        # twice for every transition batch is equivalent mathematically, but
        # is substantially slower for Qwen3.5's reference linear-attention
        # kernels and doubles tokenizer/dispatch overhead.
        batch_size = len(observations)
        embeddings = self.encoder(list(observations) + list(next_observations))
        current_q = self.posterior(embeddings[:batch_size])
        next_q = self.posterior(embeddings[batch_size:])
        prior_next = self.transition_logits(current_q, actions)
        current_reconstruction = self.observation_decoder(torch.sigmoid(current_q))
        next_reconstruction = self.observation_decoder(torch.sigmoid(next_q))
        return {
            "current_q_logits": current_q,
            "next_q_logits": next_q,
            "prior_next_logits": prior_next,
            "current_reconstruction_logits": current_reconstruction,
            "next_reconstruction_logits": next_reconstruction,
        }

    def free_energy(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        beta: float = 1.0,
        state_supervision_weight: float = 1.0,
        transition_supervision_weight: float = 1.0,
        state_positive_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        bce = nn.functional.binary_cross_entropy_with_logits
        current_reconstruction = bce(
            outputs["current_reconstruction_logits"],
            batch["observation_bow"].to(self.device),
        )
        next_reconstruction = bce(
            outputs["next_reconstruction_logits"],
            batch["next_observation_bow"].to(self.device),
        )
        pos_weight = None
        if state_positive_weight != 1.0:
            pos_weight = torch.full(
                (len(self.vocab.facts),),
                float(state_positive_weight),
                device=self.device,
            )
        posterior_state = bce(
            outputs["current_q_logits"],
            batch["state"].to(self.device),
            pos_weight=pos_weight,
        )
        posterior_next_state = bce(
            outputs["next_q_logits"],
            batch["next_state"].to(self.device),
            pos_weight=pos_weight,
        )
        transition_state = bce(
            outputs["prior_next_logits"],
            batch["next_state"].to(self.device),
            pos_weight=pos_weight,
        )
        kl = bernoulli_kl(
            outputs["next_q_logits"], outputs["prior_next_logits"]
        ).mean()
        reconstruction = 0.5 * (current_reconstruction + next_reconstruction)
        supervision = state_supervision_weight * 0.5 * (
            posterior_state + posterior_next_state
        )
        total = (
            reconstruction
            + beta * kl
            + supervision
            + transition_supervision_weight * transition_state
        )
        return {
            "loss": total,
            "reconstruction": reconstruction.detach(),
            "kl": kl.detach(),
            "state_supervision": supervision.detach(),
            "transition_supervision": transition_state.detach(),
        }

    def head_state_dict(self) -> dict[str, Any]:
        state = {
            "posterior": self.posterior.state_dict(),
            "action_embedding": self.action_embedding.state_dict(),
            "transition": self.transition.state_dict(),
            "observation_decoder": self.observation_decoder.state_dict(),
        }
        if self.encoder.qlora:
            state["encoder_lora"] = self.encoder.lora_state_dict()
        return state

    def load_head_state_dict(self, payload: dict[str, Any]) -> None:
        self.posterior.load_state_dict(payload["posterior"])
        self.action_embedding.load_state_dict(payload["action_embedding"])
        self.transition.load_state_dict(payload["transition"])
        self.observation_decoder.load_state_dict(payload["observation_decoder"])
        self.encoder.load_lora_state_dict(payload.get("encoder_lora", {}))

    def _online_vfe_parameters(self, parameter_scope: str) -> list[nn.Parameter]:
        if parameter_scope == "all":
            return [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ]
        if parameter_scope == "heads":
            modules = (
                self.posterior,
                self.action_embedding,
                self.transition,
                self.observation_decoder,
            )
        elif parameter_scope == "transition":
            # Keep the observation encoder and posterior fixed.  Updating
            # only the transition head makes test-time adaptation less likely
            # to destroy the state representation used by EFE.
            modules = (self.transition,)
        else:
            raise ValueError(
                "online VFE parameter_scope must be one of: all, heads, transition"
            )
        return [parameter for module in modules for parameter in module.parameters()]

    def enable_online_vfe(
        self,
        learning_rate: float = 1e-5,
        *,
        parameter_scope: str = "all",
    ) -> None:
        """Enable resettable test-time VFE updates.

        The initial checkpoint is kept in CPU memory so every episode starts
        from the same frozen-evaluation model. Only parameters marked
        trainable (the QLoRA adapters and structured heads) are optimized.
        """

        if learning_rate <= 0.0:
            raise ValueError("online VFE learning_rate must be positive")
        trainable = self._online_vfe_parameters(parameter_scope)
        if not trainable:
            raise RuntimeError("online VFE requires trainable QLoRA/head parameters")
        self._online_vfe_initial_state = copy.deepcopy(self.head_state_dict())
        self._online_vfe_lr = float(learning_rate)
        self._online_vfe_parameter_scope = parameter_scope
        selected = {id(parameter) for parameter in trainable}
        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) in selected)
        self.reset_online_vfe()

    def reset_online_vfe(self) -> None:
        """Restore the checkpoint and reset the online optimizer state."""

        if self._online_vfe_initial_state is None or self._online_vfe_lr is None:
            return
        self.load_head_state_dict(copy.deepcopy(self._online_vfe_initial_state))
        trainable = self._online_vfe_parameters(self._online_vfe_parameter_scope)
        self._online_vfe_optimizer = torch.optim.AdamW(
            trainable,
            lr=self._online_vfe_lr,
        )
        self.eval()

    def online_vfe_step(
        self,
        observation: str,
        action: str,
        next_observation: str,
        *,
        beta: float = 1.0,
        state_positive_weight: float = 10.0,
        gradient_clip_norm: float = 1.0,
    ) -> dict[str, float]:
        """Apply one VFE update using one observed environment transition.

        This is deliberately an online adaptation hook for evaluation
        experiments. It uses only the observation available after the action
        has executed and updates QLoRA/head parameters, never the 4-bit base
        weights. The caller is responsible for deciding the test protocol.
        """

        if self._online_vfe_optimizer is None:
            raise RuntimeError("Call enable_online_vfe before online_vfe_step")
        if beta < 0.0 or state_positive_weight <= 0.0:
            raise ValueError("invalid online VFE loss weights")

        record = TransitionRecord(
            episode_id=0,
            step=0,
            observation=observation,
            action=action,
            next_observation=next_observation,
            state=parse_observation(observation),
            next_state=parse_observation(next_observation),
        )
        sample = TransitionDataset([record], self.vocab)[0]
        batch: dict[str, Any] = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                batch[key] = value.unsqueeze(0)
            else:
                batch[key] = [value]

        self.train()
        if self._online_vfe_parameter_scope != "all":
            # The frozen encoder should stay deterministic during head-only
            # adaptation; otherwise QLoRA dropout injects noise into the
            # single-transition target even though the adapters are frozen.
            self.encoder.eval()
        # ``optimizer.zero_grad`` only clears parameters owned by the
        # optimizer.  Clear the whole module first so a parameter excluded by
        # the selected online scope cannot retain stale gradients from an
        # earlier transition.
        self.zero_grad(set_to_none=True)
        outputs = self(
            batch["observation"],
            batch["next_observation"],
            batch["action"],
        )
        losses = self.free_energy(
            outputs,
            batch,
            beta=beta,
            state_positive_weight=state_positive_weight,
        )
        losses["loss"].backward()
        trainable = self._online_vfe_parameters(self._online_vfe_parameter_scope)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, gradient_clip_norm
        )
        self._online_vfe_optimizer.step()
        self.eval()
        return {
            "loss": float(losses["loss"].detach().cpu()),
            "reconstruction": float(losses["reconstruction"].cpu()),
            "kl": float(losses["kl"].cpu()),
            "state_supervision": float(losses["state_supervision"].cpu()),
            "transition_supervision": float(
                losses["transition_supervision"].cpu()
            ),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }

    @torch.no_grad()
    def predict_next(
        self,
        observation: str,
        actions: Sequence[str],
        *,
        threshold: float = 0.5,
        top_words: int = 20,
    ) -> list[WorldModelPrediction]:
        current_q = self.posterior_logits([observation])
        current_prob = torch.sigmoid(current_q)
        action_indices = torch.tensor(
            [self.vocab.action_index(action) for action in actions],
            dtype=torch.long,
            device=self.device,
        )
        repeated_q = current_q.expand(len(actions), -1)
        prior = self.transition_logits(repeated_q, action_indices)
        next_prob = torch.sigmoid(prior)
        reconstruction = torch.sigmoid(self.observation_decoder(next_prob))
        predictions: list[WorldModelPrediction] = []
        for row in range(len(actions)):
            facts = [
                fact
                for index, fact in enumerate(self.vocab.facts)
                if float(next_prob[row, index]) >= threshold
            ]
            ranked_words = torch.argsort(reconstruction[row], descending=True)[:top_words]
            words = [self.vocab.observation_words[int(index)] for index in ranked_words]
            predictions.append(
                WorldModelPrediction(
                    posterior_state_probabilities=current_prob[0].detach().cpu(),
                    predicted_next_state_probabilities=next_prob[row].detach().cpu(),
                    predicted_facts=[facts],
                    reconstructed_words=[words],
                )
            )
        return predictions

    @torch.no_grad()
    def predict_from_state_probabilities(
        self,
        state_probabilities: Sequence[Sequence[float]] | torch.Tensor,
        actions: Sequence[str],
        *,
        threshold: float = 0.5,
        top_words: int = 20,
    ) -> list[WorldModelPrediction]:
        """Advance imagined Bernoulli states without re-encoding fake text.

        Each row is the factorized state distribution produced by the
        previous imagined step, paired with the action taken in that state.
        The transition head was trained on posterior state probabilities, so
        this lets multi-step rollouts recurse through the learned dynamics
        directly instead of treating a decoded observation as ground truth.
        """

        if not actions:
            return []
        state_tensor = torch.as_tensor(
            state_probabilities,
            dtype=torch.float32,
            device=self.device,
        )
        if state_tensor.ndim == 1:
            if len(actions) != 1:
                raise ValueError(
                    "A single state distribution can only be paired with one action"
                )
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim != 2:
            raise ValueError("state_probabilities must have shape [batch, facts]")
        if state_tensor.shape[0] != len(actions):
            raise ValueError("state and action batch sizes must match")
        if state_tensor.shape[1] != len(self.vocab.facts):
            raise ValueError(
                "state probability width must match the World Model fact vocabulary"
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if top_words < 0:
            raise ValueError("top_words must be non-negative")

        safe_state = state_tensor.clamp(1e-6, 1.0 - 1e-6)
        state_logits = torch.logit(safe_state)
        action_indices = torch.tensor(
            [self.vocab.action_index(action) for action in actions],
            dtype=torch.long,
            device=self.device,
        )
        next_logits = self.transition_logits(state_logits, action_indices)
        next_probabilities = torch.sigmoid(next_logits)
        reconstructed = torch.sigmoid(
            self.observation_decoder(next_probabilities)
        )

        predictions: list[WorldModelPrediction] = []
        for row in range(len(actions)):
            facts = [
                fact
                for index, fact in enumerate(self.vocab.facts)
                if float(next_probabilities[row, index]) >= threshold
            ]
            ranked_words = torch.argsort(
                reconstructed[row], descending=True
            )[:top_words]
            words = [
                self.vocab.observation_words[int(index)]
                for index in ranked_words
            ]
            predictions.append(
                WorldModelPrediction(
                    posterior_state_probabilities=state_tensor[row].detach().cpu(),
                    predicted_next_state_probabilities=(
                        next_probabilities[row].detach().cpu()
                    ),
                    predicted_facts=[facts],
                    reconstructed_words=[words],
                )
            )
        return predictions
