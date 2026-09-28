"""BLIP-2 backbone trained only with learnable token MaxSim retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blip2_maxsim.model.matching import maxsim_logits, maxsim_logits_low_memory
from .blip2 import Blip2Base, disabled_train


class BLIP2MaxSim(Blip2Base):
    """The clean ablation baseline used before adding new post-training ideas.

    Kept components:
      * BLIP-2 EVA-ViT image encoder;
      * BLIP-2 Q-Former and query tokens;
      * text/image projection layers;
      * the learnable ``compose_token_mat_weight`` MaxSim aggregator.

    There is no contribution decoder, evidence/anchor/direction branch, flow
    head, or auxiliary loss. The only objective is in-batch MaxSim ranking.
    """

    def __init__(
        self,
        *,
        vit_model: str = "eva_clip_g",
        image_size: int = 364,
        drop_path_rate: float = 0.0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp32",
        train_vit: bool = False,
        num_query_token: int = 32,
        cross_attention_freq: int = 2,
        embed_dim: int = 256,
        text_input_length: int = 64,
        rank_loss_weight: float = 0.5,
        rank_temperature: float = 0.07,
        maxsim_temperature: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model,
            image_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
        )
        self.train_vit = bool(train_vit)
        if not self.train_vit:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,
            self.visual_encoder.num_features,
            cross_attention_freq,
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        qformer_state = self.Qformer.state_dict()
        for name, parameter in self.Qformer.named_parameters():
            if "_query" in name:
                parameter.data.copy_(qformer_state[name.replace("_query", "")])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.compose_token_mat_weight = nn.Parameter(
            torch.eye(num_query_token),
            requires_grad=True,
        )

        self.text_input_length = int(text_input_length)
        self.rank_loss_weight = float(rank_loss_weight)
        self.rank_temperature = float(rank_temperature)
        self.maxsim_temperature = float(maxsim_temperature)
        self.latest_metrics: Dict[str, float] = {}

    def target_fea(
        self,
        target_image: torch.Tensor,
        description=None,
        fabric=None,
        device: str | torch.device = "cuda",
    ):
        """Encode target images into normalized BLIP-2 query-token features."""
        del description, fabric
        device = torch.device(device) if isinstance(device, str) else device
        is_video = target_image.ndim == 5
        batch_size = target_image.shape[0]
        num_frames = 1
        if is_video:
            batch_size, num_frames, channels, height, width = target_image.shape
            target_image = target_image.reshape(
                batch_size * num_frames, channels, height, width
            )

        if self.train_vit:
            image_embeds = self.ln_vision(self.visual_encoder(target_image))
        else:
            with torch.no_grad():
                image_embeds = self.visual_encoder(target_image)
            image_embeds = self.ln_vision(image_embeds)
        image_embeds = image_embeds.float()

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        if is_video:
            image_embeds = image_embeds.reshape(batch_size, -1, image_embeds.shape[-1])
            query_tokens = query_tokens.repeat(1, num_frames, 1)
        image_attention = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=device
        )
        output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention,
            return_dict=True,
        )
        hidden = output.last_hidden_state[:, : query_tokens.shape[1], :]
        target_tokens = F.normalize(self.vision_proj(hidden), dim=-1)
        target_mean = F.normalize(target_tokens.mean(dim=1), dim=-1)
        return target_mean, target_tokens, None

    def compose_feature(
        self,
        reference_image: torch.Tensor,
        caption,
        description=None,
        fabric=None,
        device: str | torch.device = "cuda",
    ):
        """Fuse a reference image and edit text into composed-query tokens."""
        del description, fabric
        device = torch.device(device) if isinstance(device, str) else device
        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.text_input_length,
            return_tensors="pt",
        ).to(device)

        is_video = reference_image.ndim == 5
        num_frames = 1
        if is_video:
            batch_size, num_frames, channels, height, width = reference_image.shape
            reference_image = reference_image.reshape(
                batch_size * num_frames, channels, height, width
            )
        else:
            batch_size = reference_image.shape[0]

        if self.train_vit:
            image_embeds = self.ln_vision(self.visual_encoder(reference_image))
        else:
            with torch.no_grad():
                image_embeds = self.visual_encoder(reference_image)
            image_embeds = self.ln_vision(image_embeds)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        if is_video:
            image_embeds = image_embeds.reshape(batch_size, -1, image_embeds.shape[-1])
            query_tokens = query_tokens.repeat(1, num_frames, 1)
        image_attention = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=device
        )
        query_attention = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=device
        )
        attention_mask = torch.cat([query_attention, text_tokens.attention_mask], dim=1)
        output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention,
            return_dict=True,
            output_attentions=True,
        )
        hidden = output.last_hidden_state[:, : query_tokens.shape[1], :]
        query_tokens_out = F.normalize(self.text_proj(hidden), dim=-1)
        query_mean = F.normalize(query_tokens_out.mean(dim=1), dim=-1)
        return query_mean, output.cross_attentions, query_tokens_out

    def textual_feature(self, caption, device: str | torch.device = "cuda"):
        """Encode text alone; retained as a convenient extension hook."""
        device = torch.device(device) if isinstance(device, str) else device
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.text_input_length,
            return_tensors="pt",
        ).to(device)
        output = self.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        return F.normalize(self.text_proj(output.last_hidden_state), dim=-1)[:, 0, :]

    def fine_grained_sim(
        self,
        query_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> np.ndarray:
        """Score composed queries against a target bank with low-memory MaxSim."""
        device = self.compose_token_mat_weight.device
        logits = maxsim_logits_low_memory(
            query_tokens.to(device),
            target_tokens.to(device),
            self.compose_token_mat_weight,
            self.maxsim_temperature,
        )
        return logits.detach().cpu().numpy()

    def compute_rank_loss(
        self,
        query_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross entropy to diagonal in-batch MaxSim matches."""
        logits, _ = maxsim_logits(
            query_tokens,
            target_tokens,
            self.compose_token_mat_weight,
            self.maxsim_temperature,
        )
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(
            logits / max(self.rank_temperature, 1.0e-6),
            labels,
        )
        return loss, logits

    def forward(self, batch: Dict[str, Any], fabric=None) -> torch.Tensor:
        """Train with the sole MaxSim ranking objective."""
        del fabric
        reference_image = batch["ref_img"]
        target_image = batch["tar_img"]
        device = reference_image.device
        _, _, query_tokens = self.compose_feature(
            reference_image,
            batch["edit"],
            device=device,
        )
        _, target_tokens, _ = self.target_fea(target_image, device=device)
        rank_loss, logits = self.compute_rank_loss(query_tokens, target_tokens)
        total_loss = self.rank_loss_weight * rank_loss

        with torch.no_grad():
            labels = torch.arange(logits.shape[0], device=logits.device)
            retrieval_acc = (logits.argmax(dim=1) == labels).float().mean()
            self.latest_metrics = {
                "total_loss": float(total_loss.item()),
                "rank_loss": float(rank_loss.item()),
                "inbatch_retrieval_acc": float(retrieval_acc.item()),
            }
        return total_loss

    def get_trainable_param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def load_model_state(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load either this project format or the former warmup checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = (
            checkpoint["model_state"]
            if isinstance(checkpoint, dict) and "model_state" in checkpoint
            else checkpoint
        )
        message = self.load_state_dict(state, strict=False)
        return {
            "checkpoint": checkpoint,
            "missing_keys": list(message.missing_keys),
            "unexpected_keys": list(message.unexpected_keys),
        }


def build_blip2_maxsim(
    *,
    checkpoint: str | None = None,
    **kwargs,
) -> BLIP2MaxSim:
    """Construct the model and optionally load BLIP-2 or trained weights."""
    model = BLIP2MaxSim(**kwargs)
    if checkpoint:
        if checkpoint.startswith(("http://", "https://")):
            model.load_from_pretrained(checkpoint)
        else:
            model.load_model_state(checkpoint)
    return model
