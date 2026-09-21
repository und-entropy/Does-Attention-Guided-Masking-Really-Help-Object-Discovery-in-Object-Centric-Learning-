import math
from einops import rearrange
import torch as pt
import torch.nn as nn

from .smoothsa_AGM import SmoothSA


class SsaMR(SmoothSA):
    def forward(self, input, condit=None):
        feature = self.encode_backbone(input).detach()
        b, c, h, w = feature.shape

        encode = feature.permute(0, 2, 3, 1)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)
        encode = self.encode_project(encode)

        qinit, query = self.initializ(encode, condit)
        slotz, attenta = self.aggregat(encode, query)
        attenta = rearrange(attenta, "b n (h w) -> b n h w", h=h)

        clue = rearrange(feature, "b c h w -> b (h w) c")
        att_flat = rearrange(attenta, "b s h w -> b s (h w)")
        recon, attentd = self.decode(clue, slotz, att_flat)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attentd = rearrange(attentd, "b n (h w) -> b n h w", h=h)

        self.affinity = rearrange(attenta, "b n h w -> b (h w) n")
        return feature, qinit, slotz, attenta, recon, attentd


class ARRandTransformerDecoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        posit_embed,
        project1,
        project2,
        backbone,
        readout,
        alpha=0.5,
        ranking_mode="guided",
        capture_training_vis=False,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(pt.randn(1, 1, emb_dim) * emb_dim**-0.5)
        assert hasattr(posit_embed, "pe")
        self.posit_embed = posit_embed
        self.project1 = project1
        self.project2 = project2

        assert isinstance(backbone, nn.TransformerDecoder)
        self.norm0 = backbone.layers[0].norm1
        backbone.layers[0].norm1 = nn.Identity()
        self.backbone = backbone
        self.readout = readout
        self.register_buffer("alpha", pt.tensor(alpha, dtype=pt.float), persistent=False)
        if ranking_mode not in {"guided", "within_owner", "random", "fixed"}:
            raise ValueError(f"Unsupported ranking_mode: {ranking_mode}")
        self.ranking_mode = ranking_mode
        self.capture_training_vis = bool(capture_training_vis)

        def attent_hook_forward_pre(module, args, kwargs):
            kwargs["need_weights"] = True

        def attent_hook_forward(module, args, output):
            self._attent = output[1]

        self.backbone.layers[-1].multihead_attn.register_forward_pre_hook(
            attent_hook_forward_pre, with_kwargs=True
        )
        self.backbone.layers[-1].multihead_attn.register_forward_hook(
            attent_hook_forward
        )
        if self.capture_training_vis:
            self._vis_cache = {}

    def forward(self, input, slots, attenta, smask=None, p=0.5):
        b, m, c = input.shape
        assert m == self.posit_embed.pe.size(1)
        _, n, _ = slots.shape
        device = input.device
        tokens = self.project1(input)
        memory = self.project2(slots)
        if self.capture_training_vis:
            self._vis_cache = {}

        if self.training:
            if self.ranking_mode == "random":
                # 与老师的 Rand 基线保持一致：每个样本独立 randperm，且不计算 PGM score。
                idxs = pt.vmap(
                    lambda _: pt.randperm(m, device=device), randomness="different"
                )(tokens)
                if self.capture_training_vis:
                    inverse_rank = idxs.argsort(dim=1).to(attenta.dtype)
                    vis_cache = {
                        "token_objectness": attenta.max(dim=1).values,
                        "decoder_entropy": pt.zeros_like(inverse_rank),
                        "token_scores": inverse_rank / max(m - 1, 1),
                        "guided_scores": pt.zeros_like(inverse_rank),
                    }
            elif self.ranking_mode == "fixed":
                fixed_scores = pt.arange(m, device=device, dtype=attenta.dtype)[None].expand(b, -1)
                idxs = fixed_scores.long()
                if self.capture_training_vis:
                    vis_cache = {
                        "token_objectness": attenta.max(dim=1).values,
                        "decoder_entropy": pt.zeros_like(fixed_scores),
                        "token_scores": fixed_scores / max(m - 1, 1),
                        "guided_scores": pt.zeros_like(fixed_scores),
                    }
            else:
                decoder_entropy = self.current_aligned_decoder_entropy(
                    tokens, memory, smask
                )
                idxs, vis_cache = generate_current_entropy_idxs(
                    attenta,
                    decoder_entropy,
                    float(self.alpha.item()),
                    score_variant=(
                        "within_owner" if self.ranking_mode == "within_owner" else None
                    ),
                )
            idxs_expanded = idxs[:, :, None].expand(-1, -1, c)

            idxs0 = pt.arange(0, m, device=device)[None, :]
            keep1 = pt.randint(0, m - 1, [b, 1], device=device)
            keep2 = pt.ones(b, 1, dtype=pt.long, device=device) * int(256 * 0.1) - 1
            cond = pt.rand(b, 1, device=device) < p
            keep = pt.where(cond, keep1, keep2)
            keep_mask = idxs0 < keep

            tokens_shuffled = tokens.gather(1, idxs_expanded)
            tokens_masked = tokens_shuffled.where(
                keep_mask[:, :, None], self.mask_token.expand(b, m, -1)
            )

            pe = self.posit_embed.pe[:, :m, :].expand(b, -1, -1)
            pe_shuffled = pe.gather(1, idxs_expanded)
            query = tokens_masked + pe_shuffled

            if self.capture_training_vis:
                idxs_inverse_1d = idxs.argsort(1)
                mae_mask = (~keep_mask).gather(1, idxs_inverse_1d)
                self._vis_cache.update(
                    {
                        "mae_mask": mae_mask.detach().cpu(),
                        "token_objectness": vis_cache["token_objectness"].detach().cpu(),
                        "token_decoder_entropy": vis_cache["decoder_entropy"].detach().cpu(),
                        "token_score": vis_cache["token_scores"].detach().cpu(),
                        "guided_scores": vis_cache["guided_scores"].detach().cpu(),
                    }
                )
        else:
            query = tokens + self.posit_embed.pe[:, :m, :]

        autoreg = self.backbone(
            self.norm0(query),
            memory=memory,
            memory_key_padding_mask=None if smask is None else ~smask,
        )
        recon = self.readout(autoreg)
        _, _, d = recon.shape

        if self.training:
            idxs_inverse = idxs.argsort(1)[:, :, None]
            recon = recon.gather(1, idxs_inverse.expand(-1, -1, d))
            attent = self._attent.gather(1, idxs_inverse.expand(-1, -1, n))
        else:
            attent = self._attent

        attent = rearrange(attent, "b m n -> b n m")
        return recon, attent

    @pt.no_grad()
    def current_aligned_decoder_entropy(self, tokens, memory, smask=None):
        b, m, _ = tokens.shape
        query = tokens + self.posit_embed.pe[:, :m, :].expand(b, -1, -1)
        self.backbone(
            self.norm0(query),
            memory=memory,
            memory_key_padding_mask=None if smask is None else ~smask,
        )

        dec_att = self._attent.float()
        dec_att = dec_att / dec_att.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_num = max(dec_att.shape[-1], 2)
        entropy = -(dec_att.clamp_min(1e-6) * dec_att.clamp_min(1e-6).log()).sum(dim=-1)
        entropy = entropy / math.log(slot_num)
        return entropy.detach()


@pt.no_grad()
def remove_between_owner_score(score, owner, slot_num):
    """Remove owner-mean differences while preserving within-owner ordering.

    The affine shift and scale are shared by the whole sample.  Consequently,
    every non-empty owner has the same mean score, while pairwise score
    differences inside an owner are unchanged up to one positive scale factor.
    """
    batch_size, _ = score.shape
    sums = pt.zeros(batch_size, slot_num, device=score.device, dtype=score.dtype)
    counts = pt.zeros_like(sums)
    sums.scatter_add_(1, owner, score)
    counts.scatter_add_(1, owner, pt.ones_like(score))
    owner_mean = sums / counts.clamp_min(1.0)
    residual = score - owner_mean.gather(1, owner)
    residual = residual - residual.min(dim=1, keepdim=True).values
    original_max = score.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    residual_max = residual.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    return residual / residual_max * original_max


@pt.no_grad()
def generate_current_entropy_idxs(
    attenta, decoder_entropy, alpha=0.5, score_variant=None
):
    b, s, m = attenta.shape
    eps = 1e-6

    token_saliency = attenta.max(dim=1).values

    slot_prob = attenta / (attenta.sum(dim=-1, keepdim=True) + eps)
    norm = math.log(max(m, 2))
    slot_entropy = -(slot_prob * (slot_prob + eps).log()).sum(dim=-1) / norm
    slot_compactness = 1.0 - slot_entropy

    slot_mass = attenta.mean(dim=-1)
    slot_objectness  = slot_compactness / pt.sqrt(slot_mass + eps)
    obj_min = slot_objectness.min(dim=1, keepdim=True).values
    obj_max = slot_objectness.max(dim=1, keepdim=True).values
    slot_objectness = (slot_objectness - obj_min) / (obj_max - obj_min + eps)

    token_objectness = (attenta * slot_objectness.unsqueeze(-1)).sum(dim=1)
    guided_core = token_saliency * token_objectness

    fg_mask = token_objectness > token_objectness.median(dim=1, keepdim=True).values
    owner = attenta.argmax(dim=1)

    entropy_residual = owner_relative_entropy_residual(
        decoder_entropy, owner, fg_mask, s
    )
    entropy_norm = entropy_residual / entropy_residual.max(
        dim=1, keepdim=True
    ).values.clamp_min(eps)
    entropy_norm = entropy_norm * fg_mask.float()

    fg_core_min = pt.where(
        fg_mask, guided_core, pt.full_like(guided_core, float("inf"))
    ).min(dim=1, keepdim=True).values
    fg_core_min = pt.where(
        fg_core_min.isinf(),
        guided_core.min(dim=1, keepdim=True).values,
        fg_core_min,
    )

    raw_guided_scores = guided_core + entropy_norm * fg_core_min
    if score_variant is None:
        guided_scores = raw_guided_scores
    elif score_variant == "within_owner":
        guided_scores = remove_between_owner_score(raw_guided_scores, owner, s)
    else:
        raise ValueError(f"Unsupported guided score variant: {score_variant}")

    noise = pt.rand_like(guided_scores)
    amp = guided_scores.max(dim=1, keepdim=True).values.clamp_min(eps)
    token_scores = alpha * guided_scores + (1.0 - alpha) * noise * amp
    token_scores = token_scores + pt.rand_like(token_scores) * eps
    idxs = token_scores.argsort(dim=-1, descending=False)

    return idxs, {
        "slot_mass": slot_mass,
        "slot_compactness": slot_compactness,
        "slot_objectness": slot_objectness,
        "token_saliency": token_saliency,
        "token_objectness": token_objectness,
        "decoder_entropy": decoder_entropy,
        "entropy_residual": entropy_residual,
        "guided_core": guided_core,
        "raw_guided_scores": raw_guided_scores,
        "guided_scores": guided_scores,
        "token_scores": token_scores,
    }


@pt.no_grad()
def owner_relative_entropy_residual(entropy, owner, fg_mask, slot_num):
    residual = pt.zeros_like(entropy)
    b = entropy.shape[0]

    for i in range(b):
        for sid in range(slot_num):
            mask = fg_mask[i] & (owner[i] == sid)
            if mask.sum() <= 1:
                continue

            values = entropy[i, mask]
            baseline = values.median()
            residual[i, mask] = (values - baseline).clamp_min(0)

    return residual
