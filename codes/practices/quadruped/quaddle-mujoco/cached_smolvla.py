"""Cached split-grad SmolVLA training forward.

Key idea: the VLM backbone + vision encoder are frozen. Image/language prefix
tokens never attend to state/action tokens (prefix-lm mask), so their per-layer
K/V are independent of the trainable state_proj and of the noisy actions.

We therefore:
  1. precompute & cache the frozen image embeddings (SigLIP output, the 266ms/step
     hotspot) once per frame;
  2. run the frozen image+language prefix through the 16 VLM layers under no_grad,
     keeping only per-layer post-RoPE K/V;
  3. run the trainable stream (1 state token through the VLM + 50 action tokens
     through the expert) with grad, attending to the frozen K/V.

Backward then only flows through 51 tokens instead of 291, and the frozen prefix
stores no activations. Attention uses SDPA (flash/efficient) instead of the eager
fp32 materialized implementation. Numerics match the original forward up to bf16
attention-kernel rounding (verified by equiv_test_v5.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope


class CachedSmolVLA:
    def __init__(self, policy):
        self.policy = policy
        self.m = policy.model
        self.vlm = self.m.vlm_with_expert
        self.cfg = policy.config
        self.vlm_layers = self.vlm.get_vlm_model().text_model.layers
        self.expert_layers = self.vlm.lm_expert.layers
        self.n_layers = self.vlm.num_vlm_layers
        self.every_n = self.vlm.self_attn_every_n_layers
        self.n_heads = self.vlm.num_attention_heads
        self.n_kv = self.vlm.num_key_value_heads
        self.head_dim = self.vlm.config.text_config.head_dim
        self.hidden = self.vlm.config.text_config.hidden_size
        self.chunk = self.cfg.chunk_size
        self.img_tokens = 64  # per camera after connector (verified)
        self.n_cams = len(self.cfg.image_features)
        self.lang_len = self.cfg.tokenizer_max_length

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        # q [B,Lq,H,D], k/v [B,Lk,Hkv,D], mask [B,Lq,Lk] bool (True=attend) or None.
        # Expand GQA heads explicitly so SDPA can use the flash (no mask) / mem-efficient
        # (mask) kernels; enable_gqa + bool mask otherwise falls back to the slow math path.
        if k.shape[2] != q.shape[2]:
            rep = q.shape[2] // k.shape[2]
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, :, :])
        return out.transpose(1, 2)

    @torch.no_grad()
    def frozen_prefill(self, img_emb: Tensor, lang_emb: Tensor, lang_mask: Tensor):
        """img_emb [B,n_cams*64,960] (x sqrt(hidden)), lang_emb [B,48,960], lang_mask [B,48].
        Padding language tokens are masked out everywhere in the original model, so we drop
        them and run the real image+language prefix bidirectionally with NO mask (flash SDPA).
        Returns per-layer (k,v) [B,Lf,n_kv,D] post-RoPE and an all-ones frozen pad mask [B,Lf]."""
        B = img_emb.shape[0]
        dev = img_emb.device
        n_real = int(lang_mask[0].sum().item())
        embs = torch.cat([img_emb, lang_emb[:, :n_real, :]], dim=1)
        Lf = embs.shape[1]
        pos = torch.arange(Lf, device=dev)[None, :].expand(B, -1)
        kv_list = []
        h = embs
        for l in range(self.n_layers):
            layer = self.vlm_layers[l]
            hn = layer.input_layernorm(h).to(layer.self_attn.q_proj.weight.dtype)
            q = layer.self_attn.q_proj(hn).view(B, Lf, self.n_heads, self.head_dim)
            k = layer.self_attn.k_proj(hn).view(B, Lf, self.n_kv, self.head_dim)
            v = layer.self_attn.v_proj(hn).view(B, Lf, self.n_kv, self.head_dim)
            q = apply_rope(q, pos)
            k = apply_rope(k, pos)
            kv_list.append((k, v))
            ao = self._sdpa(q, k, v, None).reshape(B, Lf, self.n_heads * self.head_dim)
            ao = ao.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(ao)
            out += h
            res = out.clone()
            out = layer.post_attention_layernorm(out)
            out = layer.mlp(out)
            out += res
            h = out
        frozen_pad = torch.ones(B, Lf, dtype=torch.bool, device=dev)
        return kv_list, frozen_pad

    def train_forward(self, kv_list, frozen_pad, state, actions, noise=None, time=None):
        """state [B,32] padded normalized, actions [B,50,32] padded normalized.
        Returns per-element losses [B,50,32] (same convention as VLAFlowMatching.forward)."""
        B = state.shape[0]
        dev = state.device
        if noise is None:
            noise = self.m.sample_noise(actions.shape, dev)
        if time is None:
            time = self.m.sample_time(B, dev)
        te = time[:, None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions

        state_emb = self.m.state_proj(state)[:, None, :]  # [B,1,960] grad
        suffix_embs, _, _ = self.m.embed_suffix(x_t, time)  # [B,50,720] grad

        Lf = frozen_pad.shape[1]
        state_pos = frozen_pad.sum(dim=1, keepdim=True)  # [B,1]
        suffix_pos_joint = state_pos + 1 + torch.arange(self.chunk, device=dev)[None, :]
        suffix_pos_cross = torch.arange(self.chunk, device=dev)[None, :].expand(B, -1)

        full_pad = torch.cat([frozen_pad,
                              torch.ones(B, 1 + self.chunk, dtype=torch.bool, device=dev)], dim=1)
        full_att = torch.cat([torch.zeros(B, Lf, dtype=torch.bool, device=dev),
                              torch.ones(B, 1 + self.chunk, dtype=torch.bool, device=dev)], dim=1)
        full_att2d = make_att_2d_masks(full_pad, full_att)
        grad_att2d = full_att2d[:, Lf:, :]  # [B,1+50, Lf+1+50]
        state_mask = grad_att2d[:, :1, : Lf + 1]
        cross_mask = grad_att2d[:, 1:, : Lf + 1]

        h_state = state_emb
        h_suffix = suffix_embs
        kv_dim = self.n_kv * self.head_dim
        for l in range(self.n_layers):
            fk, fv = kv_list[l]
            vl = self.vlm_layers[l]
            el = self.expert_layers[l]
            # state token through VLM layer
            hs = vl.input_layernorm(h_state).to(vl.self_attn.q_proj.weight.dtype)
            sq = vl.self_attn.q_proj(hs).view(B, 1, self.n_heads, self.head_dim)
            sk = vl.self_attn.k_proj(hs).view(B, 1, self.n_kv, self.head_dim)
            sv = vl.self_attn.v_proj(hs).view(B, 1, self.n_kv, self.head_dim)
            sq = apply_rope(sq, state_pos)
            sk = apply_rope(sk, state_pos)
            # suffix through expert layer
            he = el.input_layernorm(h_suffix).to(el.self_attn.q_proj.weight.dtype)
            eq = el.self_attn.q_proj(he).view(B, self.chunk, self.n_heads, self.head_dim)

            joint = (self.every_n > 0 and l % self.every_n == 0)
            if joint:
                ek = el.self_attn.k_proj(he).view(B, self.chunk, self.n_kv, self.head_dim)
                ev = el.self_attn.v_proj(he).view(B, self.chunk, self.n_kv, self.head_dim)
                eq = apply_rope(eq, suffix_pos_joint)
                ek = apply_rope(ek, suffix_pos_joint)
                K = torch.cat([fk, sk, ek], dim=1)
                V = torch.cat([fv, sv, ev], dim=1)
                Q = torch.cat([sq, eq], dim=1)
                ao = self._sdpa(Q, K, V, grad_att2d).reshape(B, 1 + self.chunk, self.n_heads * self.head_dim)
                s_ao = ao[:, :1].to(vl.self_attn.o_proj.weight.dtype)
                e_ao = ao[:, 1:].to(el.self_attn.o_proj.weight.dtype)
            else:
                # state self-attn over frozen + itself
                sK = torch.cat([fk, sk], dim=1)
                sV = torch.cat([fv, sv], dim=1)
                s_ao = self._sdpa(sq, sK, sV, state_mask).reshape(B, 1, self.n_heads * self.head_dim)
                s_ao = s_ao.to(vl.self_attn.o_proj.weight.dtype)
                # expert cross-attn to VLM K/V (frozen + state)
                vlm_K = torch.cat([fk, sk], dim=1).reshape(B, Lf + 1, kv_dim)
                vlm_V = torch.cat([fv, sv], dim=1).reshape(B, Lf + 1, kv_dim)
                ek = el.self_attn.k_proj(vlm_K.to(el.self_attn.k_proj.weight.dtype)).view(B, Lf + 1, self.n_kv, self.head_dim)
                ev = el.self_attn.v_proj(vlm_V.to(el.self_attn.v_proj.weight.dtype)).view(B, Lf + 1, self.n_kv, self.head_dim)
                eq = apply_rope(eq, suffix_pos_cross)
                eq = eq.to(ek.dtype)
                e_ao = self._sdpa(eq, ek, ev, cross_mask).reshape(B, self.chunk, self.n_heads * self.head_dim)
                e_ao = e_ao.to(el.self_attn.o_proj.weight.dtype)

            s_out = vl.self_attn.o_proj(s_ao)
            s_out += h_state
            res = s_out.clone()
            s_out = vl.post_attention_layernorm(s_out)
            s_out = vl.mlp(s_out)
            s_out += res
            h_state = s_out

            e_out = el.self_attn.o_proj(e_ao)
            e_out += h_suffix
            res = e_out.clone()
            e_out = el.post_attention_layernorm(e_out)
            e_out = el.mlp(e_out)
            e_out += res
            h_suffix = e_out

        h_suffix = self.vlm.lm_expert.norm(h_suffix)
        suffix_out = h_suffix[:, -self.chunk:].to(torch.float32)
        v_t = self.m.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def loss_from_batch(self, kv_list, frozen_pad, state, actions, action_is_pad):
        losses = self.train_forward(kv_list, frozen_pad, state, actions)
        losses = losses[:, :, : self.cfg.action_feature.shape[0]]
        if action_is_pad is not None:
            in_ep = ~action_is_pad
            losses = losses * in_ep.unsqueeze(-1)
            num_valid = (in_ep.sum() * losses.shape[-1]).clamp_min(1)
            return losses.sum() / num_valid
        return losses.mean()