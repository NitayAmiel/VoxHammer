import os
import sys

os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPCONV_ALGO"] = "native"

from types import MethodType
from typing import *

import numpy as np
import rembg
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils3d
from PIL import Image
from tqdm import tqdm

import trellis.modules.sparse as sp
from trellis.modules.spatial import patchify, unpatchify
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis.utils import postprocessing_utils


def ply_to_coords(ply_path):
    position = utils3d.io.read_ply(ply_path)[0]
    coords = ((torch.tensor(position) + 0.5) * 64).int().contiguous().cuda()
    return coords


def coords_to_voxel(coords):
    voxel = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float)
    voxel[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel.cuda()


def feats_to_slat(pipeline, feats_path):
    feats = np.load(feats_path)
    feats_tensor = sp.SparseTensor(
        feats=torch.from_numpy(feats["patchtokens"]).float(),
        coords=torch.cat(
            [
                torch.zeros(feats["patchtokens"].shape[0], 1).int(),
                torch.from_numpy(feats["indices"]).int(),
            ],
            dim=1,
        ),
    ).cuda()
    feats_encoder = pipeline.models["slat_encoder"]
    slat = feats_encoder(feats_tensor, sample_posterior=False)
    return slat


def image_rgb(img_path):
    image = Image.open(img_path)
    if image.mode == "RGB":
        image_rgb = image
    else:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image_rgb = background

    return image_rgb


def preprocess_image(img_src_path, img_tgt_path, img_mask_path):
    img_src = image_rgb(img_src_path)
    img_tgt = image_rgb(img_tgt_path)
    img_mask = Image.open(img_mask_path).convert("L")

    max_size = max(img_src.size)
    scale = min(1, 1024 / max_size)
    resize_size = (int(img_src.width * scale), int(img_src.height * scale))
    if scale < 1:
        img_src = img_src.resize(resize_size, Image.Resampling.LANCZOS)
        img_tgt = img_tgt.resize(resize_size, Image.Resampling.LANCZOS)
        img_mask = img_mask.resize(resize_size, Image.Resampling.LANCZOS)
    pre_img_src = rembg.remove(img_src, session=rembg.new_session("u2net"))
    pre_img_tgt = rembg.remove(img_tgt, session=rembg.new_session("u2net"))
    pre_img_src_np = np.array(pre_img_src)
    pre_img_tgt_np = np.array(pre_img_tgt)
    alpha_src = pre_img_src_np[:, :, 3]
    alpha_tgt = pre_img_tgt_np[:, :, 3]
    bbox_src = np.argwhere(alpha_src > 0.8 * 255)
    bbox_tgt = np.argwhere(alpha_tgt > 0.8 * 255)
    bbox = (
        min(np.min(bbox_src[:, 1]), np.min(bbox_tgt[:, 1])),
        min(np.min(bbox_src[:, 0]), np.min(bbox_tgt[:, 0])),
        max(np.max(bbox_src[:, 1]), np.max(bbox_tgt[:, 1])),
        max(np.max(bbox_src[:, 0]), np.max(bbox_tgt[:, 0])),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    pre_img_src = pre_img_src.crop(bbox)
    pre_img_tgt = pre_img_tgt.crop(bbox)
    pre_img_mask = img_mask.crop(bbox)
    pre_img_src = pre_img_src.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_tgt = pre_img_tgt.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_mask = pre_img_mask.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_src = np.array(pre_img_src).astype(np.float32) / 255
    pre_img_tgt = np.array(pre_img_tgt).astype(np.float32) / 255
    pre_img_src = pre_img_src[:, :, :3] * pre_img_src[:, :, 3:4]
    pre_img_tgt = pre_img_tgt[:, :, :3] * pre_img_tgt[:, :, 3:4]
    pre_img_src = Image.fromarray((pre_img_src * 255).astype(np.uint8))
    pre_img_tgt = Image.fromarray((pre_img_tgt * 255).astype(np.uint8))
    return pre_img_src, pre_img_tgt, pre_img_mask


def ply_to_ss_mask(coords_mask, pre_mask):
    voxel_mask = torch.ones(1, 1, 64, 64, 64, dtype=torch.float)
    voxel_mask[:, 0, coords_mask[:, 0], coords_mask[:, 1], coords_mask[:, 2]] = 0
    voxel_mask = voxel_mask.cuda()

    ss_latent_mask = voxel_mask.reshape(1, 1, 32, 2, 32, 2, 32, 2)
    ss_latent_mask = ss_latent_mask.permute(0, 1, 3, 5, 7, 2, 4, 6)
    ss_latent_mask = ss_latent_mask.reshape(1, 8, 32, 32, 32)
    ss_latent_mask = ss_latent_mask.reshape(1, 8, 16, 2, 16, 2, 16, 2)
    ss_latent_mask = ss_latent_mask.permute(0, 1, 3, 5, 7, 2, 4, 6)
    ss_latent_mask = ss_latent_mask.reshape(1, 64, 16, 16, 16)
    ss_latent_mask = ss_latent_mask.all(dim=1, keepdim=True)
    ss_latent_mask = ss_latent_mask.repeat(1, 8, 1, 1, 1).contiguous().float().cuda()

    ss_self_kv_mask = ss_latent_mask.reshape(1, 8, -1)
    ss_self_kv_mask = ss_self_kv_mask.permute(0, 2, 1)
    ss_self_kv_mask = ss_self_kv_mask.all(dim=2, keepdim=True)
    ss_self_kv_mask = ss_self_kv_mask.repeat(1, 1, 1024)
    ss_self_kv_mask = ss_self_kv_mask.reshape(1, 4096, 16, 64)
    ss_self_kv_mask = ss_self_kv_mask.permute(0, 2, 1, 3).contiguous().float().cuda()

    img_mask = transforms.ToTensor()(pre_mask)
    img_mask = (img_mask > 0).float()
    cross_kv_mask = img_mask.reshape(37, 14, 37, 14)
    cross_kv_mask = cross_kv_mask.permute(1, 3, 0, 2)
    cross_kv_mask = cross_kv_mask.reshape(1, 196, 37, 37)
    cross_kv_mask = cross_kv_mask.any(dim=1, keepdim=True)
    cross_kv_mask = cross_kv_mask.repeat(1, 1024, 1, 1)
    cross_kv_mask = cross_kv_mask.reshape(1, 1024, 1369)
    cross_kv_mask = cross_kv_mask.permute(0, 2, 1)
    cross_kv_mask = torch.cat((torch.ones(1, 5, 1024), cross_kv_mask), dim=1)
    cross_kv_mask = cross_kv_mask.reshape(1, 1374, 16, 64)
    cross_kv_mask = cross_kv_mask.permute(0, 2, 1, 3).contiguous().float().cuda()

    return voxel_mask, ss_latent_mask, ss_self_kv_mask, cross_kv_mask


def ply_to_slat_mask(coords_tgt, coords_mask):
    coords_mask = torch.cat(
        [torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1
    )
    factor = (2, 2, 2)
    coord_tgt = list(coords_tgt.unbind(dim=-1))
    coord_mask = list(coords_mask.unbind(dim=-1))
    for i, f in enumerate(factor):
        coord_tgt[i + 1] = coord_tgt[i + 1] // f
        coord_mask[i + 1] = coord_mask[i + 1] // f
    MAX = [coord_tgt[i + 1].max().item() + 1 for i in range(3)]
    OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
    code_tgt = sum([c * o for c, o in zip(coord_tgt, OFFSET)])
    code_mask = sum([c * o for c, o in zip(coord_mask, OFFSET)])
    code_tgt, idx_tgt = code_tgt.unique(return_inverse=True)
    code_mask, idx_mask = code_mask.unique(return_inverse=True)
    new_coords_tgt = torch.stack(
        [code_tgt // OFFSET[0]]
        + [(code_tgt // OFFSET[i + 1]) % MAX[i] for i in range(3)],
        dim=-1,
    )
    new_coords_mask = torch.stack(
        [code_mask // OFFSET[0]]
        + [(code_mask // OFFSET[i + 1]) % MAX[i] for i in range(3)],
        dim=-1,
    )
    slat_self_kv_mask = new_coords_mask.contiguous().cuda()

    return slat_self_kv_mask


def ss_attn_forward(
    self,
    x,
    context=None,
    indices=None,
    ss_kv=None,
    kv_mask=None,
    t_latent=None,
    order=None,
    pos=None,
    layer=None,
):
    B, L, C = x.shape
    if self._type == "self":
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(dim=2)
        if self.use_rope:
            q, k = self.rope(q, k, indices)
    else:
        Lkv = context.shape[1]
        q = self.to_q(x)
        kv = self.to_kv(context)
        q = q.reshape(B, L, self.num_heads, -1)
        kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
        k, v = kv.unbind(dim=2)

    if self.qk_rms_norm:
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    if kv_mask is None:
        ss_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"] = k.cpu()
        ss_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"] = v.cpu()
    else:
        k = k * kv_mask + ss_kv[
            f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"
        ].cuda() * (1 - kv_mask)
        v = v * kv_mask + ss_kv[
            f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"
        ].cuda() * (1 - kv_mask)
        k = k.type(q.dtype)
        v = v.type(q.dtype)
    h = F.scaled_dot_product_attention(q, k, v)
    h = h.permute(0, 2, 1, 3)
    h = h.reshape(B, L, -1)
    h = self.to_out(h)
    return h


def ss_trsfmr_forward(
    self,
    x,
    mod,
    context,
    ss_kv,
    self_kv_mask,
    cross_kv_mask,
    t_latent,
    order,
    pos,
    layer,
):
    if self.share_mod:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )
    else:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(mod).chunk(6, dim=1)
        )

    h = self.norm1(x)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    h = self.self_attn(
        h,
        ss_kv=ss_kv,
        kv_mask=self_kv_mask,
        t_latent=t_latent,
        order=order,
        pos=pos,
        layer=layer,
    )
    h = h * gate_msa.unsqueeze(1)
    x = x + h
    h = self.norm2(x)
    h = self.cross_attn(
        h,
        context,
        ss_kv=ss_kv,
        kv_mask=cross_kv_mask,
        t_latent=t_latent,
        order=order,
        pos=pos,
        layer=layer,
    )
    x = x + h
    h = self.norm3(x)
    h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    h = self.mlp(h)
    h = h * gate_mlp.unsqueeze(1)
    x = x + h
    return x


def ss_flow_forward(
    self, x, t, cond, ss_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos
):
    assert [*x.shape] == [
        x.shape[0],
        self.in_channels,
        *[self.resolution] * 3,
    ], f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
    h = patchify(x, self.patch_size)
    h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
    h = self.input_layer(h)
    h = h + self.pos_emb[None]
    t_emb = self.t_embedder(t)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    t_emb = t_emb.type(self.dtype)
    h = h.type(self.dtype)
    cond = cond.type(self.dtype)
    for layer, block in enumerate(self.blocks):
        h = block(
            h,
            t_emb,
            cond,
            ss_kv,
            self_kv_mask,
            cross_kv_mask,
            t_latent,
            order,
            pos,
            layer,
        )
    h = h.type(x.dtype)
    h = F.layer_norm(h, h.shape[-1:])
    h = self.out_layer(h)
    h = h.permute(0, 2, 1).view(
        h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3
    )
    h = unpatchify(h, self.patch_size).contiguous()
    return h


def slat_attn_forward(
    self,
    x,
    context=None,
    slat_kv=None,
    kv_mask=None,
    t_latent=None,
    order=None,
    pos=None,
    layer=None,
):
    if self._type == "self":
        qkv = self._linear(self.to_qkv, x)
        qkv = self._fused_pre(qkv, num_fused=3)
        if self.use_rope:
            qkv = self._rope(qkv)
        q, k, v = qkv.unbind(dim=1)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        if kv_mask is None:
            slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"] = k.cpu()
            slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"] = v.cpu()
        else:
            match_1_k = (k.coords.unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
            match_2_k = (
                slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"]
                .coords.cuda()
                .unsqueeze(1)
                == kv_mask.unsqueeze(0)
            ).all(dim=-1)
            idx_1_k = match_1_k.float().argmax(0)
            idx_2_k = match_2_k.float().argmax(0)
            feats_k = k.feats.clone()
            feats_k[idx_1_k] = slat_kv[
                f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"
            ].feats.cuda()[idx_2_k]
            k = k.replace(feats_k)
            k = k.type(q.dtype)

            match_1_v = (v.coords.unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
            match_2_v = (
                slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"]
                .coords.cuda()
                .unsqueeze(1)
                == kv_mask.unsqueeze(0)
            ).all(dim=-1)
            idx_1_v = match_1_v.float().argmax(0)
            idx_2_v = match_2_v.float().argmax(0)
            feats_v = v.feats.clone()
            feats_v[idx_1_v] = slat_kv[
                f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"
            ].feats.cuda()[idx_2_v]
            v = v.replace(feats_v)
            v = v.type(q.dtype)
        h = q
        q = q.feats
        k = k.feats
        v = v.feats
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
    else:
        q = self._linear(self.to_q, x)
        q = self._reshape_chs(q, (self.num_heads, -1))
        kv = self._linear(self.to_kv, context)
        kv = self._fused_pre(kv, num_fused=2)
        k, v = kv.unbind(dim=2)
        h = q
        q = q.feats
        q = q.unsqueeze(0)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if kv_mask is None:
            slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"] = k.cpu()
            slat_kv[f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"] = v.cpu()
        else:
            k = k * kv_mask + slat_kv[
                f"{t_latent}_{order}_{pos}_{layer}_{self._type}_k"
            ].cuda() * (1 - kv_mask)
            v = v * kv_mask + slat_kv[
                f"{t_latent}_{order}_{pos}_{layer}_{self._type}_v"
            ].cuda() * (1 - kv_mask)
            k = k.type(q.dtype)
            v = v.type(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.permute(0, 2, 1, 3)[0]
    h = h.replace(out)
    h = self._reshape_chs(h, (-1,))
    h = self._linear(self.to_out, h)
    return h


def slat_trsfmr_forward(
    self,
    x,
    mod,
    context,
    slat_kv,
    self_kv_mask,
    cross_kv_mask,
    t_latent,
    order,
    pos,
    layer,
):
    if self.share_mod:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )
    else:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(mod).chunk(6, dim=1)
        )

    h = x.replace(self.norm1(x.feats))
    h = h * (1 + scale_msa) + shift_msa
    h = self.self_attn(
        h,
        slat_kv=slat_kv,
        kv_mask=self_kv_mask,
        t_latent=t_latent,
        order=order,
        pos=pos,
        layer=layer,
    )
    h = h * gate_msa
    x = x + h
    h = x.replace(self.norm2(x.feats))
    h = self.cross_attn(
        h,
        context,
        slat_kv=slat_kv,
        kv_mask=cross_kv_mask,
        t_latent=t_latent,
        order=order,
        pos=pos,
        layer=layer,
    )
    x = x + h
    h = x.replace(self.norm3(x.feats))
    h = h * (1 + scale_mlp) + shift_mlp
    h = self.mlp(h)
    h = h * gate_mlp
    x = x + h
    return x


def slat_flow_forward(
    self, x, t, cond, slat_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos
):
    h = self.input_layer(x).type(self.dtype)
    t_emb = self.t_embedder(t)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    t_emb = t_emb.type(self.dtype)
    cond = cond.type(self.dtype)

    skips = []
    for block in self.input_blocks:
        h = block(h, t_emb)
        skips.append(h.feats)

    if self.pe_mode == "ape":
        h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
    for layer, block in enumerate(self.blocks):
        h = block(
            h,
            t_emb,
            cond,
            slat_kv,
            self_kv_mask,
            cross_kv_mask,
            t_latent,
            order,
            pos,
            layer,
        )

    for block, skip in zip(self.out_blocks, reversed(skips)):
        if self.use_skip_connection:
            h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
        else:
            h = block(h, t_emb)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h.type(x.dtype))
    return h


class InversionFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):
    def _inference_model(
        self,
        model,
        sample,
        t,
        cond,
        kv,
        self_kv_mask,
        cross_kv_mask,
        t_latent,
        order,
        pos,
    ):
        t = torch.tensor(
            [1000 * t] * sample.shape[0], device=sample.device, dtype=torch.float32
        )
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(
            sample, t, cond, kv, self_kv_mask, cross_kv_mask, t_latent, order, pos
        )

    def inference_model(
        self,
        model,
        sample,
        t,
        cond,
        cfg_strength,
        kv,
        self_kv_mask,
        cross_kv_mask,
        t_latent,
        order,
    ):
        cfg_interval = [0.5, 1.0]
        if cfg_interval[0] <= t_latent <= cfg_interval[1]:
            pos = 1
            pred = self._inference_model(
                model,
                sample,
                t,
                cond["cond"],
                kv,
                self_kv_mask,
                cross_kv_mask,
                t_latent,
                order,
                pos,
            )
            pos = 0
            neg_pred = self._inference_model(
                model,
                sample,
                t,
                cond["neg_cond"],
                kv,
                self_kv_mask,
                cross_kv_mask,
                t_latent,
                order,
                pos,
            )
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            pos = 1
            return self._inference_model(
                model,
                sample,
                t,
                cond["cond"],
                kv,
                self_kv_mask,
                cross_kv_mask,
                t_latent,
                order,
                pos,
            )

    def sample_once(
        self,
        model,
        sample,
        t_curr,
        t_prev,
        cond,
        cfg_strength,
        kv,
        self_kv_mask,
        cross_kv_mask,
        t_latent,
    ):
        order = 1
        pred = self.inference_model(
            model,
            sample,
            t_curr,
            cond,
            cfg_strength,
            kv,
            self_kv_mask,
            cross_kv_mask,
            t_latent,
            order,
        )
        sample_mid = sample + (t_prev - t_curr) / 2 * pred
        t_mid = t_curr + (t_prev - t_curr) / 2
        order = 2
        pred_mid = self.inference_model(
            model,
            sample_mid,
            t_mid,
            cond,
            cfg_strength,
            kv,
            self_kv_mask,
            cross_kv_mask,
            t_latent,
            order,
        )
        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        sample = (
            sample
            + (t_prev - t_curr) * pred
            - 0.5 * (t_prev - t_curr) ** 2 * first_order
        )
        return sample

    @torch.no_grad()
    def sample(
        self,
        model,
        stage,
        noise,
        cond,
        cfg_strength,
        latent=None,
        latent_mask=None,
        kv=None,
        self_kv_mask=None,
        cross_kv_mask=None,
        skip_step=None,
        noise_init=None,
    ):
        steps = 25
        rescale_t = 3.0
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        if latent_mask is None:
            inverse_bool = True
        else:
            inverse_bool = False

        if skip_step is not None:
            t_seq = t_seq[skip_step:]
            steps = steps - skip_step
        if noise_init is not None:
            noise_randn = torch.randn_like(noise)
            t_init = t_seq[0]
            sample = noise_init * (1 - t_init) + noise_randn * t_init

        if inverse_bool:
            t_seq = t_seq[::-1]
            desc = "Inversing"
            latent = {}
            kv = {}
        else:
            desc = "Sampling"

        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t_curr, t_prev in tqdm(t_pairs, desc=desc, disable=False):
            if inverse_bool:
                t_latent = t_prev
            else:
                t_latent = t_curr
                if stage == 1:
                    sample = sample * latent_mask + latent[f"{t_latent}"].cuda() * (
                        1 - latent_mask
                    )
                elif stage == 2:
                    match_1 = (
                        sample.coords.unsqueeze(1) == latent_mask.unsqueeze(0)
                    ).all(dim=-1)
                    match_2 = (
                        latent[f"{t_latent}"].coords.cuda().unsqueeze(1)
                        == latent_mask.unsqueeze(0)
                    ).all(dim=-1)
                    idx_1 = match_1.float().argmax(0)
                    idx_2 = match_2.float().argmax(0)
                    feats = sample.feats.clone()
                    feats[idx_1] = latent[f"{t_latent}"].feats.cuda()[idx_2]
                    sample = sample.replace(feats)
            sample = self.sample_once(
                model,
                sample,
                t_curr,
                t_prev,
                cond,
                cfg_strength,
                kv,
                self_kv_mask,
                cross_kv_mask,
                t_latent,
            )
            if inverse_bool:
                latent[f"{t_latent}"] = sample.cpu()
        return sample, latent, kv


def sample_sparse_structure_inverse(
    pipeline, cond_src, voxel_src, cfg_strength_stage_1_inverse, skip_step
):
    stage = 1
    flow_model = pipeline.models["sparse_structure_flow_model"]
    encoder = pipeline.models["sparse_structure_encoder"]
    z_s = encoder(voxel_src)
    sigma_min = pipeline.sparse_structure_sampler.sigma_min
    sparse_structure_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)

    if cfg_strength_stage_1_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_1_inverse

    noise, ss_latent, ss_kv = sparse_structure_sampler.sample(
        flow_model, stage, z_s, cond_src, cfg_strength, skip_step=skip_step
    )
    return noise, ss_latent, ss_kv


def sample_sparse_structure_denoise(
    pipeline,
    cond_tgt,
    noise,
    voxel_src,
    voxel_mask,
    ss_latent,
    ss_latent_mask,
    ss_kv,
    ss_self_kv_mask,
    ss_cross_kv_mask,
    cfg_strength_stage_1_forward,
    skip_step,
    re_init,
):
    stage = 1
    flow_model = pipeline.models["sparse_structure_flow_model"]
    sigma_min = pipeline.sparse_structure_sampler.sigma_min
    sparse_structure_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)

    if cfg_strength_stage_1_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_1_forward

    if re_init:
        encoder = pipeline.models["sparse_structure_encoder"]
        noise_init = encoder(voxel_src)
    else:
        noise_init = None

    z_s, ss_latent, ss_kv = sparse_structure_sampler.sample(
        flow_model,
        stage,
        noise,
        cond_tgt,
        cfg_strength,
        ss_latent,
        ss_latent_mask,
        ss_kv,
        ss_self_kv_mask,
        ss_cross_kv_mask,
        skip_step,
        noise_init,
    )
    decoder = pipeline.models["sparse_structure_decoder"]
    voxel = decoder(z_s)
    voxel = voxel * voxel_mask + voxel_src * (1 - voxel_mask)
    return voxel


def sample_slat_inverse(
    pipeline, cond_src, slat_src, coords_mask, cfg_strength_stage_2_inverse
):
    stage = 2
    coords_mask = torch.cat(
        [torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1
    )
    sparse_tensor_mask = torch.zeros(
        slat_src.coords.shape[0], dtype=torch.bool, device="cuda"
    )
    for coord in coords_mask:
        sparse_tensor_mask[torch.all(slat_src.coords == coord, dim=1)] = True
    slat_inverse = slat_src.replace(
        slat_src.feats[sparse_tensor_mask], slat_src.coords[sparse_tensor_mask]
    )

    flow_model = pipeline.models["slat_flow_model"]
    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[
        None
    ]
    slat_inverse = (slat_inverse - mean) / std

    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)

    if cfg_strength_stage_2_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_2_inverse
    noise, slat_latent, slat_kv = slat_sampler.sample(
        flow_model, stage, slat_inverse, cond_src, cfg_strength
    )
    return slat_latent, slat_kv


def sample_slat_denoise(
    pipeline,
    cond_tgt,
    coords_tgt,
    slat_src,
    coords_mask,
    slat_latent,
    slat_kv,
    slat_self_kv_mask,
    slat_cross_kv_mask,
    cfg_strength_stage_2_forward,
):
    stage = 2
    coords_mask = torch.cat(
        [torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1
    )
    flow_model = pipeline.models["slat_flow_model"]
    noise = sp.SparseTensor(
        feats=torch.randn(coords_tgt.shape[0], flow_model.in_channels).to(
            pipeline.device
        ),
        coords=coords_tgt,
    )

    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)

    if cfg_strength_stage_2_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_2_forward
    slat, slat_latent, slat_kv = slat_sampler.sample(
        flow_model,
        stage,
        noise,
        cond_tgt,
        cfg_strength,
        slat_latent,
        coords_mask,
        slat_kv,
        slat_self_kv_mask,
        slat_cross_kv_mask,
    )

    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[
        None
    ]
    slat = slat * std + mean

    match_1 = (coords_tgt.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
    match_2 = (slat_src.coords.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
    idx_1 = match_1.float().argmax(0)
    idx_2 = match_2.float().argmax(0)
    feats = slat.feats.clone()
    feats[idx_1] = slat_src.feats[idx_2]
    slat = slat.replace(feats)

    return slat


def run_edit(
    pipeline,
    render_dir,
    image_dir,
    output_path,
    skip_step=0,
    re_init=False,
    cfg=[5.0, 6.0, 0.0, 0.0],
):
    ss_flow = pipeline.models["sparse_structure_flow_model"]
    ss_flow.forward = MethodType(ss_flow_forward, ss_flow)
    for block in ss_flow.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(ss_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        self_attn_obj.forward = MethodType(ss_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        cross_attn_obj.forward = MethodType(ss_attn_forward, cross_attn_obj)

    slat_flow = pipeline.models["slat_flow_model"]
    slat_flow.forward = MethodType(slat_flow_forward, slat_flow)
    for block in slat_flow.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(slat_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        self_attn_obj.forward = MethodType(slat_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        cross_attn_obj.forward = MethodType(slat_attn_forward, cross_attn_obj)

    cfg_strength_stage_1_inverse = cfg[0]
    cfg_strength_stage_1_forward = cfg[1]
    cfg_strength_stage_2_inverse = cfg[2]
    cfg_strength_stage_2_forward = cfg[3]

    # torch.manual_seed(0)

    coords_src = ply_to_coords(os.path.join(render_dir, "voxels.ply"))
    voxel_src = coords_to_voxel(coords_src)
    slat_src = feats_to_slat(pipeline, os.path.join(render_dir, "features.npz"))
    img_src_path = os.path.join(image_dir, "2d_render.png")
    img_tgt_path = os.path.join(image_dir, "2d_edit.png")
    img_mask_path = os.path.join(image_dir, "2d_mask.png")
    ply_delete_path = os.path.join(render_dir, "voxels_delete.ply")

    pre_src, pre_tgt, pre_mask = preprocess_image(
        img_src_path, img_tgt_path, img_mask_path
    )
    cond_src = pipeline.get_cond([pre_src])
    cond_tgt = pipeline.get_cond([pre_tgt])
    coords_delete = ply_to_coords(ply_delete_path)
    coords_preserve = coords_src[~torch.isin(coords_src, coords_delete).all(dim=1)]

    voxel_mask, ss_latent_mask, ss_self_kv_mask, cross_kv_mask = ply_to_ss_mask(
        coords_preserve, pre_mask
    )
    noise, ss_latent, ss_kv = sample_sparse_structure_inverse(
        pipeline, cond_src, voxel_src, cfg_strength_stage_1_inverse, skip_step
    )
    voxel_tgt = sample_sparse_structure_denoise(
        pipeline,
        cond_tgt,
        noise,
        voxel_src,
        voxel_mask,
        ss_latent,
        ss_latent_mask,
        ss_kv,
        ss_self_kv_mask,
        cross_kv_mask,
        cfg_strength_stage_1_forward,
        skip_step,
        re_init,
    )
    coords_tgt = torch.argwhere(voxel_tgt > 0)[:, [0, 2, 3, 4]].int()

    slat_self_kv_mask = ply_to_slat_mask(coords_tgt, coords_preserve)
    slat_latent, slat_kv = sample_slat_inverse(
        pipeline, cond_src, slat_src, coords_preserve, cfg_strength_stage_2_inverse
    )
    slat_tgt = sample_slat_denoise(
        pipeline,
        cond_tgt,
        coords_tgt,
        slat_src,
        coords_preserve,
        slat_latent,
        slat_kv,
        slat_self_kv_mask,
        cross_kv_mask,
        cfg_strength_stage_2_forward,
    )
    assets_tgt = pipeline.decode_slat(slat_tgt, ["gaussian", "mesh"])

    # torch.set_grad_enabled(True)
    glb_tgt = postprocessing_utils.to_glb(
        assets_tgt["gaussian"][0],
        assets_tgt["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb_tgt.export(output_path)


if __name__ == "__main__":
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    run_edit(
        pipeline,
        render_dir="/path/to/render",
        image_dir="/path/to/image",
        output_path="/path/to/output.glb",
    )
