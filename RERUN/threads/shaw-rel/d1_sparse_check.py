#!/usr/bin/env python
"""
D1: is the sparse Shaw implementation the thing it claims to be?

Both halves of Shaw et al. (2018) S3.1 are computed from the sparse edge list,
because the dense [L, L, R] relation mask is 365 MiB per cached graph at
L=3900. A sparse gather/scatter is exactly where an indexing bug hides without
crashing, and an attention block that silently attends along the wrong edges
would still train and still produce a plausible number -- so this compares
every term against a dense reference built straight from the definition.

Checks:
  1. key bias     scores[h,i,j] += q^h_i . sum_{rho: i -rho-> j} a^K_rho[h]
  2. value term   out[h,i] += sum_j alpha^h_ij sum_{rho: i -rho-> j} a^V_rho[h]
  3. the full block's output against a fully dense re-implementation
  4. gradients reach key_rel and value_rel_h
  5. rel_key="learned" reserves no key dimensions, and leaves the shipped
     "bits" path bit-identical to what it was

    python d1_sparse_check.py
"""
from __future__ import annotations
import math
import sys

import torch

from imputer.entity_mf.model import RelationalAttentionBlock

TOL = 2e-5


def make_graph(L: int, R: int, nnz: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, L, (nnz,), generator=g)
    tgt = torch.randint(0, L, (nnz,), generator=g)
    rel = torch.randint(0, R, (nnz,), generator=g)
    # Duplicate (i, j, rho) triples are legal in the sparse list but would make
    # the dense {0,1} mask disagree by construction, so de-duplicate.
    keyed = {(int(a), int(b), int(r)) for a, b, r in zip(src, tgt, rel)}
    src, tgt, rel = (torch.tensor(x) for x in zip(*sorted(keyed)))
    edge_flat = src * L + tgt
    mask = torch.zeros(L, L, R)
    mask[src, tgt, rel] = 1.0
    return edge_flat, rel, mask


def report(name: str, got, want) -> bool:
    d = (got - want).abs().max().item()
    ok = d < TOL
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} max|diff| = {d:.3e}")
    return ok


def main() -> int:
    torch.manual_seed(7)
    B, L, D, H, R, nnz = 3, 37, 24, 4, 6, 80
    hd = D // H
    edge_flat, edge_rel, mask = make_graph(L, R, nnz)
    x = torch.randn(B, L, D)
    ok = True

    blk = RelationalAttentionBlock(
        model_dim=D, num_heads=H, num_relationships=R, dropout=0.0,
        use_per_head_rel=False, use_pointer=False,
        rel_key="learned", rel_value="learned",
    ).eval()

    # ── 1. the key bias ────────────────────────────────────────────────────────
    Qh = blk.Q(x).view(B, L, H, hd).transpose(1, 2) * (1.0 / math.sqrt(hd))
    rel_logits = torch.einsum("bhid,rhd->bhir", Qh, blk.key_rel)
    got = blk._rel_bias(rel_logits, edge_flat, edge_rel, L)
    # dense: bias[b,h,i,j] = sum_r rel_logits[b,h,i,r] * mask[i,j,r]
    want = torch.einsum("bhir,ijr->bhij", rel_logits, mask)
    ok &= report("key bias vs dense einsum", got, want)

    # ── 2. the value mass ──────────────────────────────────────────────────────
    attn = torch.softmax(torch.randn(B, H, L, L), dim=-1)
    got = blk._rel_mass(attn, edge_flat, edge_rel, L, R)
    want = torch.einsum("bhij,ijr->bhir", attn, mask)
    ok &= report("value mass vs dense einsum", got, want)

    # ── 3. the whole block ─────────────────────────────────────────────────────
    with torch.no_grad():
        got = blk(x, edge_flat, edge_rel)

    def dense_block(blk, x):
        """Shaw et al. S3.1 written out with the dense mask, no sparse tricks."""
        Bv, Lv, Dv = x.shape
        q = blk.Q(x).view(Bv, Lv, H, hd).transpose(1, 2) / math.sqrt(hd)
        k = blk.K(x).view(Bv, Lv, H, hd).transpose(1, 2)
        v = blk.V(x).view(Bv, Lv, H, hd).transpose(1, 2)
        # a_K[i,j,h,d] = sum_rho mask[i,j,rho] key_rel[rho,h,d]
        a_K = torch.einsum("ijr,rhd->ijhd", mask, blk.key_rel)
        s = torch.einsum("bhid,bhjd->bhij", q, k) \
            + torch.einsum("bhid,ijhd->bhij", q, a_K)
        al = torch.softmax(s, dim=-1)
        a_V = torch.einsum("ijr,rhd->ijhd", mask, blk.value_rel_h)
        o = torch.einsum("bhij,bhjd->bhid", al, v) \
            + torch.einsum("bhij,ijhd->bhid", al, a_V)
        return blk.out(o.transpose(1, 2).reshape(Bv, Lv, Dv))

    with torch.no_grad():
        ok &= report("full block vs dense reference", got, dense_block(blk, x))

    # ── 4. gradients reach both embeddings ─────────────────────────────────────
    blk.zero_grad()
    blk(x, edge_flat, edge_rel).sum().backward()
    for nm in ("key_rel", "value_rel_h"):
        g = getattr(blk, nm).grad
        live = g is not None and g.abs().sum().item() > 0
        print(f"  {'PASS' if live else 'FAIL'}  gradient reaches {nm:<22} "
              f"{'nonzero' if live else 'MISSING'}")
        ok &= live

    # ── 5. shapes, and the shipped path is untouched ───────────────────────────
    print(f"  INFO  learned: k_content_dim={blk.k_content_dim} (head_dim={hd}), "
          f"Q out={blk.Q.out_features}, a^K={tuple(blk.key_rel.shape)}")
    dims_ok = blk.k_content_dim == hd and blk.Q.out_features == D
    print(f"  {'PASS' if dims_ok else 'FAIL'}  learned reserves no key dims")
    ok &= dims_ok

    shipped = RelationalAttentionBlock(
        model_dim=D, num_heads=H, num_relationships=R, dropout=0.0,
        use_per_head_rel=False, use_pointer=True, scale_shared_rel=True,
    ).eval()
    bits_ok = (shipped.k_content_dim == hd
               and shipped.Q.out_features == D + R + 3
               and not hasattr(shipped, "key_rel")
               and not hasattr(shipped, "value_rel_h"))
    print(f"  {'PASS' if bits_ok else 'FAIL'}  shipped 'bits' block unchanged "
          f"(Q out={shipped.Q.out_features}, expected {D + R + 3})")
    ok &= bits_ok

    try:
        RelationalAttentionBlock(model_dim=D, num_heads=H, num_relationships=R,
                                dropout=0.0, use_rel_value=True, rel_value="learned")
    except ValueError:
        print("  PASS  use_rel_value + rel_value='learned' refused")
    else:
        print("  FAIL  the two value terms can both be on at once")
        ok = False

    print("\n" + ("ALL CHECKS PASS" if ok else "SOMETHING FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
