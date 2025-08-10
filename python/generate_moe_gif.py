#!/usr/bin/env python3
import argparse, math
import numpy as np
from PIL import Image, ImageDraw

PALETTE = [
    (110,231,183),(147,197,253),(244,114,182),(245,158,11),
    (52,211,153),(96,165,250),(167,139,250),(251,113,133),
    (34,211,238),(234,179,8),(74,222,128),(249,115,22),
]

def quad_bezier(p0, p1, p2, t):
    a = (1 - t) ** 2
    b = 2 * (1 - t) * t
    c = t ** 2
    return (a * p0[0] + b * p1[0] + c * p2[0],
            a * p0[1] + b * p1[1] + c * p2[1])

def gated_topk(x_t, W, top_k=2, noise_scale=0.18, rng=None):
    # Top-k gating with noise; returns masked-softmax weights for chosen indices.
    if rng is None:
        rng = np.random.default_rng()
    logits = x_t @ W + rng.normal(0, noise_scale, size=W.shape[1])
    k = min(top_k, W.shape[1])
    top_idx = np.argpartition(logits, -k)[-k:]
    masked = np.full_like(logits, -np.inf, dtype=float)
    masked[top_idx] = logits[top_idx]
    finite = np.isfinite(masked)
    m = np.max(masked[finite])
    exps = np.zeros_like(masked, dtype=float)
    exps[finite] = np.exp(masked[finite] - m)
    weights = exps / (exps.sum() + 1e-9)
    chosen = np.sort(top_idx)
    return weights, chosen

def main():
    ap = argparse.ArgumentParser(description='Render an MoE routing GIF (local/shared/micro).')
    ap.add_argument('--out', default='moe.gif', help='Output GIF path')
    ap.add_argument('--width', type=int, default=900)
    ap.add_argument('--height', type=int, default=520)
    ap.add_argument('--n-local', type=int, default=6, help='Number of local base experts')
    ap.add_argument('--n-shared', type=int, default=2, help='Number of shared base experts')
    ap.add_argument('--micro', type=int, default=3, help='Micro-experts per base expert')
    ap.add_argument('--topk', type=int, default=6, help='Top-k across all micro-experts')
    ap.add_argument('--embd', type=int, default=16, help='Embedding dimension')
    ap.add_argument('--frames', type=int, default=160)
    ap.add_argument('--fps', type=int, default=6)
    ap.add_argument('--dot-period', type=int, default=360, help='Frames per packet loop (higher = slower)')
    ap.add_argument('--noise', type=float, default=0.18, help='NoisyTopK scale')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--show-all', action='store_true', help='Draw faint non-top-k arcs for context')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    total_base = args.n_local + args.n_shared
    total_micro = total_base * args.micro

    # projection over all micro-experts
    W = rng.normal(0, 1, size=(args.embd, total_micro))

    canvas = (args.width, args.height)
    margin = 40
    router_x = 140
    experts_x = canvas[0] - 180
    packet_count = max(10, total_micro // 2)

    router_center = (router_x, canvas[1] // 2)
    # base expert centers
    gap = (canvas[1] - 2 * margin) / max(1, total_base - 1)
    base_centers = [(experts_x, int(margin + i * gap)) for i in range(total_base)]

    # micro positions for each base expert
    def micro_pos(base_idx):
        cx, cy = base_centers[base_idx]
        w, h = 130, 46
        row_y = cy + h // 2 + 12
        m_gap = w / (args.micro + 1)
        return [(cx - w//2 + int((j+1)*m_gap), row_y) for j in range(args.micro)]

    all_micro = [p for b in range(total_base) for p in micro_pos(b)]

    # input random walk (slower for clarity)
    x = rng.normal(0, 1, size=(args.frames, args.embd))
    for t in range(1, args.frames):
        x[t] = 0.95 * x[t-1] + 0.05 * rng.normal(0, 1, size=args.embd)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    packets = []
    for _ in range(packet_count):
        st = int(rng.integers(0, args.frames))
        dur = int(rng.integers(int(0.6*args.fps), int(1.6*args.fps)))
        base = int(rng.integers(0, total_base))
        packets.append((st, dur, base))

    def draw_frame(t):
        img = Image.new("RGB", canvas, "white")
        d = ImageDraw.Draw(img)
        d.text((margin, margin//2), f"MoE (top-k={args.topk})  frame {t+1}/{args.frames}", fill="black")
        r = 34
        d.ellipse((router_center[0]-r, router_center[1]-r, router_center[0]+r, router_center[1]+r), outline="black", width=2)
        d.text((router_center[0]-24, router_center[1]-8), "Router", fill="black")

        # base boxes
        for i, (cx, cy) in enumerate(base_centers):
            w, h = 130, 46
            d.rectangle((cx-w//2, cy-h//2, cx+w//2, cy+h//2), outline=PALETTE[i % len(PALETTE)], width=2)
            d.text((cx-w//2+8, cy+4), f"{'E' if i<args.n_local else 'S'}{i if i<args.n_local else i-args.n_local}", fill="black")
            # micro dots
            for (mx, my) in micro_pos(i):
                d.ellipse((mx-3, my-3, mx+3, my+3), fill=PALETTE[i % len(PALETTE)])

        # gating over all micro
        wts, chosen = gated_topk(x[t], W, top_k=args.topk, noise_scale=args.noise, rng=rng)

        # faint background arcs (optional)
        if args.show_all:
            for j, (mx, my) in enumerate(all_micro):
                base = j // args.micro
                p0 = (router_center[0]+28, router_center[1])
                p1 = ((p0[0]+mx)//2, (p0[1]+my)//2 - 60)
                pts = [quad_bezier(p0, p1, (mx, my), s/24.0) for s in range(25)]
                pts = [(int(px), int(py)) for px, py in pts]
                d.line(pts, fill=(200,200,200), width=1)

        # top-k arcs
        for k, j in enumerate(chosen):
            mx, my = all_micro[j]
            base = j // args.micro
            color = PALETTE[base % len(PALETTE)]
            p0 = (router_center[0]+28, router_center[1])
            p1 = ((p0[0]+mx)//2, (p0[1]+my)//2 - 60)
            pts = [quad_bezier(p0, p1, (mx, my), s/24.0) for s in range(25)]
            pts = [(int(px), int(py)) for px, py in pts]
            width = max(2, int(2 + 18 * float(wts[j])))
            for off in range(-(width//2), (width//2)+1):
                shifted = [(x1, y1+off) for x1, y1 in pts]
                d.line(shifted, fill=color, width=1)

        # moving packets
        chosen_bases = set([j // args.micro for j in chosen])
        for st, dur, base in packets:
            if st <= t < st + dur and base in chosen_bases:
                cand = [j for j in chosen if j // args.micro == base]
                if cand:
                    j = max(cand, key=lambda jj: wts[jj])
                    mx, my = all_micro[j]
                    p0 = (router_center[0]+28, router_center[1])
                    p1 = ((p0[0]+mx)//2, (p0[1]+my)//2 - 60)
                    phase = (t % args.dot_period) / args.dot_period
                    px, py = quad_bezier(p0, p1, (mx, my), phase)
                    pr = 5
                    d.ellipse((px-pr, py-pr, px+pr, py+pr), fill=PALETTE[base % len(PALETTE)])

        return img

    images = [draw_frame(t) for t in range(args.frames)]
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=int(1000/args.fps), loop=0, disposal=2)
    print(args.out)

if __name__ == '__main__':
    main()
