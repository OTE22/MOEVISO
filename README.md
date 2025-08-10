# MoE Visualizer — Top‑k, Shared & Fine‑Grained Experts

Interactive **Mixture‑of‑Experts (MoE)** visualizer (HTML/Canvas + Chart.js) and a **Python** tool
to render animated GIFs that show routing “flow” through experts.

## Features
- **Top‑k routing** with noisy gating
- **Shared experts** (global experts) + **Local experts**
- **Fine‑grained experts** (micro‑experts per expert)
- Colored arcs and moving packets; **thicker arcs = higher weight**
- Live bar chart for the current top‑k weights
- **Speed controls** (FPS & Flow) in the web app

## Repo layout
```
moe-visualizer/
├─ web/
│  ├─ index.html            # Interactive visualizer (self-contained)
│  └─ assets/
│     └─ preview.png        # Social preview (Open Graph)
├─ python/
│  └─ generate_moe_gif.py   # CLI tool to export routing as animated GIF
├─ examples/
│  └─ moe_flow_slow.gif     # Sample output (slow GIF)
├─ requirements.txt         # Python deps for GIF tool
├─ LICENSE                  # MIT
└─ README.md
```

## Running the web app
Just open `web/index.html` in your browser. It’s fully client-side and uses CDN for Chart.js.

**Controls:**
- Local experts (`N_local`), Shared experts (`S_shared`), Micro-experts per expert (`M`)
- `Top‑k` (over all micro-experts)
- Embedding dimension, Noise (NoisyTopK)
- Randomize / Compute once / Start animation / Stop
- **Playback FPS** + **Flow speed** (slower = clearer)

## Exporting a GIF (Python)
```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# basic run
python python/generate_moe_gif.py --out examples/my_moe.gif

# with custom params
python python/generate_moe_gif.py \
  --experts 8 --topk 3 --embd 16 --frames 160 --fps 6 \
  --dot-period 360 --noise 0.18 --seed 42 --width 900 --height 500
```

## Host for a preview card on LinkedIn
1. **Netlify Drop**: drag‑and‑drop the `web/` folder → get a public URL.
2. **GitHub Pages**: push repo → Settings → Pages → deploy `/web` folder (or root).

The `index.html` already includes Open Graph meta tags:
- `og:title`, `og:description`, `og:image` → `assets/preview.png`

> Tip: LinkedIn won’t animate GIFs in the card. Use the static `assets/preview.png`,
and upload your GIF as media in the post itself.

## License
MIT — see `LICENSE`.
