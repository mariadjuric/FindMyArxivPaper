from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import (
    AUTHORS_COLUMN,
    CATEGORY_COLORS,
    LABEL_COLUMN,
    PUBLISHED_COLUMN,
    SITE_DIR,
    TEXT_COLUMN,
    TITLE_COLUMN,
    URL_COLUMN,
)


def build_site(df: pd.DataFrame, embeddings: np.ndarray, site_dir: Path = SITE_DIR) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    projection = PCA(n_components=2).fit_transform(embeddings) if len(df) > 1 else np.zeros((len(df), 2))

    xs = projection[:, 0] if len(df) else np.array([0.0])
    ys = projection[:, 1] if len(df) else np.array([0.0])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    points = []
    for i, row in df.reset_index(drop=True).iterrows():
        label = str(row.get(LABEL_COLUMN, "unknown"))
        points.append(
            {
                "id": int(i),
                "x": float((projection[i, 0] - x_min) / x_span),
                "y": float((projection[i, 1] - y_min) / y_span),
                "title": str(row.get(TITLE_COLUMN, "")),
                "abstract": str(row.get(TEXT_COLUMN, "")),
                "category": label,
                "authors": str(row.get(AUTHORS_COLUMN, "")),
                "published": str(row.get(PUBLISHED_COLUMN, "")),
                "url": str(row.get(URL_COLUMN, "")),
                "color": CATEGORY_COLORS.get(label, "#94a3b8"),
            }
        )

    payload = {
        "points": points,
        "colors": {label: CATEGORY_COLORS.get(label, "#94a3b8") for label in sorted(df[LABEL_COLUMN].unique())},
        "title": "FMAP: FindMyArxivPaper Physics Atlas",
    }

    (site_dir / "data.js").write_text("window.SCIPAPER_DATA = " + json.dumps(payload) + ";\n", encoding="utf-8")
    (site_dir / "index.html").write_text(_html_template(), encoding="utf-8")
    return site_dir / "index.html"


def _html_template() -> str:
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>FMAP: FindMyArxivPaper Physics Atlas</title>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; font-family: Inter, system-ui, sans-serif; background:#0b1020; color:#f8fafc; }
    .layout { display:grid; grid-template-columns: 320px 1fr; min-height:100vh; }
    .sidebar { padding:24px; background:rgba(15,23,42,.95); border-right:1px solid rgba(148,163,184,.15); }
    h1 { margin:0 0 8px; font-size:42px; line-height:1; }
    h1 span { display:block; background:linear-gradient(90deg,#fde047,#f97316,#ec4899); -webkit-background-clip:text; color:transparent; }
    .muted { color:#94a3b8; margin-bottom:16px; }
    input { width:100%; padding:12px 14px; border-radius:14px; border:1px solid rgba(148,163,184,.2); background:#111827; color:#fff; }
    .legend { margin-top:18px; display:grid; gap:8px; }
    .legend-item { display:flex; align-items:center; gap:10px; font-size:14px; }
    .swatch { width:14px; height:14px; border-radius:999px; display:inline-block; }
    .card { margin-top:20px; background:rgba(30,41,59,.72); border:1px solid rgba(148,163,184,.14); border-radius:18px; padding:16px; min-height:200px; }
    .card h2 { font-size:24px; margin:0 0 8px; }
    .card .meta { color:#cbd5e1; font-size:14px; margin-bottom:10px; }
    .card a { color:#93c5fd; }
    .stage { position:relative; display:flex; align-items:stretch; justify-content:stretch; padding:12px; }
    canvas { width:100%; height:calc(100vh - 24px); background:#09090b; border-radius:24px; display:block; }
    .hint { position:absolute; right:28px; bottom:28px; background:rgba(15,23,42,.85); color:#e2e8f0; padding:10px 14px; border-radius:14px; font-size:13px; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } .sidebar { border-right:none; border-bottom:1px solid rgba(148,163,184,.15); } canvas { height:60vh; } }
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"sidebar\">
      <h1><span>FMAP</span> FindMyArxivPaper</h1>
      <div class=\"muted\">Interactive map of arXiv physics and astrophysics papers generated from embeddings of title + abstract.</div>
      <input id=\"search\" placeholder=\"Search by title, abstract, or author\" />
      <div id=\"legend\" class=\"legend\"></div>
      <div class=\"card\" id=\"details\">
        <h2>Select a paper</h2>
        <div class=\"meta\">Click any point on the map to inspect it.</div>
        <p>The map is a 2D PCA projection of sentence-transformer embeddings. Nearby points are semantically closer in title/abstract space.</p>
      </div>
    </aside>
    <main class=\"stage\">
      <canvas id=\"atlas\"></canvas>
      <div class=\"hint\">Search highlights papers. Click nearest point to inspect.</div>
    </main>
  </div>
  <script src=\"data.js\"></script>
  <script>
    const payload = window.SCIPAPER_DATA;
    const points = payload.points;
    const canvas = document.getElementById('atlas');
    const ctx = canvas.getContext('2d');
    const search = document.getElementById('search');
    const details = document.getElementById('details');
    const legend = document.getElementById('legend');
    let hovered = null;
    let filtered = points;

    function resize() {
      canvas.width = canvas.clientWidth * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;
      ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
      draw();
    }

    function pointXY(p) {
      const pad = 36;
      const w = canvas.clientWidth - pad * 2;
      const h = canvas.clientHeight - pad * 2;
      return { x: pad + p.x * w, y: pad + (1 - p.y) * h };
    }

    function draw() {
      ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
      ctx.fillStyle = '#09090b';
      ctx.fillRect(0,0,canvas.clientWidth,canvas.clientHeight);
      for (const p of points) {
        const {x,y} = pointXY(p);
        const active = filtered.includes(p);
        ctx.beginPath();
        ctx.fillStyle = active ? p.color : 'rgba(71,85,105,0.25)';
        ctx.arc(x,y, active ? 1.8 : 1.2, 0, Math.PI*2);
        ctx.fill();
      }
      if (hovered) {
        const {x,y} = pointXY(hovered);
        ctx.beginPath();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.arc(x,y,6,0,Math.PI*2);
        ctx.stroke();
      }
    }

    function updateLegend() {
      legend.innerHTML = '';
      Object.entries(payload.colors).forEach(([label, color]) => {
        const row = document.createElement('div');
        row.className = 'legend-item';
        row.innerHTML = `<span class=\"swatch\" style=\"background:${color}\"></span><span>${label}</span>`;
        legend.appendChild(row);
      });
    }

    function showDetails(p) {
      details.innerHTML = `
        <h2>${p.title}</h2>
        <div class=\"meta\">${p.category} · ${p.published ? p.published.slice(0,10) : 'unknown date'}</div>
        <p>${p.abstract}</p>
        <p><strong>Authors:</strong> ${p.authors || 'Unknown'}</p>
        <p><a href=\"${p.url}\" target=\"_blank\" rel=\"noreferrer\">Open on arXiv</a></p>
      `;
    }

    function nearestPoint(mx, my) {
      let best = null;
      let bestDist = Infinity;
      for (const p of filtered) {
        const {x,y} = pointXY(p);
        const d = Math.hypot(mx - x, my - y);
        if (d < bestDist) { bestDist = d; best = p; }
      }
      return bestDist < 16 ? best : null;
    }

    search.addEventListener('input', () => {
      const q = search.value.trim().toLowerCase();
      filtered = !q ? points : points.filter(p => [p.title, p.abstract, p.authors, p.category].join(' ').toLowerCase().includes(q));
      hovered = filtered[0] || null;
      draw();
      if (hovered) showDetails(hovered);
    });

    canvas.addEventListener('click', (event) => {
      const rect = canvas.getBoundingClientRect();
      const p = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
      if (p) { hovered = p; draw(); showDetails(p); }
    });

    updateLegend();
    window.addEventListener('resize', resize);
    resize();
    if (points.length) { hovered = points[0]; showDetails(points[0]); draw(); }
  </script>
</body>
</html>
"""
