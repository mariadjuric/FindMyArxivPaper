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
    similarities = embeddings @ embeddings.T if len(df) > 0 else np.zeros((0, 0))

    xs = projection[:, 0] if len(df) else np.array([0.0])
    ys = projection[:, 1] if len(df) else np.array([0.0])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    points = []
    for i, row in df.reset_index(drop=True).iterrows():
        label = str(row.get(LABEL_COLUMN, "unknown"))
        sims = similarities[i].copy() if len(df) > 1 else np.array([])
        recs = []
        if len(df) > 1:
            sims[i] = -1.0
            for idx in np.argsort(-sims)[:5]:
                if sims[idx] <= -1:
                    continue
                other = df.iloc[idx]
                recs.append(
                    {
                        "title": str(other.get(TITLE_COLUMN, "")),
                        "category": str(other.get(LABEL_COLUMN, "unknown")),
                        "url": str(other.get(URL_COLUMN, "")),
                        "match": round(float(max(0.0, min(1.0, (sims[idx] + 1) / 2))) * 100, 1),
                    }
                )
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
                "recommendations": recs,
            }
        )

    payload = {
        "points": points,
        "colors": {label: CATEGORY_COLORS.get(label, "#94a3b8") for label in sorted(df[LABEL_COLUMN].unique())},
        "title": "FMAP: FindMyArxivPaper Astro Atlas",
    }

    (site_dir / "data.js").write_text("window.FMAP_DATA = " + json.dumps(payload) + ";\n", encoding="utf-8")
    (site_dir / "index.html").write_text(_html_template(), encoding="utf-8")
    return site_dir / "index.html"


def _html_template() -> str:
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>FMAP: FindMyArxivPaper Astro Atlas</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, system-ui, sans-serif; background: #09111f; color: #f8fafc; }
    .layout { display: grid; grid-template-columns: 360px 1fr 360px; min-height: 100vh; }
    .panel { padding: 22px; background: rgba(15,23,42,.96); border-right: 1px solid rgba(148,163,184,.14); overflow: auto; }
    .panel.right { border-right: none; border-left: 1px solid rgba(148,163,184,.14); }
    .stage { position: relative; padding: 14px; }
    canvas { width: 100%; height: calc(100vh - 28px); display: block; border-radius: 24px; background: radial-gradient(circle at top, #111827, #050816 70%); }
    h1 { margin: 0 0 10px; font-size: 40px; line-height: 1; }
    h1 span { display: block; background: linear-gradient(90deg,#fde047,#fb7185,#a78bfa); -webkit-background-clip: text; color: transparent; }
    h2 { margin: 0 0 10px; font-size: 22px; }
    .muted { color: #94a3b8; font-size: 14px; line-height: 1.45; }
    input { width: 100%; margin-top: 12px; padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(148,163,184,.18); background: #111827; color: #fff; }
    .toolbar { display: flex; gap: 8px; margin-top: 10px; }
    button { border: 0; background: #1d4ed8; color: white; padding: 10px 12px; border-radius: 12px; cursor: pointer; }
    button.secondary { background: #334155; }
    .legend { margin-top: 18px; display: grid; gap: 8px; }
    .legend-item { display: flex; align-items: center; gap: 10px; font-size: 14px; }
    .swatch { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }
    .card { margin-top: 18px; background: rgba(30,41,59,.72); border: 1px solid rgba(148,163,184,.14); border-radius: 18px; padding: 16px; }
    .meta { color: #cbd5e1; font-size: 14px; margin-bottom: 10px; }
    a { color: #93c5fd; }
    .results, .recs { margin-top: 12px; display: grid; gap: 10px; }
    .result-item, .rec-item { background: rgba(15,23,42,.55); border: 1px solid rgba(148,163,184,.12); border-radius: 14px; padding: 12px; cursor: pointer; }
    .result-item:hover, .rec-item:hover { border-color: rgba(96,165,250,.65); }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: rgba(148,163,184,.16); margin-bottom: 6px; }
    .match { color: #fde68a; font-weight: 600; }
    .hint { position: absolute; right: 28px; bottom: 28px; background: rgba(15,23,42,.9); color: #e2e8f0; padding: 10px 14px; border-radius: 14px; font-size: 13px; }
    @media (max-width: 1200px) { .layout { grid-template-columns: 1fr; } .panel.right { border-left: none; border-top: 1px solid rgba(148,163,184,.14); } .panel { border-right: none; border-bottom: 1px solid rgba(148,163,184,.14); } canvas { height: 60vh; } }
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"panel\">
      <h1><span>FMAP</span> Astro Finder</h1>
      <div class=\"muted\">Astro-ph only for now. Search papers, highlight matching points on the atlas, then inspect a paper to get similar recommendations with approximate match percentages.</div>
      <input id=\"search\" placeholder=\"Search titles, abstracts, authors, categories\" />
      <div class=\"toolbar\">
        <button id=\"clearBtn\" class=\"secondary\">Clear search</button>
        <button id=\"focusBtn\">Focus matches</button>
      </div>
      <div id=\"legend\" class=\"legend\"></div>
      <div class=\"card\">
        <h2>Search results</h2>
        <div class=\"muted\" id=\"resultSummary\">Type to highlight matching astro-ph papers.</div>
        <div id=\"results\" class=\"results\"></div>
      </div>
    </aside>
    <main class=\"stage\">
      <canvas id=\"atlas\"></canvas>
      <div class=\"hint\">Search dims non-matches. Click a point or result card to inspect and get related papers.</div>
    </main>
    <aside class=\"panel right\">
      <div class=\"card\" id=\"details\">
        <h2>Select a paper</h2>
        <div class=\"meta\">Click any highlighted point or a result card.</div>
        <p class=\"muted\">This view uses a 2D PCA projection of embedding vectors from title + abstract, so nearby points are semantically closer but not exact topic clusters.</p>
      </div>
      <div class=\"card\">
        <h2>Recommended papers</h2>
        <div class=\"muted\">Similar papers for the currently selected one.</div>
        <div id=\"recs\" class=\"recs\"></div>
      </div>
    </aside>
  </div>
  <script src=\"data.js\"></script>
  <script>
    const payload = window.FMAP_DATA;
    const points = payload.points;
    const canvas = document.getElementById('atlas');
    const ctx = canvas.getContext('2d');
    const searchInput = document.getElementById('search');
    const clearBtn = document.getElementById('clearBtn');
    const focusBtn = document.getElementById('focusBtn');
    const details = document.getElementById('details');
    const legend = document.getElementById('legend');
    const resultsEl = document.getElementById('results');
    const recsEl = document.getElementById('recs');
    const resultSummary = document.getElementById('resultSummary');

    let selected = null;
    let filtered = points.slice();
    let focusMatches = false;

    function resize() {
      canvas.width = canvas.clientWidth * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      draw();
    }

    function pointXY(p) {
      const pad = 30;
      const w = canvas.clientWidth - pad * 2;
      const h = canvas.clientHeight - pad * 2;
      let x = pad + p.x * w;
      let y = pad + (1 - p.y) * h;
      if (focusMatches && filtered.length && filtered.includes(p)) {
        const cx = canvas.clientWidth / 2;
        const cy = canvas.clientHeight / 2;
        x = cx + (x - cx) * 0.72;
        y = cy + (y - cy) * 0.72;
      }
      return { x, y };
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      ctx.fillStyle = '#050816';
      ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      for (const p of points) {
        const { x, y } = pointXY(p);
        const active = filtered.includes(p);
        ctx.beginPath();
        ctx.fillStyle = active ? p.color : 'rgba(71,85,105,0.12)';
        ctx.arc(x, y, active ? 2.4 : 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
      if (selected) {
        const { x, y } = pointXY(selected);
        ctx.beginPath();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.arc(x, y, 8, 0, Math.PI * 2);
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
      selected = p;
      details.innerHTML = `
        <h2>${p.title}</h2>
        <div class=\"meta\">${p.category} · ${p.published ? p.published.slice(0,10) : 'unknown date'}</div>
        <p>${p.abstract}</p>
        <p><strong>Authors:</strong> ${p.authors || 'Unknown'}</p>
        <p><a href=\"${p.url}\" target=\"_blank\" rel=\"noreferrer\">Open on arXiv</a></p>
      `;
      renderRecommendations(p);
      draw();
    }

    function renderRecommendations(p) {
      recsEl.innerHTML = '';
      if (!p.recommendations || !p.recommendations.length) {
        recsEl.innerHTML = '<div class=\"muted\">No recommendations available.</div>';
        return;
      }
      p.recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'rec-item';
        item.innerHTML = `
          <div class=\"pill\">${rec.category}</div>
          <div><strong>${rec.title}</strong></div>
          <div class=\"match\">${rec.match}% match</div>
          ${rec.url ? `<div><a href=\"${rec.url}\" target=\"_blank\" rel=\"noreferrer\">Open on arXiv</a></div>` : ''}
        `;
        recsEl.appendChild(item);
      });
    }

    function nearestPoint(mx, my) {
      let best = null;
      let bestDist = Infinity;
      for (const p of filtered) {
        const { x, y } = pointXY(p);
        const d = Math.hypot(mx - x, my - y);
        if (d < bestDist) { bestDist = d; best = p; }
      }
      return bestDist < 18 ? best : null;
    }

    function scoreMatch(p, q) {
      const hay = [p.title, p.abstract, p.authors, p.category].join(' ').toLowerCase();
      return hay.includes(q) ? 1 : 0;
    }

    function renderResults() {
      resultsEl.innerHTML = '';
      const q = searchInput.value.trim().toLowerCase();
      if (!q) {
        resultSummary.textContent = `Showing all ${points.length} astro-ph papers.`;
        return;
      }
      resultSummary.textContent = `${filtered.length} matching papers highlighted.`;
      filtered.slice(0, 25).forEach(p => {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.innerHTML = `
          <div class=\"pill\">${p.category}</div>
          <div><strong>${p.title}</strong></div>
          <div class=\"muted\">${p.authors || 'Unknown authors'}</div>
        `;
        item.addEventListener('click', () => showDetails(p));
        resultsEl.appendChild(item);
      });
    }

    function updateFilter() {
      const q = searchInput.value.trim().toLowerCase();
      filtered = !q ? points.slice() : points.filter(p => scoreMatch(p, q));
      if (selected && !filtered.includes(selected) && q) {
        selected = filtered[0] || null;
      }
      renderResults();
      draw();
      if (selected) showDetails(selected);
    }

    searchInput.addEventListener('input', updateFilter);
    clearBtn.addEventListener('click', () => {
      searchInput.value = '';
      filtered = points.slice();
      selected = points[0] || null;
      renderResults();
      if (selected) showDetails(selected); else draw();
    });
    focusBtn.addEventListener('click', () => { focusMatches = !focusMatches; focusBtn.textContent = focusMatches ? 'Unfocus matches' : 'Focus matches'; draw(); });
    canvas.addEventListener('click', (event) => {
      const rect = canvas.getBoundingClientRect();
      const p = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
      if (p) showDetails(p);
    });

    updateLegend();
    window.addEventListener('resize', resize);
    resize();
    renderResults();
    if (points.length) showDetails(points[0]);
  </script>
</body>
</html>
"""
