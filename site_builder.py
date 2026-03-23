from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

from config import (
    AUTHORS_COLUMN,
    CATEGORY_COLORS,
    LABEL_COLUMN,
    PUBLISHED_COLUMN,
    SITE_DIR,
    TEXT_COLUMN,
    TITLE_COLUMN,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
    URL_COLUMN,
)


def build_site(df: pd.DataFrame, embeddings: np.ndarray, site_dir: Path = SITE_DIR) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    projection = _project_embeddings(embeddings)
    similarities = embeddings @ embeddings.T if len(df) > 0 else np.zeros((0, 0))

    xs = projection[:, 0] if len(df) else np.array([0.0])
    ys = projection[:, 1] if len(df) else np.array([0.0])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    label_counts = df[LABEL_COLUMN].value_counts().to_dict() if len(df) else {}

    points = []
    for i, row in df.reset_index(drop=True).iterrows():
        label = str(row.get(LABEL_COLUMN, "unknown"))
        sims = similarities[i].copy() if len(df) > 1 else np.array([])
        recs = []
        if len(df) > 1:
            sims[i] = -1.0
            for idx in np.argsort(-sims)[:8]:
                if sims[idx] <= -1:
                    continue
                other = df.iloc[idx]
                recs.append(
                    {
                        "id": int(idx),
                        "title": str(other.get(TITLE_COLUMN, "")),
                        "category": str(other.get(LABEL_COLUMN, "unknown")),
                        "url": str(other.get(URL_COLUMN, "")),
                        "authors": str(other.get(AUTHORS_COLUMN, "")),
                        "published": str(other.get(PUBLISHED_COLUMN, "")),
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
        "counts": {label: int(count) for label, count in sorted(label_counts.items())},
        "stats": {
            "papers": int(len(df)),
            "categories": int(df[LABEL_COLUMN].nunique()) if len(df) else 0,
            "projection": "UMAP",
            "umap_neighbors": int(min(UMAP_N_NEIGHBORS, max(2, len(df) - 1))) if len(df) > 1 else 0,
            "umap_min_dist": float(UMAP_MIN_DIST),
            "last_updated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        },
        "title": "FMAP: FindMyArxivPaper Astro Atlas",
    }

    (site_dir / "data.js").write_text("window.FMAP_DATA = " + json.dumps(payload) + ";\n", encoding="utf-8")
    (site_dir / "index.html").write_text(_html_template(), encoding="utf-8")
    return site_dir / "index.html"


def _project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) <= 1:
        return np.zeros((len(embeddings), 2))
    n_neighbors = min(UMAP_N_NEIGHBORS, max(2, len(embeddings) - 1))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    return reducer.fit_transform(embeddings)


def _html_template() -> str:
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>FMAP: FindMyArxivPaper Astro Atlas</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #020409;
      --panel: rgba(10, 15, 28, 0.92);
      --border: rgba(123, 146, 179, 0.18);
      --text: #ecf3ff;
      --muted: #8fa4c7;
      --shadow: 0 20px 70px rgba(0, 0, 0, 0.38);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body {
      background:
        radial-gradient(circle at 18% 12%, rgba(125, 211, 252, 0.08), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(192, 132, 252, 0.07), transparent 24%),
        radial-gradient(circle at 50% 100%, rgba(110, 231, 183, 0.06), transparent 30%),
        #020409;
    }
    .layout { display:grid; grid-template-columns:360px minmax(0,1fr) 390px; min-height:100vh; gap:16px; padding:16px; }
    .panel,.stage-shell { min-height:calc(100vh - 32px); border:1px solid var(--border); border-radius:28px; background:var(--panel); box-shadow:var(--shadow); backdrop-filter:blur(10px); overflow:hidden; }
    .panel { display:flex; flex-direction:column; }
    .panel-inner { padding:24px; overflow:auto; height:100%; }
    .brand { font-size:48px; line-height:.95; font-weight:800; margin:0 0 14px; letter-spacing:-.04em; }
    .brand .mark { display:block; background:linear-gradient(120deg,#7dd3fc 0%,#6ee7b7 52%,#c084fc 100%); -webkit-background-clip:text; color:transparent; }
    .subtle,.muted { color:var(--muted); font-size:14px; line-height:1.6; }
    .stats { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin:18px 0; }
    .stat { background:rgba(18, 28, 48, 0.82); border:1px solid var(--border); border-radius:18px; padding:14px; }
    .stat .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .stat .value { margin-top:6px; font-size:22px; font-weight:700; }
    .controls { display:grid; gap:10px; margin-top:14px; }
    input { width:100%; border-radius:16px; border:1px solid rgba(128,154,192,.18); background:rgba(13,21,38,.96); color:var(--text); padding:14px 16px; font-size:15px; outline:none; }
    input:focus { border-color: rgba(125, 211, 252, 0.55); box-shadow: 0 0 0 4px rgba(125, 211, 252, 0.08); }
    .toolbar,.stage-toolbar { display:flex; flex-wrap:wrap; gap:8px; margin-top:2px; }
    button { border:0; border-radius:14px; padding:11px 14px; background:linear-gradient(135deg,#102c46 0%,#18405e 100%); color:var(--text); cursor:pointer; font-weight:600; }
    button.secondary { background:rgba(54,69,96,.9); }
    button.active { outline:2px solid rgba(125,211,252,.55); }
    .section-title { margin:22px 0 10px; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.12em; }
    .legend { display:flex; flex-wrap:wrap; gap:8px; }
    .legend-item { border-radius:999px; border:1px solid var(--border); background:rgba(17,25,42,.88); color:var(--text); padding:8px 12px; font-size:13px; display:inline-flex; align-items:center; gap:8px; cursor:pointer; }
    .legend-item.active { border-color: rgba(125, 211, 252, 0.45); background: rgba(22, 37, 64, 0.96); }
    .swatch { width:10px; height:10px; border-radius:999px; display:inline-block; }
    .scroll-block { margin-top:12px; display:grid; gap:10px; }
    .result-item,.rec-item { background:rgba(16,24,40,.84); border:1px solid var(--border); border-radius:18px; padding:14px; cursor:pointer; transition:transform .15s ease,border-color .15s ease,background .15s ease; }
    .result-item:hover,.rec-item:hover { transform:translateY(-1px); border-color:rgba(125,211,252,.45); background:rgba(20,32,52,.96); }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:5px 10px; border-radius:999px; background:rgba(125,211,252,.08); color:#d9f4ff; font-size:12px; margin-bottom:8px; }
    .title-sm { font-weight:700; line-height:1.35; }
    .meta { color:#c5d4ea; font-size:13px; margin-top:8px; line-height:1.5; }
    .stage-shell { display:grid; grid-template-rows:auto minmax(0,1fr); background:rgba(8,11,22,.92); position:relative; }
    .stage-head { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:18px 20px 8px; color:var(--muted); font-size:13px; }
    .canvas-wrap { position:relative; padding:12px; min-height:0; }
    .atlas-canvas { width:100%; height:calc(100vh - 84px); min-height:640px; display:block; border-radius:22px; border:1px solid rgba(127,146,183,.08); }
    #atlasBase { background:radial-gradient(circle at 50% 30%, rgba(31,41,80,.22), transparent 45%), radial-gradient(circle at 50% 100%, rgba(18,31,51,.18), transparent 50%), #030611; }
    #atlasOverlay { position:absolute; inset:12px; pointer-events:none; background:transparent; }
    .hint,.hover-card { position:absolute; background:rgba(9,14,27,.92); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); backdrop-filter:blur(8px); }
    .hint { left:24px; bottom:24px; padding:11px 14px; color:#d7e3f7; font-size:13px; max-width:360px; }
    .hover-card { display:none; pointer-events:none; z-index:20; padding:12px; max-width:320px; font-size:13px; }
    .detail-card { background:rgba(16,24,40,.84); border:1px solid var(--border); border-radius:22px; padding:18px; margin-bottom:14px; }
    .detail-title { font-size:22px; line-height:1.2; font-weight:800; margin:0 0 10px; }
    .abstract { max-height:320px; overflow:auto; padding-right:4px; line-height:1.55; }
    a { color:#8bddff; text-decoration:none; } a:hover { text-decoration:underline; }
    @media (max-width:1380px) { .layout { grid-template-columns:320px minmax(0,1fr) 360px; } }
    @media (max-width:1180px) { .layout { grid-template-columns:1fr; } .panel,.stage-shell { min-height:auto; } .atlas-canvas { height:64vh; min-height:480px; } }
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"panel\"><div class=\"panel-inner\">
      <h1 class=\"brand\"><span class=\"mark\">FMAP</span></h1>
      <div class=\"subtle\">Astro-ph only. Search the map, filter by category, hover for previews, click for details, and explore nearby papers through embedding similarity.</div>
      <div class=\"subtle\" style=\"margin-top:10px;\">by mariadjuric · <a href=\"https://github.com/mariadjuric/FindMyArxivPaper\" target=\"_blank\" rel=\"noreferrer\">GitHub repo</a></div>
      <div class=\"stats\"><div class=\"stat\"><div class=\"label\">Papers</div><div class=\"value\" id=\"paperCount\">0</div></div><div class=\"stat\"><div class=\"label\">Categories</div><div class=\"value\" id=\"categoryCount\">0</div></div></div>
      <div class=\"subtle\" id=\"projectionSummary\">Projection: loading…</div>
      <div class=\"subtle\" id=\"lastUpdatedSummary\" style=\"margin-top:6px;\">Last updated: loading…</div>
      <div class=\"controls\"><input id=\"search\" placeholder=\"Search titles, abstracts, authors, categories\" /><div class=\"toolbar\"><button id=\"clearBtn\" class=\"secondary\">Clear</button><button id=\"focusBtn\">Focus matches</button><button id=\"resetViewBtn\" class=\"secondary\">Reset view</button></div></div>
      <div class=\"section-title\">Category filter</div><div id=\"legend\" class=\"legend\"></div>
      <div class=\"section-title\">Result list</div><div class=\"muted\" id=\"resultSummary\">Loading papers…</div><div id=\"results\" class=\"scroll-block\"></div>
    </div></aside>
    <main class=\"stage-shell\">
      <div class=\"stage-head\"><div>Interactive 2D UMAP projection of title+abstract embeddings</div><div class=\"stage-toolbar\"><button id=\"zoomInBtn\">Zoom in</button><button id=\"zoomOutBtn\" class=\"secondary\">Zoom out</button></div></div>
      <div class=\"canvas-wrap\">
        <canvas id=\"atlasBase\" class=\"atlas-canvas\"></canvas>
        <canvas id=\"atlasOverlay\" class=\"atlas-canvas\"></canvas>
        <div class=\"hover-card\" id=\"hoverCard\"></div>
        <div class=\"hint\">Drag to pan · wheel to zoom · shift-click a category chip to isolate it · click any point for full details. Performance mode is enabled for larger maps.</div>
      </div>
    </main>
    <aside class=\"panel\"><div class=\"panel-inner\"><div id=\"details\" class=\"detail-card\"><div class=\"detail-title\">Select a paper</div><div class=\"muted\">Use the map or result list. Hover gives a quick preview; click locks the paper and loads related recommendations.</div></div><div class=\"section-title\">Related papers</div><div id=\"recs\" class=\"scroll-block\"></div></div></aside>
  </div>
  <script src=\"data.js\"></script>
  <script>
    const payload = window.FMAP_DATA || { points: [], colors: {}, counts: {}, stats: { papers: 0, categories: 0 } };
    const points = payload.points || [];
    const byId = new Map(points.map(p => [p.id, p]));
    const baseCanvas = document.getElementById('atlasBase');
    const overlayCanvas = document.getElementById('atlasOverlay');
    const baseCtx = baseCanvas.getContext('2d', { alpha: true });
    const overlayCtx = overlayCanvas.getContext('2d', { alpha: true });
    const searchInput = document.getElementById('search');
    const clearBtn = document.getElementById('clearBtn');
    const focusBtn = document.getElementById('focusBtn');
    const resetViewBtn = document.getElementById('resetViewBtn');
    const zoomInBtn = document.getElementById('zoomInBtn');
    const zoomOutBtn = document.getElementById('zoomOutBtn');
    const details = document.getElementById('details');
    const legend = document.getElementById('legend');
    const resultsEl = document.getElementById('results');
    const recsEl = document.getElementById('recs');
    const resultSummary = document.getElementById('resultSummary');
    const hoverCard = document.getElementById('hoverCard');
    const paperCount = document.getElementById('paperCount');
    const categoryCount = document.getElementById('categoryCount');
    const projectionSummary = document.getElementById('projectionSummary');
    const lastUpdatedSummary = document.getElementById('lastUpdatedSummary');

    let selected = null, hovered = null, hoverNeighbors = [], filtered = points.slice();
    let focusMatches = false, activeCategories = new Set(Object.keys(payload.colors));
    let panX = 0, panY = 0, zoom = 1, dragging = false, lastPointer = null;
    let pointPositions = [], interactivePositions = [], grid = new Map();
    let animationQueued = false, needsBaseRedraw = true, lastHoverId = null;
    const GRID_SIZE = 28;
    const HIT_RADIUS = 16;

    paperCount.textContent = String(payload.stats?.papers || points.length || 0);
    categoryCount.textContent = String(payload.stats?.categories || Object.keys(payload.colors).length || 0);
    projectionSummary.textContent = payload.stats?.projection ? `Projection: ${payload.stats.projection} · neighbors ${payload.stats.umap_neighbors} · min_dist ${payload.stats.umap_min_dist}` : 'Projection metadata unavailable';
    lastUpdatedSummary.textContent = payload.stats?.last_updated ? `Last updated: ${payload.stats.last_updated}` : 'Last updated: unavailable';

    function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
    function canvasRect() { return baseCanvas.getBoundingClientRect(); }
    function canvasWidth() { return baseCanvas.clientWidth; }
    function canvasHeight() { return baseCanvas.clientHeight; }

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      for (const canvas of [baseCanvas, overlayCanvas]) {
        canvas.width = canvas.clientWidth * ratio;
        canvas.height = canvas.clientHeight * ratio;
      }
      baseCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
      overlayCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
      recomputePositions();
      requestDraw(true);
    }

    function transformPoint(baseX, baseY, isMatch) {
      const cx = canvasWidth() / 2, cy = canvasHeight() / 2;
      let x = baseX, y = baseY;
      if (focusMatches && isMatch) {
        x = cx + (x - cx) * 0.82;
        y = cy + (y - cy) * 0.82;
      }
      x = cx + (x - cx) * zoom + panX;
      y = cy + (y - cy) * zoom + panY;
      return { x, y };
    }

    function recomputePositions() {
      const pad = 28;
      const w = Math.max(1, canvasWidth() - pad * 2);
      const h = Math.max(1, canvasHeight() - pad * 2);
      pointPositions = new Array(points.length);
      interactivePositions = [];
      grid = new Map();
      for (const p of points) {
        const isMatch = filtered.includes(p);
        const baseX = pad + p.x * w;
        const baseY = pad + (1 - p.y) * h;
        const pos = transformPoint(baseX, baseY, isMatch);
        pointPositions[p.id] = pos;
        if (!isMatch) continue;
        interactivePositions.push(p.id);
        const gx = Math.floor(pos.x / GRID_SIZE), gy = Math.floor(pos.y / GRID_SIZE);
        const key = `${gx},${gy}`;
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(p.id);
      }
    }

    function drawField(ctx) {
      ctx.clearRect(0, 0, canvasWidth(), canvasHeight());
      ctx.fillStyle = '#030611'; ctx.fillRect(0, 0, canvasWidth(), canvasHeight());
      const grad = ctx.createRadialGradient(canvasWidth() * 0.5, canvasHeight() * 0.45, 40, canvasWidth() * 0.5, canvasHeight() * 0.45, canvasWidth() * 0.7);
      grad.addColorStop(0, 'rgba(125, 211, 252, 0.05)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad; ctx.fillRect(0, 0, canvasWidth(), canvasHeight());
      ctx.strokeStyle = 'rgba(128, 154, 192, 0.05)'; ctx.lineWidth = 1;
      for (let i = 1; i < 6; i++) {
        const x = (canvasWidth() / 6) * i, y = (canvasHeight() / 6) * i;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvasHeight()); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasWidth(), y); ctx.stroke();
      }
    }

    function drawBase() {
      drawField(baseCtx);
      const neighborSet = new Set(hoverNeighbors.map(p => p.id));
      const filteredSet = new Set(filtered.map(p => p.id));
      for (const p of points) {
        const pos = pointPositions[p.id];
        if (!pos) continue;
        const active = filteredSet.has(p.id);
        const isNeighbor = neighborSet.has(p.id);
        const radius = isNeighbor ? clamp(1.75 * zoom, 2.0, 4.2) : active ? clamp(0.95 * zoom, 1.05, 2.3) : clamp(0.42 * zoom, 0.33, 0.75);
        baseCtx.beginPath();
        baseCtx.fillStyle = isNeighbor ? '#d8f3ff' : active ? p.color : 'rgba(100, 116, 139, 0.10)';
        baseCtx.arc(pos.x, pos.y, radius, 0, Math.PI * 2); baseCtx.fill();
      }
    }

    function drawOverlay() {
      overlayCtx.clearRect(0, 0, canvasWidth(), canvasHeight());
      if (hovered && hoverNeighbors.length) {
        const origin = pointPositions[hovered.id];
        hoverNeighbors.forEach((neighbor, index) => {
          const target = pointPositions[neighbor.id];
          if (!origin || !target) return;
          overlayCtx.beginPath();
          overlayCtx.strokeStyle = `rgba(125, 211, 252, ${Math.max(0.18, 0.42 - index * 0.04)})`;
          overlayCtx.lineWidth = Math.max(1.8, 3.4 - index * 0.2);
          overlayCtx.moveTo(origin.x, origin.y); overlayCtx.lineTo(target.x, target.y); overlayCtx.stroke();
        });
      }
      if (hovered) {
        const pos = pointPositions[hovered.id];
        if (pos) { overlayCtx.beginPath(); overlayCtx.strokeStyle = 'rgba(125, 211, 252, 0.95)'; overlayCtx.lineWidth = 2.2; overlayCtx.arc(pos.x, pos.y, clamp(8 * zoom, 7, 14), 0, Math.PI * 2); overlayCtx.stroke(); }
      }
      if (selected) {
        const pos = pointPositions[selected.id];
        if (pos) { overlayCtx.beginPath(); overlayCtx.strokeStyle = '#ffffff'; overlayCtx.lineWidth = 2.6; overlayCtx.arc(pos.x, pos.y, clamp(10 * zoom, 9, 16), 0, Math.PI * 2); overlayCtx.stroke(); }
      }
    }

    function requestDraw(forceBase = false) {
      if (forceBase) needsBaseRedraw = true;
      if (animationQueued) return;
      animationQueued = true;
      requestAnimationFrame(() => {
        animationQueued = false;
        if (needsBaseRedraw) { drawBase(); needsBaseRedraw = false; }
        drawOverlay();
      });
    }

    function updateLegend() {
      legend.innerHTML = '';
      Object.entries(payload.colors).forEach(([label, color]) => {
        const count = payload.counts?.[label] || 0;
        const item = document.createElement('button');
        item.className = 'legend-item' + (activeCategories.has(label) ? ' active' : '');
        item.innerHTML = `<span class=\"swatch\" style=\"background:${color}\"></span><span>${label}</span><span class=\"muted\">${count}</span>`;
        item.addEventListener('click', (event) => {
          if (event.shiftKey) activeCategories = new Set([label]);
          else if (activeCategories.has(label)) { activeCategories.delete(label); if (!activeCategories.size) activeCategories = new Set(Object.keys(payload.colors)); }
          else activeCategories.add(label);
          updateLegend(); updateFilter();
        });
        legend.appendChild(item);
      });
    }

    function renderHover(p, x, y) {
      if (!p) { hoverNeighbors = []; hoverCard.style.display = 'none'; return; }
      hoverNeighbors = (p.recommendations || []).map(rec => byId.get(rec.id)).filter(Boolean).slice(0, 6);
      hoverCard.style.display = 'block';
      hoverCard.style.left = Math.min(x + 18, canvasWidth() - 320) + 'px';
      hoverCard.style.top = Math.min(y + 18, canvasHeight() - 170) + 'px';
      hoverCard.innerHTML = `<div class=\"pill\"><span class=\"swatch\" style=\"background:${p.color}\"></span>${p.category}</div><div class=\"title-sm\">${p.title}</div><div class=\"meta\">${p.published ? p.published.slice(0,10) : 'Unknown date'}${p.authors ? ' · ' + p.authors : ''}</div><div class=\"meta\">Vector neighborhood: ${hoverNeighbors.length} related papers highlighted</div>`;
    }

    function showDetails(p) {
      selected = p;
      details.innerHTML = `<div class=\"detail-title\">${p.title}</div><div class=\"pill\"><span class=\"swatch\" style=\"background:${p.color}\"></span>${p.category}</div><div class=\"meta\">${p.published ? p.published.slice(0,10) : 'Unknown date'}${p.authors ? ' · ' + p.authors : ''}</div><div class=\"abstract\">${p.abstract || 'No abstract available.'}</div><div class=\"meta\" style=\"margin-top:12px;\">${p.url ? `<a href=\"${p.url}\" target=\"_blank\" rel=\"noreferrer\">Open on arXiv</a>` : 'No URL available'}</div>`;
      renderRecommendations(p);
      needsBaseRedraw = true; requestDraw();
    }

    function renderRecommendations(p) {
      recsEl.innerHTML = '';
      const recs = p?.recommendations || [];
      if (!recs.length) { recsEl.innerHTML = '<div class=\"muted\">No related papers available yet.</div>'; return; }
      recs.forEach(rec => {
        const linked = byId.get(rec.id); const item = document.createElement('div'); item.className = 'rec-item';
        item.innerHTML = `<div class=\"pill\">${rec.category}</div><div class=\"title-sm\">${rec.title}</div><div class=\"meta\">${rec.authors || 'Unknown authors'}</div><div class=\"meta\">${rec.published ? rec.published.slice(0,10) : 'Unknown date'} · <strong>${rec.match}% match</strong></div>`;
        item.addEventListener('click', () => linked && showDetails(linked)); recsEl.appendChild(item);
      });
    }

    function nearestPoint(mx, my) {
      const gx = Math.floor(mx / GRID_SIZE), gy = Math.floor(my / GRID_SIZE);
      let best = null, bestDist = Infinity;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const bucket = grid.get(`${gx + dx},${gy + dy}`) || [];
          for (const id of bucket) {
            const pos = pointPositions[id];
            const d = Math.hypot(mx - pos.x, my - pos.y);
            if (d < bestDist) { bestDist = d; best = byId.get(id); }
          }
        }
      }
      return bestDist < HIT_RADIUS ? best : null;
    }

    function scoreMatch(p, q) {
      if (!q) return true;
      const terms = q.split(/\\s+/).filter(Boolean);
      const hay = [p.title, p.abstract, p.authors, p.category].join(' ').toLowerCase();
      return terms.every(term => hay.includes(term));
    }

    function updateFilter() {
      const q = searchInput.value.trim().toLowerCase();
      filtered = points.filter(p => activeCategories.has(p.category) && scoreMatch(p, q));
      if (selected && !filtered.includes(selected)) selected = filtered[0] || null;
      resultSummary.textContent = q || activeCategories.size !== Object.keys(payload.colors).length ? `${filtered.length} papers match the current filters.` : `Showing all ${points.length} astro-ph papers.`;
      recomputePositions(); renderResults();
      if (selected) showDetails(selected); else { recsEl.innerHTML = ''; needsBaseRedraw = true; requestDraw(); }
    }

    function renderResults() {
      resultsEl.innerHTML = '';
      filtered.slice(0, 40).forEach(p => {
        const item = document.createElement('div'); item.className = 'result-item';
        item.innerHTML = `<div class=\"pill\"><span class=\"swatch\" style=\"background:${p.color}\"></span>${p.category}</div><div class=\"title-sm\">${p.title}</div><div class=\"meta\">${p.authors || 'Unknown authors'}</div>`;
        item.addEventListener('mouseenter', () => { hovered = p; lastHoverId = p.id; renderHover(p, 30, 30); needsBaseRedraw = true; requestDraw(); });
        item.addEventListener('mouseleave', () => { hovered = null; hoverNeighbors = []; lastHoverId = null; renderHover(null); needsBaseRedraw = true; requestDraw(); });
        item.addEventListener('click', () => showDetails(p)); resultsEl.appendChild(item);
      });
      if (!filtered.length) resultsEl.innerHTML = '<div class=\"muted\">No papers match this search/filter combo.</div>';
    }

    function resetView() { panX = 0; panY = 0; zoom = 1; recomputePositions(); needsBaseRedraw = true; requestDraw(); }

    searchInput.addEventListener('input', updateFilter);
    clearBtn.addEventListener('click', () => { searchInput.value = ''; activeCategories = new Set(Object.keys(payload.colors)); updateLegend(); updateFilter(); });
    focusBtn.addEventListener('click', () => { focusMatches = !focusMatches; focusBtn.classList.toggle('active', focusMatches); recomputePositions(); needsBaseRedraw = true; requestDraw(); });
    resetViewBtn.addEventListener('click', resetView);
    zoomInBtn.addEventListener('click', () => { zoom = clamp(zoom * 1.18, 0.7, 6); recomputePositions(); needsBaseRedraw = true; requestDraw(); });
    zoomOutBtn.addEventListener('click', () => { zoom = clamp(zoom / 1.18, 0.7, 6); recomputePositions(); needsBaseRedraw = true; requestDraw(); });

    baseCanvas.addEventListener('wheel', (event) => { event.preventDefault(); const factor = event.deltaY > 0 ? 0.92 : 1.09; zoom = clamp(zoom * factor, 0.7, 6); recomputePositions(); needsBaseRedraw = true; requestDraw(); }, { passive: false });
    baseCanvas.addEventListener('mousedown', (event) => { dragging = true; lastPointer = { x: event.clientX, y: event.clientY }; });
    window.addEventListener('mouseup', () => { dragging = false; lastPointer = null; });

    let pendingPointer = null;
    function processPointerMove() {
      pendingPointer = null;
      const event = window.__fmap_last_move;
      if (!event) return;
      const rect = canvasRect();
      const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (dragging && lastPointer) {
        panX += event.clientX - lastPointer.x; panY += event.clientY - lastPointer.y; lastPointer = { x: event.clientX, y: event.clientY };
        recomputePositions(); needsBaseRedraw = true; requestDraw(); return;
      }
      if (!inside) {
        if (hovered || hoverNeighbors.length) { hovered = null; hoverNeighbors = []; lastHoverId = null; renderHover(null); needsBaseRedraw = true; requestDraw(); }
        return;
      }
      const mx = event.clientX - rect.left, my = event.clientY - rect.top;
      const next = nearestPoint(mx, my);
      const nextId = next ? next.id : null;
      if (nextId === lastHoverId) {
        if (next) renderHover(next, mx, my);
        return;
      }
      hovered = next; lastHoverId = nextId;
      renderHover(next, mx, my); needsBaseRedraw = true; requestDraw();
    }

    window.addEventListener('mousemove', (event) => {
      window.__fmap_last_move = event;
      if (!pendingPointer) pendingPointer = requestAnimationFrame(processPointerMove);
    });

    baseCanvas.addEventListener('click', (event) => {
      const rect = canvasRect();
      const p = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
      if (p) showDetails(p);
    });

    updateLegend();
    window.addEventListener('resize', resize);
    resize();
    filtered = points.slice();
    renderResults();
    if (points.length) { selected = points[0]; showDetails(selected); } else { requestDraw(true); }
  </script>
</body>
</html>
"""
