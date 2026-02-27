#!/usr/bin/env python3
"""
coq_visualizer.py — Interactive graph visualizer for Coq Analyzer JSON output.

Generates a self-contained HTML page with a force-directed dependency graph
(inspired by JSON Crack's node-graph style).

Usage:
    python coq_visualizer.py coq_deps.json [-o graph.html] [--open]
"""

import json
import sys
import argparse
import webbrowser
import os
from html import escape
from pathlib import Path


def generate_graph_html(data: dict) -> str:
    project = data.get("project", data.get("stats", {}).get("project", "Coq Project"))
    items = data["items"]

    # Normalize key names (analyzer JSON vs HTML export JSON use different keys)
    normalized = []
    for it in items:
        normalized.append({
            "name": it.get("name", ""),
            "qname": it.get("qname", it.get("qualified_name", "")),
            "kind": it.get("kind", ""),
            "status": it.get("status", ""),
            "file": it.get("file", ""),
            "line": it.get("line", 0),
            "deps": it.get("deps", it.get("dependencies", [])),
            "rdeps": it.get("rdeps", it.get("dependents", [])),
            "tainted": it.get("tainted", False),
            "statement": it.get("statement", ""),
        })

    data_json = json.dumps(normalized, indent=None, ensure_ascii=False)
    stats_json = json.dumps(data.get("stats", {}), indent=None, ensure_ascii=False)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Coq Graph — {escape(str(project))}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
  --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --link: #58a6ff;
  --green: #238636; --yellow: #d29922; --blue: #1f6feb;
  --purple: #8957e5; --red: #da3633; --gray: #484f58;
  --green-soft: #1a2e1f; --yellow-soft: #2e2a15; --blue-soft: #151e30;
  --purple-soft: #1e1530; --red-soft: #2d1517; --gray-soft: #1c2028;
}}
html, body {{ width: 100%; height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; }}

/* ── Top bar ── */
#topbar {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  height: 48px; background: var(--card-bg); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 16px; gap: 12px;
}}
#topbar h1 {{ font-size: 14px; font-weight: 600; white-space: nowrap; }}
#topbar h1 span {{ color: var(--link); }}
#search {{
  padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--bg); color: var(--text); font-size: 12px; width: 220px; outline: none;
}}
#search:focus {{ border-color: var(--link); }}
.tb-btn {{
  padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--card-bg); color: var(--text); cursor: pointer; font-size: 11px;
  white-space: nowrap;
}}
.tb-btn:hover {{ background: var(--border); }}
.tb-btn.active {{ background: var(--blue); border-color: var(--blue); }}
#legend {{ display: flex; gap: 8px; margin-left: auto; align-items: center; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.legend-label {{ font-size: 10px; color: var(--muted); margin-right: 4px; }}

/* ── Canvas ── */
#canvas-wrap {{ position: absolute; top: 48px; left: 0; right: 0; bottom: 0; }}
canvas {{ display: block; width: 100%; height: 100%; }}

/* ── Tooltip / detail panel ── */
#detail {{
  position: fixed; top: 60px; right: 12px; width: 320px;
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; display: none; z-index: 200; max-height: calc(100vh - 80px);
  overflow-y: auto; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}}
#detail.visible {{ display: block; }}
#detail h2 {{ font-size: 15px; margin-bottom: 4px; }}
#detail .meta {{ font-size: 11px; color: var(--muted); margin-bottom: 10px; }}
#detail pre {{
  background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px; font-size: 11px; white-space: pre-wrap; word-break: break-all;
  font-family: 'SF Mono', 'Fira Code', monospace; margin-bottom: 10px;
  max-height: 150px; overflow-y: auto;
}}
#detail h3 {{ font-size: 12px; color: var(--muted); margin: 8px 0 4px; }}
#detail .dep-link {{
  display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 4px;
  background: var(--bg); border: 1px solid var(--border); cursor: pointer;
  font-size: 11px; color: var(--link); text-decoration: none;
}}
#detail .dep-link:hover {{ border-color: var(--link); }}
#close-detail {{
  position: absolute; top: 10px; right: 12px; background: none; border: none;
  color: var(--muted); cursor: pointer; font-size: 16px;
}}

/* ── Minimap ── */
#minimap {{
  position: fixed; bottom: 12px; right: 12px; width: 180px; height: 120px;
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  z-index: 150; overflow: hidden;
}}
#minimap canvas {{ width: 100%; height: 100%; }}

/* ── Controls hint ── */
#controls-hint {{
  position: fixed; bottom: 12px; left: 12px; font-size: 10px; color: var(--muted);
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 6px 10px; z-index: 150;
}}
</style>
</head>
<body>

<div id="topbar">
  <h1><span>◆</span> {escape(str(project))}</h1>
  <input id="search" type="text" placeholder="Search nodes... (/)" autocomplete="off">
  <button class="tb-btn" id="btn-deps">Show deps on hover</button>
  <button class="tb-btn" id="btn-labels">Labels</button>
  <button class="tb-btn" id="btn-fit">Fit</button>
  <button class="tb-btn" id="btn-pause">Pause</button>
  <button class="tb-btn" id="btn-grid">Grid</button>
  <button class="tb-btn" id="btn-spread">Spread</button>
  <button class="tb-btn" id="btn-relax">Relax</button>
  <div id="legend">
    <span class="legend-dot" style="background:var(--green)"></span><span class="legend-label">proved</span>
    <span class="legend-dot" style="background:var(--yellow)"></span><span class="legend-label">admitted</span>
    <span class="legend-dot" style="background:var(--blue)"></span><span class="legend-label">defined</span>
    <span class="legend-dot" style="background:var(--purple)"></span><span class="legend-label">axiom</span>
    <span class="legend-dot" style="background:var(--red)"></span><span class="legend-label">tainted</span>
  </div>
</div>

<div id="canvas-wrap"><canvas id="graph"></canvas></div>
<div id="minimap"><canvas id="mini"></canvas></div>

<div id="detail">
  <button id="close-detail">&times;</button>
  <div id="detail-content"></div>
</div>

<div id="controls-hint">
  Scroll: zoom · Drag: pan · Click: select · Double-click: focus · /: search · Esc: clear
</div>

<script>
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  DATA
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
const ITEMS = {data_json};
const STATS = {stats_json};

const nodeMap = {{}};
ITEMS.forEach((it, i) => {{
  it._i = i;
  nodeMap[it.qname] = it;
}});

const NODE_W = 140, NODE_H = 38, NODE_R = 8;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  GRAPH LAYOUT — force-directed simulation
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
const nodes = ITEMS.map((it, i) => ({{
  x: (Math.random() - 0.5) * Math.min(ITEMS.length * 8, 2000),
  y: (Math.random() - 0.5) * Math.min(ITEMS.length * 8, 2000),
  vx: 0, vy: 0,
  item: it, idx: i,
}}));

const edges = [];
ITEMS.forEach((it, i) => {{
  it.deps.forEach(dq => {{
    const target = nodeMap[dq];
    if (target) edges.push({{ source: i, target: target._i }});
  }});
}});

// Spatial grid for O(n) repulsion approximation
const SIM = {{
  running: true,
  alpha: 1.0,
  alphaDecay: 0.0015,
  alphaMin: 0.002,
  repulsion: -280,
  linkDist: 90,
  linkStrength: 0.08,
  centerStrength: 0.005,
  damping: 0.45,
  maxVelocity: 40,
}};

// Separate isolated nodes (no edges at all) from the graph
const connectedSet = new Set();
edges.forEach(e => {{ connectedSet.add(e.source); connectedSet.add(e.target); }});
const graphNodes = nodes.filter(n => connectedSet.has(n.idx));
const islandNodes = nodes.filter(n => !connectedSet.has(n.idx));


// Lay out islands in a grid strip along the bottom
(function layoutIslands() {{
  if (islandNodes.length === 0) return;
  // Find the extent of graph nodes for positioning
  const cols = Math.ceil(Math.sqrt(islandNodes.length * 3));
  const spacing = NODE_W * 1.3;
  const startX = -(cols * spacing) / 2;
  let graphMaxY = 0;
  graphNodes.forEach(n => {{ if (n.y > graphMaxY) graphMaxY = n.y; }});
  const startY = graphMaxY + 300;

  islandNodes.forEach((n, i) => {{
    const col = i % cols;
    const row = Math.floor(i / cols);
    n.x = startX + col * spacing;
    n.y = startY + row * (NODE_H * 2.2);
    n.vx = 0; n.vy = 0;
    n._island = true;
  }});
}})();


function simulate() {{
  if (!SIM.running || SIM.alpha < SIM.alphaMin) return;
  const alpha = SIM.alpha;

  // Only simulate connected nodes
  const active = graphNodes;
  const N = active.length;
  if (N === 0) {{ SIM.alpha *= (1 - SIM.alphaDecay); return; }}

  // Compute centroid (attract to centroid, not origin — prevents orbit)
  let cx = 0, cy = 0;
  active.forEach(n => {{ cx += n.x; cy += n.y; }});
  cx /= N; cy /= N;

  active.forEach(n => {{
    n.vx += (cx - n.x) * SIM.centerStrength * alpha;
    n.vy += (cy - n.y) * SIM.centerStrength * alpha;
  }});

  // Repulsion via spatial grid
  const cellSize = 120;
  const grid = {{}};
  active.forEach(n => {{
    const gx = Math.floor(n.x / cellSize);
    const gy = Math.floor(n.y / cellSize);
    const key = gx + ',' + gy;
    if (!grid[key]) grid[key] = [];
    grid[key].push(n);
  }});

  active.forEach(n => {{
    const gx = Math.floor(n.x / cellSize);
    const gy = Math.floor(n.y / cellSize);
    for (let dx = -2; dx <= 2; dx++) {{
      for (let dy = -2; dy <= 2; dy++) {{
        const nbrs = grid[(gx+dx)+','+(gy+dy)];
        if (!nbrs) continue;
        for (const m of nbrs) {{
          if (m.idx <= n.idx) continue;
          let ex = n.x - m.x, ey = n.y - m.y;
          let d2 = ex*ex + ey*ey;
          if (d2 < 1) {{ ex = (Math.random()-0.5)*2; ey = (Math.random()-0.5)*2; d2 = 1; }}
          const f = SIM.repulsion * alpha / d2;
          n.vx += ex * f; n.vy += ey * f;
          m.vx -= ex * f; m.vy -= ey * f;
        }}
      }}
    }}
  }});

  // Soft collision: prevents "black hole cloud"
  applyMinSeparation(active, grid, cellSize, /*minDist=*/60, /*strength=*/0.11, alpha);

  // Link attraction
  edges.forEach(e => {{
    const s = nodes[e.source], t = nodes[e.target];
    let dx = t.x - s.x, dy = t.y - s.y;
    let dist = Math.sqrt(dx*dx + dy*dy) || 1;
    
    let f = (dist - SIM.linkDist) * SIM.linkStrength * alpha;

    // Clamp spring force to avoid singular collapse
    const maxLinkForce = 0.75;
    if (f >  maxLinkForce) f =  maxLinkForce;
    if (f < -maxLinkForce) f = -maxLinkForce;
    
    s.vx += (dx/dist)*f; s.vy += (dy/dist)*f;
    t.vx -= (dx/dist)*f; t.vy -= (dy/dist)*f;
  }});

  // Integrate with velocity clamping
  let totalV = 0;
  active.forEach(n => {{
    n.vx *= SIM.damping;
    n.vy *= SIM.damping;
    // Clamp velocity
    const v = Math.sqrt(n.vx*n.vx + n.vy*n.vy);
    if (v > SIM.maxVelocity) {{
      n.vx = (n.vx / v) * SIM.maxVelocity;
      n.vy = (n.vy / v) * SIM.maxVelocity;
    }}
    n.x += n.vx;
    n.y += n.vy;
    totalV += v;
  }});

  // Auto-stop when kinetic energy is negligible
  if (totalV / N < 0.15) {{
    SIM.alpha = 0;
    SIM.running = false;
    document.getElementById('btn-pause').textContent = 'Resume';
    document.getElementById('btn-pause').classList.add('active');
  }}

  if (SIM.relaxTicks && SIM.relaxTicks > 0) {{
    SIM.relaxTicks--;
    if (SIM.relaxTicks === 0) endRelaxMode();
  }}

  SIM.alpha *= (1 - SIM.alphaDecay);
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  CAMERA (pan + zoom)
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
const cam = {{ x: 0, y: 0, zoom: 1 }};

function worldToScreen(wx, wy, c) {{
  c = c || cam;
  const W = canvas.width / devicePixelRatio;
  const H = canvas.height / devicePixelRatio;
  return [
    (wx - c.x) * c.zoom + W/2,
    (wy - c.y) * c.zoom + H/2
  ];
}}
function screenToWorld(sx, sy) {{
  const W = canvas.width / devicePixelRatio;
  const H = canvas.height / devicePixelRatio;
  return [
    (sx - W/2) / cam.zoom + cam.x,
    (sy - H/2) / cam.zoom + cam.y
  ];
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  COLORS
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
const STATUS_COLORS = {{
  proved:  {{ fill: '#1a2e1f', border: '#238636', text: '#3fb950' }},
  admitted: {{ fill: '#2e2a15', border: '#d29922', text: '#e3b341' }},
  defined: {{ fill: '#151e30', border: '#1f6feb', text: '#58a6ff' }},
  assumed: {{ fill: '#1e1530', border: '#8957e5', text: '#bc8cff' }},
  aborted: {{ fill: '#2d1517', border: '#da3633', text: '#f85149' }},
}};
const DEFAULT_COLOR = {{ fill: '#1c2028', border: '#484f58', text: '#8b949e' }};

function getColor(item) {{ return STATUS_COLORS[item.status] || DEFAULT_COLOR; }}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  RENDERING
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
const miniCanvas = document.getElementById('mini');
const miniCtx = miniCanvas.getContext('2d');

let showLabels = true;
let highlightDeps = true;
let hoverNode = null;
let selectedNode = null;
let searchMatches = new Set();

function resize() {{
  const wrap = document.getElementById('canvas-wrap');
  canvas.width = wrap.clientWidth * devicePixelRatio;
  canvas.height = wrap.clientHeight * devicePixelRatio;
  canvas.style.width = wrap.clientWidth + 'px';
  canvas.style.height = wrap.clientHeight + 'px';
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

  miniCanvas.width = 180 * devicePixelRatio;
  miniCanvas.height = 120 * devicePixelRatio;
  miniCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}}
window.addEventListener('resize', resize);
resize();

function roundRect(c, x, y, w, h, r) {{
  c.beginPath();
  c.moveTo(x+r, y);
  c.lineTo(x+w-r, y); c.quadraticCurveTo(x+w, y, x+w, y+r);
  c.lineTo(x+w, y+h-r); c.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
  c.lineTo(x+r, y+h); c.quadraticCurveTo(x, y+h, x, y+h-r);
  c.lineTo(x, y+r); c.quadraticCurveTo(x, y, x+r, y);
  c.closePath();
}}

function drawArrow(c, x1, y1, x2, y2, color, alpha) {{
  // Shorten to stop at node border
  const dx = x2-x1, dy = y2-y1;
  const dist = Math.sqrt(dx*dx+dy*dy) || 1;
  const ux = dx/dist, uy = dy/dist;
  const ax = x2 - ux*NODE_W*0.52, ay = y2 - uy*NODE_H*0.55;

  c.save();
  c.globalAlpha = alpha;
  c.strokeStyle = color;
  c.lineWidth = 1.2;
  c.beginPath();
  c.moveTo(x1, y1);
  c.lineTo(ax, ay);
  c.stroke();

  // Arrowhead
  const headLen = 7;
  const angle = Math.atan2(ay-y1, ax-x1);
  c.fillStyle = color;
  c.beginPath();
  c.moveTo(ax, ay);
  c.lineTo(ax - headLen*Math.cos(angle-0.35), ay - headLen*Math.sin(angle-0.35));
  c.lineTo(ax - headLen*Math.cos(angle+0.35), ay - headLen*Math.sin(angle+0.35));
  c.closePath();
  c.fill();
  c.restore();
}}

function getHighlightSet() {{
  const focus = hoverNode || selectedNode;
  if (!focus || !highlightDeps) return null;
  const s = new Set();
  s.add(focus.idx);
  focus.item.deps.forEach(d => {{ const n = nodeMap[d]; if(n) s.add(n._i); }});
  focus.item.rdeps.forEach(d => {{ const n = nodeMap[d]; if(n) s.add(n._i); }});
  return s;
}}

function getHighlightEdges() {{
  const focus = hoverNode || selectedNode;
  if (!focus || !highlightDeps) return null;
  const s = new Set();
  edges.forEach((e, i) => {{
    if (e.source === focus.idx || e.target === focus.idx) s.add(i);
  }});
  return s;
}}

function render() {{
  const W = canvas.width / devicePixelRatio;
  const H = canvas.height / devicePixelRatio;
  ctx.clearRect(0, 0, W, H);

  const hl = getHighlightSet();
  const hlEdges = getHighlightEdges();
  const z = cam.zoom;
  const showText = z > 0.25 && showLabels;
  const showNodes = true;

  // Draw edges
  edges.forEach((e, ei) => {{
    const s = nodes[e.source], t = nodes[e.target];
    const [sx, sy] = worldToScreen(s.x, s.y);
    const [tx, ty] = worldToScreen(t.x, t.y);

    // Cull off-screen
    if (Math.max(sx,tx) < -50 || Math.min(sx,tx) > W+50) return;
    if (Math.max(sy,ty) < -50 || Math.min(sy,ty) > H+50) return;

    const isHl = hlEdges && hlEdges.has(ei);
    const dimmed = hl && !isHl;

    const color = isHl
      ? (e.source === (hoverNode||selectedNode).idx ? '#58a6ff' : '#3fb950')
      : '#21262d';
    drawArrow(ctx, sx, sy, tx, ty, color, dimmed ? 0.08 : (isHl ? 0.9 : 0.25));
  }});

  // Draw nodes
  nodes.forEach(n => {{
    const [sx, sy] = worldToScreen(n.x, n.y);
    const hw = NODE_W*z/2, hh = NODE_H*z/2;

    if (sx + hw < -10 || sx - hw > W+10 || sy + hh < -10 || sy - hh > H+10) return;

    const c = getColor(n.item);
    const dimmed = hl && !hl.has(n.idx);
    const isSearch = searchMatches.size > 0 && searchMatches.has(n.idx);

    ctx.save();
    ctx.globalAlpha = dimmed ? 0.12 : 1;

    // Shadow for selected/hovered
    if ((n === selectedNode || n === hoverNode) && !dimmed) {{
      ctx.shadowColor = c.border;
      ctx.shadowBlur = 14 * z;
    }}

    const nw = NODE_W * z, nh = NODE_H * z;
    roundRect(ctx, sx - nw/2, sy - nh/2, nw, nh, NODE_R * z);
    ctx.fillStyle = c.fill;
    ctx.fill();
    ctx.strokeStyle = n.item.tainted ? '#da3633' : c.border;
    ctx.lineWidth = (n === selectedNode || n === hoverNode) ? 2.2*z : 1.2*z;
    ctx.stroke();

    // Search highlight ring
    if (isSearch && !dimmed) {{
      ctx.strokeStyle = '#f0883e';
      ctx.lineWidth = 2.5 * z;
      roundRect(ctx, sx - nw/2 - 3*z, sy - nh/2 - 3*z, nw+6*z, nh+6*z, (NODE_R+2)*z);
      ctx.stroke();
    }}

    ctx.shadowBlur = 0;

    // Tainted double border
    if (n.item.tainted && !dimmed) {{
      roundRect(ctx, sx - nw/2 + 2*z, sy - nh/2 + 2*z, nw - 4*z, nh - 4*z, (NODE_R-1)*z);
      ctx.strokeStyle = 'rgba(218,54,51,0.3)';
      ctx.lineWidth = 1*z;
      ctx.stroke();
    }}

    // Label
    if (showText && z > 0.15) {{
      const fontSize = Math.max(8, Math.min(12 * z, 14));
      ctx.font = `600 ${{fontSize}}px -apple-system, sans-serif`;
      ctx.fillStyle = c.text;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const label = n.item.name.length > 18 ? n.item.name.slice(0,16)+'…' : n.item.name;
      ctx.fillText(label, sx, sy - 2*z);

      ctx.font = `${{Math.max(6, 9*z)}}px -apple-system, sans-serif`;
      ctx.fillStyle = dimmed ? 'rgba(139,148,158,0.3)' : '#8b949e';
      ctx.fillText(n.item.kind, sx, sy + 10*z);
    }}

    ctx.restore();
  }});

  renderMinimap();
}}

function renderMinimap() {{
  const mw = 180, mh = 120;
  miniCtx.clearRect(0, 0, mw, mh);
  miniCtx.fillStyle = '#0d1117';
  miniCtx.fillRect(0, 0, mw, mh);

  if (nodes.length === 0) return;

  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;
  nodes.forEach(n => {{
    if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
  }});
  const pad = 50;
  minX -= pad; maxX += pad; minY -= pad; maxY += pad;
  const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
  const scale = Math.min(mw / rangeX, mh / rangeY);

  nodes.forEach(n => {{
    const mx = (n.x - minX) * scale;
    const my = (n.y - minY) * scale;
    const c = getColor(n.item);
    miniCtx.fillStyle = n._island ? '#484f58' : c.border;
    miniCtx.fillRect(mx-1, my-1, 2.5, 2.5);
  }});

  // Viewport rectangle
  const W = canvas.width / devicePixelRatio;
  const H = canvas.height / devicePixelRatio;
  const [tlx, tly] = screenToWorld(0, 0);
  const [brx, bry] = screenToWorld(W, H);
  const rx = (tlx - minX) * scale, ry = (tly - minY) * scale;
  const rw = (brx - tlx) * scale, rh = (bry - tly) * scale;
  miniCtx.strokeStyle = '#58a6ff';
  miniCtx.lineWidth = 1;
  miniCtx.strokeRect(rx, ry, rw, rh);
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  INTERACTION
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
let isDragging = false, dragStartX, dragStartY, camStartX, camStartY;
let isDraggingNode = false, dragNode = null;

function hitTest(mx, my) {{
  const [wx, wy] = screenToWorld(mx, my);
  for (let i = nodes.length-1; i >= 0; i--) {{
    const n = nodes[i];
    if (Math.abs(wx - n.x) < NODE_W/2 && Math.abs(wy - n.y) < NODE_H/2) return n;
  }}
  return null;
}}

canvas.addEventListener('mousedown', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const hit = hitTest(mx, my);

  if (hit && e.button === 0) {{
    isDraggingNode = true;
    dragNode = hit;
    canvas.style.cursor = 'grabbing';
  }} else {{
    isDragging = true;
    dragStartX = e.clientX; dragStartY = e.clientY;
    camStartX = cam.x; camStartY = cam.y;
    canvas.style.cursor = 'grabbing';
  }}
}});

canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;

  if (isDraggingNode && dragNode) {{
    const [wx, wy] = screenToWorld(mx, my);
    dragNode.x = wx; dragNode.y = wy;
    dragNode.vx = 0; dragNode.vy = 0;
    return;
  }}
  if (isDragging) {{
    cam.x = camStartX - (e.clientX - dragStartX) / cam.zoom;
    cam.y = camStartY - (e.clientY - dragStartY) / cam.zoom;
    return;
  }}

  const hit = hitTest(mx, my);
  if (hit !== hoverNode) {{
    hoverNode = hit;
    canvas.style.cursor = hit ? 'pointer' : 'default';
  }}
}});

canvas.addEventListener('mouseup', e => {{
  if (isDraggingNode) {{
    isDraggingNode = false; dragNode = null;
    canvas.style.cursor = 'default';
    return;
  }}
  if (isDragging) {{
    isDragging = false;
    canvas.style.cursor = 'default';
  }}
}});

canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const hit = hitTest(mx, my);
  if (hit) {{
    selectedNode = hit;
    showDetail(hit.item);
  }} else {{
    selectedNode = null;
    document.getElementById('detail').classList.remove('visible');
  }}
}});

canvas.addEventListener('dblclick', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const hit = hitTest(mx, my);
  if (hit) {{
    cam.x = hit.x; cam.y = hit.y;
    cam.zoom = 1.5;
  }}
}});

canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const [wx, wy] = screenToWorld(mx, my);
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  const newZoom = Math.max(0.05, Math.min(8, cam.zoom * factor));
  cam.x = wx - (mx - canvas.width/devicePixelRatio/2) / newZoom;
  cam.y = wy - (my - canvas.height/devicePixelRatio/2) / newZoom;
  cam.zoom = newZoom;
}}, {{ passive: false }});

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  DETAIL PANEL
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function escapeHtml(t) {{ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }}

function showDetail(item) {{
  const el = document.getElementById('detail');
  const c = document.getElementById('detail-content');
  const col = getColor(item);

  const depLinks = item.deps.map(d => {{
    const n = nodeMap[d];
    return `<span class="dep-link" data-q="${{d}}">${{n ? n.name : d}}</span>`;
  }}).join('') || '<span style="color:#8b949e">None</span>';

  const rdepLinks = item.rdeps.map(d => {{
    const n = nodeMap[d];
    return `<span class="dep-link" data-q="${{d}}">${{n ? n.name : d}}</span>`;
  }}).join('') || '<span style="color:#8b949e">None</span>';

  c.innerHTML = `
    <h2 style="color:${{col.text}}">${{escapeHtml(item.name)}}</h2>
    <div class="meta">
      ${{item.kind}} · ${{item.status}}${{item.tainted ? ' · <span style="color:#da3633">⚠ tainted</span>' : ''}}
      <br>${{escapeHtml(item.file)}}:${{item.line}}
    </div>
    ${{item.statement ? `<pre>${{escapeHtml(item.statement)}}</pre>` : ''}}
    <h3>Dependencies (${{item.deps.length}})</h3>
    <div>${{depLinks}}</div>
    <h3>Used by (${{item.rdeps.length}})</h3>
    <div>${{rdepLinks}}</div>
  `;
  el.classList.add('visible');

  c.querySelectorAll('.dep-link').forEach(a => {{
    a.addEventListener('click', () => {{
      const qn = a.dataset.q;
      const target = nodeMap[qn];
      if (!target) return;
      const tn = nodes[target._i];
      selectedNode = tn;
      cam.x = tn.x; cam.y = tn.y;
      showDetail(tn.item);
    }});
  }});
}}

document.getElementById('close-detail').addEventListener('click', () => {{
  document.getElementById('detail').classList.remove('visible');
  selectedNode = null;
}});

function snapToGrid() {{
  // Pause simulation and kill momentum
  SIM.running = false;
  SIM.alpha = 0;

  // Update Pause button UI to match paused state
  const pauseBtn = document.getElementById('btn-pause');
  if (pauseBtn) {{
    pauseBtn.textContent = 'Resume';
    pauseBtn.classList.add('active');
  }}

  // Choose which nodes to grid: usually just the connected graph
  const target = graphNodes.length ? graphNodes : nodes;

  // Sort for stability (so repeated clicks look the same)
  target.sort((a, b) => {{
    // higher degree first; tie-break by qname
    const da = (a.item.deps?.length || 0) + (a.item.rdeps?.length || 0);
    const db = (b.item.deps?.length || 0) + (b.item.rdeps?.length || 0);
    if (db !== da) return db - da;
    return (a.item.qname || '').localeCompare(b.item.qname || '');
  }});

  // Grid geometry (world units)
  const spacingX = NODE_W * 1.35;
  const spacingY = NODE_H * 2.2;
  const cols = Math.max(1, Math.ceil(Math.sqrt(target.length)));

  // Center the grid around current centroid to avoid camera jump
  let cx = 0, cy = 0;
  target.forEach(n => {{ cx += n.x; cy += n.y; }});
  cx /= target.length; cy /= target.length;

  const gridW = (cols - 1) * spacingX;
  const rows = Math.ceil(target.length / cols);
  const gridH = (rows - 1) * spacingY;

  const startX = cx - gridW / 2;
  const startY = cy - gridH / 2;

  target.forEach((n, i) => {{
    const col = i % cols;
    const row = Math.floor(i / cols);
    n.x = startX + col * spacingX;
    n.y = startY + row * spacingY;
    n.vx = 0; n.vy = 0;
  }});
}}

function buildAdjacency(active, edges) {{
  const adj = new Map();
  for (const n of active) adj.set(n.idx, []);
  for (const e of edges) {{
    // edges use global indices into `nodes`
    const a = e.source, b = e.target;
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push(b);
    adj.get(b).push(a);
  }}
  return adj;
}}

// Deterministic-ish BFS "ring expansion" seeding
function seedRingLayout(active, edges, opts = {{}}) {{
  const {{
    baseRadius = 260,
    ringStep = 210,
    jitter = 10,
    preferHighDegree = true,
  }} = opts;

  if (!active || active.length === 0) return;

  const activeSet = new Set(active.map(n => n.idx));
  const adjAll = buildAdjacency(active, edges);

  // pick a root (highest degree within active)
  let root = active[0];
  if (preferHighDegree) {{
    root = active.reduce((best, n) => {{
      const db = (adjAll.get(best.idx)?.filter(x => activeSet.has(x)).length || 0);
      const dn = (adjAll.get(n.idx)?.filter(x => activeSet.has(x)).length || 0);
      return (dn > db) ? n : best;
    }}, active[0]);
  }}

  // center around current centroid (avoid camera jump)
  let cx = 0, cy = 0;
  active.forEach(n => {{ cx += n.x; cy += n.y; }});
  cx /= active.length; cy /= active.length;

  const byIdx = new Map(active.map(n => [n.idx, n]));
  const placed = new Set();

  root.x = cx; root.y = cy;
  root.vx = 0; root.vy = 0;
  placed.add(root.idx);

  const q = [root.idx];
  const layerOf = new Map([[root.idx, 0]]);

  const degree = (id) => (adjAll.get(id)?.filter(x => activeSet.has(x)).length || 0);

  while (q.length) {{
    const u = q.shift();
    const uNode = byIdx.get(u);
    const uLayer = layerOf.get(u) || 0;

    let nbrs = (adjAll.get(u) || [])
      .filter(v => activeSet.has(v) && !placed.has(v));

    if (!nbrs.length) continue;

    nbrs.sort((a, b) => degree(b) - degree(a));

    const radius = baseRadius + uLayer * ringStep;
    const step = (Math.PI * 2) / nbrs.length;

    for (let i = 0; i < nbrs.length; i++) {{
      const v = nbrs[i];
      const vNode = byIdx.get(v);
      if (!vNode) continue;

      const ang = i * step;
      const jx = (Math.random() - 0.5) * jitter;
      const jy = (Math.random() - 0.5) * jitter;

      vNode.x = uNode.x + radius * Math.cos(ang) + jx;
      vNode.y = uNode.y + radius * Math.sin(ang) + jy;
      vNode.vx = 0; vNode.vy = 0;

      placed.add(v);
      layerOf.set(v, uLayer + 1);
      q.push(v);
    }}
  }}

  // If anything unplaced (rare inside active), scatter them around a big circle
  const leftovers = active.filter(n => !placed.has(n.idx));
  if (leftovers.length) {{
    const R = baseRadius + (Math.max(...layerOf.values()) + 2) * ringStep;
    const step2 = (Math.PI * 2) / leftovers.length;
    for (let i = 0; i < leftovers.length; i++) {{
      leftovers[i].x = cx + R * Math.cos(i * step2);
      leftovers[i].y = cy + R * Math.sin(i * step2);
      leftovers[i].vx = 0; leftovers[i].vy = 0;
    }}
  }}
}}

// Soft min-distance constraint using the SAME grid you already build
function applyMinSeparation(active, grid, cellSize, minDist, strength, alpha) {{
  const minD2 = minDist * minDist;

  for (const n of active) {{
    const gx = Math.floor(n.x / cellSize);
    const gy = Math.floor(n.y / cellSize);

    for (let dx = -1; dx <= 1; dx++) {{
      for (let dy = -1; dy <= 1; dy++) {{
        const nbrs = grid[(gx + dx) + ',' + (gy + dy)];
        if (!nbrs) continue;

        for (const m of nbrs) {{
          if (m.idx <= n.idx) continue;

          let ex = n.x - m.x, ey = n.y - m.y;
          let d2 = ex*ex + ey*ey;
          if (d2 < 1) {{ ex = (Math.random()-0.5)*2; ey = (Math.random()-0.5)*2; d2 = 1; }}

          if (d2 < minD2) {{
            const d = Math.sqrt(d2);
            const push = (minDist - d) * strength * alpha;
            const ux = ex / d, uy = ey / d;

            n.vx += ux * push; n.vy += uy * push;
            m.vx -= ux * push; m.vy -= uy * push;
          }}
        }}
      }}
    }}
  }}
}}

// Relax-mode parameter override (temporary)
function beginRelaxMode() {{
  if (SIM._relaxSaved) return; // already relaxing

  SIM._relaxSaved = {{
    linkStrength: SIM.linkStrength,
    repulsion: SIM.repulsion,
    centerStrength: SIM.centerStrength,
    damping: SIM.damping,
    alphaDecay: SIM.alphaDecay,
  }};

  // these are the knobs that stop black-hole collapse
  SIM.linkStrength = SIM.linkStrength * 0.35;     // much weaker springs
  SIM.repulsion = SIM.repulsion * 2.0;            // stronger separation
  SIM.centerStrength = SIM.centerStrength * 0.25; // weaker gravity-to-centroid
  SIM.damping = Math.min(SIM.damping, 0.48);      // more damping (0.48–0.55 works well)
  SIM.alphaDecay = Math.max(SIM.alphaDecay, 0.012); // cool faster

  SIM.relaxTicks = 260; // ~ a few seconds worth of ticks depending on your loop
}}

function endRelaxMode() {{
  if (!SIM._relaxSaved) return;
  const s = SIM._relaxSaved;
  SIM.linkStrength = s.linkStrength;
  SIM.repulsion = s.repulsion;
  SIM.centerStrength = s.centerStrength;
  SIM.damping = s.damping;
  SIM.alphaDecay = s.alphaDecay;
  SIM._relaxSaved = null;
  SIM.relaxTicks = 0;
}}

function setPaused(paused) {{
  SIM.running = !paused;
  const btn = document.getElementById('btn-pause');
  if (btn) {{
    btn.textContent = paused ? 'Resume' : 'Pause';
    btn.classList.toggle('active', paused);
  }}
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  TOOLBAR BUTTONS
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
document.getElementById('btn-labels').addEventListener('click', function() {{
  showLabels = !showLabels;
  this.classList.toggle('active', showLabels);
}});
document.getElementById('btn-labels').classList.add('active');

document.getElementById('btn-deps').addEventListener('click', function() {{
  highlightDeps = !highlightDeps;
  this.classList.toggle('active', highlightDeps);
}});
document.getElementById('btn-deps').classList.add('active');

document.getElementById('btn-fit').addEventListener('click', () => {{
  if (nodes.length === 0) return;
  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;
  nodes.forEach(n => {{
    if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
  }});
  cam.x = (minX + maxX) / 2;
  cam.y = (minY + maxY) / 2;
  const W = canvas.width/devicePixelRatio, H = canvas.height/devicePixelRatio;
  cam.zoom = Math.min(W / (maxX - minX + 200), H / (maxY - minY + 200), 2);
}});

document.getElementById('btn-pause').addEventListener('click', function() {{
  SIM.running = !SIM.running;
  if (SIM.running) {{
    SIM.alpha = Math.max(SIM.alpha, 0.15);
  }}
  this.textContent = SIM.running ? 'Pause' : 'Resume';
  this.classList.toggle('active', !SIM.running);
}});

document.getElementById('btn-grid').addEventListener('click', () => {{
  snapToGrid();
}});

document.getElementById('btn-spread').addEventListener('click', () => {{
  const active = graphNodes;
  if (!active || active.length === 0) return;

  // pause + spread + redraw
  setPaused(true);
  seedRingLayout(active, edges, {{ baseRadius: 260, ringStep: 210 }});
}});

document.getElementById('btn-relax').addEventListener('click', () => {{
  const active = graphNodes;
  if (!active || active.length === 0) return;

  // Seed wide, then run in relax mode for a bit
  seedRingLayout(active, edges, {{ baseRadius: 260, ringStep: 210 }});

  beginRelaxMode();
  SIM.alpha = 1.0;
  SIM.running = true;

  // Update Pause button UI (since we’re running)
  const btn = document.getElementById('btn-pause');
  if (btn) {{
    btn.textContent = 'Pause';
    btn.classList.remove('active');
  }}
}});

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  SEARCH
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
document.getElementById('search').addEventListener('input', function() {{
  const q = this.value.toLowerCase().trim();
  searchMatches.clear();
  if (q) {{
    nodes.forEach(n => {{
      if (n.item.name.toLowerCase().includes(q) ||
          n.item.qname.toLowerCase().includes(q) ||
          n.item.file.toLowerCase().includes(q))
        searchMatches.add(n.idx);
    }});
    // Center on first match
    if (searchMatches.size > 0) {{
      const first = nodes[[...searchMatches][0]];
      cam.x = first.x; cam.y = first.y;
    }}
  }}
}});

document.addEventListener('keydown', e => {{
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {{
    e.preventDefault();
    document.getElementById('search').focus();
  }}
  if (e.key === 'Escape') {{
    document.getElementById('search').blur();
    document.getElementById('search').value = '';
    searchMatches.clear();
    selectedNode = null;
    document.getElementById('detail').classList.remove('visible');
  }}
}});

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  MAIN LOOP
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function loop() {{
  simulate();
  render();
  requestAnimationFrame(loop);
}}

// Auto-fit after initial layout settles a bit
setTimeout(() => document.getElementById('btn-fit').click(), 800);

loop();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Coq Dependency Graph Visualizer — interactive force-directed graph"
    )
    parser.add_argument("json_file", help="JSON file from coq_analyzer.py")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML file (default: <input>_graph.html)")
    parser.add_argument("--open", action="store_true",
                        help="Open in browser after generating")

    args = parser.parse_args()

    if not os.path.isfile(args.json_file):
        print(f"Error: {args.json_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_path = args.output or args.json_file.rsplit(".", 1)[0] + "_graph.html"

    html = generate_graph_html(data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Graph written to {out_path}", file=sys.stderr)

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")
        print("Opened in browser.", file=sys.stderr)


if __name__ == "__main__":
    main()


"""
# Generate the JSON with the analyzer first
python coq_analyzer.py /path/to/project -o coq_deps.html --json

# Then visualize it
python coq_visualizer.py coq_deps.json --open
"""