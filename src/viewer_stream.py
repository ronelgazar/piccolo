"""MJPEG streaming server for external viewers.

Runs a lightweight Flask server in a daemon thread.  The main application
pushes processed frames into :meth:`update_frame`; any number of browser
clients can view the live feed at ``http://<host>:<port>/``.

Two stream endpoints are exposed:

* ``/video_feed`` – the full side-by-side stereo view (same as surgeon).
* ``/video_left``  – left eye only.
* ``/video_right`` – right eye only.

Control API:

* ``POST /api/action/<name>`` – inject a command (zoom_in, zoom_out, etc.)
* ``GET  /api/status``        – current zoom, alignment, FPS info as JSON.
"""

from __future__ import annotations

import queue
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app import PiccoloApp
from .config import StreamCfg
from .annotation import AnnotationOverlay

# ---------------------------------------------------------------------------
# Inline HTML template – full control panel + live stream
# ---------------------------------------------------------------------------
_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Piccolo – Surgery Viewer</title>
<style>
  :root { --bg:#111; --panel:#1a1a2e; --accent:#0d47a1; --ok:#00c853;
          --warn:#ff9100; --text:#eee; --muted:#888; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:10px 20px; background:var(--panel); display:flex;
           align-items:center; gap:14px; flex-shrink:0; }
  header h1 { font-size:1.1rem; font-weight:600; }
  .badge { background:var(--ok); color:#000; font-size:0.7rem; padding:2px 8px;
           border-radius:10px; font-weight:700; }
  .main { display:flex; flex:1; overflow:hidden; }

  /* ─── stream view ─── */
  .stream-area { flex:1; display:flex; flex-direction:column; min-width:0; }
  .stream-tabs { display:flex; gap:6px; padding:8px 16px; flex-shrink:0; }
  .stream-tabs button { padding:5px 14px; border:1px solid #444; border-radius:6px;
                         background:#222; color:#ccc; cursor:pointer; font-size:0.8rem; }
  .stream-tabs button.active { background:var(--accent); border-color:#1565c0; color:#fff; }
  .tab-sep { border-left:1px solid #444; height:20px; align-self:center; margin:0 2px; }
  .stream-info { padding:0 16px 4px; font-size:0.72rem; color:var(--muted); min-height:18px; }
  .stream-img { flex:1; display:flex; justify-content:center; align-items:center;
                padding:6px; overflow:hidden; }
  .stream-img img { max-width:100%; max-height:100%; object-fit:contain;
                    border:1px solid #333; border-radius:4px; }
  .stream-img canvas { position:absolute; cursor:crosshair; border-radius:4px; }

  /* ─── control panel ─── */
  .controls { width:280px; flex-shrink:0; background:var(--panel); padding:14px;
              display:flex; flex-direction:column; gap:14px; overflow-y:auto;
              border-left:1px solid #333; }
  .ctrl-group h3 { font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;
                   color:var(--muted); margin-bottom:8px; }
  .btn-row { display:flex; gap:6px; flex-wrap:wrap; }
  .ctrl-btn { padding:8px 0; flex:1; min-width:70px; border:1px solid #444;
              border-radius:6px; background:#222; color:#ccc; font-size:0.85rem;
              cursor:pointer; text-align:center; transition:background .15s; }
  .ctrl-btn:hover { background:#333; }
  .ctrl-btn:active { background:var(--accent); color:#fff; }
  .ctrl-btn.hold { user-select:none; }  /* for zoom/convergence */
  .ctrl-btn.danger { border-color:#c62828; color:#ef5350; }
  .ctrl-btn.danger:hover { background:#c62828; color:#fff; }
  .ctrl-btn.toggle-on { background:#1b5e20; border-color:#2e7d32; color:#a5d6a7; }

  /* status display */
  .status { font-size:0.75rem; color:var(--muted); line-height:1.6; }
  .status .val { color:var(--text); font-weight:600; font-variant-numeric:tabular-nums; }
  .kbd { display:inline-block; background:#333; border:1px solid #555; border-radius:3px;
         padding:0 5px; font-size:0.7rem; font-family:monospace; color:#bbb;
         margin-left:4px; vertical-align:middle; }
  .footer { padding:6px 16px; font-size:0.7rem; color:var(--muted); flex-shrink:0; }
</style>
</head>
<body>

<header>
  <h1>Piccolo – Surgery Viewer</h1>
  <span class="badge" id="live-badge">LIVE</span>
</header>

<div class="main">
  <!-- Stream view -->
  <div class="stream-area">
    <div class="stream-tabs">
      <button class="active" onclick="sw('/video_feed',this)">SBS 3D</button>
      <button onclick="sw('/video_annotated',this)">Annotated SBS</button>
      <button onclick="sw('/video_fused_annotated',this)">Fused Annotated</button>
      <button onclick="sw('/video_anaglyph',this)">Anaglyph 3D</button>
      <button onclick="sw('/video_fused_3d',this)">Fused 3D</button>
      <span class="tab-sep"></span>
      <button onclick="sw('/video_left',this)">Left</button>
      <button onclick="sw('/video_right',this)">Right</button>
    </div>
    <div class="stream-info" id="stream-info">Side-by-side 3D &ndash; for Goovis / VR headsets</div>
    <div class="stream-img" style="position:relative;">
      <img id="stream" src="/video_feed" alt="live stream" style="display:block;max-width:100%;max-height:100%;"/>
      <canvas id="zoom-crosshair" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;"></canvas>
    </div>
  </div>

  <!-- Control panel -->
  <div class="controls">
    <!-- Zoom -->
    <div class="ctrl-group">
      <h3>Zoom</h3>
      <div class="btn-row">
        <button class="ctrl-btn hold" data-action="zoom_out"
                onpointerdown="holdStart('zoom_out')" onpointerup="holdStop()"
                onpointerleave="holdStop()">&#x2212; Zoom<span class="kbd">-</span></button>
        <button class="ctrl-btn hold" data-action="zoom_in"
                onpointerdown="holdStart('zoom_in')" onpointerup="holdStop()"
                onpointerleave="holdStop()">&#x2b; Zoom<span class="kbd">+</span></button>
      </div>
    </div>

    <!-- Zoom (Custom Center) -->
    <div class="ctrl-group">
      <h3>Zoom (Custom Center)</h3>
      <div style="margin-bottom:6px;font-size:0.8em;color:var(--muted);">Click the image to set zoom center</div>
      <div class="btn-row">
        <button class="ctrl-btn hold" data-action="zoom_out"
                onpointerdown="customZoom('out')" onpointerup="holdStop()"
                onpointerleave="holdStop()">&#x2212; Zoom<span class="kbd">-</span></button>
        <button class="ctrl-btn hold" data-action="zoom_in"
                onpointerdown="customZoom('in')" onpointerup="holdStop()"
                onpointerleave="holdStop()">&#x2b; Zoom<span class="kbd">+</span></button>
      </div>
    </div>

    <!-- Convergence -->
    <div class="ctrl-group">
      <h3>Convergence</h3>
      <div class="btn-row">
        <button class="ctrl-btn hold" data-action="converge_out"
                onpointerdown="holdStart('converge_out')" onpointerup="holdStop()"
                onpointerleave="holdStop()">&#x2190; Out<span class="kbd">[</span></button>
        <button class="ctrl-btn hold" data-action="converge_in"
                onpointerdown="holdStart('converge_in')" onpointerup="holdStop()"
                onpointerleave="holdStop()">&#x2192; In<span class="kbd">]</span></button>
      </div>
    </div>

    <!-- Alignment -->
    <div class="ctrl-group">
      <h3>Auto-Alignment</h3>
      <div class="btn-row">
        <button class="ctrl-btn" id="btn-align" onclick="sendAction('toggle_alignment')">
          Toggle<span class="kbd">A</span></button>
        <button class="ctrl-btn" onclick="sendAction('force_align')">Re-scan</button>
      </div>
    </div>

    <!-- Calibration -->
    <div class="ctrl-group">
      <h3>Calibration</h3>
      <div class="btn-row">
        <button class="ctrl-btn" id="btn-calib" onclick="sendAction('toggle_calibration')">
          Start<span class="kbd">C</span></button>
        <button class="ctrl-btn" id="btn-calib-next" onclick="sendAction('calib_next')">
          Next Eye<span class="kbd">N</span></button>
      </div>
      <div class="btn-row" style="margin-top:6px">
        <button class="ctrl-btn hold" data-action="calib_nudge_left"
                onpointerdown="holdStart('calib_nudge_left')" onpointerup="holdStop()"
                onpointerleave="holdStop()">&#x2190; Nudge<span class="kbd">&larr;</span></button>
        <button class="ctrl-btn hold" data-action="calib_nudge_right"
                onpointerdown="holdStart('calib_nudge_right')" onpointerup="holdStop()"
                onpointerleave="holdStop()">Nudge &#x2192;<span class="kbd">&rarr;</span></button>
      </div>
      <div class="btn-row" style="margin-top:6px">
        <button class="ctrl-btn" onclick="sendAction('reset_nudge')">
          Reset Nudge</button>
      </div>
      <div class="status" id="calib-status" style="margin-top:6px;min-height:18px;"></div>
    </div>

    <!-- Reset / Quit -->
    <div class="ctrl-group">
      <h3>System</h3>
      <div class="btn-row">
        <button class="ctrl-btn" onclick="sendAction('reset')">Reset<span class="kbd">R</span></button>
        <button class="ctrl-btn danger" onclick="if(confirm('Quit Piccolo?'))sendAction('quit')">
          Quit<span class="kbd">Esc</span></button>
      </div>
    </div>

    <!-- Status -->
    <div class="ctrl-group">
      <h3>Status</h3>
      <div class="status" id="status-area">
        Loading…
      </div>
    </div>
  </div>
</div>

<!-- Fused 3D Alignment Slider -->
<div class="ctrl-group" id="fused3d-slider-group" style="display:none;">
  <h3>Fused 3D Alignment</h3>
  <div>
    <label for="alignment-slider">Adjust Alignment:</label>
    <input type="range" id="alignment-slider" min="-400" max="400" value="0" step="1" oninput="updateAlignment(this.value)">
    <span id="alignment-value">0</span>
  </div>
</div>

<div class="footer">
  Stream URL: <code>http://&lt;host&gt;:{{ port }}/video_feed</code> &nbsp;|&nbsp;
  <a href="/annotate" style="color:#0d47a1">&#9998; Open Annotation Tool</a> &nbsp;|&nbsp;
  Keyboard shortcuts work when this page has focus.
</div>

<script>
/* ── stream switcher ── */
function sw(src, btn) {
  document.getElementById('stream').src = src;
  document.querySelectorAll('.stream-tabs button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const info = document.getElementById('stream-info');
  if (src === '/video_anaglyph') info.textContent = 'Requires Red/Cyan 3D glasses';
  else if (src === '/video_feed') info.textContent = 'Side-by-side 3D \u2013 for Goovis / VR headsets';
  else if (src === '/video_annotated') info.textContent = 'Annotated SBS (with overlay)';
  else if (src === '/video_fused_annotated') info.textContent = 'Fused Annotated (true 3D merge)';
  else info.textContent = '';

  // Show/hide Fused 3D slider
  const sliderGroup = document.getElementById('fused3d-slider-group');
  if (sliderGroup) {
    sliderGroup.style.display = (src === '/video_fused_3d') ? '' : 'none';
    // Reset slider value display
    const slider = document.getElementById('alignment-slider');
    const valueSpan = document.getElementById('alignment-value');
    if (slider && valueSpan) valueSpan.innerText = slider.value;
  }
}

function updateAlignment(value) {
  document.getElementById('alignment-value').innerText = value;
  fetch('/api/adjust_alignment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ offset: parseInt(value) })
  });
}

/* ── single action ── */
function sendAction(name) {
  fetch('/api/action/' + name, { method: 'POST' }).catch(() => {});
}

/* ── hold-to-repeat for zoom / convergence ── */
let holdTimer = null;
function holdStart(action) {
  sendAction(action);
  holdTimer = setInterval(() => sendAction(action), 60);
}
function holdStop() {
  if (holdTimer) { clearInterval(holdTimer); holdTimer = null; }
}

/* ── keyboard shortcuts (when page is focused) ── */
const KEY_MAP = {
  '=': 'zoom_in', '+': 'zoom_in',
  '-': 'zoom_out', '_': 'zoom_out',
  ']': 'converge_in', '[': 'converge_out',
  'a': 'toggle_alignment', 'A': 'toggle_alignment',
  'c': 'toggle_calibration', 'C': 'toggle_calibration',
  'n': 'calib_next', 'N': 'calib_next',
  'ArrowLeft': 'calib_nudge_left',
  'ArrowRight': 'calib_nudge_right',
  'r': 'reset', 'R': 'reset',
  'Escape': 'quit',
};
const HOLD_ACTIONS = new Set(['zoom_in', 'zoom_out', 'converge_in', 'converge_out',
                              'calib_nudge_left', 'calib_nudge_right']);
let heldKeys = {};
document.addEventListener('keydown', e => {
  if (e.repeat) return;
  const action = KEY_MAP[e.key];
  if (!action) return;
  e.preventDefault();
  if (HOLD_ACTIONS.has(action)) {
    sendAction(action);
    heldKeys[e.key] = setInterval(() => sendAction(action), 60);
  } else {
    sendAction(action);
  }
});
document.addEventListener('keyup', e => {
  if (heldKeys[e.key]) { clearInterval(heldKeys[e.key]); delete heldKeys[e.key]; }
});

/* ── periodic status poll ── */
function updateStatus() {
  fetch('/api/status').then(r => r.json()).then(d => {
    const el = document.getElementById('status-area');
    const alignBtn = document.getElementById('btn-align');
    const calibBtn = document.getElementById('btn-calib');
    let html = '';
    html += `FPS: <span class="val">${d.fps.toFixed(0)}</span> &nbsp; `;
    html += `Loop: <span class="val">${d.loop_ms.toFixed(1)} ms</span><br>`;
    html += `Zoom: <span class="val">${d.zoom.toFixed(2)}x</span><br>`;
    html += `Convergence: <span class="val">${d.convergence_offset}</span><br>`;
    html += `Alignment: <span class="val">${d.alignment.enabled ? 'ON' : 'OFF'}</span>`;
    if (d.alignment.enabled && d.alignment.method !== 'none') {
      html += ` (${d.alignment.method})`;
      html += `<br>&nbsp; dy=<span class="val">${d.alignment.dy.toFixed(1)}px</span>`;
      html += ` rot=<span class="val">${d.alignment.dtheta_deg.toFixed(2)}°</span>`;
      html += ` [${d.alignment.n_matches} matches]`;
    }
    html += `<br>Calibration: <span class="val">${d.calibration ? 'ACTIVE' : 'off'}</span>`;
    if (d.calibration && d.calibration_phase) {
      html += ` &mdash; <span class="val">${d.calibration_phase.toUpperCase()}</span>`;
    }
    if (d.nudge_left !== undefined) {
      html += `<br>Nudge L: <span class="val">${d.nudge_left}px</span>`;
      html += ` &nbsp; R: <span class="val">${d.nudge_right}px</span>`;
    }
    el.innerHTML = html;
    // Toggle button styles
    alignBtn.classList.toggle('toggle-on', d.alignment.enabled);
    alignBtn.textContent = d.alignment.enabled ? 'ON' : 'OFF';
    calibBtn.classList.toggle('toggle-on', d.calibration);
    calibBtn.textContent = d.calibration ? 'Active…' : 'Start';
    // Calibration sub-status
    const calibEl = document.getElementById('calib-status');
    if (d.calibration && d.calibration_phase) {
      calibEl.innerHTML = `Phase: <span class="val">${d.calibration_phase.toUpperCase()}</span>`;
    } else {
      calibEl.innerHTML = '';
    }
  }).catch(() => {});
}
setInterval(updateStatus, 500);
updateStatus();

/* Visual feedback for alignment and convergence */
function showAlignmentFeedback(alignmentData) {
  const feedbackElement = document.getElementById('alignment-feedback');
  if (!feedbackElement) {
    const newElement = document.createElement('div');
    newElement.id = 'alignment-feedback';
    newElement.style.position = 'absolute';
    newElement.style.bottom = '10px';
    newElement.style.right = '10px';
    newElement.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    newElement.style.color = 'white';
    newElement.style.padding = '10px';
    newElement.style.borderRadius = '5px';
    newElement.style.fontSize = '0.8rem';
    document.body.appendChild(newElement);
  }
  const feedbackText = `Alignment: dy=${alignmentData.dy.toFixed(2)}px, dtheta=${alignmentData.dtheta.toFixed(2)}°`;
  feedbackElement.textContent = feedbackText;
}

/* Update alignment feedback periodically */
function updateAlignmentFeedback() {
  fetch('/api/status').then(r => r.json()).then(d => {
    if (d.alignment) {
      showAlignmentFeedback(d.alignment);
    }
  }).catch(() => {});
}
setInterval(updateAlignmentFeedback, 500);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Annotation page HTML template
# ---------------------------------------------------------------------------
_ANNOTATE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Piccolo – Annotate</title>
<style>
  :root { --bg:#111; --panel:#1a1a2e; --accent:#0d47a1; --ok:#00c853;
          --text:#eee; --muted:#888; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
         display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  header { padding:10px 20px; background:var(--panel); display:flex;
           align-items:center; gap:14px; flex-shrink:0; }
  header h1 { font-size:1.1rem; font-weight:600; }
  header a { color:var(--accent); text-decoration:none; font-size:0.85rem; margin-left:auto; }
  header a:hover { text-decoration:underline; }
  .badge { background:var(--ok); color:#000; font-size:0.7rem; padding:2px 8px;
           border-radius:10px; font-weight:700; }
  .main { display:flex; flex:1; overflow:hidden; }

  /* canvas area */
  .canvas-area { flex:1; position:relative; display:flex; justify-content:center;
                 align-items:center; padding:6px; overflow:hidden; }
  .canvas-area img { max-width:100%; max-height:100%; object-fit:contain;
                     border:1px solid #333; border-radius:4px; display:block; }
  .canvas-area canvas { position:absolute; cursor:crosshair; border-radius:4px; }

  /* toolbar */
  .toolbar { width:260px; flex-shrink:0; background:var(--panel); padding:14px;
             display:flex; flex-direction:column; gap:12px; overflow-y:auto;
             border-left:1px solid #333; }
  .tool-group h3 { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                   color:var(--muted); margin-bottom:6px; }
  .tool-row { display:flex; gap:5px; flex-wrap:wrap; }
  .tool-btn { padding:7px 0; flex:1; min-width:55px; border:1px solid #444;
              border-radius:6px; background:#222; color:#ccc; font-size:0.8rem;
              cursor:pointer; text-align:center; transition:background .15s; }
  .tool-btn:hover { background:#333; }
  .tool-btn.active { background:var(--accent); border-color:#1565c0; color:#fff; }
  .tool-btn.danger { border-color:#c62828; color:#ef5350; }
  .tool-btn.danger:hover { background:#c62828; color:#fff; }
  .tool-btn.send { border-color:#1b5e20; color:#a5d6a7; }
  .tool-btn.send.active { background:#1b5e20; border-color:#2e7d32; color:#fff; }

  /* color/width pickers */
  .color-row { display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
  .color-swatch { width:28px; height:28px; border-radius:50%; border:2px solid transparent;
                  cursor:pointer; transition:border-color .15s; }
  .color-swatch.active { border-color:#fff; }
  .color-custom { width:28px; height:28px; border:none; padding:0; background:none; cursor:pointer; }
  .width-row { display:flex; gap:8px; align-items:center; }
  .width-row input { flex:1; accent-color:var(--accent); }
  .width-row span { font-size:0.8rem; color:var(--text); min-width:30px; text-align:right; }

  /* text input */
  .text-input { width:100%; padding:6px 8px; border:1px solid #444; border-radius:6px;
                background:#222; color:#eee; font-size:0.85rem; display:none; }
  .text-input.visible { display:block; }

  /* annotations list */
  .ann-list { font-size:0.75rem; color:var(--muted); max-height:140px; overflow-y:auto; line-height:1.5; }
  .ann-count { font-size:0.75rem; color:var(--muted); }
</style>
</head>
<body>

<header>
  <h1>Piccolo – Annotate</h1>
  <span class="badge" id="send-badge" style="background:#555">NOT SENT</span>
  <a href="/">&#x2190; Back to Viewer</a>
</header>

<div class="main">
  <!-- Canvas over stream -->
  <div class="canvas-area" id="canvas-area">
    <img id="bg-stream" src="/video_left" alt="live feed"/>
    <canvas id="draw-canvas"></canvas>
  </div>

  <!-- Toolbar -->
  <div class="toolbar">
    <!-- Tools -->
    <div class="tool-group">
      <h3>Tool</h3>
      <div class="tool-row" id="tool-row">
        <button class="tool-btn active" data-tool="freehand">&#9998; Pen</button>
        <button class="tool-btn" data-tool="line">&#x2571; Line</button>
        <button class="tool-btn" data-tool="arrow">&#x2794; Arrow</button>
        <button class="tool-btn" data-tool="circle">&#x25CB; Circle</button>
        <button class="tool-btn" data-tool="rect">&#x25A1; Rect</button>
        <button class="tool-btn" data-tool="text">T Text</button>
      </div>
    </div>

    <!-- Color -->
    <div class="tool-group">
      <h3>Color</h3>
      <div class="color-row" id="color-row">
        <div class="color-swatch active" style="background:#00ff00" data-color="#00ff00"></div>
        <div class="color-swatch" style="background:#ff0000" data-color="#ff0000"></div>
        <div class="color-swatch" style="background:#00bfff" data-color="#00bfff"></div>
        <div class="color-swatch" style="background:#ffff00" data-color="#ffff00"></div>
        <div class="color-swatch" style="background:#ff00ff" data-color="#ff00ff"></div>
        <div class="color-swatch" style="background:#ffffff" data-color="#ffffff"></div>
        <input type="color" class="color-custom" id="custom-color" value="#00ff00"
               title="Custom color"/>
      </div>
    </div>

    <!-- Width -->
    <div class="tool-group">
      <h3>Width</h3>
      <div class="width-row">
        <input type="range" id="line-width" min="1" max="12" value="2"/>
        <span id="width-val">2px</span>
      </div>
    </div>

    <!-- Text input (shown only when text tool active) -->
    <div class="tool-group" id="text-group">
      <h3>Text</h3>
      <input type="text" class="text-input visible" id="text-content"
             placeholder="Type annotation text…" value=""/>
    </div>

    <!-- Actions -->
    <div class="tool-group">
      <h3>Actions</h3>
      <div class="tool-row">
        <button class="tool-btn" id="btn-undo" onclick="doUndo()">&#x21B6; Undo</button>
        <button class="tool-btn danger" id="btn-clear" onclick="doClear()">&#x2715; Clear</button>
      </div>
      <div class="tool-row" style="margin-top:6px">
        <button class="tool-btn send" id="btn-send" onclick="toggleSend()">
          &#x25B6; Send to Screen</button>
      </div>
    </div>

    <!-- Stream selector (single-eye views only — SBS would break
         coordinate mapping since annotations use normalised single-eye coords) -->
    <div class="tool-group">
      <h3>View</h3>
      <div class="tool-row">
        <button class="tool-btn active" onclick="switchBg('/video_left','left',this)">Left Eye</button>
        <button class="tool-btn" onclick="switchBg('/video_right','right',this)">Right Eye</button>
      </div>
    </div>

    <!-- Info -->
    <div class="tool-group">
      <h3>Annotations</h3>
      <div class="ann-count" id="ann-count">0 annotations</div>
    </div>

    <!-- Disparity correction for cross-eye alignment -->
    <div class="tool-group">
      <h3>Disparity</h3>
      <div style="display:flex;align-items:center;gap:8px">
        <input type="range" id="disp-slider" min="-400" max="600" value="0" step="1"
               style="flex:1;height:28px;cursor:pointer;accent-color:var(--accent,#0af)"
               oninput="setDisparity(this.value)">
        <span id="disp-val" style="min-width:48px;text-align:right;font-size:1em;font-weight:600">0px</span>
      </div>
      <div style="font-size:.75em;color:#aaa;margin-top:2px">
        Shift annotations on the opposite eye to align with scene
      </div>
    </div>
  </div>
</div>

<!-- Fused 3D Alignment Slider -->
<div class="ctrl-group" id="fused3d-slider-group" style="display:none;">
  <h3>Fused 3D Alignment</h3>
  <div>
    <label for="alignment-slider">Adjust Alignment:</label>
    <input type="range" id="alignment-slider" min="-50" max="50" value="0" step="1" oninput="updateAlignment(this.value)">
    <span id="alignment-value">0</span>
  </div>
</div>

<script>
/* ── State ── */
let currentTool = 'freehand';
let currentColor = '#00ff00';
let currentWidth = 2;
let isSending = false;
let currentEye = 'left';  // tracks which eye stream is being viewed
let isDrawing = false;
let startPt = null;
let freehandPts = [];
let localAnnotations = [];  // mirrors server state for canvas redraw

const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
const bgImg = document.getElementById('bg-stream');
const area = document.getElementById('canvas-area');

/* ── Resize canvas to match image ── */
function resizeCanvas() {
  const rect = bgImg.getBoundingClientRect();
  canvas.style.left = rect.left + 'px';
  canvas.style.top = rect.top + 'px';
  canvas.width = rect.width;
  canvas.height = rect.height;
  redrawAll();
}
window.addEventListener('resize', resizeCanvas);
bgImg.addEventListener('load', resizeCanvas);
// Poll for image size changes (MJPEG streams)
setInterval(resizeCanvas, 2000);
setTimeout(resizeCanvas, 500);

/* ── Tool selection ── */
document.getElementById('tool-row').addEventListener('click', e => {
  const btn = e.target.closest('[data-tool]');
  if (!btn) return;
  document.querySelectorAll('#tool-row .tool-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentTool = btn.dataset.tool;
  document.getElementById('text-group').style.display = currentTool === 'text' ? '' : 'none';
});
// Initially hide text group if not text tool
document.getElementById('text-group').style.display = 'none';

/* ── Color selection ── */
document.getElementById('color-row').addEventListener('click', e => {
  const sw = e.target.closest('.color-swatch');
  if (!sw) return;
  document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
  sw.classList.add('active');
  currentColor = sw.dataset.color;
  document.getElementById('custom-color').value = currentColor;
});
document.getElementById('custom-color').addEventListener('input', e => {
  currentColor = e.target.value;
  document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
});

/* ── Width ── */
document.getElementById('line-width').addEventListener('input', e => {
  currentWidth = parseInt(e.target.value);
  document.getElementById('width-val').textContent = currentWidth + 'px';
});

/* ── Coordinate helpers ── */
function canvasToNorm(x, y) {
  return [x / canvas.width, y / canvas.height];
}
function normToCanvas(nx, ny) {
  return [nx * canvas.width, ny * canvas.height];
}

/* ── Hex → BGR for OpenCV ── */
function hexToBGR(hex) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return [b, g, r];
}

/* ── Drawing on canvas (local preview) ── */
function drawAnnotation(ann) {
  const pts = ann.points || [];
  const color = ann.css_color || currentColor;
  const width = ann.width || 2;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  if (ann.type === 'freehand' && pts.length >= 2) {
    ctx.beginPath();
    const [sx, sy] = normToCanvas(pts[0][0], pts[0][1]);
    ctx.moveTo(sx, sy);
    for (let i = 1; i < pts.length; i++) {
      const [px, py] = normToCanvas(pts[i][0], pts[i][1]);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
  } else if (ann.type === 'line' && pts.length >= 2) {
    const [x1,y1] = normToCanvas(pts[0][0], pts[0][1]);
    const [x2,y2] = normToCanvas(pts[pts.length-1][0], pts[pts.length-1][1]);
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  } else if (ann.type === 'arrow' && pts.length >= 2) {
    const [x1,y1] = normToCanvas(pts[0][0], pts[0][1]);
    const [x2,y2] = normToCanvas(pts[pts.length-1][0], pts[pts.length-1][1]);
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    // Arrowhead
    const angle = Math.atan2(y2-y1, x2-x1);
    const headLen = 12 + width * 2;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4));
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4));
    ctx.stroke();
  } else if (ann.type === 'circle' && pts.length >= 2) {
    const [cx,cy] = normToCanvas(pts[0][0], pts[0][1]);
    const [ex,ey] = normToCanvas(pts[pts.length-1][0], pts[pts.length-1][1]);
    const r = Math.hypot(ex-cx, ey-cy);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.stroke();
  } else if (ann.type === 'rect' && pts.length >= 2) {
    const [x1,y1] = normToCanvas(pts[0][0], pts[0][1]);
    const [x2,y2] = normToCanvas(pts[pts.length-1][0], pts[pts.length-1][1]);
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
  } else if (ann.type === 'text' && pts.length >= 1) {
    const [tx,ty] = normToCanvas(pts[0][0], pts[0][1]);
    const text = ann.text || '';
    ctx.font = `${14 + width * 2}px system-ui, sans-serif`;
    ctx.fillText(text, tx, ty);
  }
}

function redrawAll() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  localAnnotations.forEach(a => drawAnnotation(a));
}

/* ── Mouse / pointer events ── */
canvas.addEventListener('pointerdown', e => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  isDrawing = true;
  startPt = canvasToNorm(x, y);

  if (currentTool === 'freehand') {
    freehandPts = [startPt];
  } else if (currentTool === 'text') {
    const text = document.getElementById('text-content').value.trim();
    if (text) {
      const ann = {
        type: 'text', points: [startPt], text: text,
        color: hexToBGR(currentColor), css_color: currentColor, width: currentWidth
      };
      sendAnnotation(ann);
      localAnnotations.push(ann);
      redrawAll();
    }
    isDrawing = false;
  }
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', e => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const pt = canvasToNorm(x, y);

  if (currentTool === 'freehand') {
    freehandPts.push(pt);
    // Draw incrementally
    if (freehandPts.length >= 2) {
      const prev = freehandPts[freehandPts.length - 2];
      const [px, py] = normToCanvas(prev[0], prev[1]);
      const [cx, cy] = normToCanvas(pt[0], pt[1]);
      ctx.strokeStyle = currentColor;
      ctx.lineWidth = currentWidth;
      ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(cx, cy); ctx.stroke();
    }
  } else {
    // Preview: redraw all + the shape being drawn
    redrawAll();
    const preview = {
      type: currentTool, points: [startPt, pt],
      css_color: currentColor, width: currentWidth
    };
    drawAnnotation(preview);
  }
});

canvas.addEventListener('pointerup', e => {
  if (!isDrawing) return;
  isDrawing = false;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const endPt = canvasToNorm(x, y);

  let ann = null;
  if (currentTool === 'freehand' && freehandPts.length >= 2) {
    ann = { type: 'freehand', points: freehandPts,
            color: hexToBGR(currentColor), css_color: currentColor, width: currentWidth };
  } else if (['line','arrow','circle','rect'].includes(currentTool)) {
    ann = { type: currentTool, points: [startPt, endPt],
            color: hexToBGR(currentColor), css_color: currentColor, width: currentWidth };
  }

  if (ann) {
    sendAnnotation(ann);
    localAnnotations.push(ann);
    redrawAll();
  }
  freehandPts = [];
  startPt = null;
});

/* ── API calls ── */
function sendAnnotation(ann) {
  // Send only what the server needs (no css_color)
  const payload = { type: ann.type, points: ann.points, color: ann.color,
                    width: ann.width, source_eye: currentEye };
  if (ann.text) payload.text = ann.text;
  fetch('/api/annotations/add', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  }).then(() => updateCount()).catch(() => {});
}

function doUndo() {
  fetch('/api/annotations/undo', { method: 'POST' })
    .then(() => { localAnnotations.pop(); redrawAll(); updateCount(); })
    .catch(() => {});
}

function doClear() {
  fetch('/api/annotations/clear', { method: 'POST' })
    .then(() => { localAnnotations = []; redrawAll(); updateCount(); })
    .catch(() => {});
}

function toggleSend() {
  isSending = !isSending;
  fetch('/api/annotations/send', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ enabled: isSending })
  }).catch(() => {});
  const btn = document.getElementById('btn-send');
  const badge = document.getElementById('send-badge');
  btn.classList.toggle('active', isSending);
  btn.innerHTML = isSending ? '&#x25A0; Stop Sending' : '&#x25B6; Send to Screen';
  badge.textContent = isSending ? 'ON SCREEN' : 'NOT SENT';
  badge.style.background = isSending ? 'var(--ok)' : '#555';
}

function updateCount() {
  document.getElementById('ann-count').textContent = localAnnotations.length + ' annotations';
}

function switchBg(src, eye, btn) {
  bgImg.src = src;
  currentEye = eye;
  btn.parentElement.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  setTimeout(resizeCanvas, 300);
}

/* ── Disparity adjustment ── */
function setDisparity(val) {
  document.getElementById('disp-val').textContent = val + 'px';
  fetch('/api/annotations/disparity', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ offset: parseInt(val) })
  }).catch(() => {});
}

/* ── Keyboard shortcuts ── */
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;  // don't intercept text input
  if (e.ctrlKey && e.key === 'z') { e.preventDefault(); doUndo(); }
  if (e.key === 'Delete' || (e.ctrlKey && e.key === 'x')) { e.preventDefault(); doClear(); }
  // Tool shortcuts
  const toolKeys = { 'p':'freehand', 'l':'line', 'a':'arrow', 'o':'circle', 'b':'rect', 't':'text' };
  if (toolKeys[e.key]) {
    document.querySelectorAll('#tool-row .tool-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tool === toolKeys[e.key]);
    });
    currentTool = toolKeys[e.key];
    document.getElementById('text-group').style.display = currentTool === 'text' ? '' : 'none';
  }
});

/* ── Sync annotation count on load ── */
fetch('/api/annotations/list').then(r => r.json()).then(d => {
  localAnnotations = (d.annotations || []).map(a => {
    // Reconstruct css_color from BGR
    const bgr = a.color || [0,255,0];
    a.css_color = '#' + [bgr[2],bgr[1],bgr[0]].map(c => c.toString(16).padStart(2,'0')).join('');
    return a;
  });
  redrawAll();
  updateCount();
  // Sync send state
  isSending = d.show_on_screen || false;
  const btn = document.getElementById('btn-send');
  const badge = document.getElementById('send-badge');
  btn.classList.toggle('active', isSending);
  btn.innerHTML = isSending ? '&#x25A0; Stop Sending' : '&#x25B6; Send to Screen';
  badge.textContent = isSending ? 'ON SCREEN' : 'NOT SENT';
  badge.style.background = isSending ? 'var(--ok)' : '#555';
}).catch(() => {});
</script>
</body>
</html>
"""


class ViewerStream:
    """MJPEG streaming server that runs in a background thread."""

    def __init__(self, cfg: StreamCfg, app: PiccoloApp):
        self.cfg = cfg
        self.app = app
        self.cam_l = app.cam_l
        self.cam_r = app.cam_r
        self._frame_sbs: np.ndarray | None = None
        self._frame_left: np.ndarray | None = None
        self._frame_right: np.ndarray | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._client_count: int = 0  # track active MJPEG clients

        # Annotation overlay (shared with app.py via .annotations)
        self.annotations = AnnotationOverlay()

        self._app = self._create_app()

        # Command queue: web UI pushes action names, main loop drains
        self._cmd_queue: queue.Queue[str] = queue.Queue()

        # Status dict updated by the main loop for the web UI to read
        self._status: dict = {}
        self._status_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Frame update (called from main loop)
    # ------------------------------------------------------------------

    def update_frame(
        self,
        sbs: np.ndarray | None = None,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ):
        # Must copy – the caller's buffers are pre-allocated and get
        # overwritten on the next frame.  Without a copy the MJPEG
        # encoder thread reads a buffer that the main thread is mutating
        # concurrently, producing corrupt / blank frames.
        #
        # When no browser clients are connected we still store a cheap
        # reference so the first client to connect sees a frame instantly
        # instead of waiting for the next main-loop iteration.
        if self._client_count <= 0:
            # No clients: just keep a reference (no expensive copy)
            self._frame_sbs = sbs
            self._frame_left = left
            self._frame_right = right
            return
        with self._lock:
            if sbs is not None:
                self._frame_sbs = sbs.copy()
            if left is not None:
                self._frame_left = left.copy()
            if right is not None:
                self._frame_right = right.copy()

    # ------------------------------------------------------------------
    # Command queue (web UI → main loop)
    # ------------------------------------------------------------------

    def push_command(self, action_name: str):
        """Push a command from the web UI (thread-safe)."""
        self._cmd_queue.put_nowait(action_name)

    def drain_commands(self) -> list[str]:
        """Drain all pending commands (called from main loop)."""
        cmds: list[str] = []
        while True:
            try:
                cmds.append(self._cmd_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    # ------------------------------------------------------------------
    # Status (main loop → web UI)
    # ------------------------------------------------------------------

    def update_status(self, status: dict):
        """Update the status dict (called from main loop)."""
        with self._status_lock:
            self._status = status

    def get_status(self) -> dict:
        with self._status_lock:
            return dict(self._status)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="viewer-stream"
        )
        self._thread.start()

    def _run_server(self):
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)  # silence per-request logs
        self._app.run(
            host=self.cfg.host,
            port=self.cfg.port,
            threaded=True,
            use_reloader=False,
        )

    # ------------------------------------------------------------------
    # Flask app factory
    # ------------------------------------------------------------------

    def _create_app(self) -> Flask:
        app = Flask(__name__)
        server = self  # capture reference

        @app.route("/")
        def index():
            return render_template_string(_INDEX_HTML, port=server.cfg.port)

        @app.route("/annotate")
        def annotate_page():
            return render_template_string(_ANNOTATE_HTML, port=server.cfg.port)

        @app.route("/video_feed")
        def video_feed():
            return Response(
                server._generate("sbs"),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video_left")
        def video_left():
            return Response(
                server._generate("left"),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video_right")
        def video_right():
            return Response(
                server._generate("right"),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video_anaglyph")
        def video_anaglyph():
            return Response(
                server._generate("anaglyph"),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video_annotated")
        def video_annotated():
            return Response(
                server._generate("annotated"),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video_fused_annotated")
        def video_fused_annotated():
            return Response(
                server._generate("fused_annotated"),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video_fused_3d")
        def video_fused_3d():
            """Stream the fused 3D image construction simulation."""
            return Response(server.generate_fused_3d(), mimetype='multipart/x-mixed-replace; boundary=frame')

        # ── Control API ──

        @app.route("/api/action/<action_name>", methods=["POST"])
        def api_action(action_name: str):
            """Inject an action into the main loop."""
            valid = {
                "zoom_in", "zoom_out", "converge_in", "converge_out",
                "toggle_calibration", "toggle_alignment", "force_align",
                "calib_next", "calib_nudge_left", "calib_nudge_right",
                "reset_nudge", "reset", "quit",
            }
            if action_name not in valid:
                return jsonify({"error": f"Unknown action: {action_name}",
                                "valid": sorted(valid)}), 400
            server.push_command(action_name)
            return jsonify({"ok": True, "action": action_name})

        @app.route("/api/status")
        def api_status():
            """Return current system status as JSON."""
            return jsonify(server.get_status())

        # ── Annotation API ──

        @app.route("/api/annotations/add", methods=["POST"])
        def api_ann_add():
            data = request.get_json(silent=True)
            if not data or "type" not in data:
                return jsonify({"error": "Missing annotation data"}), 400
            server.annotations.add(data)
            return jsonify({"ok": True, "count": server.annotations.count})

        @app.route("/api/annotations/undo", methods=["POST"])
        def api_ann_undo():
            server.annotations.undo()
            return jsonify({"ok": True, "count": server.annotations.count})

        @app.route("/api/annotations/clear", methods=["POST"])
        def api_ann_clear():
            server.annotations.clear()
            return jsonify({"ok": True, "count": 0})

        @app.route("/api/annotations/list")
        def api_ann_list():
            return jsonify({
                "annotations": server.annotations.get_all(),
                "show_on_screen": server.annotations.show_on_screen,
                "count": server.annotations.count,
            })

        @app.route("/api/annotations/send", methods=["POST"])
        def api_ann_send():
            data = request.get_json(silent=True) or {}
            server.annotations.show_on_screen = bool(data.get("enabled", True))
            state = "ON" if server.annotations.show_on_screen else "OFF"
            return jsonify({"ok": True, "show_on_screen": server.annotations.show_on_screen})

        @app.route("/api/annotations/disparity", methods=["POST"])
        def api_ann_disparity():
            data = request.get_json(silent=True) or {}
            offset = int(data.get("offset", 0))
            # Clamp the offset to a reasonable range
            offset = max(-400, min(400, offset))
            server.annotations.disparity_offset = offset
            return jsonify({"ok": True, "disparity_offset": offset})

        @app.route("/api/adjust_alignment", methods=["POST"])
        def api_adjust_alignment():
            data = request.get_json(silent=True) or {}
            offset = int(data.get("offset", 0))
            # Clamp the offset to a reasonable range
            offset = max(-50, min(50, offset))
            server.cfg.alignment_offset = offset
            return jsonify({"ok": True, "offset": offset})

        @app.route("/api/zoom_center", methods=["POST"])
        def api_zoom_center():
            data = request.get_json(silent=True) or {}
            center = int(data.get("center", 50))
            center_y = int(data.get("center_y", 50))
            # Clamp to 0-100
            center = max(0, min(100, center))
            center_y = max(0, min(100, center_y))
            # Set on processor (via app)
            if hasattr(server, 'app') and hasattr(server.app, 'processor'):
                server.app.processor.set_joint_zoom_center(center)
                if hasattr(server.app.processor, 'set_joint_zoom_center_y'):
                    server.app.processor.set_joint_zoom_center_y(center_y)
            server.cfg.joint_zoom_center = center
            server.cfg.joint_zoom_center_y = center_y
            return jsonify({"ok": True, "zoom_center": center, "zoom_center_y": center_y})

        return app

    def _generate(self, which: str):
        """MJPEG generator for Flask streaming response."""
        self._client_count += 1
        try:
            while True:
                # ── Grab frame(s) under lock ──
                with self._lock:
                    if which == "anaglyph":
                        raw_l = self._frame_left
                        raw_r = self._frame_right
                        left = raw_l.copy() if raw_l is not None else None
                        right = raw_r.copy() if raw_r is not None else None
                        frame = None
                    elif which == "annotated":
                        raw = self._frame_sbs
                        frame = raw.copy() if raw is not None else None
                        if frame is not None:
                            self.annotations.render_on_sbs(frame)
                    elif which == "fused_annotated":
                        raw_l = self._frame_left
                        raw_r = self._frame_right
                        if raw_l is not None and raw_r is not None:
                            frame = self.annotations.render_on_fused(raw_l, raw_r)
                        else:
                            frame = None
                    else:
                        if which == "sbs":
                            raw = self._frame_sbs
                        elif which == "left":
                            raw = self._frame_left
                        else:
                            raw = self._frame_right
                        # Snapshot: copy to avoid race with main-loop buffer.
                        frame = raw.copy() if raw is not None else None

                # ── Compose anaglyph outside the lock ──
                if which == "anaglyph":
                    if left is None or right is None:
                        time.sleep(0.05)
                        continue
                    # True anaglyph: left → red, right → cyan (grayscale)
                    frame = np.empty_like(left)
                    lg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                    rg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                    frame[:, :, 2] = lg   # Red   ← left eye
                    frame[:, :, 1] = rg   # Green ← right eye
                    frame[:, :, 0] = rg   # Blue  ← right eye

                if frame is None:
                    time.sleep(0.05)
                    continue

                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]
                )
                if not ok:
                    time.sleep(0.02)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
                # Cap stream frame-rate to ~30 fps to save bandwidth
                time.sleep(0.033)
        finally:
            self._client_count -= 1

    def generate_fused_3d(self):
        """Generate frames for the Fused 3D simulation (optimized for speed and smoothness)."""
        import time
        while True:
            t_start = time.time()
            with self._lock:
                if self.app.cam_l is None or self.app.cam_r is None:
                    time.sleep(0.01)
                    continue

                frame_l = self.app.cam_l.read()
                frame_r = self.app.cam_r.read()

                if frame_l is None or frame_r is None:
                    time.sleep(0.01)
                    continue

                # Apply horizontal offset from slider to right frame
                rows, cols, _ = frame_r.shape
                offset = getattr(self.cfg, 'alignment_offset', 0)
                translation_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
                aligned_frame_r = cv2.warpAffine(frame_r, translation_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

                # Fuse frames in color (simple average)
                fused_frame = cv2.addWeighted(frame_l, 0.5, aligned_frame_r, 0.5, 0)

                # Encode the frame as JPEG
                success, buffer = cv2.imencode('.jpg', fused_frame)
                if not success:
                    time.sleep(0.01)
                    continue

                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # Throttle to avoid CPU overload and allow smoother streaming
            t_elapsed = time.time() - t_start
            if t_elapsed < 0.01:
                time.sleep(0.01 - t_elapsed)
