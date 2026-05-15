"""
Local HTML trace viewer — run with:
    python -m uvicorn trace_viewer:app --port 8001 --reload
Then open http://localhost:8001
"""

import json
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from trace_store import TraceStore

app = FastAPI(title="LangGraph Trace Viewer")
_HERE = os.path.dirname(os.path.abspath(__file__))
store = TraceStore(db_path=os.path.join(_HERE, "data", "traces.db"))


@app.get("/api/runs")
def get_runs(limit: int = 100):
    return store.get_runs(limit=limit)


@app.get("/api/runs/{run_id}/traces")
def get_node_traces(run_id: str):
    return store.get_node_traces(run_id)


@app.get("/api/token-stats")
def get_token_stats():
    return store.get_token_stats()


@app.get("/", response_class=HTMLResponse)
def trace_viewer():
    return HTMLResponse(content=VIEWER_HTML)


VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LangGraph Trace Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  h1 { padding: 16px 24px; font-size: 18px; border-bottom: 1px solid #2d3748; background: #1a1f2e; color: #90cdf4; }
  .layout { display: flex; height: calc(100vh - 53px); }
  .sidebar { width: 340px; min-width: 240px; border-right: 1px solid #2d3748; overflow-y: auto; background: #141820; }
  .main { flex: 1; overflow-y: auto; padding: 20px; }
  .run-item { padding: 12px 16px; border-bottom: 1px solid #1e2535; cursor: pointer; transition: background 0.15s; }
  .run-item:hover { background: #1e2535; }
  .run-item.active { background: #1e2a45; border-left: 3px solid #4299e1; }
  .run-query { font-size: 13px; color: #e2e8f0; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .run-meta { font-size: 11px; color: #718096; display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
  .badge { padding: 1px 7px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .badge-complete { background: #1c4532; color: #68d391; }
  .badge-degraded { background: #44337a; color: #d6bcfa; }
  .badge-blocked  { background: #742a2a; color: #fc8181; }
  .badge-failed   { background: #742a2a; color: #fc8181; }
  .badge-unknown  { background: #2d3748; color: #a0aec0; }
  .section-title { font-size: 13px; font-weight: 600; color: #90cdf4; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .timeline { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
  .node-box { padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500; border: 1px solid transparent; transition: all 0.15s; min-width: 80px; text-align: center; }
  .node-box:hover { filter: brightness(1.2); }
  .node-box.success { background: #1c4532; color: #68d391; border-color: #276749; }
  .node-box.error   { background: #742a2a; color: #fc8181; border-color: #9b2c2c; }
  .node-box.pending { background: #2d3748; color: #a0aec0; border-color: #4a5568; }
  .node-box.selected { outline: 2px solid #4299e1; }
  .node-duration { font-size: 10px; color: #718096; margin-top: 2px; }
  .node-tokens  { font-size: 10px; color: #805ad5; margin-top: 1px; }
  .detail-panel { background: #1a1f2e; border-radius: 8px; padding: 16px; margin-top: 8px; font-size: 13px; }
  .detail-panel h3 { font-size: 14px; color: #90cdf4; margin-bottom: 10px; }
  .kv-row { display: flex; gap: 12px; padding: 4px 0; border-bottom: 1px solid #2d3748; }
  .kv-key { color: #718096; min-width: 120px; font-size: 12px; }
  .kv-val { color: #e2e8f0; font-size: 12px; word-break: break-all; }
  .json-block { background: #0f1117; border-radius: 4px; padding: 10px; font-family: monospace; font-size: 11px; color: #a0aec0; overflow-x: auto; white-space: pre-wrap; max-height: 200px; overflow-y: auto; margin-top: 6px; }
  .tool-call { background: #1e2535; border-radius: 4px; padding: 8px; margin-top: 4px; }
  .tool-call-name { color: #f6ad55; font-weight: 600; font-size: 12px; }
  .empty { color: #4a5568; font-style: italic; padding: 40px; text-align: center; }
  .stat-bar { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
  .stat { background: #1a1f2e; border-radius: 6px; padding: 10px 16px; flex: 1; min-width: 120px; }
  .stat-label { font-size: 11px; color: #718096; margin-bottom: 4px; }
  .stat-value { font-size: 20px; font-weight: 700; color: #90cdf4; }
  .token-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
  .token-table th { text-align: left; padding: 6px 10px; color: #718096; border-bottom: 1px solid #2d3748; }
  .token-table td { padding: 6px 10px; border-bottom: 1px solid #1e2535; }
  .error-msg { color: #fc8181; font-size: 12px; margin-top: 4px; }
  #loading { color: #718096; padding: 20px; font-size: 13px; }
</style>
</head>
<body>
<h1>⛓ LangGraph Trace Viewer</h1>
<div class="layout">
  <div class="sidebar" id="run-list"><div id="loading">Loading runs...</div></div>
  <div class="main" id="main-panel">
    <div class="empty">Select a run from the left panel to view its trace.</div>
  </div>
</div>
<script>
const API = '';
let allRuns = [];
let selectedRun = null;
let selectedNode = null;

async function fetchJSON(url) {
  const r = await fetch(API + url);
  return r.json();
}

function badge(status) {
  const s = (status || 'unknown').toLowerCase();
  return `<span class="badge badge-${s}">${s}</span>`;
}

function formatMs(ms) {
  if (!ms) return '-';
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms/1000).toFixed(1)}s`;
}

function renderRunList(runs) {
  const el = document.getElementById('run-list');
  if (!runs.length) { el.innerHTML = '<div class="empty">No runs recorded yet.</div>'; return; }
  el.innerHTML = runs.map((r, i) => `
    <div class="run-item" onclick="selectRun('${r.run_id}', this)">
      <div class="run-query">${escHtml(r.user_query || '(no query)')}</div>
      <div class="run-meta">
        ${badge(r.final_status)}
        <span>${formatMs(r.duration_ms)}</span>
        <span>${(r.total_tokens || 0).toLocaleString()} tok</span>
        <span>${(r.start_time || '').replace('T', ' ').slice(0, 16)}</span>
      </div>
    </div>
  `).join('');
}

async function selectRun(runId, el) {
  document.querySelectorAll('.run-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');
  selectedRun = runId;
  selectedNode = null;

  const [run, traces, tokenStats] = await Promise.all([
    Promise.resolve(allRuns.find(r => r.run_id === runId)),
    fetchJSON(`/api/runs/${runId}/traces`),
    fetchJSON('/api/token-stats'),
  ]);

  renderMainPanel(run, traces, tokenStats);
}

function renderMainPanel(run, traces, tokenStats) {
  const panel = document.getElementById('main-panel');
  if (!run) { panel.innerHTML = '<div class="empty">Run not found.</div>'; return; }

  const totalTok = traces.reduce((s, t) => s + (t.total_tokens || 0), 0);
  const totalMs  = traces.reduce((s, t) => s + (t.duration_ms  || 0), 0);

  panel.innerHTML = `
    <div class="section-title">Run Summary</div>
    <div class="stat-bar">
      <div class="stat"><div class="stat-label">Status</div><div class="stat-value" style="font-size:14px">${badge(run.final_status)}</div></div>
      <div class="stat"><div class="stat-label">Nodes</div><div class="stat-value">${traces.length}</div></div>
      <div class="stat"><div class="stat-label">Duration</div><div class="stat-value">${formatMs(run.duration_ms || totalMs)}</div></div>
      <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value">${(run.total_tokens || totalTok).toLocaleString()}</div></div>
    </div>

    <div class="section-title">Trajectory</div>
    <div class="timeline" id="timeline">
      ${traces.map((t, i) => `
        <div class="node-box ${t.status || 'pending'}" onclick="selectNode(${i})" id="nb-${i}">
          <div>${escHtml(t.node_name)}</div>
          <div class="node-duration">${formatMs(t.duration_ms)}</div>
          ${t.total_tokens ? `<div class="node-tokens">${t.total_tokens} tok</div>` : ''}
        </div>
      `).join('')}
    </div>

    <div id="node-detail"></div>

    <div class="section-title" style="margin-top:24px">Token Usage by Node (all runs)</div>
    <table class="token-table">
      <thead><tr><th>Node</th><th>Avg Input</th><th>Avg Output</th><th>Avg Total</th><th>Max</th><th>Calls</th></tr></thead>
      <tbody>
        ${tokenStats.map(s => `
          <tr>
            <td>${escHtml(s.node_name)}</td>
            <td>${s.avg_input_tokens}</td>
            <td>${s.avg_output_tokens}</td>
            <td>${s.avg_total_tokens}</td>
            <td>${s.max_total_tokens}</td>
            <td>${s.call_count}</td>
          </tr>
        `).join('') || '<tr><td colspan="6" style="color:#4a5568">No LLM traces yet.</td></tr>'}
      </tbody>
    </table>
  `;

  window._traces = traces;
}

function selectNode(idx) {
  document.querySelectorAll('.node-box').forEach(e => e.classList.remove('selected'));
  const nb = document.getElementById(`nb-${idx}`);
  if (nb) nb.classList.add('selected');

  const t = window._traces[idx];
  if (!t) return;

  const detail = document.getElementById('node-detail');
  const toolCallsHtml = (t.tool_calls || []).map(tc => `
    <div class="tool-call">
      <div class="tool-call-name">⚙ ${escHtml(tc.name || '')}</div>
      <div class="json-block">${escHtml(JSON.stringify(tc.args || {}, null, 2))}</div>
    </div>
  `).join('');

  detail.innerHTML = `
    <div class="detail-panel">
      <h3>${escHtml(t.node_name)} <span style="font-weight:400;color:#718096">(node ${t.node_index})</span></h3>
      <div class="kv-row"><span class="kv-key">Status</span><span class="kv-val">${badge(t.status)}${t.error_message ? `<div class="error-msg">${escHtml(t.error_message)}</div>` : ''}</span></div>
      <div class="kv-row"><span class="kv-key">Duration</span><span class="kv-val">${formatMs(t.duration_ms)}</span></div>
      <div class="kv-row"><span class="kv-key">Model</span><span class="kv-val">${t.model_name || '-'}</span></div>
      <div class="kv-row"><span class="kv-key">Tokens</span><span class="kv-val">in: ${t.input_tokens ?? '-'} / out: ${t.output_tokens ?? '-'} / total: ${t.total_tokens ?? '-'}</span></div>
      <div class="kv-row"><span class="kv-key">Started</span><span class="kv-val">${(t.start_time || '').replace('T', ' ').slice(0, 19)} UTC</span></div>

      ${toolCallsHtml ? `<div style="margin-top:10px"><div class="kv-key" style="margin-bottom:4px">Tool calls:</div>${toolCallsHtml}</div>` : ''}

      <div style="margin-top:12px">
        <div class="kv-key" style="margin-bottom:4px">Output state:</div>
        <div class="json-block">${escHtml(JSON.stringify(t.output_snapshot || {}, null, 2))}</div>
      </div>
    </div>
  `;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function init() {
  try {
    allRuns = await fetchJSON('/api/runs?limit=100');
    renderRunList(allRuns);
  } catch(e) {
    document.getElementById('run-list').innerHTML = '<div class="empty">Could not connect to API.<br>Is the server running?</div>';
  }
}

init();
setInterval(async () => {
  try {
    allRuns = await fetchJSON('/api/runs?limit=100');
    renderRunList(allRuns);
  } catch(e) {}
}, 30000);
</script>
</body>
</html>"""