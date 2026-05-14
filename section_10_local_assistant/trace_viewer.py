"""
Standalone trace viewer — run separately:
    python -m uvicorn trace_viewer:app --port 8001 --reload
"""

import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from infrastructure.trace_store import TraceStore

app   = FastAPI(title="LangGraph Trace Viewer")
store = TraceStore()


@app.get("/api/runs")
def get_runs(limit: int = 100):
    return store.get_runs(limit=limit)

@app.get("/api/runs/{run_id}/traces")
def get_traces(run_id: str):
    return store.get_node_traces(run_id)

@app.get("/api/token-stats")
def token_stats():
    return store.get_token_stats()

@app.get("/", response_class=HTMLResponse)
def viewer():
    return HTMLResponse(VIEWER_HTML)


VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trace Viewer</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}
  h1{padding:14px 22px;font-size:16px;border-bottom:1px solid #2d3748;background:#1a1f2e;color:#90cdf4}
  .layout{display:flex;height:calc(100vh - 49px)}
  .sidebar{width:330px;border-right:1px solid #2d3748;overflow-y:auto;background:#141820}
  .main{flex:1;overflow-y:auto;padding:20px}
  .run-item{padding:11px 15px;border-bottom:1px solid #1e2535;cursor:pointer;transition:background .15s}
  .run-item:hover{background:#1e2535}
  .run-item.active{background:#1e2a45;border-left:3px solid #4299e1}
  .run-q{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px}
  .run-meta{font-size:11px;color:#718096;display:flex;gap:8px;flex-wrap:wrap}
  .badge{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
  .badge-complete{background:#1c4532;color:#68d391}
  .badge-degraded{background:#44337a;color:#d6bcfa}
  .badge-blocked,.badge-failed{background:#742a2a;color:#fc8181}
  .badge-unknown{background:#2d3748;color:#a0aec0}
  .sec{font-size:12px;font-weight:600;color:#90cdf4;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}
  .stat-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}
  .stat{background:#1a1f2e;border-radius:6px;padding:9px 15px;flex:1;min-width:100px}
  .stat-l{font-size:11px;color:#718096;margin-bottom:3px}
  .stat-v{font-size:20px;font-weight:700;color:#90cdf4}
  .timeline{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:20px}
  .nb{padding:7px 11px;border-radius:6px;cursor:pointer;font-size:11px;font-weight:500;border:1px solid transparent;text-align:center;transition:filter .15s}
  .nb:hover{filter:brightness(1.2)}
  .nb.success{background:#1c4532;color:#68d391;border-color:#276749}
  .nb.error{background:#742a2a;color:#fc8181;border-color:#9b2c2c}
  .nb.pending{background:#2d3748;color:#a0aec0;border-color:#4a5568}
  .nb.selected{outline:2px solid #4299e1}
  .nb-dur{font-size:10px;color:#718096;margin-top:1px}
  .nb-tok{font-size:10px;color:#805ad5;margin-top:1px}
  .detail{background:#1a1f2e;border-radius:8px;padding:15px;margin-top:8px;font-size:13px}
  .detail h3{font-size:14px;color:#90cdf4;margin-bottom:10px}
  .kv{display:flex;gap:10px;padding:4px 0;border-bottom:1px solid #2d3748}
  .kk{color:#718096;min-width:120px;font-size:12px}
  .kv_{color:#e2e8f0;font-size:12px;word-break:break-all}
  .jb{background:#0f1117;border-radius:4px;padding:9px;font-family:monospace;font-size:11px;color:#a0aec0;overflow-x:auto;white-space:pre-wrap;max-height:200px;overflow-y:auto;margin-top:5px}
  .tc{background:#1e2535;border-radius:4px;padding:7px;margin-top:4px}
  .tc-name{color:#f6ad55;font-weight:600;font-size:12px}
  .empty{color:#4a5568;font-style:italic;padding:40px;text-align:center}
  .tok-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
  .tok-table th{text-align:left;padding:5px 9px;color:#718096;border-bottom:1px solid #2d3748}
  .tok-table td{padding:5px 9px;border-bottom:1px solid #1e2535}
  .err{color:#fc8181;font-size:12px;margin-top:3px}
</style>
</head>
<body>
<h1>⛓ LangGraph Trace Viewer</h1>
<div class="layout">
  <div class="sidebar" id="sidebar"><div class="empty">Loading…</div></div>
  <div class="main"   id="main"><div class="empty">Select a run to view its trace.</div></div>
</div>
<script>
let allRuns=[];
async function api(u){const r=await fetch(u);return r.json()}
function badge(s){const c=(s||'unknown').toLowerCase();return`<span class="badge badge-${c}">${c}</span>`}
function ms(v){if(!v)return'-';return v<1000?Math.round(v)+'ms':(v/1000).toFixed(1)+'s'}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

function renderSidebar(runs){
  const el=document.getElementById('sidebar');
  if(!runs.length){el.innerHTML='<div class="empty">No runs yet.</div>';return}
  el.innerHTML=runs.map(r=>`
    <div class="run-item" onclick="selectRun('${r.run_id}',this)">
      <div class="run-q">${esc(r.user_query||'(no query)')}</div>
      <div class="run-meta">${badge(r.final_status)}<span>${ms(r.duration_ms)}</span><span>${(r.total_tokens||0).toLocaleString()} tok</span><span>${(r.start_time||'').slice(0,16).replace('T',' ')}</span></div>
    </div>`).join('')
}

async function selectRun(id,el){
  document.querySelectorAll('.run-item').forEach(e=>e.classList.remove('active'));
  if(el)el.classList.add('active');
  const run=allRuns.find(r=>r.run_id===id);
  const [traces,stats]=await Promise.all([api(`/api/runs/${id}/traces`),api('/api/token-stats')]);
  renderMain(run,traces,stats);
}

function renderMain(run,traces,stats){
  const p=document.getElementById('main');
  if(!run){p.innerHTML='<div class="empty">Run not found.</div>';return}
  const totalTok=traces.reduce((s,t)=>s+(t.total_tokens||0),0);
  p.innerHTML=`
    <div class="sec">Run Summary</div>
    <div class="stat-row">
      <div class="stat"><div class="stat-l">Status</div><div class="stat-v" style="font-size:14px">${badge(run.final_status)}</div></div>
      <div class="stat"><div class="stat-l">Nodes</div><div class="stat-v">${traces.length}</div></div>
      <div class="stat"><div class="stat-l">Duration</div><div class="stat-v">${ms(run.duration_ms)}</div></div>
      <div class="stat"><div class="stat-l">Tokens</div><div class="stat-v">${(run.total_tokens||totalTok).toLocaleString()}</div></div>
    </div>
    <div class="sec">Trajectory</div>
    <div class="timeline" id="tl">
      ${traces.map((t,i)=>`
        <div class="nb ${t.status||'pending'}" onclick="selectNode(${i})" id="nb-${i}">
          <div>${esc(t.node_name)}</div>
          <div class="nb-dur">${ms(t.duration_ms)}</div>
          ${t.total_tokens?`<div class="nb-tok">${t.total_tokens} tok</div>`:''}
        </div>`).join('')}
    </div>
    <div id="nd"></div>
    <div class="sec" style="margin-top:22px">Token Usage by Node (all runs)</div>
    <table class="tok-table">
      <thead><tr><th>Node</th><th>Avg In</th><th>Avg Out</th><th>Avg Total</th><th>Max</th><th>Calls</th></tr></thead>
      <tbody>${stats.map(s=>`<tr><td>${esc(s.node_name)}</td><td>${s.avg_input_tokens}</td><td>${s.avg_output_tokens}</td><td>${s.avg_total_tokens}</td><td>${s.max_total_tokens}</td><td>${s.call_count}</td></tr>`).join('')||'<tr><td colspan="6" style="color:#4a5568">No LLM traces yet.</td></tr>'}</tbody>
    </table>`;
  window._traces=traces;
}

function selectNode(i){
  document.querySelectorAll('.nb').forEach(e=>e.classList.remove('selected'));
  const nb=document.getElementById('nb-'+i);if(nb)nb.classList.add('selected');
  const t=window._traces[i];if(!t)return;
  const tcs=(t.tool_calls||[]).map(tc=>`<div class="tc"><div class="tc-name">⚙ ${esc(tc.name||'')}</div><div class="jb">${esc(JSON.stringify(tc.args||{},null,2))}</div></div>`).join('');
  document.getElementById('nd').innerHTML=`
    <div class="detail">
      <h3>${esc(t.node_name)} <span style="font-weight:400;color:#718096">(#${t.node_index})</span></h3>
      <div class="kv"><span class="kk">Status</span><span class="kv_">${badge(t.status)}${t.error_message?`<div class="err">${esc(t.error_message)}</div>`:''}</span></div>
      <div class="kv"><span class="kk">Duration</span><span class="kv_">${ms(t.duration_ms)}</span></div>
      <div class="kv"><span class="kk">Model</span><span class="kv_">${t.model_name||'-'}</span></div>
      <div class="kv"><span class="kk">Tokens</span><span class="kv_">in: ${t.input_tokens??'-'} / out: ${t.output_tokens??'-'} / total: ${t.total_tokens??'-'}</span></div>
      <div class="kv"><span class="kk">Started</span><span class="kv_">${(t.start_time||'').replace('T',' ').slice(0,19)} UTC</span></div>
      ${tcs?`<div style="margin-top:9px"><div class="kk" style="margin-bottom:3px">Tool calls:</div>${tcs}</div>`:''}
      <div style="margin-top:11px"><div class="kk" style="margin-bottom:3px">Output state:</div><div class="jb">${esc(JSON.stringify(t.output_snapshot||{},null,2))}</div></div>
    </div>`;
}

async function init(){
  try{allRuns=await api('/api/runs?limit=100');renderSidebar(allRuns)}
  catch(e){document.getElementById('sidebar').innerHTML='<div class="empty">API unavailable.</div>'}
}
init();
setInterval(async()=>{try{allRuns=await api('/api/runs?limit=100');renderSidebar(allRuns)}catch(e){}},30000);
</script>
</body>
</html>"""