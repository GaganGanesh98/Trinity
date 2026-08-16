"""
FastAPI wrapper around the RAG query engine.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /              — landing page
    POST /query         — {"question": "..."} → {"answer": "...", "sources": [...]}
    GET  /graph/stats   — Neo4j GraphRAG node/edge counts
    GET  /health        — liveness probe
    GET  /docs          — modern API reference (Scalar)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

import requests

from nis2.assessor import assess
from nis2.checkpoints import CHECKPOINTS
from nis2.fetch import FetchError, fetch_document
from nis2.scope import ScopeInput, determine, sectors
from rag.indexer import load_index, query, query_with_sources

logger = logging.getLogger(__name__)

_STARTED_AT = time.time()
_INDEX_READY = False


# ── Lifespan: warm up the index on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _INDEX_READY
    logger.info("Loading RAG index on startup...")
    try:
        load_index()
        _INDEX_READY = True
        logger.info("Index loaded.")
    except FileNotFoundError:
        logger.warning(
            "No persisted index found — run `python main.py --rebuild-index` first."
        )
    yield


app = FastAPI(
    title="ENISA RAG Agent API",
    version="0.2.0",
    description="Query ENISA cybersecurity publications via hybrid RAG.",
    lifespan=lifespan,
    docs_url=None,   # replaced by the custom Scalar reference below
    redoc_url=None,
)

# Local/demo API — wide open CORS so the docs page and any frontend can call it
# from a different origin/port. Tighten via an explicit allow-list before any
# real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    use_hybrid: bool = Field(True, description="Use BM25+vector hybrid retrieval")
    include_sources: bool = Field(True, description="Return source citations")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the top ransomware trends in the 2024 threat landscape?",
                "use_hybrid": True,
                "include_sources": True,
            }
        }
    )


class SourceInfo(BaseModel):
    title: str
    source_url: str
    score: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "ENISA Threat Landscape 2024",
                "source_url": "https://www.enisa.europa.eu/publications/enisa-threat-landscape-2024",
                "score": 0.8421,
            }
        }
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": (
                    "Ransomware remained the top threat in 2024, with double extortion "
                    "and RaaS models driving most incidents..."
                ),
                "sources": [
                    {
                        "title": "ENISA Threat Landscape 2024",
                        "source_url": "https://www.enisa.europa.eu/publications/enisa-threat-landscape-2024",
                        "score": 0.8421,
                    }
                ],
            }
        }
    )


class GraphStatsResponse(BaseModel):
    nodes_total: int
    relationships_total: int
    nodes_by_type: dict[str, int]
    relationships_by_type: dict[str, int]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nodes_total": 842,
                "relationships_total": 1310,
                "nodes_by_type": {"ThreatActor": 61, "Technique": 214, "Malware": 97},
                "relationships_by_type": {"USES": 402, "TARGETS": 288, "MITIGATES": 176},
            }
        }
    )


# ── Docs & landing page ───────────────────────────────────────────────────────

@app.get("/docs", include_in_schema=False)
async def api_reference() -> HTMLResponse:
    """Modern API reference (Scalar), served from the auto-generated OpenAPI schema."""
    return HTMLResponse(
        """
        <!doctype html>
        <html>
          <head>
            <title>ENISA RAG Agent API — Reference</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <style>body{margin:0;background:#0b0f19;}</style>
          </head>
          <body>
            <script id="api-reference" data-url="/openapi.json"></script>
            <script>
              document.getElementById('api-reference').dataset.configuration =
                JSON.stringify({ theme: 'purple', darkMode: true, layout: 'modern' });
            </script>
            <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
          </body>
        </html>
        """
    )


_LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ENISA RAG Agent</title>
  <style>
    :root{
      --bg:#0a0e1a; --panel:rgba(20,26,40,.72); --panel-solid:#121826;
      --border:rgba(255,255,255,.08); --border-hi:rgba(139,92,246,.5);
      --text:#e8ebf2; --muted:#9aa4b8; --faint:#6b7488;
      --accent:#8b5cf6; --accent2:#22d3ee; --good:#22c55e; --warn:#f59e0b; --bad:#ef4444;
      --radius:16px;
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0; min-height:100vh; color:var(--text); line-height:1.5;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
      -webkit-font-smoothing:antialiased;
      background:
        radial-gradient(1100px 560px at 12% -12%, rgba(139,92,246,.22), transparent 60%),
        radial-gradient(900px 520px at 112% 4%, rgba(34,211,238,.16), transparent 58%),
        var(--bg);
      background-attachment:fixed;
    }
    a{color:inherit}
    .wrap{max-width:940px; margin:0 auto; padding:0 22px}
    /* nav */
    nav{
      position:sticky; top:0; z-index:20; backdrop-filter:blur(12px);
      background:rgba(10,14,26,.6); border-bottom:1px solid var(--border);
    }
    nav .inner{display:flex; align-items:center; justify-content:space-between; height:58px}
    .brand{display:flex; align-items:center; gap:10px; font-weight:700; letter-spacing:.2px}
    .logo{width:26px;height:26px;border-radius:8px;
      background:conic-gradient(from 210deg,var(--accent),var(--accent2),var(--accent));
      box-shadow:0 0 18px rgba(139,92,246,.5)}
    .navlinks{display:flex; gap:6px; align-items:center}
    .navlinks a{font-size:13.5px;color:var(--muted);text-decoration:none;padding:8px 12px;border-radius:9px}
    .navlinks a:hover{color:var(--text);background:rgba(255,255,255,.05)}
    /* hero */
    header{padding:68px 0 34px}
    .pill{display:inline-flex;align-items:center;gap:9px;font-size:12.5px;color:var(--muted);
      background:var(--panel);border:1px solid var(--border);padding:6px 13px;border-radius:999px;margin-bottom:22px}
    .dot{width:8px;height:8px;border-radius:50%;background:var(--warn);box-shadow:0 0 10px var(--warn);transition:.3s}
    h1{font-size:clamp(34px,6vw,52px);line-height:1.08;margin:0 0 16px;letter-spacing:-.02em;font-weight:800;
      background:linear-gradient(92deg,#fff 20%,#c9b8ff 55%,var(--accent2));
      -webkit-background-clip:text;background-clip:text;color:transparent}
    p.lead{color:var(--muted);font-size:17px;max-width:640px;margin:0 0 28px}
    .cta{display:flex;gap:11px;flex-wrap:wrap}
    .btn{padding:11px 18px;border-radius:11px;text-decoration:none;font-size:14px;font-weight:600;
      border:1px solid var(--border);display:inline-flex;align-items:center;gap:7px;cursor:pointer;
      transition:transform .14s ease,border-color .14s ease,background .14s ease}
    .btn:hover{transform:translateY(-1px);border-color:var(--border-hi)}
    .btn.primary{background:linear-gradient(92deg,var(--accent),var(--accent2));color:#08101c;border:none}
    .btn.primary:hover{filter:brightness(1.06)}
    .btn.ghost{background:var(--panel);color:var(--text)}
    /* stat tiles */
    .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:38px 0 8px}
    .tile{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:15px 16px}
    .tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin-bottom:7px}
    .tile .v{font-size:22px;font-weight:750;font-variant-numeric:tabular-nums}
    .tile .v small{font-size:12px;color:var(--muted);font-weight:500}
    /* console */
    section{padding:30px 0}
    .h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin:0 0 14px;font-weight:700}
    .console{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
      padding:18px;box-shadow:0 20px 60px -30px rgba(0,0,0,.8)}
    .ask{display:flex;gap:10px;align-items:flex-end}
    textarea{flex:1;resize:vertical;min-height:52px;max-height:200px;background:#0c1120;color:var(--text);
      border:1px solid var(--border);border-radius:12px;padding:13px 15px;font-size:15px;font-family:inherit;line-height:1.45}
    textarea:focus{outline:none;border-color:var(--border-hi);box-shadow:0 0 0 3px rgba(139,92,246,.15)}
    .examples{display:flex;gap:8px;flex-wrap:wrap;margin:13px 0 2px}
    .ex{font-size:12.5px;color:var(--muted);background:rgba(255,255,255,.04);border:1px solid var(--border);
      padding:6px 11px;border-radius:999px;cursor:pointer;transition:.14s}
    .ex:hover{color:var(--text);border-color:var(--border-hi);background:rgba(139,92,246,.1)}
    #answer{margin-top:18px;border-top:1px solid var(--border);padding-top:18px;display:none}
    #answer.show{display:block;animation:fade .3s ease}
    @keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
    .ansMeta{font-size:12px;color:var(--faint);margin-bottom:8px;display:flex;gap:10px;align-items:center}
    .ansBody{white-space:pre-wrap;font-size:15px;color:#eef1f7}
    .loading{display:flex;align-items:center;gap:11px;color:var(--muted);font-size:14px}
    .spin{width:15px;height:15px;border:2px solid var(--border);border-top-color:var(--accent2);border-radius:50%;animation:sp .8s linear infinite}
    @keyframes sp{to{transform:rotate(360deg)}}
    .err{color:#fca5a5;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);padding:12px 14px;border-radius:10px;font-size:13.5px}
    .srcTitle{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin:18px 0 9px;font-weight:700}
    .src{display:flex;justify-content:space-between;align-items:center;gap:12px;text-decoration:none;
      background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:10px;padding:10px 13px;margin-bottom:7px;transition:.14s}
    .src:hover{border-color:var(--border-hi);background:rgba(139,92,246,.07)}
    .src .t{font-size:13.5px;color:var(--text)}
    .src .score{font-size:12px;color:var(--accent2);font-variant-numeric:tabular-nums;background:rgba(34,211,238,.1);padding:3px 9px;border-radius:999px;white-space:nowrap}
    /* endpoint + tech */
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px 17px;transition:.14s}
    .card:hover{border-color:var(--border-hi);transform:translateY(-2px)}
    .card .m{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.04em;padding:2px 7px;border-radius:6px;margin-bottom:9px}
    .m.post{background:rgba(34,197,94,.15);color:#4ade80}
    .m.get{background:rgba(34,211,238,.15);color:#67e8f9}
    .card code{color:var(--text);font-size:14px}
    .card p{margin:8px 0 0;color:var(--muted);font-size:13px}
    .chips{display:flex;flex-wrap:wrap;gap:8px}
    .chip{font-size:12px;padding:6px 12px;border-radius:999px;background:var(--panel);border:1px solid var(--border);color:var(--muted)}
    footer{border-top:1px solid var(--border);padding:26px 0 60px;color:var(--faint);font-size:12.5px;
      display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}
    footer a{color:var(--accent2);text-decoration:none}
    @media(max-width:680px){.stats{grid-template-columns:repeat(2,1fr)}.ask{flex-direction:column}.ask .btn{width:100%;justify-content:center}}
  </style>
</head>
<body>
  <nav><div class="wrap inner">
    <div class="brand"><span class="logo"></span> ENISA RAG Agent</div>
    <div class="navlinks">
      <a href="/docs">API Reference</a>
      <a href="/graph/stats">Graph</a>
      <a href="https://github.com/GaganGanesh98" target="_blank" rel="noopener">GitHub</a>
    </div>
  </div></nav>

  <div class="wrap">
    <header>
      <span class="pill"><span class="dot" id="dot"></span><span id="statusText">connecting…</span><span style="color:var(--faint)">·</span><span id="uptime">v0.2.0</span></span>
      <h1>Ask the ENISA<br>cybersecurity corpus</h1>
      <p class="lead">Hybrid retrieval-augmented QA over ENISA publications — BM25 + vector fusion,
        cross-encoder reranking, and a Neo4j knowledge-graph layer. Runs fully local via Ollama, no API keys.</p>
      <div class="cta">
        <a class="btn primary" href="#console">Try it live ↓</a>
        <a class="btn ghost" href="/docs">API Reference</a>
        <a class="btn ghost" href="/openapi.json">OpenAPI</a>
      </div>

      <div class="stats">
        <div class="tile"><div class="k">Vector index</div><div class="v" id="tIndex">—</div></div>
        <div class="tile"><div class="k">Graph nodes</div><div class="v" id="tNodes">—</div></div>
        <div class="tile"><div class="k">Graph edges</div><div class="v" id="tEdges">—</div></div>
        <div class="tile"><div class="k">Retrieval</div><div class="v">Hybrid <small>+graph</small></div></div>
      </div>
    </header>

    <section id="console">
      <div class="h2">Live query console</div>
      <div class="console">
        <div class="ask">
          <textarea id="q" placeholder="e.g. What are the top ransomware trends in the ENISA Threat Landscape 2024?"></textarea>
          <button class="btn primary" id="askBtn" onclick="ask()">Ask →</button>
        </div>
        <div class="examples" id="examples"></div>
        <div id="answer">
          <div class="ansMeta" id="ansMeta"></div>
          <div class="ansBody" id="ansBody"></div>
          <div id="sources"></div>
        </div>
      </div>
    </section>

    <section>
      <div class="h2">Endpoints</div>
      <div class="grid">
        <div class="card"><span class="m post">POST</span> <code>/query</code><p>Ask a question, get an answer with cited sources.</p></div>
        <div class="card"><span class="m get">GET</span> <code>/graph/stats</code><p>Node/edge counts from the GraphRAG knowledge graph.</p></div>
        <div class="card"><span class="m get">GET</span> <code>/health</code><p>Liveness probe + index status.</p></div>
      </div>
    </section>

    <section>
      <div class="h2">Stack</div>
      <div class="chips">
        <span class="chip">FastAPI</span><span class="chip">LlamaIndex</span><span class="chip">Ollama · llama3.2</span>
        <span class="chip">BM25 + Vector fusion</span><span class="chip">Cross-encoder rerank</span>
        <span class="chip">Neo4j GraphRAG</span><span class="chip">qwen2.5 extraction</span>
      </div>
    </section>

    <footer>
      <span>Built by Gagan Ganesh · local-first energy &amp; security AI</span>
      <span><a href="https://github.com/GaganGanesh98" target="_blank" rel="noopener">github.com/GaganGanesh98</a></span>
    </footer>
  </div>

  <script>
    const $ = s => document.querySelector(s);
    const EXAMPLES = [
      "What are the top ransomware trends in the ENISA Threat Landscape 2024?",
      "What does NIS2 require for technical cybersecurity measures?",
      "Which cybersecurity threats affect the finance sector?",
    ];
    async function jget(url, opts){
      const r = await fetch(url, opts);
      const body = await r.json().catch(() => ({}));
      if(!r.ok) throw new Error(body.detail || ("HTTP " + r.status));
      return body;
    }
    async function loadHealth(){
      try{
        const h = await jget('/health');
        const ready = !!h.index_ready;
        $('#dot').style.background = ready ? 'var(--good)' : 'var(--warn)';
        $('#dot').style.boxShadow = '0 0 10px ' + (ready ? 'var(--good)' : 'var(--warn)');
        $('#statusText').textContent = ready ? 'index ready' : 'index not built';
        $('#uptime').textContent = 'up ' + h.uptime_seconds + 's';
        $('#tIndex').innerHTML = ready ? 'Ready' : '<small>run --rebuild-index</small>';
      }catch(e){
        $('#statusText').textContent = 'offline';
        $('#dot').style.background = 'var(--bad)';
      }
    }
    async function loadGraph(){
      try{
        const g = await jget('/graph/stats');
        $('#tNodes').textContent = g.nodes_total.toLocaleString();
        $('#tEdges').textContent = g.relationships_total.toLocaleString();
      }catch(e){
        $('#tNodes').innerHTML = '<small>not built</small>';
        $('#tEdges').innerHTML = '<small>—</small>';
      }
    }
    function renderExamples(){
      $('#examples').innerHTML = EXAMPLES.map((q,i) =>
        '<span class="ex" data-i="'+i+'">'+q+'</span>').join('');
      document.querySelectorAll('.ex').forEach(el =>
        el.onclick = () => { $('#q').value = EXAMPLES[el.dataset.i]; $('#q').focus(); });
    }
    async function ask(){
      const q = $('#q').value.trim();
      if(q.length < 3){ $('#q').focus(); return; }
      const btn = $('#askBtn'); btn.disabled = true; btn.textContent = 'Asking…';
      $('#answer').classList.add('show');
      $('#ansMeta').textContent = '';
      $('#sources').innerHTML = '';
      $('#ansBody').innerHTML = '<div class="loading"><span class="spin"></span>Thinking — the LLM runs locally via Ollama. The first (cold) answer can take a minute or two while the model loads; ~30–40s once warm.</div>';
      const t0 = performance.now();
      try{
        const res = await jget('/query', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ question:q, include_sources:true })
        });
        const secs = ((performance.now()-t0)/1000).toFixed(1);
        $('#ansMeta').innerHTML = '<span>✓ answered in '+secs+'s</span><span>· hybrid retrieval</span>';
        $('#ansBody').textContent = res.answer || '(empty answer)';
        if(res.sources && res.sources.length){
          $('#sources').innerHTML = '<div class="srcTitle">Sources</div>' + res.sources.map(s =>
            '<a class="src" href="'+(s.source_url||'#')+'" target="_blank" rel="noopener">'
            + '<span class="t">'+ (s.title||'Untitled') +'</span>'
            + '<span class="score">'+ (s.score!=null? s.score : '') +'</span></a>').join('');
        }
      }catch(e){
        $('#ansBody').innerHTML = '<div class="err">'+ e.message +'</div>';
      }finally{
        btn.disabled = false; btn.textContent = 'Ask →';
      }
    }
    $('#q').addEventListener('keydown', e => {
      if((e.metaKey||e.ctrlKey) && e.key === 'Enter') ask();
    });
    renderExamples(); loadHealth(); loadGraph();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing() -> HTMLResponse:
    """Interactive landing page. All dynamic data is fetched client-side from
    /health, /graph/stats, and /query, so this is a static (offline-friendly) shell."""
    return HTMLResponse(_LANDING_HTML)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    # Sync `def` on purpose: query_with_sources() is blocking and the hybrid
    # QueryFusionRetriever runs nested async internally. FastAPI runs sync path
    # operations in a threadpool, which both avoids blocking the event loop for
    # the ~30s call and lets the retriever's own asyncio.run() work.
    try:
        if req.include_sources:
            result = query_with_sources(req.question)
            return QueryResponse(
                answer=result["answer"],
                sources=[SourceInfo(**s) for s in result["sources"]],
            )
        else:
            answer = query(req.question, use_hybrid=req.use_hybrid)
            return QueryResponse(answer=answer)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready: {e}. Run `python main.py --rebuild-index`.",
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — policy documents are small


@app.post("/assess")
def handle_assess(
    file: UploadFile = File(..., description="Security policy to assess (PDF, TXT or MD)"),
    checkpoints: str | None = Form(
        None,
        description="Optional comma-separated checkpoint ids, e.g. 'NIS2-03,NIS2-05'. "
        "Omit to assess all thirteen.",
    ),
):
    """Assess an uploaded document against the NIS2 Article 21 requirement domains.

    Returns a per-domain report: status, severity, rationale, and a **verbatim**
    quote from the uploaded document supporting each finding.

    The upload is chunked and embedded in memory and discarded when the request
    ends — it is never written to the vector store, and embeddings and generation
    both run on the local Ollama server.

    Not legal advice: this flags gaps for a human to review, it does not certify
    compliance.
    """
    # Sync `def` for the same reason as /query — the assessment is a long
    # blocking call that FastAPI will run in its threadpool.
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    ids = [c.strip() for c in checkpoints.split(",") if c.strip()] if checkpoints else None
    try:
        report = assess(data, file.filename or "upload", checkpoint_ids=ids)
    except ValueError as e:
        # Unsupported type, unreadable file, or unknown checkpoint id — all of
        # these are the caller's input, not a server fault.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Assessment failed")
        raise HTTPException(status_code=500, detail=str(e))

    return report.model_dump()


class AssessUrlRequest(BaseModel):
    url: str = Field(description="Public link to a policy: PDF, text file, or HTML page.")
    checkpoints: list[str] | None = Field(
        None, description="Optional subset of checkpoint ids, e.g. ['NIS2-03']."
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"url": "https://example.com/security-policy.pdf"}}
    )


@app.post("/assess/url")
def handle_assess_url(req: AssessUrlRequest):
    """Assess a policy linked by URL instead of uploaded.

    Covers a published policy page, a direct PDF link, and "anyone with the link"
    shares from Google Drive/Docs, which are rewritten to their direct-download
    form. HTML pages are reduced to text before assessment.

    The fetch is restricted to public http(s) hosts: URLs resolving to loopback,
    private or link-local addresses are refused, redirects are re-validated at
    every hop, and the response is size-capped. As with upload, nothing is stored.
    """
    try:
        data, filename = fetch_document(req.url)
    except FetchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch the URL: {e}")

    try:
        report = assess(data, filename, checkpoint_ids=req.checkpoints)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Assessment failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {**report.model_dump(), "source_url": req.url}


@app.get("/assess/checkpoints")
def list_checkpoints():
    """The NIS2 requirement domains an assessment covers, with expected evidence."""
    return [
        {
            "id": c.id,
            "domain": c.domain,
            "article": c.article,
            "obligation": c.obligation,
            "evidence_expected": list(c.evidence),
        }
        for c in CHECKPOINTS
    ]


class ScopeRequest(BaseModel):
    """Answers to the scope questionnaire. Everything except sector is optional."""

    sector: str = Field(description="Annex I/II sector, e.g. 'health'. See GET /scope/sectors.")
    subsector: str | None = None
    employees: int | None = Field(None, ge=0)
    annual_turnover_eur: float | None = Field(None, ge=0)
    balance_sheet_eur: float | None = Field(None, ge=0)
    flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Art. 2(2) size-cap criteria, e.g. {'is_dns_or_tld_provider': true}.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sector": "health",
                "subsector": "healthcare providers",
                "employees": 800,
                "annual_turnover_eur": 90000000,
                "balance_sheet_eur": 70000000,
                "flags": {},
            }
        }
    )


@app.post("/scope")
def handle_scope(req: ScopeRequest):
    """Determine whether an entity falls in NIS2 scope, and as what.

    Rule-based end to end — no LLM is involved, so the answer is reproducible
    and every step cites the article it follows from. Nothing is stored.

    Indicative only, not legal advice: Member States may extend scope in national
    transposition, and Art. 2(2)(d)–(g) contain judgement-based tests that are
    returned as caveats rather than decided automatically.
    """
    result = determine(
        ScopeInput(
            sector=req.sector,
            subsector=req.subsector,
            employees=req.employees,
            annual_turnover_eur=req.annual_turnover_eur,
            balance_sheet_eur=req.balance_sheet_eur,
            flags=req.flags,
        )
    )
    return {
        "classification": result.classification.value,
        "in_scope": result.in_scope,
        "size": result.size.value if result.size else None,
        "annex": result.annex,
        "reasoning": result.reasoning,
        "caveats": result.caveats,
        "obligations": result.obligations,
        "disclaimer": result.disclaimer,
    }


@app.get("/scope/sectors")
def scope_sectors():
    """Annex I/II sectors, subsectors, and the Art. 2(2) size-cap criteria."""
    return sectors()


@app.get("/nis2", response_class=HTMLResponse, include_in_schema=False)
def nis2_ui() -> HTMLResponse:
    """Browser UI for the scope check and the policy assessment."""
    page = Path(__file__).parent / "static" / "nis2.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="UI asset missing: static/nis2.html")
    return HTMLResponse(page.read_text())


@app.get("/graph/stats", response_model=GraphStatsResponse)
def graph_stats():
    """Node/edge counts by type for the Neo4j knowledge graph (GraphRAG layer).

    Sync `def`: does blocking Neo4j I/O, so let FastAPI run it in the threadpool."""
    try:
        from rag.graph_indexer import get_graph_stats

        return GraphStatsResponse(**get_graph_stats())
    except Exception as e:
        logger.exception("Graph stats failed")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Graph unavailable: {e}. Ensure Neo4j is running "
                "(`docker compose up -d neo4j`) and built "
                "(`python main.py --rebuild-graph`)."
            ),
        )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": _INDEX_READY,
        "uptime_seconds": int(time.time() - _STARTED_AT),
    }
