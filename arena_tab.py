"""
arena_tab.py  ─  Sansad AI · Model Arena  (Tab 04)
====================================================
Head-to-head benchmark: runs the same inference + Q&A workflow
from app.py across four LLM providers and renders comparison charts.

INTEGRATION (already done in app.py — no manual edits needed):
    from arena_tab import render_arena_tab
    ...
    with tab4:
        render_arena_tab(ai_engine)
"""

from __future__ import annotations

import time
import re
import json
import statistics
import concurrent.futures
from dataclasses import dataclass, field

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from openai import OpenAI
import os
import io
from pathlib import Path
from datetime import datetime

# ─── SHARED CONFIG (mirrors Tab 2 exactly) ───────────────────────────────────

ARENA_MODELS: list[str] = [

    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/google/gemma-3-12b-it",
    "@hf/mistral/mistral-7b-instruct-v0.2",
    "@cf/ibm-granite/granite-4.0-h-micro",
]

MODEL_COLORS: dict[str, str] = {
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast":     "#4ECDC4",
    "@cf/meta/llama-4-scout-17b-16e-instruct":      "#26DE81",
    "@cf/google/gemma-3-12b-it":                    "#FED330",
    "@hf/mistral/mistral-7b-instruct-v0.2":         "#A55EEA",
    "@cf/ibm-granite/granite-4.0-h-micro":          "#45AAF2",
}

MODEL_LABELS: dict[str, str] = {
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast":     "Llama 3.3 · 70B",
    "@cf/meta/llama-4-scout-17b-16e-instruct":      "Llama 4 · 17B",
    "@cf/google/gemma-3-12b-it":                    "Gemma 3 · 12B",
    "@hf/mistral/mistral-7b-instruct-v0.2":         "Mistral · 7B",
    "@cf/ibm-granite/granite-4.0-h-micro":          "Granite 4.0 · Micro",
}

PROMPT_TYPES = ["summary", "timeline", "ministry", "gaps"]
PROMPT_LABELS = {
    "summary":  "Executive Summary",
    "timeline": "Progression Timeline",
    "ministry": "Ministry Engagement",
    "gaps":     "Policy Gaps",
}
INFERENCE_PROMPTS = {
    "summary":  "Based strictly on the provided parliamentary data, write a 3-paragraph executive summary of what has been discussed regarding this topic.",
    "timeline": "Analyze the policy progression. Identify key years and how the conversation evolved. Format as a markdown timeline.",
    "ministry": "Identify which government ministries are engaged in this topic and summarize their specific jurisdictions or responses.",
    "gaps":     "Identify 'Policy Gaps'. What questions are repeatedly asked? What issues remain unresolved based on these records?",
}
SYS_PROMPT = (
    "You are an expert AI public policy analyst. "
    "Ground all inferences strictly in the provided text to prevent hallucinations. "
    "Do not include reasoning traces, thinking tags, or chain-of-thought in your response."
)
SYS_PROMPT_QA = (
    "You are Sansad AI, an expert conversational public policy analyst. "
    "You have access to parliamentary records in <context> tags. "
    "If the context contains the answer, synthesize it clearly. "
    "If not, use general knowledge but state that explicitly."
)

_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0", family="'Fira Code', monospace"),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2D3139"),
)

# ─── DATA MODEL ──────────────────────────────────────────────────────────────

@dataclass
class ModelRun:
    model_id: str

    # Inference (4 prompt types)
    inf_texts:    dict[str, str]   = field(default_factory=dict)
    inf_latency:  dict[str, float] = field(default_factory=dict)
    inf_in_tok:   dict[str, int]   = field(default_factory=dict)
    inf_out_tok:  dict[str, int]   = field(default_factory=dict)

    # Q&A
    qa_results:  list[dict]  = field(default_factory=list)   # [{q, a, lat, out_tok}]

    errors: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.model_id]

    @property
    def color(self) -> str:
        return MODEL_COLORS[self.model_id]

    @property
    def avg_inf_lat(self) -> float:
        v = list(self.inf_latency.values())
        return statistics.mean(v) if v else 0.0

    @property
    def avg_qa_lat(self) -> float:
        v = [r["lat"] for r in self.qa_results if r.get("lat")]
        return statistics.mean(v) if v else 0.0

    @property
    def total_out_tok(self) -> int:
        return sum(self.inf_out_tok.values()) + sum(r.get("out_tok", 0) for r in self.qa_results)

    @property
    def avg_tps(self) -> float:
        pairs = (
            [(self.inf_out_tok.get(pt, 0), self.inf_latency.get(pt, 0)) for pt in PROMPT_TYPES]
            + [(r.get("out_tok", 0), r.get("lat", 0)) for r in self.qa_results]
        )
        valid = [(t, l) for t, l in pairs if l > 0 and t > 0]
        return statistics.mean(t / l for t, l in valid) if valid else 0.0


# ─── API HELPERS ─────────────────────────────────────────────────────────────

def _call(client, model_id, messages, max_tokens=4000, temperature=0.3):
    """Returns (text, latency, in_tok, out_tok). SDK handles 429 retries."""
    time.sleep(10)  # generous stagger for free-tier rate limits
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_id, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        lat     = time.perf_counter() - t0
        text    = resp.choices[0].message.content or ""
        
        # Extract internal reasoning for models like GLM-4 or DeepSeek R1
        try:
            reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
            if reasoning:
                text = f"**<think>**\n{reasoning}\n**</think>**\n\n" + text
        except Exception:
            pass
            
        in_tok  = (resp.usage.prompt_tokens     if resp.usage else 0)
        out_tok = (resp.usage.completion_tokens if resp.usage else 0)
        return text, lat, in_tok, out_tok
    except Exception as e:
        return f"⚠️ Error: {e}", time.perf_counter() - t0, 0, 0


def _run_one_inference(client, model_id, pt, context):
    text, lat, in_tok, out_tok = _call(
        client, model_id,
        [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user",   "content": f"{INFERENCE_PROMPTS[pt]}\n\nDATA:\n{context}"},
        ],
        max_tokens=4000, temperature=0.3,
    )
    return model_id, pt, text, lat, in_tok, out_tok


def _run_one_qa(client, model_id, question, rag_context):
    sys_msg = f"{SYS_PROMPT_QA}\n\n<context>\n{rag_context}\n</context>"
    text, lat, in_tok, out_tok = _call(
        client, model_id,
        [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": question},
        ],
        max_tokens=4000, temperature=0.5,
    )
    return model_id, question, text, lat, in_tok, out_tok


# ─── CHART HELPERS ───────────────────────────────────────────────────────────

def _scorecard_html(run: ModelRun) -> str:
    return f"""
    <div style="background:#14161C; border-top:3px solid {run.color};
                padding:0.85rem; border-radius:4px; text-align:center;">
        <div style="font-family:'Fira Code',monospace; font-size:0.65rem;
                    color:{run.color}; text-transform:uppercase; letter-spacing:0.08em;">
            {run.label}
        </div>
        <div style="font-size:1.4rem; font-weight:700; color:#fff; margin:0.35rem 0 0.1rem;">
            {run.avg_tps:.1f}
            <span style="font-size:0.7rem; color:#8B949E;">tok/s</span>
        </div>
        <div style="font-size:0.72rem; color:#8B949E;">
            Inf {run.avg_inf_lat:.2f}s &nbsp;|&nbsp; Q&A {run.avg_qa_lat:.2f}s
        </div>
        <div style="font-size:0.72rem; color:#8B949E; margin-top:0.15rem;">
            {run.total_out_tok:,} out-tokens &nbsp;|&nbsp; {len(run.errors)} err
        </div>
    </div>"""


def _latency_grouped(runs: list[ModelRun]) -> go.Figure:
    labels = [r.label for r in runs]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Inference avg (s)", x=labels,
        y=[r.avg_inf_lat for r in runs],
        marker=dict(color=[r.color for r in runs]),
        text=[f"{r.avg_inf_lat:.2f}" for r in runs], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Q&A avg (s)", x=labels,
        y=[r.avg_qa_lat for r in runs],
        marker=dict(color=[r.color for r in runs], opacity=0.45, pattern_shape="/"),
        text=[f"{r.avg_qa_lat:.2f}" for r in runs], textposition="outside",
    ))
    fig.update_layout(barmode="group", title="Latency: Inference vs Q&A",
                      yaxis_title="seconds", **_LAYOUT)
    return fig


def _tps_bar(runs: list[ModelRun]) -> go.Figure:
    labels = [r.label for r in runs]
    vals   = [r.avg_tps for r in runs]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=[r.color for r in runs]),
        text=[f"{v:.1f} tok/s" for v in vals], textposition="outside",
    ))
    fig.update_layout(title="Throughput (tokens/sec)", xaxis_title="tok/s", **_LAYOUT)
    return fig


def _token_stacked(runs: list[ModelRun]) -> go.Figure:
    labels = [r.label for r in runs]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Inference tokens", x=labels,
        y=[sum(r.inf_out_tok.values()) for r in runs],
        marker=dict(color=[r.color for r in runs]),
    ))
    fig.add_trace(go.Bar(
        name="Q&A tokens", x=labels,
        y=[sum(x.get("out_tok", 0) for x in r.qa_results) for r in runs],
        marker=dict(color=[r.color for r in runs], opacity=0.5, pattern_shape="x"),
    ))
    fig.update_layout(barmode="stack", title="Output Tokens by Task", **_LAYOUT)
    return fig


def _radar(runs: list[ModelRun]) -> go.Figure:
    def norm(vals, low_better):
        mn, mx = min(vals), max(vals)
        rng = mx - mn or 1
        return [(mx - v) / rng if low_better else (v - mn) / rng for v in vals]

    cats = ["Inf Speed", "Q&A Speed", "Throughput", "Output Vol", "Reliability"]
    inf_l  = norm([r.avg_inf_lat  for r in runs], True)
    qa_l   = norm([r.avg_qa_lat   for r in runs], True)
    tps    = norm([r.avg_tps      for r in runs], False)
    out_t  = norm([r.total_out_tok for r in runs], False)
    rel    = norm([len(r.errors)  for r in runs], True)

    fig = go.Figure()
    for i, r in enumerate(runs):
        scores = [inf_l[i], qa_l[i], tps[i], out_t[i], rel[i]]
        vals   = scores + [scores[0]]
        thetas = cats   + [cats[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=thetas, fill="toself", name=r.label,
            line=dict(color=r.color, width=2),
            fillcolor="rgba({},{},{},0.15)".format(
                int(r.color[1:3], 16),
                int(r.color[3:5], 16),
                int(r.color[5:7], 16),
            ),
        ))
    fig.update_layout(
        title="Performance Radar (normalised, higher = better)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], color="#8B949E"),
            angularaxis=dict(color="#E2E8F0"),
        ),
        **_LAYOUT,
    )
    return fig


def _inf_heatmap(runs: list[ModelRun]) -> go.Figure:
    z = [[r.inf_latency.get(pt, 0) for pt in PROMPT_TYPES] for r in runs]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[PROMPT_LABELS[pt] for pt in PROMPT_TYPES],
        y=[r.label for r in runs],
        colorscale="RdYlGn_r",
        text=[[f"{v:.2f}s" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Inference Latency Heatmap (red = slow)",
                      **_LAYOUT)
    return fig


def _timeline_chart(runs: list[ModelRun]) -> go.Figure:
    fig = go.Figure()
    for r in runs:
        labels = [f"Inf/{PROMPT_LABELS[pt]}" for pt in PROMPT_TYPES if pt in r.inf_latency]
        vals   = [r.inf_latency[pt] for pt in PROMPT_TYPES if pt in r.inf_latency]
        labels += [f"Q{i+1}" for i in range(len(r.qa_results))]
        vals   += [q.get("lat", 0) for q in r.qa_results]
        fig.add_trace(go.Scatter(
            x=labels, y=vals, mode="lines+markers", name=r.label,
            line=dict(color=r.color, width=2), marker=dict(size=7),
        ))
    fig.update_layout(title="Per-call Latency Timeline",
                      yaxis_title="seconds", **_LAYOUT)
    return fig


def _build_score_chart(runs: list[ModelRun]) -> go.Figure:
    """Rebuild the score comparison bar chart for export."""
    labels = [r.label for r in runs]
    colors = [r.color for r in runs]
    prog   = [getattr(r, "scores", {}).get("progress", 0) for r in runs]
    impact = [getattr(r, "scores", {}).get("impact", 0) for r in runs]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Progress", x=labels, y=prog,
                         marker=dict(color=colors), text=prog, textposition="outside"))
    fig.add_trace(go.Bar(name="Impact", x=labels, y=impact,
                         marker=dict(color=colors, opacity=0.5, pattern_shape="/"),
                         text=impact, textposition="outside"))
    fig.update_layout(barmode="group", title="Progress vs Impact Scores per Model",
                      yaxis=dict(range=[0, 115]), **_LAYOUT)
    return fig


def _build_error_chart(runs: list[ModelRun]) -> go.Figure:
    """Rebuild the error count bar chart for export."""
    labels = [r.label for r in runs]
    fig = go.Figure(go.Bar(
        x=labels, y=[len(r.errors) for r in runs],
        marker=dict(color=[r.color for r in runs]),
        text=[len(r.errors) for r in runs], textposition="outside",
    ))
    fig.update_layout(title="Error Count per Model", yaxis_title="errors", **_LAYOUT)
    return fig


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────

def render_arena_tab(ai_engine) -> None:
    st.markdown("""
    <div style="border-bottom:1px solid #2D3139; padding-bottom:1rem; margin-bottom:1.5rem;">
        <h2 style="font-family:'Crimson Text',serif; font-size:2rem; color:#fff; margin:0;">
            ⚔️ Model Arena
        </h2>
        <p style="font-family:'Fira Code',monospace; color:#FF4F00; font-size:0.78rem;
                  text-transform:uppercase; letter-spacing:0.1em; margin:0.3rem 0 0;">
            Full pipeline benchmark · 4 models · inference + scoring + Q&A
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "**What this tab does:** runs all 4 arena models through the *complete* "
        "Sansad AI workflow — policy scoring (progress/impact JSON), all 4 inference "
        "prompt types, and your custom Q&A questions — then compares every output and "
        "every latency/token metric side by side.",
        icon="ℹ️",
    )

    # ── Guard ──────────────────────────────────────────────────────────────────
    records = st.session_state.get("app_state", {}).get("scraped_records", [])

    # ── Build context (same logic as Tab 2) ───────────────────────────────────
    raw_context = "\n".join([
        f"Date: {r.get('date','Unknown')}, Ministry: {r.get('ministry','Unknown')}, Title: {r.get('title','Unknown')}"
        for r in records[:20]
        if any([r.get("date"), r.get("ministry"), r.get("title")])
    ])

    # Fallback: if no scraped metadata, pull real content from Pinecone
    if not raw_context:
        _fb_keywords = st.session_state.get("app_state", {}).get("keywords_expanded", [])
        _fb_query = " ".join(_fb_keywords) if _fb_keywords else "parliamentary policy questions"
        try:
            _fb_docs = ai_engine.retrieve(_fb_query, top_k=10)
            if _fb_docs:
                raw_context = "\n---\n".join([d["text"] for d in _fb_docs])
                st.caption(f"ℹ️ Using {len(_fb_docs)} documents retrieved from Pinecone (no local metadata in session).")
        except Exception:
            pass

    if not raw_context:
        st.warning(
            "⚠️ No parliamentary data found — neither in session nor in Pinecone. "
            "Run **[ 01_DATA_ACQUISITION ]** first, then return here."
        )
        return

    # ── Q&A questions config ───────────────────────────────────────────────────
    st.markdown("### ① Configure Q&A Questions")
    default_qs = [
        "What is the current implementation status of this policy?",
        "Which ministry has been most active in responding to this topic?",
        "What are the key unresolved issues raised by parliamentarians?",
    ]
    qa_inputs: list[str] = []
    for i, dq in enumerate(default_qs):
        v = st.text_input(f"Question {i+1}", value=dq, key=f"arena4_q_{i}")
        if v.strip():
            qa_inputs.append(v.strip())
    extra = st.text_input("Custom question (optional)", key="arena4_q_extra")
    if extra.strip():
        qa_inputs.append(extra.strip())

    st.markdown("---")

    # ── Launch ─────────────────────────────────────────────────────────────────
    col_run, col_clear = st.columns([3, 1])
    run_btn   = col_run.button("🚀 Launch Full Arena Run", type="primary", width="stretch")
    clear_btn = col_clear.button("🗑 Clear Results", width="stretch")

    if clear_btn:
        st.session_state.pop("arena4_runs", None)
        st.rerun()

    if run_btn:
        cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        cf_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not cf_account or not cf_token:
            st.error("CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN missing from environment. Add them to your .env file.")
            return

        client = OpenAI(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/v1",
            api_key=cf_token,
            max_retries=0,
        )

        rag_context = raw_context

        def _parse_scores(raw: str) -> dict:
            cleaned = re.sub(r"\**<think>\**.*?\**</think>\**", "", raw, flags=re.DOTALL)
            cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned).replace("```", "").strip()
            # Strategy 1: greedy outermost { ... }
            match = re.search(r"\{(.+)\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads("{" + match.group(1) + "}")
                    if "progress" in data or "impact" in data:
                        return data
                except Exception:
                    pass
            # Strategy 2: whole string
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
            # Strategy 3: regex field extraction
            def _gi(k): m = re.search(rf'"{k}"\s*:\s*(\d+)', cleaned); return int(m.group(1)) if m else None
            def _gs(k): m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', cleaned); return m.group(1) if m else None
            p, i, r = _gi("progress"), _gi("impact"), _gs("reason")
            if p is not None or i is not None:
                return {"progress": p or 0, "impact": i or 0, "reason": r or "Extracted via fallback."}
            raise ValueError(f"No scores in: {cleaned[:150]}")

        def _score(model_id):
            for attempt, prompt_text in enumerate([
                (
                    "You are a JSON-only API. Respond with NOTHING except a single JSON object.\n"
                    "No explanation, no markdown, no code fences — raw JSON only.\n"
                    "Do not include reasoning traces or thinking tags.\n"
                    "Required keys: \"progress\" (integer 0-100), \"impact\" (integer 0-100), "
                    "\"reason\" (string, max 20 words).\n\nParliamentary data:\n" + raw_context
                ),
                (
                    "Analyze the parliamentary data and rate this policy.\n"
                    "Answer as a JSON object with fields:\n"
                    "  progress: number 0-100 (implementation status)\n"
                    "  impact: number 0-100 (socio-economic impact)\n"
                    "  reason: one short sentence\n"
                    "Do not include reasoning traces or thinking tags.\n"
                    "Example: {\"progress\": 45, \"impact\": 60, \"reason\": \"Moderate progress noted.\"}\n\n"
                    "Data:\n" + raw_context
                ),
            ]):
                time.sleep(15)  # generous stagger for free-tier rate limits
                t0 = time.perf_counter()
                try:
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.1, max_tokens=300,
                    )
                    lat = time.perf_counter() - t0
                    raw = resp.choices[0].message.content
                    if isinstance(raw, dict):
                        data = raw
                    else:
                        raw = str(raw or "")
                        if not raw.strip():
                            continue
                        data = _parse_scores(raw)
                    return model_id, {
                        "progress": max(0, min(100, int(data.get("progress", 0)))),
                        "impact":   max(0, min(100, int(data.get("impact",   0)))),
                        "reason":   str(data.get("reason", "—")),
                        "lat": lat,
                    }
                except Exception as e:
                    if attempt == 1:
                        try:
                            reason = f"Unparseable: {(resp.choices[0].message.content or '')[:80]}"
                        except Exception:
                            reason = f"Error: {e}"[:200]
                        return model_id, {"progress": 0, "impact": 0, "reason": reason, "lat": 0.0}
            return model_id, {"progress": 0, "impact": 0, "reason": "All attempts failed.", "lat": 0.0}

        # Build ModelRun shells
        runs: dict[str, ModelRun] = {mid: ModelRun(model_id=mid) for mid in ARENA_MODELS}

        overall = st.progress(0.0)
        status  = st.empty()
        total_tasks = len(ARENA_MODELS) * (1 + len(PROMPT_TYPES) + len(qa_inputs))
        done = 0

        # ① Scoring (rate-limit safe: max 1 concurrent)
        status.markdown("⚙️ **Phase 1/3** — Scoring policy parameters …")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            for mid, sc in ex.map(_score, ARENA_MODELS):
                runs[mid].inf_texts["_score"]   = sc["reason"]
                runs[mid].inf_latency["_score"] = sc["lat"]
                runs[mid].inf_out_tok["_score"] = 0
                runs[mid].inf_in_tok["_score"]  = 0
                # Store scores on the run object for later display
                runs[mid].__dict__.setdefault("scores", {})
                runs[mid].scores = sc          # type: ignore[attr-defined]
                done += 1
                overall.progress(done / total_tasks)

        # ② Inference (rate-limit safe: max 2 concurrent)
        inf_tasks = [(mid, pt) for mid in ARENA_MODELS for pt in PROMPT_TYPES]
        status.markdown(f"⚙️ **Phase 2/3** — Policy inference ({len(inf_tasks)} calls) …")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            futures = [ex.submit(_run_one_inference, client, mid, pt, raw_context) for mid, pt in inf_tasks]
            for fut in concurrent.futures.as_completed(futures):
                mid, pt, text, lat, in_tok, out_tok = fut.result()
                runs[mid].inf_texts[pt]   = text
                runs[mid].inf_latency[pt] = lat
                runs[mid].inf_in_tok[pt]  = in_tok
                runs[mid].inf_out_tok[pt] = out_tok
                if text.startswith("⚠️"):
                    runs[mid].errors.append(f"inference/{pt}: {text}")
                done += 1
                overall.progress(done / total_tasks)

        # ③ Q&A
        status.markdown("⚙️ **Phase 3/3** — Q&A …")
        qa_tasks = [(mid, q) for mid in ARENA_MODELS for q in qa_inputs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            futures = [ex.submit(_run_one_qa, client, mid, q, rag_context) for mid, q in qa_tasks]
            for fut in concurrent.futures.as_completed(futures):
                mid, q, text, lat, in_tok, out_tok = fut.result()
                runs[mid].qa_results.append({"q": q, "a": text, "lat": lat,
                                             "in_tok": in_tok, "out_tok": out_tok})
                if text.startswith("⚠️"):
                    runs[mid].errors.append(f"qa/{q[:30]}: {text}")
                done += 1
                overall.progress(done / total_tasks)

        # Sort qa_results by question order
        for mid in ARENA_MODELS:
            runs[mid].qa_results.sort(key=lambda x: qa_inputs.index(x["q"]) if x["q"] in qa_inputs else 99)

        overall.progress(1.0)
        status.success("✅ Arena run complete!")
        st.session_state["arena4_runs"] = runs

    # ── Display ────────────────────────────────────────────────────────────────
    if "arena4_runs" not in st.session_state:
        return

    runs: dict[str, ModelRun] = st.session_state["arena4_runs"]
    run_list = [runs[mid] for mid in ARENA_MODELS]

    # ── Section A: Score comparison ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🏆 Section A — Policy Scores")
    st.caption("Progress (implementation status) and Impact (socio-economic) scores — each model's assessment.")

    sc_cols = st.columns(len(run_list))
    for col, r in zip(sc_cols, run_list):
        sc = getattr(r, "scores", {})
        with col:
            st.markdown(f"""
            <div style="background:#14161C; border-top:3px solid {r.color};
                        padding:0.8rem; border-radius:4px;">
                <div style="font-family:'Fira Code',monospace; font-size:0.62rem;
                            color:{r.color}; text-transform:uppercase; letter-spacing:0.07em;">
                    {r.label}
                </div>
                <div style="display:flex; gap:1.2rem; margin-top:0.5rem;">
                    <div>
                        <div style="font-size:0.6rem; color:#8B949E;">Progress</div>
                        <div style="font-size:1.5rem; font-weight:700; color:#fff;">
                            {sc.get('progress', 0)}<span style="font-size:0.65rem;color:#8B949E;">/100</span>
                        </div>
                    </div>
                    <div>
                        <div style="font-size:0.6rem; color:#8B949E;">Impact</div>
                        <div style="font-size:1.5rem; font-weight:700; color:#fff;">
                            {sc.get('impact', 0)}<span style="font-size:0.65rem;color:#8B949E;">/100</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(sc.get("progress", 0) / 100)
            st.caption(sc.get("reason", "—"))

    # Score comparison chart
    prog_vals   = [getattr(r, "scores", {}).get("progress", 0) for r in run_list]
    impact_vals = [getattr(r, "scores", {}).get("impact",   0) for r in run_list]
    labels      = [r.label for r in run_list]
    colors      = [r.color for r in run_list]

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(name="Progress", x=labels, y=prog_vals,
                            marker=dict(color=colors),
                            text=prog_vals, textposition="outside"))
    fig_sc.add_trace(go.Bar(name="Impact",   x=labels, y=impact_vals,
                            marker=dict(color=colors, opacity=0.5, pattern_shape="/"),
                            text=impact_vals, textposition="outside"))
    fig_sc.update_layout(barmode="group", title="Progress vs Impact Scores per Model",
                         yaxis=dict(range=[0, 115]), **_LAYOUT)
    st.plotly_chart(fig_sc, width="stretch")

    # ── Section B: Performance metrics ────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Section B — Performance Metrics")

    # Scorecards
    mc = st.columns(len(run_list))
    for col, r in zip(mc, run_list):
        with col:
            st.markdown(_scorecard_html(r), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1: st.plotly_chart(_latency_grouped(run_list), width="stretch")
    with r1c2: st.plotly_chart(_tps_bar(run_list),         width="stretch")

    r2c1, r2c2 = st.columns(2)
    with r2c1: st.plotly_chart(_token_stacked(run_list), width="stretch")
    with r2c2:
        err_counts = [len(r.errors) for r in run_list]
        fig_err = go.Figure(go.Bar(
            x=[r.label for r in run_list], y=err_counts,
            marker=dict(color=[r.color for r in run_list]),
            text=err_counts, textposition="outside",
        ))
        fig_err.update_layout(title="Error Count per Model", yaxis_title="errors", **_LAYOUT)
        st.plotly_chart(fig_err, width="stretch")

    st.plotly_chart(_radar(run_list), width="stretch")

    # Full metrics table
    st.markdown("### 📋 Metrics Summary Table")
    rows = [{
        "Model":                r.label,
        "Avg Inf Lat (s)":      round(r.avg_inf_lat, 3),
        "Avg Q&A Lat (s)":      round(r.avg_qa_lat, 3),
        "Avg tok/s":            round(r.avg_tps, 1),
        "Inf Out Tokens":       sum(r.inf_out_tok.get(pt, 0) for pt in PROMPT_TYPES),
        "Q&A Out Tokens":       sum(x.get("out_tok", 0) for x in r.qa_results),
        "Total Out Tokens":     r.total_out_tok,
        "Errors":               len(r.errors),
    } for r in run_list]
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    # ── Section C: Inference comparison ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🧠 Section C — Inference Output Comparison")

    inf_tabs = st.tabs([PROMPT_LABELS[pt] for pt in PROMPT_TYPES])
    for tab_obj, pt in zip(inf_tabs, PROMPT_TYPES):
        with tab_obj:
            inf_cols = st.columns(len(run_list))
            for col, r in zip(inf_cols, run_list):
                with col:
                    lat = r.inf_latency.get(pt)
                    lat_str = f"{lat:.2f}s" if lat else "—"
                    tok = r.inf_out_tok.get(pt, "—")
                    st.markdown(f"""
                    <div style="font-family:'Fira Code',monospace; font-size:0.62rem;
                                color:{r.color}; border-bottom:1px solid #2D3139;
                                padding-bottom:0.3rem; margin-bottom:0.5rem;">
                        {r.label} &nbsp;·&nbsp; {lat_str} &nbsp;·&nbsp; {tok} tok
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(r.inf_texts.get(pt, "_Not run_"))

    st.plotly_chart(_inf_heatmap(run_list), width="stretch")

    # ── Section D: Q&A comparison ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 💬 Section D — Q&A Output Comparison")

    all_questions = [x["q"] for x in run_list[0].qa_results] if run_list else []
    for q_idx, question in enumerate(all_questions):
        with st.expander(f"**Q{q_idx+1}:** {question}", expanded=(q_idx == 0)):
            qa_cols = st.columns(len(run_list))
            for col, r in zip(qa_cols, run_list):
                with col:
                    entry = r.qa_results[q_idx] if q_idx < len(r.qa_results) else {}
                    lat_str = f"{entry.get('lat', 0):.2f}s"
                    tok_str = str(entry.get("out_tok", "—"))
                    st.markdown(f"""
                    <div style="font-family:'Fira Code',monospace; font-size:0.62rem;
                                color:{r.color}; border-bottom:1px solid #2D3139;
                                padding-bottom:0.3rem; margin-bottom:0.5rem;">
                        {r.label} &nbsp;·&nbsp; {lat_str} &nbsp;·&nbsp; {tok_str} tok
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(entry.get("a", "_Not answered_"))

    # ── Section E: Timeline + error log ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## ⏱ Section E — Per-call Latency Timeline")
    st.plotly_chart(_timeline_chart(run_list), width="stretch")

    if any(r.errors for r in run_list):
        st.markdown("---")
        st.markdown("## ⚠️ Error Log")
        for r in run_list:
            if r.errors:
                with st.expander(f"`{r.label}` — {len(r.errors)} error(s)"):
                    for e in r.errors:
                        st.error(e)

    # ── Section F: Export & Save All Results ───────────────────────────────
    st.markdown("---")
    st.markdown("## 💾 Section F — Export & Save All Results")
    st.caption("Download benchmark data as CSV and charts as interactive HTML files.")

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Build export DataFrames ───────────────────────────────────────────
    df_exp_scores = pd.DataFrame([{
        "Model": r.label,
        "Progress (/100)": getattr(r, "scores", {}).get("progress", 0),
        "Impact (/100)": getattr(r, "scores", {}).get("impact", 0),
        "Reason": getattr(r, "scores", {}).get("reason", ""),
        "Score Latency (s)": round(r.inf_latency.get("_score", 0), 3),
    } for r in run_list])

    df_exp_metrics = pd.DataFrame([{
        "Model": r.label,
        "Avg Inf Latency (s)": round(r.avg_inf_lat, 3),
        "Avg Q&A Latency (s)": round(r.avg_qa_lat, 3),
        "Avg Throughput (tok/s)": round(r.avg_tps, 1),
        "Inf Output Tokens": sum(r.inf_out_tok.get(pt, 0) for pt in PROMPT_TYPES),
        "Q&A Output Tokens": sum(x.get("out_tok", 0) for x in r.qa_results),
        "Total Output Tokens": r.total_out_tok,
        "Errors": len(r.errors),
    } for r in run_list])

    inf_exp_rows = []
    for r in run_list:
        for pt in PROMPT_TYPES:
            inf_exp_rows.append({
                "Model": r.label, "Prompt Type": PROMPT_LABELS[pt],
                "Latency (s)": round(r.inf_latency.get(pt, 0), 3),
                "Output Tokens": r.inf_out_tok.get(pt, 0),
                "Input Tokens": r.inf_in_tok.get(pt, 0),
                "Output Text": r.inf_texts.get(pt, ""),
            })
    df_exp_inference = pd.DataFrame(inf_exp_rows)

    qa_exp_rows = []
    for r in run_list:
        for qa in r.qa_results:
            qa_exp_rows.append({
                "Model": r.label, "Question": qa.get("q", ""),
                "Answer": qa.get("a", ""),
                "Latency (s)": round(qa.get("lat", 0), 3),
                "Output Tokens": qa.get("out_tok", 0),
                "Input Tokens": qa.get("in_tok", 0),
            })
    df_exp_qa = pd.DataFrame(qa_exp_rows)

    # ── Download Data CSVs ────────────────────────────────────────────────
    st.markdown("#### 📥 Download Data")
    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        st.download_button(
            "📊 Scores CSV", df_exp_scores.to_csv(index=False).encode("utf-8"),
            f"arena_scores_{_ts}.csv", "text/csv",
            width="stretch", key="arena_dl_sc")
    with dl2:
        st.download_button(
            "⚡ Metrics CSV", df_exp_metrics.to_csv(index=False).encode("utf-8"),
            f"arena_metrics_{_ts}.csv", "text/csv",
            width="stretch", key="arena_dl_mt")
    with dl3:
        st.download_button(
            "🧠 Inference CSV", df_exp_inference.to_csv(index=False).encode("utf-8"),
            f"arena_inference_{_ts}.csv", "text/csv",
            width="stretch", key="arena_dl_inf")
    with dl4:
        st.download_button(
            "💬 Q&A CSV", df_exp_qa.to_csv(index=False).encode("utf-8"),
            f"arena_qa_{_ts}.csv", "text/csv",
            width="stretch", key="arena_dl_qa")

    # ── Download Charts (Interactive HTML) ────────────────────────────────
    st.markdown("#### 📥 Download Charts")
    _charts = {
        "score_comparison": _build_score_chart(run_list),
        "latency_comparison": _latency_grouped(run_list),
        "throughput": _tps_bar(run_list),
        "token_usage": _token_stacked(run_list),
        "radar": _radar(run_list),
        "inference_heatmap": _inf_heatmap(run_list),
        "latency_timeline": _timeline_chart(run_list),
        "error_count": _build_error_chart(run_list),
    }
    ch_row1 = st.columns(4)
    ch_row2 = st.columns(4)
    ch_cols = ch_row1 + ch_row2
    for i, (cname, cfig) in enumerate(_charts.items()):
        with ch_cols[i]:
            st.download_button(
                f"📈 {cname.replace('_', ' ').title()}",
                cfig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                f"arena_{cname}_{_ts}.html", "text/html",
                width="stretch", key=f"arena_dl_ch_{cname}")
