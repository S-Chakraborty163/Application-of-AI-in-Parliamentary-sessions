import streamlit as st
import time
import json
import requests
import concurrent.futures
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
import os
import torch
import torch.nn.functional as F

# Import arena configurations
from arena_tab import (
    ARENA_MODELS, MODEL_LABELS, MODEL_COLORS,
    PROMPT_TYPES, PROMPT_LABELS, SYS_PROMPT, INFERENCE_PROMPTS,
    _run_one_inference
)

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two 1D vectors."""
    t1 = torch.tensor(emb1)
    t2 = torch.tensor(emb2)
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

def render_quality_eval_tab(ai_engine):
    st.markdown("## 🥇 Semantic Quality Evaluation (Gold Standard Matrix)")
    st.markdown("""
    This tab tests the **actual quality and meaning** of the open-source models' outputs by mathematically comparing them against a "Golden Answer". 
    We use a powerful **Cloudflare Model** (like Llama 3.3 70B) as the ultimate Judge to generate the perfect reference answer, and then we use semantic embeddings (Cosine Similarity) to see which open-source model matches it closest!
    """)
    
    with st.expander("⚙️ Evaluation Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            gold_model_id = st.text_input(
                "Gold Standard Model (Cloudflare ID)", 
                value="@cf/meta/llama-3.3-70b-instruct-fp8-fast", 
                help="The model used to generate the perfect baseline."
            )
            or_key = st.text_input(
                "Cloudflare API Token",
                value=os.environ.get("CLOUDFLARE_API_TOKEN", ""),
                type="password"
            )
            cf_acc = st.text_input(
                "Cloudflare Account ID",
                value=os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
                type="password"
            )
            
        with col2:
            query = st.text_input(
                "Evaluation Context Query", 
                value="digital india progress and hurdles",
                help="We will fetch documents matching this keyword and run the evaluation on them."
            )
            top_k = st.number_input("Number of Chunks", min_value=1, max_value=20, value=5)
            
    if st.button("🚀 Launch Semantic Evaluation Matrix", type="primary"):
        if not gold_model_id or not or_key or not cf_acc:
            st.error("Please provide your Cloudflare API keys and select a Gold Standard model.")
            return
            
        # Init OpenAI client for Cloudflare
        client = OpenAI(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/v1",
            api_key=or_key,
            max_retries=0,
        )
        
        status = st.empty()
        overall = st.progress(0.0)
        
        # 1. Fetch Context
        status.markdown("⚙️ **Phase 1/4** — Fetching RAG Context …")
        try:
            chunks = ai_engine.retrieve(query, top_k=top_k)
            raw_context = "\n\n".join(c["text"] for c in chunks)
        except Exception as e:
            st.error(f"RAG Error: {e}")
            return
            
        if not raw_context.strip():
            st.warning("No context found. Try a different query.")
            return
            
        overall.progress(0.1)
            
        status.markdown(f"⚙️ **Phase 2/4** — Generating Golden Answers via {gold_model_id} …")
        golden_answers = {}
        
        # We use the Cloudflare wrapper to generate the Golden standard!
        for i, pt in enumerate(PROMPT_TYPES):
            _, _, text, _, _, _ = _run_one_inference(client, gold_model_id, pt, raw_context)
            golden_answers[pt] = text
                
        overall.progress(0.4)
        
        # 3. Generate Arena Answers
        status.markdown(f"⚙️ **Phase 3/4** — Generating Arena Answers ({len(ARENA_MODELS) * len(PROMPT_TYPES)} calls) …")
        arena_answers = {mid: {} for mid in ARENA_MODELS}
        
        inf_tasks = [(mid, pt) for mid in ARENA_MODELS for pt in PROMPT_TYPES]
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: # Keep low to avoid rate limits
            futures = {ex.submit(_run_one_inference, client, mid, pt, raw_context): (mid, pt) for mid, pt in inf_tasks}
            for fut in concurrent.futures.as_completed(futures):
                mid, pt = futures[fut]
                _, _, text, lat, in_tok, out_tok = fut.result()
                arena_answers[mid][pt] = text
                done += 1
                overall.progress(0.4 + (0.4 * (done / len(inf_tasks))))
                
        # 4. Compute Semantic Similarity
        status.markdown("⚙️ **Phase 4/4** — Computing Mathematical Semantic Similarity …")
        
        # Embed golden answers
        golden_embs = {}
        for pt in PROMPT_TYPES:
            if not golden_answers[pt].startswith("⚠️"):
                golden_embs[pt] = ai_engine.embedder.encode(golden_answers[pt])
                
        similarity_scores = {mid: {} for mid in ARENA_MODELS}
        
        for mid in ARENA_MODELS:
            for pt in PROMPT_TYPES:
                ans = arena_answers[mid].get(pt, "")
                if ans.startswith("⚠️") or pt not in golden_embs or len(ans.strip()) < 10:
                    similarity_scores[mid][pt] = 0.0
                else:
                    emb = ai_engine.embedder.encode(ans)
                    score = cosine_similarity(golden_embs[pt], emb)
                    similarity_scores[mid][pt] = score
                    
        overall.progress(1.0)
        status.success("✅ Semantic Evaluation Complete!")
        
        # --- DISPLAY RESULTS ---
        st.markdown("---")
        st.markdown(f"### 🏆 Semantic Leaderboard (vs {gold_model_id})")
        
        # Calculate Average Similarity
        avg_scores = []
        for mid in ARENA_MODELS:
            scores = [similarity_scores[mid][pt] for pt in PROMPT_TYPES if similarity_scores[mid][pt] > 0]
            avg = sum(scores) / len(scores) if scores else 0
            avg_scores.append({"Model": MODEL_LABELS[mid], "Average Semantic Match": avg})
            
        df_avg = pd.DataFrame(avg_scores).sort_values(by="Average Semantic Match", ascending=False)
        
        col_chart, col_data = st.columns([2, 1])
        with col_chart:
            fig = px.bar(
                df_avg, x="Average Semantic Match", y="Model", orientation='h',
                color="Model", color_discrete_map={MODEL_LABELS[m]: MODEL_COLORS[m] for m in ARENA_MODELS},
                text_auto='.1%', range_x=[0, 1]
            )
            fig.update_layout(showlegend=False, template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_data:
            df_display = df_avg.copy()
            df_display["Average Semantic Match"] = df_display["Average Semantic Match"].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_display, hide_index=True, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🎯 Prompt-Specific Semantic Match Heatmap")
        
        z_data = []
        for mid in ARENA_MODELS:
            z_data.append([similarity_scores[mid][pt] for pt in PROMPT_TYPES])
            
        fig_heat = go.Figure(go.Heatmap(
            z=z_data,
            x=[PROMPT_LABELS[pt] for pt in PROMPT_TYPES],
            y=[MODEL_LABELS[mid] for mid in ARENA_MODELS],
            colorscale="Viridis",
            text=[[f"{v:.1%}" for v in row] for row in z_data],
            texttemplate="%{text}",
            zmin=0, zmax=1
        ))
        fig_heat.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🔍 Side-by-Side Golden Answers")
        
        tabs = st.tabs([PROMPT_LABELS[pt] for pt in PROMPT_TYPES])
        for tab, pt in zip(tabs, PROMPT_TYPES):
            with tab:
                st.info(f"**{gold_model_id} (Gold Standard)**\n\n{golden_answers[pt]}")
                st.markdown("#### Cloudflare Models")
                
                # Sort models by similarity for this specific prompt
                sorted_models = sorted(ARENA_MODELS, key=lambda m: similarity_scores[m][pt], reverse=True)
                
                for mid in sorted_models:
                    score = similarity_scores[mid][pt]
                    with st.expander(f"{MODEL_LABELS[mid]} - {score:.1%} Match"):
                        st.write(arena_answers[mid][pt])
