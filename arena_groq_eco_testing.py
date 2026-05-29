import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import io

# Import your actual production architecture
from app import PolicyAI

st.set_page_config(page_title="Sansad AI | Advanced Master Arena", layout="wide")

st.title("⚔️ End-to-End LLM Evaluation Arena")
st.markdown("Select a topic to race the dashboards, then automatically run the full 5-question Q&A suite to evaluate the models over sequential iterations.")

# ─── 1. CORE INITIALIZATION & EVAL SUITE ───
@st.cache_resource
def load_engine():
    return PolicyAI()

with st.spinner("Booting up Sansad AI Backend..."):
    ai_engine = load_engine()

# Persistent session state keys
if 'test_history_dash' not in st.session_state:
    st.session_state.test_history_dash = []
if 'test_history_qa' not in st.session_state:
    st.session_state.test_history_qa = []
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None
if 'dash_cached_results' not in st.session_state:
    st.session_state.dash_cached_results = {}

# Corrected 2026 Model Fleet with exact vendor prefixes
arena_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b"
]

dashboard_prompts = {
    "summary": "Based strictly on the provided parliamentary data, write a 3-paragraph executive summary.",
    "timeline": "Analyze the policy progression. Identify key years and format as a markdown timeline.",
    "ministry": "Identify which government ministries are engaged in this topic and summarize their responses.",
    "gaps": "Identify 'Policy Gaps'. What issues remain unresolved based on these records?"
}

# The 50-Question Master Suite
eval_suite = {
    "Unemployment & Economy": [
        "What are the primary factors contributing to graduate unemployment in urban areas as discussed in the recent sessions?",
        "Extract a timeline of initiatives launched by the Ministry of Skill Development regarding the gig economy.",
        "Compare the state-wise youth unemployment trends mentioned in the context and format the output strictly as a Markdown table.",
        "Summarize the debate around AI automation and its projected impact on BPO and IT jobs over the next decade.",
        "Draft a 100-word executive summary of the proposed social security protections for gig and platform workers."
    ],
    "Air Pollution & Environment": [
        "Identify the key policies implemented by the MoEFCC to combat stubble burning in the Delhi NCR region.",
        "Based strictly on the provided text, what specific policy gaps exist in the current AQI prediction and monitoring infrastructure?",
        "Create a Markdown table comparing emission reduction targets versus actual achievements for the top 5 most polluted cities.",
        "Summarize the parliamentary committee's recommendations on curbing vehicular pollution and adopting BS-VI norms.",
        "Format the latest National Clean Air Programme (NCAP) fund allocations by state as a structured JSON object."
    ],
    "Agriculture & Farmers": [
        "List all the government interventions mentioned regarding Minimum Support Price (MSP) adjustments for Rabi crops.",
        "Create a structured JSON output listing the top three challenges in implementing the Pradhan Mantri Fasal Bima Yojana (crop insurance).",
        "Summarize the parliamentary debate on the long-term environmental impact of current chemical fertilizer subsidies.",
        "Identify the policy gaps in linking rural farmers directly to the e-NAM (National Agriculture Market) platform.",
        "Draft a two-paragraph synthesis of the discussion on adopting climate-resilient crop varieties."
    ],
    "Healthcare Infrastructure": [
        "Which states are reported to have the highest doctor shortages in rural primary health centers?",
        "Detail the progression of the AIIMS expansion phases and their current operational status.",
        "Create a Markdown table of infant mortality rate trends comparing Tier-2 cities with rural districts.",
        "Synthesize the debate on the regulation of private hospital pricing and out-of-pocket expenditures during public health emergencies.",
        "List the top five infrastructural deficits identified in rural health sub-centers (e.g., electricity, cold storage)."
    ],
    "Education & NEP": [
        "Summarize the budget allocations and target metrics for digital education initiatives under NEP 2020.",
        "Draft a 150-word executive summary contrasting the outcomes of government versus private school digital infrastructure.",
        "Create a Markdown table detailing the phased rollout timeline of higher education accreditation reforms.",
        "Identify the policy gaps discussed regarding the integration of vocational training in middle schools.",
        "Format the financial grants given to central universities as a JSON object, categorized by state."
    ],
    "Cybersecurity & Privacy": [
        "Extract all mentioned statistics regarding data breaches in Public Sector Undertakings (PSUs) over the last three years.",
        "Identify the proposed regulatory frameworks for Artificial Intelligence and map them to their corresponding implementing agencies.",
        "Summarize the debate on the privacy implications of linking biometric data to essential government services.",
        "Format the budget allocations for the National Cyber Security Coordinator as a JSON object.",
        "Detail the timeline, objectives, and primary concerns raised regarding the upcoming Digital India Act."
    ],
    "Women Safety & Justice": [
        "What are the key focus areas of the latest women's reservation bills discussed in the Lok Sabha?",
        "Provide a state-wise analysis of female workforce participation rates as debated in the recent sessions.",
        "Create a Markdown table mapping the allocation versus the actual utilization of the Nirbhaya Fund by state.",
        "Identify the policy gaps highlighted in the implementation of the Maternity Benefit Act in the private sector.",
        "List the specific financial initiatives launched to promote women-led startups in the technology and manufacturing sectors."
    ],
    "Renewable Energy": [
        "Outline the phase-wise transition goals from coal dependency to solar missions mentioned by the Ministry of New and Renewable Energy.",
        "Create a markdown table showing the EV infrastructure targets versus actual charging station implementation across Tier-1 cities.",
        "Extract the state-wise renewable energy capacity additions and format them as a structured JSON object.",
        "Identify the policy gaps in the current electric vehicle battery swapping network regulations.",
        "Synthesize the discussions on the integration of green hydrogen into the industrial and transportation sectors."
    ],
    "Urban Infrastructure": [
        "List the smart city performance metrics that have repeatedly failed to meet their targets according to the references.",
        "Summarize the parliamentary debates on traffic congestion and metro expansion in a strictly chronological timeline.",
        "Create a Markdown table comparing the budget allocations for Tier-1 versus Tier-2 smart city projects.",
        "Identify the policy gaps discussed regarding urban flood management and drainage infrastructure during monsoons.",
        "Detail the debate on municipal bond issuances as a mechanism to independently fund urban infrastructure projects."
    ],
    "Water & Rivers": [
        "Extract the inter-state river disputes currently under tribunal review and list the specific states involved in each.",
        "How is the Ministry of Jal Shakti addressing rapid groundwater depletion in the context of increasingly erratic rainfall predictions?",
        "Create a Markdown table detailing the progress of the Jal Jeevan Mission across the bottom five performing states.",
        "Synthesize the parliamentary discussions on the downstream impact of climate change on Himalayan glacial runoff.",
        "List the proposed technologies and financial models for large-scale desalination plants in coastal, water-scarce regions."
    ]
}

# ─── 2. STAGE 1 & 2: KEYWORD SEARCH & DASHBOARD INFERENCES ───
st.header("🎯 Step 1 & 2: Ingestion & Dashboard Analysis")

col_input, col_action = st.columns([3, 1])
with col_input:
    selected_topic = st.selectbox("Select a Benchmark Topic:", list(eval_suite.keys()))
with col_action:
    st.write("##")
    run_pipeline = st.button("Initialize & Race Dashboards", type="primary", use_container_width=True)

if run_pipeline and selected_topic:
    st.session_state.current_topic = selected_topic
    
    with st.status(f"Executing Ingestion Pipeline for '{selected_topic}'...") as status:
        st.write("Simulating Web Scraper & Vector Embedding...")
        time.sleep(1)
        status.update(label="Ingestion complete. Documents grounded in vector space.", state="complete")

    with st.spinner("Launching parallel execution threads for Dashboard Inferences..."):
        broad_docs = ai_engine.retrieve(selected_topic, top_k=4)
        dash_context = "\n---\n".join([d["text"] for d in broad_docs]) if broad_docs else "No specific records found."

        def run_dashboard_for_model(model_name):
            def fetch_single_insight(prompt_type):
                sys_prompt = "You are an expert AI public policy analyst. Ground inferences strictly in the text."
                user_prompt = f"{dashboard_prompts[prompt_type]}\n\nDATA:\n{dash_context}"
                try:
                    resp = ai_engine.llm_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                        max_tokens=800, temperature=0.3
                    )
                    return prompt_type, resp.choices[0].message.content, resp.usage.total_tokens
                except Exception as e:
                    return prompt_type, f"Error: {str(e)}", 0

            start_time = time.time()
            insights = {}
            total_tokens = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as inner_exec:
                futures = [inner_exec.submit(fetch_single_insight, p_type) for p_type in dashboard_prompts.keys()]
                for f in concurrent.futures.as_completed(futures):
                    p_type, text, tokens = f.result()
                    insights[p_type] = text
                    total_tokens += tokens
            
            latency = time.time() - start_time
            return {
                "model": model_name, "success": True if total_tokens > 0 else False,
                "insights": insights, "latency": round(latency, 2), "tokens": total_tokens
            }

        dash_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(arena_models)) as out_exec:
            future_to_model = {out_exec.submit(run_dashboard_for_model, m): m for m in arena_models}
            for f in concurrent.futures.as_completed(future_to_model):
                m_name = future_to_model[f]
                res = f.result()
                dash_results[m_name] = res
                
                if res["success"]:
                    st.session_state.test_history_dash.append({
                        "Topic": selected_topic, "Model": m_name,
                        "Dashboard Wait Time (s)": res["latency"], "Total Tokens": res["tokens"]
                    })
        
        st.session_state.dash_cached_results = dash_results

if st.session_state.dash_cached_results:
    st.markdown("### 📊 Tab 2 Performance: Side-by-Side Inferences")
    cols_dash = st.columns(len(arena_models))
    for idx, m_name in enumerate(arena_models):
        with cols_dash[idx]:
            res = st.session_state.dash_cached_results.get(m_name, {"success": False})
            st.subheader(m_name.split('/')[-1].upper())
            if res["success"]:
                st.metric(label="Dashboard Render Time", value=f"{res['latency']} s")
                tab_sum, tab_line, tab_min, tab_gap = st.tabs(["Summary", "Timeline", "Ministries", "Gaps"])
                tab_sum.markdown(res["insights"]["summary"])
                tab_line.markdown(res["insights"]["timeline"])
                tab_min.markdown(res["insights"]["ministry"])
                tab_gap.markdown(res["insights"]["gaps"])
            else:
                st.error("Dashboard Inference Crashed")

st.divider()

# ─── 3. STAGE 3: THE AUTOMATED CHATBOT ARENA ───
if st.session_state.current_topic:
    st.header(f"💬 Step 3: Automated Q&A Arena (Topic: {st.session_state.current_topic})")
    st.write("Clicking the button below will automatically retrieve context and race the models across all 5 predefined benchmark questions for this topic.")
    
    questions_to_run = eval_suite[st.session_state.current_topic]
    
    if st.button("🚀 Launch Full Q&A Suite (5 Questions)", type="primary"):
        
        # Clear out previous topic QA results to prevent overlapping metrics on individual reports
        st.session_state.test_history_qa = [r for r in st.session_state.test_history_qa if r["Topic"] != st.session_state.current_topic]
        
        for q_idx, qa_query in enumerate(questions_to_run):
            st.markdown(f"### Q{q_idx+1}: {qa_query}")
            
            with st.spinner(f"Retrieving vector context and executing models for Q{q_idx+1}..."):
                qa_docs = ai_engine.retrieve(qa_query, top_k=3)
                qa_context = "\n---\n".join([d["text"] for d in qa_docs]) if qa_docs else "No specific records found."
                
                sys_prompt = f"""You are Sansad AI. You have access to these records: <context>{qa_context}</context>. 
                If the context contains the answer, extract it accurately. If not, fallback to general knowledge but explicitly state it is outside the database."""

                def race_chatbot(model_name, query):
                    start_time = time.time()
                    try:
                        resp = ai_engine.llm_client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}],
                            max_tokens=600, temperature=0.1
                        )
                        latency = time.time() - start_time
                        tps = resp.usage.total_tokens / latency if latency > 0 else 0
                        return {
                            "model": model_name, "success": True, "answer": resp.choices[0].message.content,
                            "latency": round(latency, 2), "tps": round(tps, 2), "tokens": resp.usage.total_tokens
                        }
                    except Exception as e:
                        return {"model": model_name, "success": False, "error": str(e)}

                qa_results = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(arena_models)) as executor:
                    future_to_qa = {executor.submit(race_chatbot, m, qa_query): m for m in arena_models}
                    for f in concurrent.futures.as_completed(future_to_qa):
                        m_name = future_to_qa[f]
                        res = f.result()
                        qa_results[m_name] = res
                        
                        if res["success"]:
                            # --- CRUCIAL EXTRACTION: Added explicit Question ID tracking for line plots ---
                            st.session_state.test_history_qa.append({
                                "Topic": st.session_state.current_topic,
                                "Question": f"Q{q_idx+1}",
                                "Query": qa_query,
                                "Model": m_name,
                                "Latency (s)": res["latency"],
                                "Speed (TPS)": res["tps"],
                                "Tokens": res["tokens"]
                            })

            cols_qa = st.columns(len(arena_models))
            for idx, m_name in enumerate(arena_models):
                with cols_qa[idx]:
                    res = qa_results.get(m_name, {"success": False})
                    st.markdown(f"**{m_name.split('/')[-1].upper()}**")
                    if res["success"]:
                        st.caption(f"⏱️ **{res['latency']}s** | ⚡ **{res['tps']} TPS** | 🪙 **{res['tokens']}** Tkns")
                        st.info(res["answer"])
                    else:
                        st.error(f"Failed: {res.get('error', 'Unknown')}")
            st.divider()

st.divider()

# ─── 4. GRAPH GENERATION & TOPIC SPECIFIC SYSTEM REPORTING ───
if st.session_state.test_history_dash or st.session_state.test_history_qa:
    # Build clean string labels for saving files safely
    safe_topic_string = st.session_state.current_topic.lower().replace(" ", "_").replace("&", "and")
    
    st.header(f"📊 Performance Report: {st.session_state.current_topic}")
    
    tab_dash_report, tab_qa_report = st.tabs(["Dashboard Analytical Metrics", "Chatbot Q&A Metrics"])
    
    with tab_dash_report:
        # Filter metrics to display data ONLY for the current topic run
        df_dash_all = pd.DataFrame(st.session_state.test_history_dash)
        df_dash = df_dash_all[df_dash_all["Topic"] == st.session_state.current_topic]
        
        if not df_dash.empty:
            st.dataframe(df_dash, use_container_width=True)
            
            fig_dash, ax_dash = plt.subplots(figsize=(8, 4))
            sns.barplot(data=df_dash, x='Model', y='Dashboard Wait Time (s)', ax=ax_dash, palette='crest', errorbar=None)
            ax_dash.set_title(f'Dashboard Rendering Latency for {st.session_state.current_topic}')
            st.pyplot(fig_dash)
            
            csv_dash = df_dash.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Save {st.session_state.current_topic} Dashboard Report (CSV)",
                data=csv_dash,
                file_name=f'{safe_topic_string}_dashboard_metrics.csv',
                mime='text/csv'
            )

    with tab_qa_report:
        # THE FIX: Check if the Q&A history actually has data before building the DataFrame
        if st.session_state.test_history_qa:
            df_qa_all = pd.DataFrame(st.session_state.test_history_qa)
            df_qa = df_qa_all[df_qa_all["Topic"] == st.session_state.current_topic]
            
            if not df_qa.empty:
                st.dataframe(df_qa, use_container_width=True)
                
                # ─── THE NEW VISUALIZATION SUITE ───
                sns.set_theme(style="darkgrid")
                fig_qa, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Chart 1: The Question-by-Question Progression Line Plot
                sns.lineplot(
                    data=df_qa, x='Question', y='Latency (s)', hue='Model', 
                    marker='o', linewidth=2.5, markersize=8, ax=ax1
                )
                ax1.set_title('Sequential Latency Progression (Question by Question)')
                ax1.set_ylabel('Latency (Seconds)')
                ax1.set_xlabel('Testing Timeline Progress')
                
                # Chart 2: Interesting Comparison (Throughput TPS vs Latency)
                sns.scatterplot(
                    data=df_qa, x='Latency (s)', y='Speed (TPS)', hue='Model', 
                    style='Model', s=200, alpha=0.8, ax=ax2
                )
                ax2.set_title('Efficiency Tradeoff: Throughput (TPS) vs Latency')
                ax2.set_ylabel('Generation Speed (Tokens/Second)')
                ax2.set_xlabel('Response Delay (Seconds)')
                
                st.pyplot(fig_qa)
                
                # Target Exports
                col_save_csv, col_save_png = st.columns(2)
                with col_save_csv:
                    csv_qa = df_qa.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Save {st.session_state.current_topic} Q&A Data Log (CSV)",
                        data=csv_qa, file_name=f'{safe_topic_string}_qa_performance_report.csv', mime='text/csv'
                    )
                with col_save_png:
                    buf = io.BytesIO()
                    fig_qa.savefig(buf, format="png", bbox_inches="tight")
                    st.download_button(
                        label=f"🖼️ Save {st.session_state.current_topic} Analytical Charts (PNG)",
                        data=buf.getvalue(), file_name=f"{safe_topic_string}_performance_charts.png", mime="image/png"
                    )
        else:
            # Show a friendly message instead of crashing
            st.info("No Q&A data yet. Click 'Launch Full Q&A Suite' above to generate these charts.")