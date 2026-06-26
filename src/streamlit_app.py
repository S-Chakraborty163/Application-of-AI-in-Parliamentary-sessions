import streamlit as st
import time
import pandas as pd
import duckdb
import plotly.express as px
from graph_rag_retriever import answer_question, generate_followups, generate_local_inference

st.set_page_config(page_title="Parliamentary AI Assistant", page_icon="🏛️", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .chat-bubble { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; }
    .chat-bubble.user { background-color: #2b313e; }
    .chat-bubble.bot { background-color: #1e2129; border: 1px solid #3d4554; }
    .graph-fact { font-size: 0.85rem; color: #a3a8b8; background-color: #2c3240; padding: 0.2rem 0.5rem; border-radius: 0.3rem; margin: 0.2rem 0; display: inline-block; }
    .stSpinner > div > div { border-top-color: #4CAF50 !important; }
    .metric-card { background-color: #1e2129; padding: 1rem; border-radius: 0.5rem; border: 1px solid #3d4554; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Parliamentary AI Assistant")
st.caption("100% Local: Mistral 7B Analytics + Llama Chat")

# Initialize state
if "keyword" not in st.session_state:
    st.session_state.keyword = None
    st.session_state.inference = None
    st.session_state.messages = []

# --- STAGE 1: Keyword Inference ---
if not st.session_state.keyword:
    st.markdown("### Step 1: Deep Analytical Inference")
    st.markdown("Enter a keyword to map the parliamentary landscape (e.g., Agriculture, Cyber Security, Healthcare).")
    
    keyword_input = st.text_input("Enter Keyword:", placeholder="Type a topic...")
    if st.button("Generate Topic Inference"):
        if keyword_input.strip():
            with st.spinner(f"Routing to Local Mistral 7B for '{keyword_input}'..."):
                inference_result = generate_local_inference(keyword_input)
                st.session_state.keyword = keyword_input
                st.session_state.inference = inference_result
                st.rerun()

# --- STAGE 2: Q/A & Follow-up Chat ---
else:
    # Sidebar control
    with st.sidebar:
        st.markdown(f"**Current Topic:** {st.session_state.keyword}")
        if st.button("Start Over"):
            st.session_state.keyword = None
            st.session_state.messages = []
            st.rerun()

    main_tab1, main_tab2 = st.tabs(["📊 Policy Inference Dashboard", "💬 Q&A Chat System"])
    with main_tab1:
        # Display the broad inference summary
        st.markdown(f"### 📊 Deep Analysis: {st.session_state.keyword}")
        
        analysis = st.session_state.inference.get("analysis", {})
        
        # 6-Part Structure Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📑 Summary", "⏱️ Timeline", "🏛️ Ministries", "❓ Gaps", "⚔️ Controversies", "📈 Visualizations"])
        
        with tab1:
            colA, colB = st.columns([2, 1])
            with colA:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"**Executive Summary:**\n\n{analysis.get('Summary', 'No summary available.')}")
                st.markdown("</div>", unsafe_allow_html=True)
            with colB:
                sentiment = analysis.get('Sentiment', 'Neutral Inquiry')
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"**AI Sentiment / Urgency:**\n\n### {sentiment}")
                st.progress(1.0 if 'Critical' in sentiment or 'Urgent' in sentiment else (0.5 if 'Neutral' in sentiment else 0.2))
                st.markdown("</div>", unsafe_allow_html=True)
            
        with tab2:
            st.markdown("**Chronological Events:**")
            timeline = analysis.get('Timeline', [])
            if isinstance(timeline, str):
                st.markdown(f"- {timeline}")
            else:
                for event in timeline:
                    st.markdown(f"- {event}")
            
        with tab3:
            st.markdown("**Key Ministries & Stakeholders:**")
            mins = analysis.get('Ministries', [])
            if isinstance(mins, str):
                st.markdown(f"- {mins}")
            else:
                for item in mins:
                    st.markdown(f"- {item}")
            
        with tab4:
            st.markdown("**Information Gaps & Missing Context:**")
            gaps = analysis.get('Gaps', [])
            if isinstance(gaps, str):
                st.markdown(f"- {gaps}")
            else:
                for item in gaps:
                    st.markdown(f"- {item}")
            
        with tab5:
            st.markdown("**Debates & Controversies:**")
            controversies = analysis.get('Controversies', [])
            if isinstance(controversies, str):
                st.markdown(f"- {controversies}")
            else:
                for item in controversies:
                    st.markdown(f"- {item}")
    
        with tab6:
            st.markdown("#### Macro Trends & Engagement")
            # Query DuckDB for real data
            try:
                con = duckdb.connect("parliament_v2.duckdb", read_only=True)
                # We search for records containing the keyword in the title or text
                query = f"SELECT * FROM parliamentary_documents WHERE title ILIKE '%{st.session_state.keyword}%' LIMIT 1000"
                df = con.execute(query).df()
                
                if not df.empty:
                    import json
                    import re
                    
                    # Extract house and ministry from metadata JSON
                    df['metadata_dict'] = df['metadata'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
                    df['house'] = df['metadata_dict'].apply(lambda x: x.get('house', ''))
                    
                    # Extract Ministry from raw_markdown
                    def extract_ministry(text):
                        if not text: return ""
                        m = re.search(r"MINISTRY OF\s+([A-Z &]+)", str(text))
                        if m: return m.group(1).strip()
                        return ""
                    df['ministry'] = df['raw_markdown'].apply(extract_ministry)
                    
                    # Extract Member Names from raw_markdown
                    def extract_members(text):
                        if not text: return ""
                        matches = re.findall(r"(?:SHRI|SHRIMATI|DR\.|PROF\.)\s+([A-Z\s\.]+)(?:$|\n|:)", str(text))
                        if not matches:
                            m2 = re.search(r"\*\d+\.\s+([A-Z\s\.]+):", str(text))
                            if m2: return m2.group(1)
                        return ",".join([m.strip() for m in matches if len(m) > 3])
                    df['member_name'] = df['raw_markdown'].apply(extract_members)
                    
                    # Extract Question Type from raw_markdown
                    def extract_type(text):
                        return 'STARRED (ORAL)' if 'STARRED' in str(text).upper() else 'UNSTARRED (WRITTEN)'
                    df['clean_type'] = df['raw_markdown'].apply(extract_type)
                    
                    viz_col1, viz_col2, viz_col3 = st.columns(3)
                    
                    # Chart A: House Breakdown (Donut)
                    with viz_col1:
                        if 'house' in df.columns:
                            house_counts = df['house'].value_counts().reset_index()
                            house_counts.columns = ['House', 'Count']
                            fig_house = px.pie(
                                house_counts, values='Count', names='House', hole=0.6,
                                color_discrete_sequence=['#00F0FF', '#FF4F00']
                            )
                            fig_house.update_layout(
                                title="House Distribution", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E2E8F0'), margin=dict(l=20, r=20, t=40, b=20), showlegend=False
                            )
                            fig_house.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_house, use_container_width=True)
                            st.caption("💡 Shows which House is driving the conversation.")
                    
                    # Chart B: Ministry Engagement (Bar)
                    with viz_col2:
                        if 'ministry' in df.columns:
                            df_min = df[df['ministry'].astype(str).str.strip() != '']
                            min_counts = df_min['ministry'].value_counts().head(5).reset_index()
                            min_counts.columns = ['Ministry', 'Questions']
                            min_counts['Ministry'] = min_counts['Ministry'].apply(lambda x: (x[:25] + '...') if len(x) > 25 else x)
                            
                            fig_min = px.bar(
                                min_counts, x='Questions', y='Ministry', orientation='h',
                                color_discrete_sequence=['#FF4F00']
                            )
                            fig_min.update_layout(
                                title="Top 5 Ministries", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E2E8F0'), yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_min, use_container_width=True)
                            st.caption("💡 Identifies primary bureaucratic jurisdictions.")
                            
                    # Chart C: Topic Progression (Area Chart)
                    with viz_col3:
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            df['YearMonth'] = df['date'].dt.to_period('M').astype(str)
                            df_time = df[df['YearMonth'] != 'NaT']
                            time_counts = df_time.groupby('YearMonth').size().reset_index(name='Volume')
                            time_counts = time_counts.sort_values('YearMonth')
                            
                            fig_time = px.area(
                                time_counts, x='YearMonth', y='Volume', color_discrete_sequence=['#00F0FF']
                            )
                            fig_time.update_layout(
                                title="Discussion Volume Over Time", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E2E8F0'), margin=dict(l=20, r=20, t=40, b=20), xaxis_title=None, yaxis_title=None
                            )
                            st.plotly_chart(fig_time, use_container_width=True)
                            st.caption("💡 Visualizes the urgency lifecycle.")
    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### Political & Regional Dynamics")
                    viz_col4, viz_col5, viz_col6 = st.columns(3)
                    
                    # Chart D: Question Type (Starred vs Unstarred)
                    with viz_col4:
                        type_counts = df['clean_type'].value_counts().reset_index()
                        type_counts.columns = ['Type', 'Count']
                        fig_type = px.pie(
                            type_counts, values='Count', names='Type', hole=0.6,
                            color_discrete_map={'STARRED (ORAL)': '#FF4F00', 'UNSTARRED (WRITTEN)': '#2D3139'}
                        )
                        fig_type.update_layout(
                            title="Urgency (Starred vs Unstarred)", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#E2E8F0'), margin=dict(l=20, r=20, t=40, b=20), showlegend=False
                        )
                        fig_type.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_type, use_container_width=True)
                        st.caption("💡 Starred questions force Ministers to debate on the floor.")
    
                    # Chart E: Regional Focus (State-wise Treemap)
                    with viz_col5:
                        states_list = [
                            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", 
                            "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", 
                            "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", 
                            "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu", "Kashmir", "Ladakh"
                        ]
                        df['region'] = df['title'].astype(str).apply(
                            lambda x: next((s for s in states_list if s.lower() in x.lower()), "National / General")
                        )
                        region_counts = df[df['region'] != "National / General"]['region'].value_counts().reset_index()
                        region_counts.columns = ['State', 'Mentions']
                        
                        if not region_counts.empty:
                            fig_region = px.treemap(
                                region_counts, path=['State'], values='Mentions',
                                color='Mentions', color_continuous_scale='sunset'
                            )
                            fig_region.update_layout(
                                title="Regional Priority Focus", margin=dict(l=10, r=10, t=40, b=10),
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0')
                            )
                            st.plotly_chart(fig_region, use_container_width=True)
                        else:
                            st.info("Topic is entirely National.")
                        st.caption("💡 Reveals geographic imbalances in funding and attention.")
    
                    # Chart F: Top Parliamentarians
                    with viz_col6:
                        df_mems = df[df['member_name'].astype(str).str.strip() != '']
                        all_mems = df_mems['member_name'].astype(str).str.replace(';', ',').str.split(',')
                        mem_counts = all_mems.explode().str.strip().value_counts().head(5).reset_index()
                        mem_counts.columns = ['Member', 'Questions']
                        
                        if not mem_counts.empty:
                            fig_mem = px.bar(
                                mem_counts, x='Questions', y='Member', orientation='h', color_discrete_sequence=['#00F0FF']
                            )
                            fig_mem.update_layout(
                                title="Top Questioning MPs", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E2E8F0'), yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_mem, use_container_width=True)
                        else:
                            st.info("No specific MP names identified.")
                        st.caption("💡 Highlights lawmakers championing or scrutinizing this topic.")
                else:
                    st.info("Not enough data to generate charts.")
            except Exception as e:
                st.error(f"Error loading charts: {e}")
    
        with st.expander("🔍 View Raw Extraction Evidence (ChromaDB)"):
            timeline = st.session_state.inference.get("Timeline", [])
            if isinstance(timeline, str):
                st.write(timeline)
            else:
                for item in timeline:
                    st.markdown(f"- {item}")
            for para in st.session_state.inference.get("vector_paragraphs", []):
                st.info(para, icon="📄")
    
    with main_tab2:
        st.markdown("### 💬 Detailed Q&A")
        st.caption("Ask specific questions about the topic above.")

        # Render previous chat messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "metadata" in msg:
                    with st.expander("🔍 View AI Reasoning & Evidence"):
                        st.markdown("**1. Semantic Connections (Knowledge Graph)**")
                        if msg["metadata"].get("graph_facts"):
                            import re
                            for fact in msg["metadata"]["graph_facts"]:
                                m = re.match(r"(.*?)\s*-\[(.*?)\]->\s*(.*)", fact)
                                if m:
                                    src, rel, dst = m.groups()
                                    rel_clean = rel.replace("_", " ").title()
                                    st.markdown(f"🔹 **{src}** is structurally linked to **{dst}** (Relationship: *{rel_clean}*)")
                                else:
                                    st.markdown(f"🔹 {fact}")
                        else:
                            st.info("No explicit graph relationships were used.")
                        
                        st.markdown("**2. Source Documents (Vector DB)**")
                        for para in msg["metadata"].get("vector_paragraphs", []):
                            st.info(para, icon="📄")
                if "follow_ups" in msg:
                    if i == len(st.session_state.messages) - 1:
                        st.markdown("**💡 Suggested Follow-up Questions:**")
                        import re
                        lines = [line.strip() for line in msg["follow_ups"].split('\n') if line.strip()]
                        for idx, line in enumerate(lines):
                            clean_q = re.sub(r"^(\d+\.|\-|\*)\s*", "", line).strip()
                            if clean_q:
                                if st.button(clean_q, key=f"btn_{i}_{idx}"):
                                    st.session_state.next_question = clean_q
                                    st.rerun()
                    else:
                        st.info("**Suggested Follow-up Questions (Mistral 7B):**\n\n" + msg["follow_ups"], icon="💡")

        # Chat input
        prompt = st.chat_input("Ask a follow-up question...")
        
        if "next_question" in st.session_state:
            prompt = st.session_state.next_question
            del st.session_state.next_question
            
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Step 1: Answer Generation (Mistral 7B)
                with st.spinner("Retrieving facts & generating answer..."):
                    result = answer_question(prompt, st.session_state.messages)
                    answer = result["answer"]

                st.markdown(answer)
                with st.expander("🔍 View AI Reasoning & Evidence"):
                    st.markdown("**1. Semantic Connections (Knowledge Graph)**")
                    if result.get("graph_facts"):
                        import re
                        for fact in result["graph_facts"]:
                            m = re.match(r"(.*?)\s*-\[(.*?)\]->\s*(.*)", fact)
                            if m:
                                src, rel, dst = m.groups()
                                rel_clean = rel.replace("_", " ").title()
                                st.markdown(f"🔹 **{src}** is structurally linked to **{dst}** (Relationship: *{rel_clean}*)")
                            else:
                                st.markdown(f"🔹 {fact}")
                    else:
                        st.info("No explicit graph relationships were used.")
                        
                    st.markdown("**2. Source Documents (Vector DB)**")
                    for para in result.get("vector_paragraphs", []):
                        st.info(para, icon="📄")

                # Step 2: Insight Generation (Mistral 7B)
                with st.spinner("Mistral 7B is analyzing the response for critical follow-ups..."):
                    follow_ups = generate_followups(answer)
                    st.info("**Suggested Follow-up Questions (Mistral 7B):**\n\n" + follow_ups, icon="💡")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": result,
                "follow_ups": follow_ups
            })
            st.rerun()
