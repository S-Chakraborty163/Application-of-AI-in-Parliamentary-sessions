"""
Unified Sansad AI Policy Analyzer
=================================
A production-grade Streamlit application for scraping, analyzing, 
and chatting with Indian Parliament Q&A documents.
"""

import os
import re
import time
import json
import shutil
import logging
import tempfile
from pathlib import Path
from datetime import datetime
import itertools
import sansad_scraper
from arena_tab import render_arena_tab
from quality_eval_tab import render_quality_eval_tab
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import pdfplumber
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import io
import hashlib
import concurrent.futures
import fitz
import torch
from pinecone import Pinecone, ServerlessSpec 

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load API Keys
load_dotenv(override=True)
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")

# ─── LOGGING & CONFIG ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Sansad Policy AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── AESTHETIC & UI INJECTION ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Chivo:wght@300;400;700&family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Fira+Code:wght@400;500&display=swap');

    :root {
        --bg-deep: #0B0C10;
        --bg-panel: #14161C;
        --text-main: #E2E8F0;
        --text-muted: #8B949E;
        --accent-orange: #FF4F00;
        --accent-cyan: #00F0FF;
        --border-color: #2D3139;
    }

    /* Global Typography */
    .stApp {
        background-color: var(--bg-deep);
        color: var(--text-main);
        font-family: 'Chivo', sans-serif;
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Crimson Text', serif !important;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
    }

    /* Dashboard Header */
    .policy-header {
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 2rem;
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-out;
    }
    .policy-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .policy-header p {
        font-family: 'Fira Code', monospace;
        color: var(--accent-orange);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: var(--bg-panel);
        border-top: 2px solid var(--accent-cyan);
        padding: 1.5rem;
        border-radius: 4px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Fira Code', monospace;
        font-size: 0.9rem;
        color: var(--text-muted);
        border: none !important;
        padding-bottom: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
    }

    /* Chat Elements */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stChatMessageContent"] {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0px !important;
        border-left: 3px solid var(--accent-orange) !important;
        color: var(--text-main) !important;
        font-family: 'Chivo', sans-serif !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Document References */
    .ref-chip {
        display: inline-block;
        background: rgba(0, 240, 255, 0.1);
        border: 1px solid var(--accent-cyan);
        color: var(--accent-cyan);
        padding: 2px 8px;
        font-family: 'Fira Code', monospace;
        font-size: 0.75rem;
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 2px;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ─── CLOUD PERSISTENCE (MOCK) ────────────────────────────────────────────────
class CloudSyncManager:
    """Simulates syncing local SQLite/Chroma state to a Cloud Bucket (e.g., S3)."""
    def __init__(self, cloud_dir=".cloud_backup", local_dir="chroma_db"):
        self.cloud_dir = Path(cloud_dir)
        self.local_dir = Path(local_dir)
        self.cloud_dir.mkdir(exist_ok=True)
        
    def push_to_cloud(self):
        """Simulate uploading vector DB and chat history to cloud."""
        if self.local_dir.exists():
            shutil.copytree(self.local_dir, self.cloud_dir / "chroma_db", dirs_exist_ok=True)
        with open(self.cloud_dir / "chat_history.json", "w") as f:
            json.dump(st.session_state.get("messages", []), f)
            
    def pull_from_cloud(self):
        """Simulate downloading from cloud on new session."""
        if (self.cloud_dir / "chroma_db").exists():
            shutil.copytree(self.cloud_dir / "chroma_db", self.local_dir, dirs_exist_ok=True)
        hist_file = self.cloud_dir / "chat_history.json"
        if hist_file.exists():
            with open(hist_file, "r") as f:
                return json.load(f)
        return []

# ─── SMART KEYWORD PARSER ────────────────────────────────────────────────────
def expand_keywords(user_input: str) -> list[str]:
    """Expands queries like 'drone and/or UAV' into comprehensive search terms."""
    input_lower = user_input.lower().strip()
    
    # Common policy expansions
    expansions_map = {
        # Technology & Innovation
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "uav": "unmanned aerial vehicle",
        "ev": "electric vehicle",
        "iot": "internet of things",
        "r&d": "research and development",
        "it": "information technology",

        # Ministries & Departments
        "meity": "ministry of electronics and information technology",
        "moefcc": "ministry of environment, forest and climate change",
        "mha": "ministry of home affairs",
        "mea": "ministry of external affairs",
        "morth": "ministry of road transport and highways",
        "mof": "ministry of finance",
        "mod": "ministry of defence",

        # Governance, Acts & Schemes
        "niti": "national institution for transforming india",
        "gst": "goods and services tax",
        "mnrega": "mahatma gandhi national rural employment guarantee act",
        "mgnrega": "mahatma gandhi national rural employment guarantee act",
        "rti": "right to information",
        "pil": "public interest litigation",
        "cag": "comptroller and auditor general",
        "pmjay": "pradhan mantri jan arogya yojana",
        "pmkisan": "pradhan mantri kisan samman nidhi",

        # Economy & Business
        "msme": "micro, small and medium enterprises",
        "fdi": "foreign direct investment",
        "gdp": "gross domestic product",
        "psu": "public sector undertaking",
        "pse": "public sector enterprise",
        "rbi": "reserve bank of india",
        "sebi": "securities and exchange board of india",
        "upi": "unified payments interface",
        "npa": "non-performing asset",

        # Defense & Space
        "isro": "indian space research organisation",
        "drdo": "defence research and development organisation",
        "hal": "hindustan aeronautics limited",
        "loc": "line of control",
        "lac": "line of actual control",

        # Social & Demographic
        "sc": "scheduled caste",
        "st": "scheduled tribe",
        "obc": "other backward classes",
        "ews": "economically weaker section",
        "bpl": "below poverty line",
        "ngo": "non-governmental organization"
    }
    
    if " and/or " in input_lower:
        parts = [p.strip() for p in input_lower.split(" and/or ")]
        combinations = []
        combinations.extend(parts)
        combinations.append(" ".join(parts))
        
        # Add acronym expansions
        for part in parts:
            if part in expansions_map:
                combinations.append(expansions_map[part])
                
        return list(set(combinations))
        
    return [input_lower]

# ─── RAG & AI ENGINE ─────────────────────────────────────────────────────────
class PolicyAI:
    def __init__(self):
        # 1. GPU Setup for Embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading embedding model on: {self.device.upper()}")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        
        # 2. Cloud Vector Database Setup (Pinecone)
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY is missing from .env file!")
            
        self.pc = Pinecone(api_key=pinecone_key)
        self.index_name = "sansad-policy-index"
        
        # Create the index if it doesn't exist online yet
        if self.index_name not in self.pc.list_indexes().names():
            log.info("Creating new Pinecone index...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384, # This matches the MiniLM-L6-v2 output size
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)
        
        # 3. LLM Initialization (Cloudflare AI)
        self.llm_client = OpenAI(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
            api_key=CLOUDFLARE_API_TOKEN,
            max_retries=0,
        ) if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN else None

    def ingest_pdf_from_url(self, pdf_url: str, metadata: dict) -> int:
        """Downloads PDF to RAM, extracts text, and stores directly in Pinecone."""
        if not pdf_url: return 0
        
        pdf_hash = hashlib.md5(pdf_url.encode('utf-8')).hexdigest()
        
        # Check if already indexed in Pinecone by looking up the first chunk ID
        try:
            fetch_res = self.index.fetch(ids=[f"{pdf_hash}_0_0"])
            if fetch_res.get("vectors"): 
                return 0 # Already indexed, skip
        except Exception:
            pass

        chunks = []
        try:
            # Download and read in RAM
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            doc = fitz.open(stream=response.content, filetype="pdf")
            for i, page in enumerate(doc):
                text = page.get_text()
                if text:
                    words = text.split()
                    for j in range(0, len(words), 100):
                        chunk_text = " ".join(words[j:j+120])
                        chunks.append({
                            "id": f"{pdf_hash}_{i}_{j}",
                            "text": chunk_text,
                            "meta": {**metadata, "pdf_hash": pdf_hash, "page": i+1}
                        })
            doc.close()
        except Exception as e:
            log.error(f"Failed to process {pdf_url}: {e}")
            return 0

        # Store in Pinecone
        if chunks:
            # Batch encode on the GPU
            embeddings = self.embedder.encode([c["text"] for c in chunks]).tolist()
            
            # Format vectors for Pinecone
            vectors = []
            for c, emb in zip(chunks, embeddings):
                meta = c["meta"].copy()
                meta["text"] = c["text"] # Store the actual text in metadata for retrieval
                vectors.append({
                    "id": c["id"],
                    "values": emb,
                    "metadata": meta
                })
            
            # Upload to Pinecone in batches of 100 (Pinecone's recommended limit)
            for i in range(0, len(vectors), 100):
                self.index.upsert(vectors=vectors[i:i+100])
                
        return len(chunks)

    def retrieve(self, query: str, top_k=6):
        """Retrieves the most relevant chunks from the Pinecone cloud index."""
        # Convert the query to a vector
        query_embedding = self.embedder.encode([query]).tolist()[0]
        
        # Search the cloud database
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if not results.get("matches"): 
            return []
            
        # Reformat the results so the Chat UI can read them perfectly
        return [{"text": match["metadata"]["text"], "meta": match["metadata"]} 
                for match in results["matches"]]

    def generate_inference(self, prompt_type: str, context: str) -> str:
        """Utilizes Llama-3.3-70B for deep policy inference."""
        if not self.llm_client: return "API Key missing."
        
        prompts = {
            "summary": "Based strictly on the provided parliamentary data, write a 3-paragraph executive summary of what has been discussed regarding this topic.",
            "timeline": "Analyze the policy progression. Identify key years and how the conversation evolved. Format as a markdown timeline.",
            "ministry": "Identify which government ministries are engaged in this topic and summarize their specific jurisdictions or responses.",
            "gaps": "Identify 'Policy Gaps'. What questions are repeatedly asked? What issues remain unresolved based on these records?"
        }
        
        sys_prompt = "You are an expert AI public policy analyst. Ground all inferences strictly in the provided text to prevent hallucinations."
        user_prompt = f"{prompts[prompt_type]}\n\nDATA:\n{context}"
        
        try:
            response = self.llm_client.chat.completions.create(
                model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating inference: {e}"

# ─── STATE INITIALIZATION ────────────────────────────────────────────────────
if "app_state" not in st.session_state:
    cloud = CloudSyncManager()
    st.session_state.app_state = {
        "messages": cloud.pull_from_cloud(),
        "scraped_records": [],
        "scraping_active": False,
        "partial_ready": False,
        "keywords_expanded": []
    }
# --- UPDATED: Cached AI Engine ---
@st.cache_resource
def get_ai_engine():
    return PolicyAI()

ai_engine = get_ai_engine()

cloud_manager = CloudSyncManager()

# ─── UI LAYOUT ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="policy-header">
    <h1>Sansad AI Inference Engine</h1>
    <p>Parliamentary Data Retrieval & Policy Analysis System</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "[ 01_DATA_ACQUISITION ]",
    "[ 02_POLICY_INFERENCE ]",
    "[ 03_QUERY_INTERFACE ]",
    "[ 04_MODEL_ARENA ]",
    "[ 05_EVALUATION_MATRIX ]",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: SCRAPE & INDEX
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Search Parameters")
        raw_query = st.text_input("Enter Policy Topic (e.g., 'Drone and/or UAV', 'Artificial Intelligence')", placeholder="Type keyword...")
        source_opt = st.radio("Target House", ["Both Houses", "Lok Sabha", "Rajya Sabha"], horizontal=True)
        
    with col2:
        st.markdown("### Cloud Sync Status")
        st.success("🟢 Active & Connected")
        if st.button("Force Sync to Cloud", width="stretch"):
            cloud_manager.push_to_cloud()
            st.toast("Data synchronized to cloud storage.")

    if st.button("Initialize Acquisition Sequence", type="primary"):
        st.session_state.app_state["scraping_active"] = True
        st.session_state.app_state["keywords_expanded"] = expand_keywords(raw_query)
        st.session_state.app_state["scraped_records"] = []
        st.session_state.app_state["partial_ready"] = False
        st.rerun()

    if st.session_state.app_state.get("scraping_active"):
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text = st.empty()
        col_pause, col_partial = st.columns(2)
        
        interrupt = col_pause.button("🛑 Halt Acquisition")
        partial_view = col_partial.button("⚡ View Partial Analysis Now")
        
        if interrupt or partial_view:
            st.session_state.app_state["scraping_active"] = False
            if partial_view:
                st.session_state.app_state["partial_ready"] = True
            st.rerun()

        # Simulated Asynchronous Scraping & Indexing Loop
        # In a real environment, this connects to the sansad_scraper functions.
        # REAL SCRAPING & IN-MEMORY INTEGRATION
        keywords = st.session_state.app_state["keywords_expanded"]
        status_text.markdown(f"**Executing Search:** Querying Parliament databases for `{', '.join(keywords)}`...")
        
        http_session = requests.Session()
        all_records = []
        
        try:
            # 1. Scrape Lok Sabha (ALL sessions)
            if source_opt in ["Both Houses", "Lok Sabha"]:
                status_text.markdown("**Status:** Scraping all Lok Sabha sessions. This may take a few minutes...")
                for kw in keywords:
                    ls_data = sansad_scraper.ls_scrape(http_session, kw, all_loksabhas=True)
                    all_records.extend(ls_data)
                    
            # 2. Scrape Rajya Sabha (ALL sessions)
            if source_opt in ["Both Houses", "Rajya Sabha"]:
                status_text.markdown("**Status:** Scraping all Rajya Sabha sessions...")
                for kw in keywords:
                    rs_data = sansad_scraper.rs_scrape(http_session, kw)
                    all_records.extend(rs_data)

            # 3. Deduplicate records
            unique_records = {r.get("pdf_url"): r for r in all_records if r.get("pdf_url")}.values()
            final_records = list(unique_records)
            
            # 4. Process PDFs directly into memory (CONCURRENTLY)
            status_text.markdown(f"**Status:** Extracting and indexing {len(final_records)} documents in memory. Please wait...")
            
            progress_bar = st.progress(0)
            completed = 0
            
            # We use max_workers=4 to prevent maxing out your physical RAM
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all PDF ingestion tasks
                futures = [
                    executor.submit(ai_engine.ingest_pdf_from_url, record["pdf_url"], record)
                    for record in final_records
                ]
                
                # Update progress bar as each thread finishes
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    progress_bar.progress(completed / len(final_records))

            # 5. Update State
            st.session_state.app_state["scraped_records"] = final_records
            status_text.success(f"**Complete:** Successfully processed {len(final_records)} historical documents directly into AI memory.")
            
        except Exception as e:
            st.error(f"Acquisition Error: {str(e)}")
            
        st.session_state.app_state["scraping_active"] = False
        st.session_state.app_state["partial_ready"] = True
        cloud_manager.push_to_cloud()
        time.sleep(2)
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: AI INFERENCE DASHBOARD (Expanded Visual Analytics & Scoring)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.app_state.get("partial_ready") and ai_engine.index.describe_index_stats().get('total_vector_count', 0) == 0:
        st.info("No data available. Please run data acquisition in Tab 1.")
    else:
        st.markdown("### Policy Inference Dashboard")
        
        records = st.session_state.app_state.get("scraped_records", [])
        
        # --- 1. QUANTITATIVE VISUALIZATIONS ---
        if records:
            df = pd.DataFrame(records)
            
            st.markdown("#### 1. Macro Trends & Engagement")
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
                    st.plotly_chart(fig_house, width="stretch")
                    st.caption("💡 **Inference:** Shows which House is driving the conversation. Heavy Lok Sabha presence usually indicates direct constituency pressure.")
            
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
                    st.plotly_chart(fig_min, width="stretch")
                    st.caption("💡 **Inference:** Identifies the primary bureaucratic jurisdictions responsible for answering and executing this policy.")
                    
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
                    st.plotly_chart(fig_time, width="stretch")
                    st.caption("💡 **Inference:** Visualizes the urgency lifecycle. Spikes correlate with major news events, budgets, or crises.")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- ROW 2: Political & Regional Dynamics ---
            st.markdown("#### 2. Political & Regional Dynamics")
            viz_col4, viz_col5, viz_col6 = st.columns(3)
            
            # Chart D: Question Type (Starred vs Unstarred)
            with viz_col4:
                if 'type' in df.columns:
                    df['clean_type'] = df['type'].astype(str).str.upper().apply(
                        lambda x: 'STARRED (ORAL)' if 'STARRED' in x and 'UNSTARRED' not in x else ('UNSTARRED (WRITTEN)' if 'UNSTARRED' in x else 'OTHER')
                    )
                    type_counts = df[df['clean_type'] != 'OTHER']['clean_type'].value_counts().reset_index()
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
                    st.plotly_chart(fig_type, width="stretch")
                    st.caption("💡 **Inference:** Starred questions force Ministers to debate on the floor, indicating high political friction/importance.")

            # Chart E: Regional Focus (State-wise Treemap)
            with viz_col5:
                # List of Indian States & UTs for keyword extraction
                states_list = [
                    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", 
                    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", 
                    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", 
                    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu", "Kashmir", "Ladakh"
                ]
                # Extract states mentioned in the question titles
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
                    st.plotly_chart(fig_region, width="stretch")
                else:
                    st.info("No specific state mentions found; topic is entirely National.")
                st.caption("💡 **Inference:** Reveals geographic imbalances. Identifies which states are demanding the most attention or funding.")

            # Chart F: Top Parliamentarians
            with viz_col6:
                if 'members' in df.columns or 'member_name' in df.columns:
                    mem_col = 'members' if 'members' in df.columns else 'member_name'
                    df_mems = df[df[mem_col].astype(str).str.strip() != '']
                    all_mems = df_mems[mem_col].astype(str).str.replace(';', ',').str.split(',')
                    mem_counts = all_mems.explode().str.strip().value_counts().head(5).reset_index()
                    mem_counts.columns = ['Member', 'Questions']
                    
                    fig_mem = px.bar(
                        mem_counts, x='Questions', y='Member', orientation='h', color_discrete_sequence=['#00F0FF']
                    )
                    fig_mem.update_layout(
                        title="Top Questioning MPs", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#E2E8F0'), yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_mem, width="stretch")
                    st.caption("💡 **Inference:** Highlights the specific lawmakers championing or scrutinizing this topic.")

            st.markdown("---")
            
        # ── Build context (shared by scoring + inference) ─────────────────────
        raw_context = "\n".join([
            f"Date: {r.get('date','Unknown')}, Ministry: {r.get('ministry','Unknown')}, Title: {r.get('title','Unknown')}"
            for r in records[:20]
            if any([r.get('date'), r.get('ministry'), r.get('title')])
        ])

        # Fallback: if no scraped metadata, pull real content from Pinecone
        if not raw_context:
            _fb_keywords = st.session_state.app_state.get("keywords_expanded", [])
            _fb_query = " ".join(_fb_keywords) if _fb_keywords else "parliamentary policy questions"
            try:
                _fb_docs = ai_engine.retrieve(_fb_query, top_k=10)
                if _fb_docs:
                    raw_context = "\n---\n".join([d["text"] for d in _fb_docs])
                    st.caption(f"ℹ️ Using {len(_fb_docs)} documents retrieved from Pinecone (no local metadata in session).")
            except Exception:
                pass
            if not raw_context:
                raw_context = "No specific metadata found."

        ALL_ARENA_MODELS = [
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/meta/llama-3-8b-instruct",
            "@cf/google/gemma-3-12b-it",
            "@hf/mistral/mistral-7b-instruct-v0.2",
            "@cf/ibm-granite/granite-4.0-h-micro",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "@cf/zai-org/glm-4.7-flash",
        ]
        MODEL_COLORS_TAB2 = {
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast":     "#4ECDC4",
            "@cf/meta/llama-3-8b-instruct":                 "#FF6B6B",
            "@cf/google/gemma-3-12b-it":                    "#F7B731",
            "@hf/mistral/mistral-7b-instruct-v0.2":         "#A55EEA",
            "@cf/ibm-granite/granite-4.0-h-micro":          "#00F0FF",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b": "#45AAF2",
            "@cf/zai-org/glm-4.7-flash":                    "#FA8231",
        }
        MODEL_LABELS_TAB2 = {
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast":   "Llama 3.3 · 70B",
            "@cf/meta/llama-3-8b-instruct":                 "Llama 3.0 · 8B",
            "@cf/google/gemma-3-12b-it":                    "Gemma 3 · 12B",
            "@hf/mistral/mistral-7b-instruct-v0.2":         "Mistral · 7B",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b": "DeepSeek R1 · 32B",
            "@cf/zai-org/glm-4.7-flash":                    "GLM-4.7 · Flash",
        }

        # ── Model selector for batch runs ──────────────────────────────────────
        already_scored = list(st.session_state.get("ai_scores_arena", {}).keys())
        remaining = [m for m in ALL_ARENA_MODELS if m not in already_scored]
        default_sel = remaining[:3] if remaining else ALL_ARENA_MODELS[:3]

        st.markdown("##### 🎯 Select Models for This Batch")
        if already_scored:
            st.success(f"✅ Already scored: {', '.join(MODEL_LABELS_TAB2.get(m, m) for m in already_scored)}")
        ARENA_MODELS_TAB2 = st.multiselect(
            "Pick models to run (swap API key between batches)",
            options=ALL_ARENA_MODELS,
            default=default_sel,
            format_func=lambda m: MODEL_LABELS_TAB2.get(m, m),
            key="model_selector_tab2",
        )
        if not ARENA_MODELS_TAB2:
            st.warning("Select at least one model to run.")
            ARENA_MODELS_TAB2 = []

        def _safe_json_scores(client, model_id: str, context: str) -> dict:
            """Call model for JSON scores. Robust multi-strategy parser, never throws."""

            def _parse_scores(raw: str) -> dict:
                """Try every strategy to extract progress/impact/reason from raw text."""
                # Strategy 1: strip think blocks, strip fences, try greedy outermost {...}
                cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
                cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned).replace("```", "").strip()
                match = re.search(r"\{(.+)\}", cleaned, re.DOTALL)  # greedy — gets outermost
                if match:
                    try:
                        data = json.loads("{" + match.group(1) + "}")
                        if "progress" in data or "impact" in data:
                            return data
                    except Exception:
                        pass

                # Strategy 2: try parsing the whole cleaned string
                try:
                    data = json.loads(cleaned)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

                # Strategy 3: extract individual fields with regex fallback
                def _grab_int(key):
                    m = re.search(rf'"{key}"\s*:\s*(\d+)', cleaned)
                    return int(m.group(1)) if m else None
                def _grab_str(key):
                    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', cleaned)
                    return m.group(1) if m else None

                prog = _grab_int("progress")
                imp  = _grab_int("impact")
                rsn  = _grab_str("reason")
                if prog is not None or imp is not None:
                    return {
                        "progress": prog or 0,
                        "impact":   imp  or 0,
                        "reason":   rsn  or "Extracted via fallback parser.",
                    }
                raise ValueError(f"No parseable scores found in: {cleaned[:200]}")

            # Two-shot: first attempt with strict JSON instruction,
            # second with a more relaxed instruction if the first fails.
            for attempt, prompt_text in enumerate([
                (
                    "You are a JSON-only API. Respond with NOTHING except a single JSON object.\n"
                    "No explanation, no markdown, no code fences — raw JSON only.\n"
                    "Do not include reasoning traces or thinking tags.\n"
                    "Required keys: \"progress\" (integer 0-100), \"impact\" (integer 0-100), "
                    "\"reason\" (string, max 20 words).\n\n"
                    "Parliamentary data to score:\n" + context
                ),
                (
                    "Analyze the following parliamentary data and rate this policy.\n"
                    "Give your answer as a JSON object with these fields:\n"
                    "  progress: a number from 0 to 100 for implementation status\n"
                    "  impact: a number from 0 to 100 for socio-economic impact\n"
                    "  reason: one short sentence\n"
                    "Do not include reasoning traces or thinking tags.\n"
                    "Example: {\"progress\": 45, \"impact\": 60, \"reason\": \"Moderate progress with high public benefit.\"}\n\n"
                    "Data:\n" + context
                ),
            ]):
                try:
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.1,
                        max_tokens=300,
                    )
                    raw = resp.choices[0].message.content
                    if isinstance(raw, dict):
                        data = raw
                    else:
                        raw = str(raw or "")
                        if not raw.strip():
                            continue
                        data = _parse_scores(raw)
                    return {
                        "progress": max(0, min(100, int(data.get("progress", 0)))),
                        "impact":   max(0, min(100, int(data.get("impact",   0)))),
                        "reason":   str(data.get("reason", "No reason provided.")),
                    }
                except Exception as e:
                    if attempt == 1:
                        try:
                            reason = f"Unparseable output: {(resp.choices[0].message.content or '')[:80]}"
                        except Exception:
                            # If it's a 429, show the actual error so the user knows
                            reason = f"Error: {e}"[:200]
                        return {"progress": 0, "impact": 0, "reason": reason}
            return {"progress": 0, "impact": 0, "reason": "All parse attempts failed."}

        # --- 3. AI PROGRESS & IMPACT ASSESSMENT ─────────────────────────────
        st.markdown("#### 3. AI Progress & Impact Assessment")
        st.caption("Select models above, run a batch, swap API key, run the next batch — results accumulate.")

        # Check which selected models still need scoring
        existing_scores = st.session_state.get("ai_scores_arena", {})
        models_to_score = [m for m in ARENA_MODELS_TAB2 if m not in existing_scores]

        if models_to_score:
            if ai_engine.llm_client:
                run_col, skip_col = st.columns([3, 1])
                run_scoring = run_col.button(f"▶️ Score {len(models_to_score)} model(s)", type="primary", key="run_score_btn")
                skip_col.caption(f"~{len(models_to_score) * 20}s estimated")

                if run_scoring:
                    with st.spinner(f"Scoring {len(models_to_score)} models sequentially …"):
                        def _score_model(mid):
                            time.sleep(15)  # generous stagger for free-tier rate limits
                            return mid, _safe_json_scores(ai_engine.llm_client, mid, raw_context)

                        new_scores: dict[str, dict] = {}
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                            for mid, scores in ex.map(_score_model, models_to_score):
                                new_scores[mid] = scores

                        # MERGE into existing scores (accumulate across batches)
                        merged = dict(existing_scores)
                        merged.update(new_scores)
                        st.session_state.ai_scores_arena = merged
                        st.rerun()
            else:
                st.warning("No LLM client available — check CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.")
        else:
            if ARENA_MODELS_TAB2:
                st.info("✅ All selected models are already scored. Select different models or hit Re-score.")

        if "ai_scores_arena" in st.session_state:
            arena_scores = st.session_state.ai_scores_arena
            scored_models = [m for m in ALL_ARENA_MODELS if m in arena_scores]

            if scored_models:
                # — Score cards (one per scored model) ─────────────────────────
                score_cols = st.columns(len(scored_models))
                for col, mid in zip(score_cols, scored_models):
                    sc = arena_scores.get(mid, {"progress": 0, "impact": 0, "reason": "—"})
                    color = MODEL_COLORS_TAB2[mid]
                    with col:
                        st.markdown(f"""
                        <div style="background:#14161C; border-top:3px solid {color};
                                    padding:0.8rem; border-radius:4px; margin-bottom:0.5rem;">
                            <div style="font-family:'Fira Code',monospace; font-size:0.65rem;
                                        color:{color}; text-transform:uppercase; letter-spacing:0.08em;">
                                {MODEL_LABELS_TAB2[mid]}
                            </div>
                            <div style="display:flex; gap:1rem; margin-top:0.4rem;">
                                <div>
                                    <div style="font-size:0.65rem; color:#8B949E;">Progress</div>
                                    <div style="font-size:1.4rem; font-weight:700; color:#fff;">
                                        {sc['progress']}<span style="font-size:0.7rem; color:#8B949E;">/100</span>
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size:0.65rem; color:#8B949E;">Impact</div>
                                    <div style="font-size:1.4rem; font-weight:700; color:#fff;">
                                        {sc['impact']}<span style="font-size:0.7rem; color:#8B949E;">/100</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(sc["progress"] / 100)
                        st.caption(sc["reason"])

                st.markdown("<br>", unsafe_allow_html=True)

                # — Grouped bar chart comparing progress vs impact ─────────────
                import plotly.graph_objects as go
                labels  = [MODEL_LABELS_TAB2[m] for m in scored_models]
                prog_vals   = [arena_scores.get(m, {}).get("progress", 0) for m in scored_models]
                impact_vals = [arena_scores.get(m, {}).get("impact",   0) for m in scored_models]
                colors = [MODEL_COLORS_TAB2[m] for m in scored_models]

                fig_scores = go.Figure()
                fig_scores.add_trace(go.Bar(
                    name="Implementation Progress",
                    x=labels, y=prog_vals,
                    marker=dict(color=colors),
                    text=prog_vals, textposition="outside",
                ))
                fig_scores.add_trace(go.Bar(
                    name="Socio-Economic Impact",
                    x=labels, y=impact_vals,
                    marker=dict(color=colors, opacity=0.5, pattern_shape="/"),
                    text=impact_vals, textposition="outside",
                ))
                fig_scores.update_layout(
                    barmode="group",
                    title="Progress & Impact Scores — Model Comparison",
                    yaxis=dict(range=[0, 110], title="Score / 100"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_scores, width="stretch")

                # Keep legacy key so other code that may read ai_scores still works
                first_mid = scored_models[0]
                st.session_state.ai_scores = arena_scores.get(first_mid, {})

        if st.button("🔄 Re-score with fresh API calls", key="rescore_btn"):
            st.session_state.pop("ai_scores_arena", None)
            st.session_state.pop("ai_scores", None)
            st.rerun()

        st.markdown("---")

        # --- 4. QUALITATIVE AI INFERENCES ─────────────────────────────────────
        st.markdown("#### 4. Qualitative Policy Inferences")
        st.caption("Each tab shows model outputs side-by-side so you can compare depth, style and accuracy.")

        PROMPT_TYPES  = ["summary", "timeline", "ministry", "gaps"]
        PROMPT_LABELS = {
            "summary":  "Executive Summary",
            "timeline": "Progression Timeline",
            "ministry": "Ministry Engagement",
            "gaps":     "Policy Gaps",
        }
        INFERENCE_PROMPTS_TAB2 = {
            "summary":  "Based strictly on the provided parliamentary data, write a 3-paragraph executive summary of what has been discussed regarding this topic.",
            "timeline": "Analyze the policy progression. Identify key years and how the conversation evolved. Format as a markdown timeline.",
            "ministry": "Identify which government ministries are engaged in this topic and summarize their specific jurisdictions or responses.",
            "gaps":     "Identify 'Policy Gaps'. What questions are repeatedly asked? What issues remain unresolved based on these records?",
        }
        SYS_PROMPT_TAB2 = (
            "You are an expert AI public policy analyst. "
            "Ground all inferences strictly in the provided text to prevent hallucinations. "
            "Do not include reasoning traces, thinking tags, or chain-of-thought in your response."
        )

        def _run_inference(model_id: str, prompt_type: str, context: str) -> tuple:
            """Returns (model_id, prompt_type, text, latency). SDK handles 429 retries."""
            user_prompt = f"{INFERENCE_PROMPTS_TAB2[prompt_type]}\n\nDATA:\n{context}"
            time.sleep(10)  # generous stagger for free-tier rate limits
            t0 = time.perf_counter()
            try:
                resp = ai_engine.llm_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT_TAB2},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=1200,
                    temperature=0.3,
                )
                text = resp.choices[0].message.content or "_(empty response)_"
            except Exception as e:
                text = f"⚠️ Error: {e}"
            lat = time.perf_counter() - t0
            return model_id, prompt_type, text, lat

        if "dash_results_arena" not in st.session_state:
            if ai_engine.llm_client and ARENA_MODELS_TAB2:
                n_calls = len(ARENA_MODELS_TAB2) * 4
                inf_col, est_col = st.columns([3, 1])
                run_inf = inf_col.button(f"▶️ Run Inference ({n_calls} calls)", type="primary", key="run_inf_btn")
                est_col.caption(f"~{n_calls * 15}s estimated")

                if run_inf:
                    with st.spinner(f"Running {len(ARENA_MODELS_TAB2)} models × 4 prompt types …"):
                        tasks = [
                            (mid, pt)
                            for mid in ARENA_MODELS_TAB2
                            for pt  in PROMPT_TYPES
                        ]
                        arena_inference: dict[str, dict[str, dict]] = {
                            mid: {} for mid in ARENA_MODELS_TAB2
                        }
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                            futures = [ex.submit(_run_inference, mid, pt, raw_context) for mid, pt in tasks]
                            for fut in concurrent.futures.as_completed(futures):
                                mid, pt, text, lat = fut.result()
                                arena_inference[mid][pt] = {"text": text, "latency": lat}
                        st.session_state.dash_results_arena = arena_inference
                        st.rerun()
            elif not ARENA_MODELS_TAB2:
                st.info("Select models above to run inferences.")
            else:
                st.warning("No LLM client — cannot run inferences.")

        if "dash_results_arena" in st.session_state:
            arena_inf = st.session_state.dash_results_arena
            dash_tabs = st.tabs([PROMPT_LABELS[pt] for pt in PROMPT_TYPES])

            for tab_obj, pt in zip(dash_tabs, PROMPT_TYPES):
                with tab_obj:
                    inf_cols = st.columns(len(ARENA_MODELS_TAB2))
                    for col, mid in zip(inf_cols, ARENA_MODELS_TAB2):
                        with col:
                            color = MODEL_COLORS_TAB2[mid]
                            result = arena_inf.get(mid, {}).get(pt, {})
                            text   = result.get("text", "_(not run)_")
                            lat    = result.get("latency")
                            lat_str = f"{lat:.2f}s" if lat else "—"
                            st.markdown(f"""
                            <div style="font-family:'Fira Code',monospace; font-size:0.65rem;
                                        color:{color}; border-bottom:1px solid #2D3139;
                                        padding-bottom:0.3rem; margin-bottom:0.6rem;">
                                {MODEL_LABELS_TAB2[mid]} &nbsp;·&nbsp; {lat_str}
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(text)

            # — Latency heatmap across models × prompt types ──────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### ⏱ Inference Latency Heatmap (seconds)")
            import plotly.graph_objects as go
            heat_data = []
            for mid in ARENA_MODELS_TAB2:
                row = []
                for pt in PROMPT_TYPES:
                    row.append(round(arena_inf.get(mid, {}).get(pt, {}).get("latency", 0), 2))
                heat_data.append(row)

            fig_heat = go.Figure(go.Heatmap(
                z=heat_data,
                x=[PROMPT_LABELS[pt] for pt in PROMPT_TYPES],
                y=[MODEL_LABELS_TAB2[m] for m in ARENA_MODELS_TAB2],
                colorscale="RdYlGn_r",
                text=heat_data,
                texttemplate="%{text}s",
                showscale=True,
            ))
            fig_heat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"), margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_heat, width="stretch")

        if st.button("🔄 Re-run all inferences", key="rerun_inf_btn"):
            st.session_state.pop("dash_results_arena", None)
            st.rerun()

        # ── 5. EXPORT & SAVE RESULTS ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 5. Export & Save Results")
        st.caption("Download all Tab 2 benchmark data and charts.")

        from datetime import datetime as _ExpDt
        import plotly.graph_objects as go

        _exp_ts = _ExpDt.now().strftime("%Y%m%d_%H%M%S")
        _has_scores = "ai_scores_arena" in st.session_state
        _has_infer = "dash_results_arena" in st.session_state

        if _has_scores or _has_infer:
            _exp_dfs = {}
            _exp_figs = {}

            if _has_scores:
                _sc_data = st.session_state.ai_scores_arena
                _exp_dfs["scores"] = pd.DataFrame([{
                    "Model": MODEL_LABELS_TAB2[mid],
                    "Progress (/100)": _sc_data.get(mid, {}).get("progress", 0),
                    "Impact (/100)": _sc_data.get(mid, {}).get("impact", 0),
                    "Reason": _sc_data.get(mid, {}).get("reason", ""),
                } for mid in ARENA_MODELS_TAB2])

                # Recreate score comparison chart for download
                _lbl = [MODEL_LABELS_TAB2[m] for m in ARENA_MODELS_TAB2]
                _clr = [MODEL_COLORS_TAB2[m] for m in ARENA_MODELS_TAB2]
                _prg = [_sc_data.get(m, {}).get("progress", 0) for m in ARENA_MODELS_TAB2]
                _imp = [_sc_data.get(m, {}).get("impact", 0) for m in ARENA_MODELS_TAB2]
                _fig_sc_exp = go.Figure()
                _fig_sc_exp.add_trace(go.Bar(name="Progress", x=_lbl, y=_prg,
                    marker=dict(color=_clr), text=_prg, textposition="outside"))
                _fig_sc_exp.add_trace(go.Bar(name="Impact", x=_lbl, y=_imp,
                    marker=dict(color=_clr, opacity=0.5, pattern_shape="/"),
                    text=_imp, textposition="outside"))
                _fig_sc_exp.update_layout(barmode="group",
                    title="Progress & Impact Scores — Model Comparison",
                    yaxis=dict(range=[0, 110]), plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"),
                    margin=dict(l=20, r=20, t=50, b=20))
                _exp_figs["score_comparison"] = _fig_sc_exp

            if _has_infer:
                _inf_data = st.session_state.dash_results_arena
                _inf_export = []
                for mid in ARENA_MODELS_TAB2:
                    for pt in PROMPT_TYPES:
                        res = _inf_data.get(mid, {}).get(pt, {})
                        _inf_export.append({
                            "Model": MODEL_LABELS_TAB2[mid],
                            "Prompt Type": PROMPT_LABELS[pt],
                            "Latency (s)": round(res.get("latency", 0), 3),
                            "Output Text": res.get("text", ""),
                        })
                _exp_dfs["inference"] = pd.DataFrame(_inf_export)

                # Recreate latency heatmap for download
                _h_data = []
                for mid in ARENA_MODELS_TAB2:
                    _h_data.append([
                        round(_inf_data.get(mid, {}).get(pt, {}).get("latency", 0), 2)
                        for pt in PROMPT_TYPES
                    ])
                _fig_hm_exp = go.Figure(go.Heatmap(
                    z=_h_data,
                    x=[PROMPT_LABELS[pt] for pt in PROMPT_TYPES],
                    y=[MODEL_LABELS_TAB2[m] for m in ARENA_MODELS_TAB2],
                    colorscale="RdYlGn_r", text=_h_data,
                    texttemplate="%{text}s", showscale=True))
                _fig_hm_exp.update_layout(
                    title="Inference Latency Heatmap (red = slow)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"), margin=dict(l=20, r=20, t=40, b=20))
                _exp_figs["inference_heatmap"] = _fig_hm_exp

            # Data download buttons
            _dl_cols = st.columns(max(len(_exp_dfs), 1))
            for idx, (name, df) in enumerate(_exp_dfs.items()):
                with _dl_cols[idx]:
                    st.download_button(
                        f"📊 {name.title()} CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        f"tab2_{name}_{_exp_ts}.csv", "text/csv",
                        width="stretch", key=f"t2_dl_{name}")

            # Chart download buttons
            if _exp_figs:
                _ch_cols = st.columns(max(len(_exp_figs), 1))
                for idx, (name, fig) in enumerate(_exp_figs.items()):
                    with _ch_cols[idx]:
                        st.download_button(
                            f"📈 {name.replace('_', ' ').title()}",
                            fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                            f"tab2_{name}_{_exp_ts}.html", "text/html",
                            width="stretch", key=f"t2_dl_ch_{name}")
        else:
            st.info("Run the scoring and inference sections above to generate exportable data.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: CHAT INTERFACE (Conversational & Flexible RAG)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Deep Query Interface")
    
    # 1. Render Chat History
    for msg in st.session_state.app_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "refs" in msg and msg["refs"]:
                for ref in msg["refs"]:
                    st.markdown(f"""
                    <a href="{ref['url']}" target="_blank" style="text-decoration: none;">
                        <span class='ref-chip'>🔗 SOURCE: {ref['text']}</span>
                    </a>
                    """, unsafe_allow_html=True)

    # 2. Handle New Queries
    if query := st.chat_input("Ask a policy question or just say hello..."):
        # Append user query to UI
        st.session_state.app_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # A. Retrieve Data (But don't hard-fail if empty)
                docs = ai_engine.retrieve(query)
                refs = []
                
                if docs:
                    context = "\n---\n".join([d["text"] for d in docs])
                    # Extract URLs for clickable links
                    seen_urls = set()
                    for d in docs:
                        url = d['meta'].get('pdf_url', '#')
                        if url not in seen_urls:
                            refs.append({
                                "text": f"{d['meta'].get('ministry', 'Document')} (Pg {d['meta'].get('page', '?')})",
                                "url": url
                            })
                            seen_urls.add(url)
                else:
                    context = "No specific parliamentary records were found for this exact query in the current database."

                # B. Build the Conversational Prompt
                system_instruction = f"""You are Sansad AI, an expert, conversational public policy analyst. 
                You have access to the following parliamentary records: 
                
                <context>
                {context}
                </context>
                
                INSTRUCTIONS:
                - If the context contains the answer, synthesize it clearly and professionally.
                - If the context is partial or missing, DO NOT just say "I don't know." Use your general knowledge as an AI to explain the concept, but explicitly mention that this information is outside the current parliamentary database.
                - If the user is just chatting (e.g., "Hi", "Who are you?"), be polite and conversational.
                - Answer directly without repeating the user's prompt.
                """

                # C. Compile the Chat History (Keep last 6 messages to maintain context without overloading tokens)
                llm_messages = [{"role": "system", "content": system_instruction}]
                
                # Fetch recent history (skip 'refs' as the LLM only needs the text)
                recent_history = st.session_state.app_state["messages"][-6:]
                for msg in recent_history:
                    llm_messages.append({"role": msg["role"], "content": msg["content"]})

                # D. Generate Flexible Answer using Llama-3.3-70B
                try:
                    resp = ai_engine.llm_client.chat.completions.create(
                        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                        messages=llm_messages,
                        max_tokens=1024,
                        temperature=0.5 # Slightly higher temperature for conversational flexibility
                    )
                    response_text = resp.choices[0].message.content
                except Exception as e:
                    response_text = f"API Error: {e}"
                
                # E. Render AI Response & Links
                st.markdown(response_text)
                
                for ref in refs:
                    st.markdown(f"""
                    <a href="{ref['url']}" target="_blank" style="text-decoration: none;">
                        <span class='ref-chip'>🔗 SOURCE: {ref['text']}</span>
                    </a>
                    """, unsafe_allow_html=True)
                
                # Append assistant response to state
                st.session_state.app_state["messages"].append({
                    "role": "assistant", 
                    "content": response_text,
                    "refs": refs
                })
                
                cloud_manager.push_to_cloud()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL ARENA
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    render_arena_tab(ai_engine)

with tab5:
    render_quality_eval_tab(ai_engine)