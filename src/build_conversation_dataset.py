"""
build_conversation_dataset.py
-------------------------------
Pipeline:
  1. Load custom_fqg_training_data.jsonl (962 rows, 3 follow-up questions each)
  2. Score each row for quality -> keep top 300-400 rows
  3. For each kept row, generate answers to all 3 follow-up questions via Groq
  4. Build multi-turn conversation JSONL for round-2 fine-tuning

Output: conversation_dataset.jsonl
"""

import json
import time
import os
import re
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── OpenRouter credentials ───────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "meta-llama/llama-3.1-8b-instruct:free"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://localhost",
    "X-Title": "Sansad-v2-Dataset"
}

INPUT_FILE  = "custom_fqg_training_data.jsonl"
OUTPUT_FILE = "conversation_dataset.jsonl"
TARGET_ROWS = 350   # aim for 300-400

# ── Quality Scoring ─────────────────────────────────────────────────────────
COMPLEXITY_KEYWORDS = [
    "reconcile", "implications", "mechanisms", "disparities", "accountability",
    "provisions", "allocated", "scrutiny", "counterargument", "exacerbate",
    "socio-economic", "implementation", "parliamentary", "ministry", "committee",
    "legislation", "constitutional", "transparency", "jurisdiction", "mandate",
    "oversight", "fiscal", "revenue", "expenditure", "amendment", "clause",
    "evaluate", "analyze", "assess", "compare", "investigate", "address",
    "shortfall", "allocation", "distribution", "monitoring", "reform"
]

WEAK_PATTERNS = [
    r"^what is",
    r"^who is",
    r"^when was",
    r"^where is",
    r"^how many",
]

def score_question(q: str) -> float:
    """Score a single follow-up question for quality (0-100)."""
    score = 0.0
    q_lower = q.lower().strip()

    # 1. Length score (longer = more analytical) — max 35 pts
    length = len(q)
    if length > 300:
        score += 35
    elif length > 200:
        score += 25
    elif length > 150:
        score += 15
    elif length > 100:
        score += 8

    # 2. Complexity keyword presence — max 30 pts
    hits = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in q_lower)
    score += min(hits * 5, 30)

    # 3. Multi-clause structure (commas + conjunctions) — max 15 pts
    clause_markers = [" and ", " while ", " whereas ", " furthermore ", " additionally ", ", "]
    clause_count = sum(q_lower.count(m) for m in clause_markers)
    score += min(clause_count * 3, 15)

    # 4. Question contains a specific actionable ask — max 10 pts
    if any(w in q_lower for w in ["how does", "can the minister", "what measures", "what mechanisms",
                                    "in what ways", "to what extent", "given that", "given the"]):
        score += 10

    # 5. Penalize weak/trivial patterns — minus 20 pts
    for pattern in WEAK_PATTERNS:
        if re.match(pattern, q_lower):
            score -= 20
            break

    # 6. Penalize very short questions
    if length < 80:
        score -= 15

    return max(score, 0.0)

def score_row(row: dict) -> float:
    """Score an entire row (average of 3 follow-up questions)."""
    q1 = row.get("followup_q1", "")
    q2 = row.get("followup_q2", "")
    q3 = row.get("followup_q3", "")
    scores = [score_question(q) for q in [q1, q2, q3] if q]
    return sum(scores) / len(scores) if scores else 0.0

# ── Removed Groq call ────────────────────────────────────────────────────────

# ── Answer generator ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Indian Parliamentary Research Assistant with deep knowledge of Lok Sabha and Rajya Sabha proceedings, government policies, legislative frameworks, and parliamentary procedures.

When answering questions:
- Be precise, analytical, and authoritative
- Reference specific parliamentary mechanisms, ministries, or legislative provisions where relevant
- Provide structured, well-reasoned answers (3-5 paragraphs)
- Use formal parliamentary language
- Never fabricate specific statistics, but frame answers using established parliamentary knowledge

Here is an example of the quality, depth, and tone expected in your responses:

--- EXAMPLE ---
User: What specific clauses or provisions in the National Rural Drinking Water Programme (NRDWP) ensure that funds allocated to States for addressing fluoride-rich water issues are utilized effectively and efficiently, and what mechanisms are in place to monitor and evaluate the implementation of these schemes?

Assistant: The National Rural Drinking Water Programme (NRDWP) is a flagship programme of the Ministry of Jal Shakti, Government of India, aimed at providing safe and adequate drinking water to rural areas. The programme has specific provisions to address the issue of fluoride-rich water, which is a significant concern in many parts of the country. To ensure effective and efficient utilization of funds allocated to States for addressing fluoride-rich water issues, the NRDWP has several clauses and provisions in place.

One of the key provisions is the requirement for States to prepare a State-specific plan for mitigation of fluoride-rich water, which is to be approved by the National Rural Drinking Water Council. This plan must include a detailed strategy for addressing the issue, including the identification of affected areas, the type of mitigation measures to be undertaken, and the budget required. The plan must also specify the roles and responsibilities of various stakeholders, including the State Government, local authorities, and implementing agencies. Furthermore, the NRDWP guidelines stipulate that at least 10% of the total allocation to a State must be earmarked for mitigation of fluoride-rich water, ensuring that a significant portion of the funds are dedicated to addressing this critical issue.

The NRDWP also has a robust monitoring and evaluation framework in place to ensure that the funds allocated for addressing fluoride-rich water issues are utilized effectively. The programme is monitored through a three-tier system, comprising the National Rural Drinking Water Council, the State Water and Sanitation Mission, and the District Water and Sanitation Mission. These bodies are responsible for overseeing the implementation of the programme, including the mitigation of fluoride-rich water, and for ensuring that the funds are utilized in accordance with the approved plan. Additionally, the Ministry of Jal Shakti has established a web-based monitoring system, known as the Integrated Management Information System (IMIS), which tracks the progress of the programme, including the implementation of fluoride mitigation measures, and provides real-time data on the utilization of funds.

The NRDWP also provides for regular evaluation and review of the programme, including the mitigation of fluoride-rich water, through various mechanisms. These include annual reviews by the National Rural Drinking Water Council, mid-term evaluations by independent agencies, and impact assessments by reputable research institutions. These evaluations provide valuable feedback on the effectiveness of the programme and help identify areas for improvement, enabling the Government to take corrective action and ensure that the funds allocated for addressing fluoride-rich water issues are utilized efficiently and effectively. Moreover, the programme is also subject to parliamentary oversight, with the Ministry of Jal Shakti required to submit regular reports to the Parliament on the implementation of the NRDWP, including the progress made in addressing fluoride-rich water issues.

In conclusion, the NRDWP has a comprehensive framework in place to ensure that funds allocated to States for addressing fluoride-rich water issues are utilized effectively and efficiently. The programme's provisions, including the requirement for State-specific plans, the earmarking of funds, and the robust monitoring and evaluation framework, all contribute to this goal.
--- END EXAMPLE ---"""

def generate_answer(question: str) -> str:
    """Generate a high-quality answer for a parliamentary follow-up question using OpenRouter."""
    user_prompt = f"""Answer the following parliamentary research question with depth and precision:

{question}

Provide a comprehensive analytical answer that a senior parliamentary researcher would give."""

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.3,
    }

    for attempt in range(6):
        try:
            resp = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"\n⚠️  Rate limit. Sleeping {wait}s (attempt {attempt+1}/6)...")
                time.sleep(wait)
                continue

            data = resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            
            print(f"\n⚠️  API Error: {data}")
            return ""

        except Exception as e:
            print(f"\n❌ Exception: {e}")
            time.sleep(10)

    return ""

# ── Main pipeline ────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Parliamentary Conversation Dataset Builder")
    print("=" * 65)

    # Step 1: Load and score all rows
    print(f"\n📂 Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"   Loaded {len(rows)} rows")

    # Step 2: Score and filter
    print(f"\n🔍 Scoring rows for quality...")
    scored = [(row, score_row(row)) for row in rows]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_rows = [row for row, score in scored[:TARGET_ROWS] if score > 20]
    print(f"   ✅ Selected {len(top_rows)} high-quality rows (score threshold > 20)")

    # Show score distribution
    scores = [s for _, s in scored[:TARGET_ROWS]]
    print(f"   Score range: {min(scores):.1f} – {max(scores):.1f}")
    print(f"   Average score: {sum(scores)/len(scores):.1f}")

    # Step 3: Check for already-completed rows
    completed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed_ids.add(d.get("source_qa_id"))
        print(f"\n📋 Resuming — {len(completed_ids)} rows already done")

    remaining = [r for r in top_rows if r.get("qa_id") not in completed_ids]
    print(f"   Generating answers for {len(remaining)} remaining rows...")

    if not remaining:
        print("\n✅ All rows already completed!")
        return

    # Step 4: Generate answers and build conversation dataset
    print(f"\n🤖 Generating answers via Google Gemini (1.5 Flash)...")
    print(f"   Estimated time: ~{len(remaining) * 3 * 1.5 / 60:.0f} minutes\n")

    skipped = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for row in tqdm(remaining, desc="Building Dataset"):
            qa_id   = row.get("qa_id", "?")
            q1 = row.get("followup_q1", "").strip()
            q2 = row.get("followup_q2", "").strip()
            q3 = row.get("followup_q3", "").strip()

            if not all([q1, q2, q3]):
                skipped += 1
                continue

            # Generate 3 answers
            a1 = generate_answer(q1)
            a2 = generate_answer(q2)
            a3 = generate_answer(q3)

            if not all([a1, a2, a3]):
                print(f"\n⚠️  Skipping row {qa_id} — empty answer returned")
                skipped += 1
                continue

            # Format as multi-turn conversation
            # Structure: 3 complete Q&A exchanges in one conversation
            conversation = {
                "source_qa_id": qa_id,
                "messages": [
                    {"role": "user",      "content": q1},
                    {"role": "assistant", "content": a1},
                    {"role": "user",      "content": q2},
                    {"role": "assistant", "content": a2},
                    {"role": "user",      "content": q3},
                    {"role": "assistant", "content": a3},
                ]
            }

            out_f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
            out_f.flush()

    # Step 5: Summary
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        final_count = sum(1 for line in f if line.strip())

    print(f"\n{'='*65}")
    print(f"✅ DONE! conversation_dataset.jsonl — {final_count} conversation rows")
    print(f"   Total Q&A exchanges: {final_count * 3}")
    print(f"   Skipped rows: {skipped}")
    print(f"\n📁 Output: {OUTPUT_FILE}")
    print(f"   → Upload this to Kaggle for Round 2 fine-tuning!")
    print("=" * 65)

if __name__ == "__main__":
    main()
