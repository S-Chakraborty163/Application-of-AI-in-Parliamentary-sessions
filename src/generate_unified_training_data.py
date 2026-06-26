import os
import json
import requests
import random
import time
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

# We import the RAG functions to get the actual graph and vector facts
from graph_rag_retriever import get_vector_context, extract_entities_from_question, get_graph_context

load_dotenv()

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
    raise ValueError("Cloudflare credentials missing in .env file.")

# Cloudflare API Endpoint
API_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
HEADERS = {
    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
    "Content-Type": "application/json"
}
# The model is specified in the URL for Cloudflare, but we keep DEFAULT_MODEL to avoid changing the payload if it includes it.
DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


print("Loading HuggingFace Reference Datasets for Few-Shot Style Transfer...")
hf_dataset = load_dataset('BondFoundry/bondfoundry-legal-sample', split='train')

# Filter for deep, complex examples (long responses)
deep_examples = []
for row in hf_dataset:
    instruction = row.get("instruction", "")
    response = row.get("response", "")
    # We only want highly detailed analytical responses
    if len(response) > 500:
        deep_examples.append(f"Instruction: {instruction}\nResponse: {response}")

STYLE_TRANSFER_SYSTEM_PROMPT_BASE = """You are an elite legal and parliamentary expert. 
Your task is to generate training data for a conversational AI model. 
You will be provided with a Question, Vector DB Text, and Neo4j Knowledge Graph Facts.

You must output a highly intelligent, deeply analytical answer followed immediately by exactly 3 highly specific follow-up questions. 

TRUE FEW-SHOT STYLE TRANSFER:
Below are two real-world examples of literal transcripts of senior partners and experts having deep analytical conversations.
You MUST analyze their sentence structure, authoritative tone, and analytical depth. You MUST mimic this exact linguistic architecture when you generate the answer to the parliamentary question.

=== REFERENCE EXAMPLE 1 ===
{ref_1}

=== REFERENCE EXAMPLE 2 ===
{ref_2}
===========================

FORMAT REQUIREMENT:
You must output exactly this format and nothing else:

[Detailed, intelligent answer here based on the facts provided, using the exact tone and depth of the reference examples]

Follow-ups:
- [Follow up 1]
- [Follow up 2]
- [Follow up 3]
"""

def generate_unified_row(question):
    # Dynamically select 2 random deep examples for this specific generation
    refs = random.sample(deep_examples, min(2, len(deep_examples)))
    ref_1 = refs[0] if len(refs) > 0 else ""
    ref_2 = refs[1] if len(refs) > 1 else ""
    
    system_prompt = STYLE_TRANSFER_SYSTEM_PROMPT_BASE.format(ref_1=ref_1, ref_2=ref_2)

    # 1. Gather RAG Context
    vector_paragraphs = get_vector_context(question)
    if isinstance(vector_paragraphs, list):
        vector_paragraphs = "\n".join(vector_paragraphs)
        
    entities = extract_entities_from_question(question)
    graph_facts = get_graph_context(entities)
    
    context_str = "KNOWLEDGE GRAPH FACTS:\n"
    for fact in graph_facts:
        context_str += f"- {fact}\n"
    context_str += f"\nDOCUMENT TEXT:\n{vector_paragraphs}\n"
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nGenerate the Expert Answer and Follow-ups."
    
    # 2. Hit Cloudflare API
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            if "result" in res_json and "response" in res_json["result"]:
                return res_json["result"]["response"]
            elif "result" in res_json and "choices" in res_json["result"]:
                return res_json["result"]["choices"][0]["message"]["content"]
            else:
                print(f"Unexpected Cloudflare response format: {res_json}")
                return None
        else:
            print(f"Cloudflare Error: {response.text}")
            if response.status_code == 429:
                print("Hit Cloudflare rate limit. Sleeping for 60 seconds...")
                time.sleep(60)
            return None
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

def main():
    print("Loading base questions...")
    questions = []
    with open("custom_fqg_training_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for key in ["followup_q1", "followup_q2", "followup_q3"]:
                q = data.get(key, "")
                if q and len(q) > 15:
                    questions.append(q)
                
    output_file = "unified_master_dataset.jsonl"
    
    # Check if we are resuming
    completed_questions = set()
    mode = "w"
    if os.path.exists(output_file):
        print("Found existing dataset. Resuming from where we left off...")
        mode = "a"
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    q = data["messages"][0]["content"]
                    completed_questions.add(q)
                except Exception:
                    pass
                    
    # Filter out already completed questions
    remaining_questions = [q for q in questions if q not in completed_questions]
    target_total = 500
    needed = target_total - len(completed_questions)
    
    if needed <= 0:
        print(f"Dataset already has {len(completed_questions)} rows. Finished!")
        return

    sampled_q = random.sample(remaining_questions, min(needed, len(remaining_questions)))
    
    print(f"Already completed: {len(completed_questions)} rows. Generating {len(sampled_q)} more rows via Cloudflare Workers AI (Llama 3.3 70B)...")
    
    with open(output_file, mode, encoding="utf-8") as f:
        for q in tqdm(sampled_q, desc="Generating Dataset"):
            generation = generate_unified_row(q)
            if generation:
                row = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": generation}
                    ]
                }
                f.write(json.dumps(row) + "\n")
            
            # Tiny sleep to avoid hitting limits
            time.sleep(0.5)
            
    print(f"\nSuccessfully generated {output_file}!")

if __name__ == "__main__":
    main()
