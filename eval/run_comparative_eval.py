import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import re

# Import the RAG components from your system
from graph_rag_retriever import extract_entities_from_question, get_graph_context, get_vector_context

load_dotenv(override=True)
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS_TO_TEST = [
    "sansad-v2:latest", 
    "mistral:latest", 
    "llama3.1:latest",
    "qwen2.5:latest"
]

def get_ollama_response(model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        if response.status_code == 200:
            latency = time.time() - start_time
            content = response.json()["message"]["content"]
            return content, latency
    except Exception as e:
        print(f"Ollama error on {model}: {e}")
    return "", time.time() - start_time

def generate_initial_question(topic):
    return f"You are an expert on Indian Parliamentary proceedings. Please explain the key arguments, discussions, and outcomes regarding this topic from the 2019-2024 sessions: '{topic}'"

def build_rag_prompt(model, question, chat_history):
    # Always get Vector Database context
    vector_paragraphs = get_vector_context(question)
    
    context_str = ""
    # Only sansad-v2 gets the full Knowledge Graph Graph RAG
    if "sansad-v2" in model:
        entities = extract_entities_from_question(question)
        graph_facts = get_graph_context(entities)
        context_str += "KNOWLEDGE GRAPH FACTS:\n"
        for fact in graph_facts:
            context_str += f"- {fact}\n"
            
    context_str += "\nORIGINAL DOCUMENT TEXT:\n"
    for p in vector_paragraphs:
        context_str += f"- {p}\n"
        
    system_prompt = "You are an elite Parliamentary Research Assistant. Provide highly engaging, detailed, and deeply insightful answers. "
    if "sansad-v2" in model:
        system_prompt += "Use the provided Knowledge Graph Facts and Document Text to ground your answers in reality."
    else:
        system_prompt += "Use the provided Document Text to ground your answers in reality."
        
    final_prompt = f"System Instruction: {system_prompt}\n\nContext:\n{context_str}\n\n"
    if chat_history:
        final_prompt += "Previous Conversation:\n"
        for msg in chat_history:
            final_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
    final_prompt += f"\nLatest Question: {question}\nAnswer:"
    return final_prompt

def generate_model_followup(model, previous_context):
    prompt = f"You are a critical Indian citizen in a debate. Based on the expert's last response:\n'{previous_context}'\nAsk a sharp, critical follow-up question challenging their points or asking for specific constitutional/legal evidence. 1-2 sentences only."
    msg, lat = get_ollama_response(model, [{"role": "user", "content": prompt}])
    return msg, lat

def double_blind_judge(trace):
    prompt = "You are the Double-Blind Intelligence Judge for the Parliamentary AI Arena.\n\n"
    prompt += "CRITICAL CONTEXT: The AI model being evaluated was trained/RAG-augmented EXCLUSIVELY on an Indian Parliamentary database covering sessions from the last 5 years (2019-2024). You must judge it based on this specific temporal and regional context.\n\n"
    prompt += "Evaluate the following AI model based strictly on its multi-turn conversation trace. Provide scores (0-100) for Faithfulness, Relevance, FollowUpQuality, and Formatting.\n\n"
    prompt += "CRITERIA:\n"
    prompt += "- Faithfulness: Did the Expert hallucinate any Indian constitutional articles, acts, or statistics? Did they bring in data outside the 2019-2024 parliamentary context that they shouldn't know? Deduct heavily for inaccuracies or out-of-scope knowledge.\n"
    prompt += "- Relevance: Did the Expert effectively answer the difficult follow-up question, or did it dodge it? Deduct heavily for dodging.\n"
    prompt += "- FollowUpQuality: Evaluate the User's follow-up question (Turn 3). Was it a high-quality, sharp, and contextually deep challenge to the Expert's first answer? (0-100)\n"
    prompt += "- Formatting: Did it maintain a professional debate structure? (Pass 100 / Fail 0)\n"
    prompt += "- Citation Accuracy (0-100): Did the Expert explicitly mention or cite specific parliamentary sessions, dates, or constitutional articles in their response, or was it just a generic summary?\n"
    prompt += "- Entity Grounding (0-100): Did the response correctly identify and weave in specific Indian politicians, ministries, or bills related to the topic?\n"
    prompt += "- Persuasiveness (0-100): How strongly did the Expert defend their arguments against the User's critical follow-up question?\n"
    prompt += "- Tone & Domain Expertise (0-100): Does the model speak with the authoritative, formal, and precise tone expected of a seasoned Indian Parliamentary Expert, or does it sound like a generic AI?\n\n"
    
    prompt += f"--- ANONYMIZED TRACE ---\n{trace}\n\n"
        
    prompt += "OUTPUT STRICT JSON ONLY containing the scores. Example format:\n"
    prompt += '{"faithfulness": 90, "relevance": 85, "followup_quality": 95, "formatting": 100, "citation_accuracy": 90, "entity_grounding": 88, "persuasiveness": 92, "domain_expertise": 95, "rationale": "..."}\n'

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct"
    
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a strict, JSON-only judge."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024
    }
    
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                text = response.json().get("result", {}).get("response", "")
                
                # Sometimes Cloudflare returns the response as a natively parsed dictionary
                if isinstance(text, dict):
                    return text
                
                # Force string conversion if Cloudflare returns something weird like a list
                if not isinstance(text, str):
                    text = str(text)
                    
                start_idx = text.find('{')
                if start_idx != -1:
                    json_str = text[start_idx:]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        print(f"[Judge Warning] JSON truncated! Extracting raw scores via Regex...")
                        fallback = {}
                        for k in ["faithfulness", "relevance", "followup_quality", "formatting", "citation_accuracy", "entity_grounding", "persuasiveness"]:
                            m = re.search(rf'"{k}"\s*:\s*(\d+)', text, re.IGNORECASE)
                            fallback[k] = int(m.group(1)) if m else 0
                            
                        # Handle potential key hallucination for domain_expertise
                        m_domain = re.search(r'"(?:tone_)?domain_expertise"\s*:\s*(\d+)', text, re.IGNORECASE)
                        fallback["domain_expertise"] = int(m_domain.group(1)) if m_domain else 0
                        
                        fallback["rationale"] = "Rationale truncated by API limit."
                        return fallback
                else:
                    print(f"[Judge Error] No JSON object found in response:\n{text}")
            else:
                print(f"[Judge Error] HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[Judge Error] Request Failed: {e}")
            
        time.sleep(2)
        
    return None

def main():
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        print("Cloudflare API keys missing. The Judge cannot run.")
        return

    with open("final_150_eval_topics.md", "r", encoding="utf-8") as f:
        topics = [line.split(". ", 1)[1].strip() for line in f.readlines() if line.strip() and ". " in line]
        
    print(f"Loaded {len(topics)} topics. Starting rigorous Double-Blind Evaluation...")
    
    results = []
    
    # Check if CSV already exists to resume
    if os.path.exists("double_blind_results.csv"):
        existing_df = pd.read_csv("double_blind_results.csv")
        results = existing_df.to_dict('records')
        print(f"Resuming from existing CSV with {len(results)} records.")
    
    # We evaluate ONE MODEL ENTIRELY before moving to the next.
    # This prevents Ollama from constantly swapping 7B models in and out of VRAM, saving massive amounts of RAM and time.
    for model in MODELS_TO_TEST:
        print(f"\n=========================================")
        print(f"STARTING EVALUATION FOR: {model}")
        print(f"=========================================\n")
        
        for i, topic in enumerate(tqdm(topics, desc=f"Evaluating {model}")):
            # Skip if we already evaluated this topic for this model
            already_done = any(r["Model"] == model and r["Topic"] == topic for r in results)
            if already_done:
                continue
                
            # Turn 1: Neutral initial question
            initial_q = generate_initial_question(topic)
            
            # Turn 2: Expert Answer (With RAG Context)
            t1_prompt = build_rag_prompt(model, initial_q, [])
            ans1, lat1 = get_ollama_response(model, [{"role": "user", "content": t1_prompt}])
            
            # Turn 3: Follow up generated BY THE MODEL UNDER TEST
            follow_up_q, lat_fu = generate_model_followup(model, previous_context=ans1)
            
            # Turn 4: Final Expert Answer (With RAG Context)
            chat_hist = [
                {"role": "user", "content": initial_q},
                {"role": "assistant", "content": ans1}
            ]
            t2_prompt = build_rag_prompt(model, follow_up_q, chat_hist)
            ans2, lat2 = get_ollama_response(model, [{"role": "user", "content": t2_prompt}])
            
            total_latency = lat1 + lat_fu + lat2
            trace_text = f"User: {initial_q}\nExpert: {ans1}\nUser: {follow_up_q}\nExpert: {ans2}"
            
            # Double Blind Grading (One trace at a time)
            scores = double_blind_judge(trace_text)
            if not scores:
                print(f"Judge failed to return scores for {model} on topic {i}. Saving un-scored trace.")
                scores = {"faithfulness": 0, "relevance": 0, "followup_quality": 0, "formatting": 0, "citation_accuracy": 0, "entity_grounding": 0, "persuasiveness": 0, "domain_expertise": 0, "rationale": "Judge API Failed"}
                
            quality = (scores.get("faithfulness", 0) + scores.get("relevance", 0) + scores.get("followup_quality", 0) + scores.get("formatting", 0) + 
                       scores.get("citation_accuracy", 0) + scores.get("entity_grounding", 0) + scores.get("persuasiveness", 0) + scores.get("domain_expertise", 0)) / 8
            eff = quality / total_latency if total_latency > 0 else 0
            
            results.append({
                "Topic": topic,
                "Model": model,
                "Faithfulness": scores.get("faithfulness", 0),
                "Relevance": scores.get("relevance", 0),
                "FollowUp Quality": scores.get("followup_quality", 0),
                "Formatting": scores.get("formatting", 0),
                "Citation Accuracy": scores.get("citation_accuracy", 0),
                "Entity Grounding": scores.get("entity_grounding", 0),
                "Persuasiveness": scores.get("persuasiveness", 0),
                "Domain Expertise": scores.get("domain_expertise", 0),
                "Quality Score": round(quality, 2),
                "Latency (s)": round(total_latency, 2),
                "Cognitive Efficiency": round(eff, 2),
                "Judge Rationale": scores.get("rationale", ""),
                "Raw Trace": trace_text
            })
            
            # Save incrementally
            df = pd.DataFrame(results)
            df.to_csv("double_blind_results.csv", index=False)
            
    print("\nEvaluation Complete! Results saved to double_blind_results.csv")

if __name__ == "__main__":
    main()
