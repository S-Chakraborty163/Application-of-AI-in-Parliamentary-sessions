import os
import json
import requests
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(override=True)

# Connect to Local Databases
neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "sansad123")))
chroma_client = chromadb.PersistentClient(path="./parliament_chroma_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="parliament_vectors", embedding_function=emb_fn)

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "sansad-v2"

def ask_llm(prompt, system_prompt="You are a helpful AI assistant.", model_override=None):
    payload = {
        "model": model_override or OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    return ""

def extract_entities_from_question(question):
    """Uses LLM to quickly extract key entities from the user's question to search the graph."""
    if len(question.split()) < 3:
        return [] # Prevent 1B model from hallucinating entities on short acronyms/keywords
    
    prompt = f"Extract the 2 or 3 most important entities (Names, Ministries, Bills, Topics) from this question. Output ONLY a comma-separated list. Question: {question}"
    result = ask_llm(prompt)
    return [e.strip() for e in result.split(",") if e.strip()]

def get_graph_context(entities):
    """Queries Neo4j for the relationships surrounding the extracted entities."""
    context = []
    with neo4j_driver.session() as session:
        for entity in entities:
            # We use a fuzzy text search or a general match
            # Use regex word boundaries to prevent 'nep' matching 'Nepal'
            import re
            safe_entity = re.escape(entity)
            query = f"""
            MATCH (n)-[r]-(m)
            WHERE n.id =~ '(?i).*\\\\b{safe_entity}\\\\b.*'
            RETURN n.id AS source, type(r) AS rel, m.id AS target
            LIMIT 5
            """
            try:
                results = session.run(query)
                for record in results:
                    context.append(f"{record['source']} -[{record['rel']}]-> {record['target']}")
            except Exception:
                pass
    return list(set(context))

def get_vector_context(question):
    """Queries ChromaDB for the most semantically similar paragraphs."""
    try:
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        if results and results['documents']:
            return results['documents'][0]
    except Exception as e:
        print(f"Vector search failed: {e}")
    return []

def answer_question(question, chat_history=None):
    if chat_history is None:
        chat_history = []
        
    print(f"\n--- Processing Question: '{question}' ---")
    
    search_query = question
    if chat_history:
        print("0. Rewriting Query for Context...")
        history_str = ""
        for msg in chat_history[-4:]:
            if msg.get("role") in ["user", "assistant"]:
                history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        rewrite_prompt = f"Chat History:\n{history_str}\n\nLatest Question: {question}\n\nRewrite the latest question into a standalone, detailed search query that can be understood without the chat history. Output ONLY the rewritten query, with no quotes or preamble."
        search_query = ask_llm(rewrite_prompt, system_prompt="You are a search query rewriting AI.", model_override="mistral").strip('\"\'\n ')
        print(f"   -> Rewritten Query: '{search_query}'")
    
    print("1. Extracting Entities...")
    entities = extract_entities_from_question(search_query)
    
    print("2. Traversing Knowledge Graph (Neo4j)...")
    graph_facts = get_graph_context(entities)
    
    print("3. Searching Vector Database (ChromaDB)...")
    vector_paragraphs = get_vector_context(search_query)
    
    # Build Context
    context_str = "KNOWLEDGE GRAPH FACTS:\n"
    for fact in graph_facts:
        context_str += f"- {fact}\n"
        
    context_str += "\nORIGINAL DOCUMENT TEXT:\n"
    for p in vector_paragraphs:
        context_str += f"- {p}\n"
        
    # Final Generation
    print("4. Generating Cognitive Answer...")
    system_prompt = "You are an elite Parliamentary Research Assistant and an expert conversational partner. Your goal is to provide highly engaging, detailed, and deeply insightful answers to the user. You must maintain a natural, conversational tone that encourages the user to keep exploring the topic. Use the provided Knowledge Graph Facts and Document Text to ground your answers in reality, but weave them into a rich, comprehensive, and high-quality narrative. If the specific facts are not in the context, use your conversational skills to explain what you DO know based on the context, rather than just saying 'I don't know'."
    
    final_prompt = f"Context:\n{context_str}\n\n"
    if chat_history:
        final_prompt += "Previous Conversation:\n"
        for msg in chat_history[-4:]:
            if msg.get("role") in ["user", "assistant"]:
                final_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
                
    final_prompt += f"\nLatest Question: {question}\nAnswer:"
    
    # Reverting to base mistral for now until we train the unified model
    answer = ask_llm(final_prompt, system_prompt)
    
    return {
        "answer": answer,
        "entities": entities,
        "graph_facts": graph_facts,
        "vector_paragraphs": vector_paragraphs
    }

def generate_local_inference(keyword):
    """Hits the local Mistral 7B model to generate a strictly structured 4-part JSON inference using Vector context ONLY."""
    print(f"\n--- Generating Local Mistral Inference for: '{keyword}' ---")
    
    # 1. Pull Context (Vector ONLY, no messy graph triplets)
    vector_paragraphs = get_vector_context(f"Overview of {keyword} policies and discussions")
    
    if not isinstance(vector_paragraphs, list):
        vector_paragraphs = [vector_paragraphs]
        
    context_str = "ORIGINAL DOCUMENT TEXT:\n" + "\n".join([f"- {p}" for p in vector_paragraphs])
    
    # 2. Local Mistral API Call
    prompt = f"""You are a top-tier Parliamentary Data Analyst.
Read the following Context regarding '{keyword}'.

Context:
{context_str}

Analyze this data and return ONLY a valid JSON object with exactly these 4 keys. Replace the placeholder text with your actual analysis. Do NOT include markdown code blocks, just the raw JSON string.

{{
    "Summary": "Write a highly detailed, comprehensive 3-4 paragraph executive summary. Provide deep analytical insights, context, and policy implications.",
    "Timeline": ["List the chronological sequence of events. If a specific date is mentioned in the text, include it (e.g., 'March 2023 - Description'). Do NOT use placeholder text like 'DD/MM/YYYY' or '20XX'. If no date is mentioned, simply describe the event without a date."],
    "Ministries": ["List distinct Ministries, Departments, or Politicians involved, followed by a brief 1-sentence explanation of their specific role"],
    "Gaps": ["Identify missing contextual information, unanswered questions, or policy ambiguities that are omitted from the text"],
    "Controversies": ["List points of contention, differing opinions, ethical concerns, or specific criticisms raised by stakeholders/MLAs"],
    "Sentiment": "Rate the overall tone as either 'Highly Critical', 'Neutral Inquiry', 'Supportive', or 'Urgent Crisis', with a 1-sentence justification."
}}"""

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        result = response.json()
        raw_text = result.get("response", "{}")
        
        data = json.loads(raw_text)
            
        return {
            "analysis": data,
            "vector_paragraphs": vector_paragraphs
        }
    except Exception as e:
        print(f"Local Inference failed: {e}")
        return {
            "analysis": {
                "Summary": "Failed to extract summary.",
                "Timeline": ["Failed to extract timeline"],
                "Ministries": ["Failed to extract ministries"],
                "Gaps": ["Failed to extract gaps"],
                "Controversies": ["Failed to extract controversies"],
                "Sentiment": "Unknown"
            },
            "vector_paragraphs": vector_paragraphs
        }

def generate_followups(context_answer):
    """Generates 3 insightful follow-up questions using the fine-tuned Mistral 7B model."""
    print("\n--- Generating Follow-Up Questions (Mistral 7B) ---")
    payload = {
        "model": "sansad-v2",
        "prompt": f"Based on the following answer, generate exactly 3 insightful follow-up questions that the user could ask to dig deeper into the topic. Output ONLY the 3 questions formatted as a bulleted list, with no other text, conversational preamble, or instruction tags.\n\nAnswer:\n{context_answer}",
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        return ""
    return ""

if __name__ == "__main__":
    # Test our Hybrid RAG!
    result = answer_question("How is Amit Shah connected to national security or bills?")
    print("\n================ FINAL ANSWER ================\n")
    print(result["answer"])
    print("\n==============================================\n")
