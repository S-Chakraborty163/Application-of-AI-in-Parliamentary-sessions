import os
import duckdb
import requests
import json
import time
import re
import concurrent.futures
import threading
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv(override=True)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# Connect to Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Connect to DuckDB
duck_conn = duckdb.connect("parliament_v2.duckdb")

# Create tracking table for graph extraction
duck_conn.execute("""
    CREATE TABLE IF NOT EXISTS graph_status (
        doc_id INTEGER PRIMARY KEY,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Lock for DuckDB concurrent writes
duck_lock = threading.Lock()

# System Prompt for Graph Extraction
SYS_PROMPT = """You are an expert Parliamentary Data Analyst and Knowledge Graph Architect.
Your task is to extract Entities and Relationships from the provided Parliamentary Document.

Identify the following node types:
- Person (e.g., MP names, Ministers)
- Ministry (e.g., Ministry of Agriculture)
- Topic (e.g., Healthcare, Farmers Protest, GST)
- Bill (e.g., Finance Bill, 2021)
- Location (e.g., Maharashtra, Mumbai, India)

Identify logical relationships between them, such as:
- (Person)-[:BELONGS_TO]->(Ministry)
- (Person)-[:RAISED_ISSUE]->(Topic)
- (Person)-[:DISCUSSED]->(Bill)
- (Topic)-[:AFFECTS]->(Location)

You MUST output ONLY a raw JSON object in exactly this format, with NO markdown formatting, NO backticks, and NO additional text:
{
  "nodes": [
    {"id": "Amit Shah", "label": "Person"},
    {"id": "Ministry of Home Affairs", "label": "Ministry"}
  ],
  "edges": [
    {"source": "Amit Shah", "target": "Ministry of Home Affairs", "type": "BELONGS_TO"}
  ]
}
"""

def extract_graph_data(markdown_text):
    prompt = f"Document Text:\n{markdown_text}\n\nExtract the graph JSON:"
    payload = {
        "model": "llama3.2:1b",
        "system": SYS_PROMPT,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    
    # We do not use a short timeout because local inference might take 5-10 seconds per doc
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    
    result = response.json()
    output_text = result.get("response", "{}")
    
    # Clean output (remove markdown blocks if the LLM hallucinated them, though format="json" usually prevents this)
    output_text = output_text.strip()
    if output_text.startswith("```json"):
        output_text = output_text[7:]
    if output_text.startswith("```"):
        output_text = output_text[3:]
    if output_text.endswith("```"):
        output_text = output_text[:-3]
        
    return json.loads(output_text.strip())

def push_to_neo4j(graph_data):
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    with neo4j_driver.session() as session:
        # Create Nodes
        for node in nodes:
            # Clean labels to prevent Neo4j Syntax errors
            raw_label = str(node.get("label", "Entity"))
            clean_label = re.sub(r'[^a-zA-Z0-9_]', '_', raw_label).strip('_')
            if not clean_label:
                clean_label = "Entity"
            
            node_id = str(node.get("id", "")).replace('"', "'")
            
            # Use MERGE to avoid duplicates
            query = f'MERGE (n:{clean_label} {{id: "{node_id}"}})'
            session.run(query)
            
        # Create Edges
        for edge in edges:
            source = str(edge.get("source", "")).replace('"', "'")
            target = str(edge.get("target", "")).replace('"', "'")
            # Clean relationship type
            raw_type = str(edge.get("type", "RELATED_TO"))
            clean_type = re.sub(r'[^a-zA-Z0-9_]', '_', raw_type).upper().strip('_')
            if not clean_type:
                clean_type = "RELATED_TO"
                
            query = f'''
            MATCH (a {{id: "{source}"}})
            MATCH (b {{id: "{target}"}})
            MERGE (a)-[r:{clean_type}]->(b)
            '''
            session.run(query)

def process_document(doc_id, title, raw_markdown):
    print(f"Extracting Graph for Doc #{doc_id}: {title[:50]}...")
    try:
        chunk = raw_markdown[:3000] 
        graph_data = extract_graph_data(chunk)
        push_to_neo4j(graph_data)
        
        with duck_lock:
            duck_conn.execute("INSERT INTO graph_status (doc_id) VALUES (?)", [doc_id])
        print(f"  -> Success ({doc_id})")
        
    except json.JSONDecodeError:
        print(f"  -> Error on doc {doc_id}: LLM returned invalid JSON.")
    except Exception as e:
        err_str = str(e).encode('ascii', 'ignore').decode('ascii')
        print(f"  -> Error on doc {doc_id}: {err_str[:200]}")

def main():
    print("Initiating Phase 3: Knowledge Graph Extraction...")
    
    # Get unprocessed documents
    docs = duck_conn.execute("""
        SELECT doc_id, title, raw_markdown 
        FROM parliamentary_documents 
        WHERE doc_id NOT IN (SELECT doc_id FROM graph_status)
    """).fetchall()
    
    print(f"Found {len(docs)} unprocessed documents for Graph Extraction.")
    
    # Run heavily concurrent extraction using 3 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_document, doc_id, title, raw_markdown) for doc_id, title, raw_markdown in docs]
        concurrent.futures.wait(futures)
        
    print("\nExtraction script finished. Run 'MATCH (n) RETURN n' in Neo4j Browser to view your graph!")

if __name__ == "__main__":
    main()
