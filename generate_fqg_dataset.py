import duckdb
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

DB_PATH = "parliament_v2.duckdb"

def build_prompt_with_patterns(conn):
    """Fetches pattern examples to create a powerful few-shot prompt."""
    examples = conn.execute("SELECT context, followup_question FROM external_pattern_dataset LIMIT 3").fetchall()
    
    prompt = "You are a world-class professional Parliamentary AI. Your task is to read Parliamentary Context, and generate exactly 3 highly professional Follow-Up questions designed to deeply expand the conversation.\n\n"
    prompt += "Use these examples of human expert follow-up patterns as inspiration for the TONE and depth of your questions:\n"
    
    for idx, (ctx, fq) in enumerate(examples, 1):
        prompt += f"\n[Expert Example {idx}]\nContext: {ctx[:200]}...\nFollow-Up Question: {fq}\n"
        
    prompt += """
    
Now, review the following Parliamentary Context and generated Answer.
Generate exactly 3 professional Follow-Up Questions adhering to these cognitive levels:
1. Procedural/Policy: Digging into clauses or hurdles.
2. Impact/Causal: Exploring long term socio-economic impact.
3. Contradiction/Debate: Asking about opposing views or gaps.

Output ONLY a JSON object:
{
    "q1": "...",
    "q2": "...",
    "q3": "..."
}
"""
    return prompt

def generate_fqg_dataset():
    print("Initiating Phase 3: Synthetic FQG Dataset Generation...")
    load_dotenv(override=True)
    cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    cf_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    client = OpenAI(base_url=f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/v1", api_key=cf_token, max_retries=3)
    
    conn = duckdb.connect(DB_PATH)
    sys_prompt = build_prompt_with_patterns(conn)
    
    # Get unprocessed documents
    docs = conn.execute("""
        SELECT doc_id, title, raw_markdown 
        FROM parliamentary_documents 
        WHERE doc_id NOT IN (SELECT doc_id FROM synthetic_fqg_dataset)
        LIMIT 1500
    """).fetchall()
    
    print(f"Found {len(docs)} new documents. Generating Follow-Up Questions...")
    
    for doc_id, title, raw_markdown in docs:
        print(f"Generating for Doc #{doc_id}: {title[:50]}...")
        
        # We simulate a "base answer" by summarizing the document first, or just passing the doc
        # For the dataset, we just pass the raw markdown context
        user_msg = f"Parliamentary Context:\n{raw_markdown[:2000]}\n"
        
        try:
            res = client.chat.completions.create(
                model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3
            )
            
            text = res.choices[0].message.content
            
            # Parse JSON
            if isinstance(text, dict):
                data = text
            else:
                import re, ast
                match = re.search(r'\{.*\}', str(text), re.DOTALL)
                if match:
                    try:
                        data = ast.literal_eval(match.group(0))
                    except:
                        data = json.loads(match.group(0).replace("'", '"'))
                else:
                    raise ValueError("No JSON found")
                    
            # Insert into DuckDB
            conn.execute(f"""
                INSERT INTO synthetic_fqg_dataset (doc_id, base_question, base_answer, followup_q1, followup_q2, followup_q3)
                VALUES ({doc_id}, 'Provide a summary of the document.', 'Summary based on context.', 
                        '{str(data.get("q1", "")).replace("'", "''")}', 
                        '{str(data.get("q2", "")).replace("'", "''")}', 
                        '{str(data.get("q3", "")).replace("'", "''")}')
            """)
            print("  -> Success")
            
        except Exception as e:
            print(f"  -> Error on doc {doc_id}: {e}")
            
        time.sleep(1) # Rate limit
        
    # Export to JSONL for Fine-Tuning
    print("\nExporting final dataset to 'custom_fqg_training_data.jsonl'...")
    conn.execute("COPY (SELECT * FROM synthetic_fqg_dataset) TO 'custom_fqg_training_data.jsonl' (FORMAT JSON);")
    
    conn.close()
    print("SUCCESS: Custom Dataset Generation Complete!")

if __name__ == "__main__":
    generate_fqg_dataset()
