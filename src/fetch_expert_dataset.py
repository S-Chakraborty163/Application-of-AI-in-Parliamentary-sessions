import duckdb
from datasets import load_dataset
import pandas as pd

DB_PATH = "parliament_v2.duckdb"

def fetch_expert_dataset():
    print("Connecting to HuggingFace to download OpenAssistant/oasst1 expert dialogues...")
    # Load a small slice of the dataset to save memory
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    # We want to extract multi-turn conversations where the assistant asks questions
    # or provides deep context. For simplicity in the PoC, we will grab high-quality threads.
    
    df = dataset.to_pandas()
    # Filter for english
    df = df[df['lang'] == 'en']
    
    # We need to reconstruct conversation threads.
    # OASST1 uses message_id and parent_id
    print(f"Loaded {len(df)} English messages. Reconstructing expert QA pairs...")
    
    # Simplified extraction: just find a user prompt and the assistant's reply.
    # We will simulate "Context" -> "Follow up question" pattern
    # For a real pipeline, we'd look for User -> Assistant -> User (Follow up)
    
    # Let's find user messages that act as follow-ups (messages with a parent that is an assistant message)
    assistant_msgs = df[df['role'] == 'assistant']
    user_msgs = df[df['role'] == 'prompter']
    
    # Join user messages to their parent assistant messages
    # Parent (Assistant Context) -> Child (User Follow-up)
    follow_ups = pd.merge(user_msgs, assistant_msgs, left_on='parent_id', right_on='message_id', suffixes=('_followup', '_context'))
    
    # Filter for substantial follow-ups
    follow_ups = follow_ups[follow_ups['text_followup'].str.endswith('?')]
    follow_ups = follow_ups[follow_ups['text_followup'].str.len() > 30] # meaningful questions
    
    print(f"Found {len(follow_ups)} high-quality professional follow-up examples.")
    
    # Take a sample of 1000 for the pattern training
    sample = follow_ups.head(1000)
    
    # Insert into DuckDB
    print("Inserting into DuckDB external_pattern_dataset...")
    conn = duckdb.connect(DB_PATH)
    
    for _, row in sample.iterrows():
        context = row['text_context'].replace("'", "''")
        question = row['text_followup'].replace("'", "''")
        
        conn.execute(f"""
            INSERT INTO external_pattern_dataset (source_dataset, conversation_turn, context, followup_question)
            VALUES ('OpenAssistant/oasst1', 2, '{context}', '{question}')
        """)
        
    conn.close()
    print("SUCCESS: Successfully saved expert patterns to database!")

if __name__ == "__main__":
    fetch_expert_dataset()
