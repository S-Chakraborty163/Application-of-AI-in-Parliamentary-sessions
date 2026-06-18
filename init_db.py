import duckdb
import os

DB_PATH = "parliament_v2.duckdb"

def init_db():
    print("Initializing DuckDB Persistent Warehouse...")
    
    # Connect to DuckDB (creates the file if it doesn't exist)
    conn = duckdb.connect(DB_PATH)
    
    # 1. Table for Parliamentary Documents (Markdown converted via MarkItDown)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_doc_id;
        CREATE TABLE IF NOT EXISTS parliamentary_documents (
            doc_id INTEGER DEFAULT nextval('seq_doc_id') PRIMARY KEY,
            source_type VARCHAR,      -- 'LS_Debate', 'RS_Debate', 'Bill', 'QA'
            session_number VARCHAR,   -- e.g., '17th Lok Sabha'
            date DATE,
            title VARCHAR,
            raw_markdown TEXT,        -- The full markdown output from MarkItDown
            metadata JSON             -- Original URL, PDF path, Ministry involved
        );
    """)
    
    # 2. Table for the Generated Follow-Up QA Dataset
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_fqg_id;
        CREATE TABLE IF NOT EXISTS synthetic_fqg_dataset (
            qa_id INTEGER DEFAULT nextval('seq_fqg_id') PRIMARY KEY,
            doc_id INTEGER,
            base_question TEXT,
            base_answer TEXT,
            followup_q1 TEXT,
            followup_q2 TEXT,
            followup_q3 TEXT,
            cognitive_level VARCHAR,  -- e.g., 'Procedural', 'Impact', 'Contradiction'
            FOREIGN KEY (doc_id) REFERENCES parliamentary_documents(doc_id)
        );
    """)
    
    # 3. Table for External Pattern Training Data (Professional Q&A)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_ext_id;
        CREATE TABLE IF NOT EXISTS external_pattern_dataset (
            ext_id INTEGER DEFAULT nextval('seq_ext_id') PRIMARY KEY,
            source_dataset VARCHAR,
            conversation_turn INTEGER,
            context TEXT,
            followup_question TEXT
        );
    """)

    print(f"SUCCESS: DuckDB successfully initialized at {os.path.abspath(DB_PATH)}")
    print("Tables Created:")
    print(" - parliamentary_documents (ZSTD compressed Markdown)")
    print(" - synthetic_fqg_dataset (Custom generated dataset)")
    print(" - external_pattern_dataset (HuggingFace expert dialogues)")
    
    conn.close()

if __name__ == "__main__":
    init_db()
