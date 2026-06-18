import os
import duckdb

def main():
    print("Initiating DuckDB Export for CocoIndex...")
    
    # Create the output directory
    output_dir = "parliament_markdowns"
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to DuckDB
    duck_conn = duckdb.connect("parliament_v2.duckdb")
    
    # Fetch all documents
    print("Fetching documents from database...")
    docs = duck_conn.execute("""
        SELECT doc_id, title, raw_markdown 
        FROM parliamentary_documents
    """).fetchall()
    
    total = len(docs)
    print(f"Found {total} documents. Starting export...")
    
    for i, (doc_id, title, raw_markdown) in enumerate(docs):
        # Clean up title for YAML
        safe_title = str(title).replace('"', "'")
        
        # Create YAML frontmatter so CocoIndex captures the metadata!
        frontmatter = f"---\ndoc_id: {doc_id}\ntitle: \"{safe_title}\"\n---\n\n"
        
        # Ensure we don't duplicate the title if the markdown already starts with it
        content = frontmatter + raw_markdown
        
        file_path = os.path.join(output_dir, f"doc_{doc_id}.md")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        if (i + 1) % 5000 == 0:
            print(f"Exported {i + 1} / {total} documents...")
            
    print(f"\nSuccessfully exported all {total} documents to ./{output_dir}/")

if __name__ == "__main__":
    main()
