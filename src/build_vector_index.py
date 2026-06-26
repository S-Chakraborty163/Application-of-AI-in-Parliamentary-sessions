import os
import json
import hashlib
import chromadb
from chromadb.utils import embedding_functions

def get_file_hash(filepath):
    """Calculate MD5 hash of a file to detect changes."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_frontmatter(content):
    """Extract doc_id and title from the YAML frontmatter."""
    lines = content.split('\n')
    doc_id = ""
    title = ""
    body = []
    in_frontmatter = False
    
    for i, line in enumerate(lines):
        if line.strip() == '---':
            if i == 0:
                in_frontmatter = True
                continue
            elif in_frontmatter:
                in_frontmatter = False
                continue
        
        if in_frontmatter:
            if line.startswith('doc_id:'):
                doc_id = line.split(':', 1)[1].strip()
            elif line.startswith('title:'):
                title = line.split(':', 1)[1].strip().strip('"').strip("'")
        else:
            body.append(line)
            
    return doc_id, title, '\n'.join(body)

def chunk_text(text, chunk_size=1000, overlap=200):
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def main():
    print("Initiating Incremental Vector Indexing...")
    
    # 1. Setup Local ChromaDB
    chroma_client = chromadb.PersistentClient(path="./parliament_chroma_db")
    
    # Using a fast, local embedding model
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = chroma_client.get_or_create_collection(
        name="parliament_vectors",
        embedding_function=emb_fn
    )
    
    # 2. Setup Incremental State Tracker (The "CocoIndex" style delta logic)
    status_file = "vector_index_status.json"
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            indexed_files = json.load(f)
    else:
        indexed_files = {}

    data_dir = "parliament_markdowns"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(".md")]
    print(f"Found {len(files)} markdown files in {data_dir}.")
    
    new_chunks = 0
    
    for i, filename in enumerate(files):
        filepath = os.path.join(data_dir, filename)
        file_hash = get_file_hash(filepath)
        
        # Check if file is already indexed and hasn't changed
        if filename in indexed_files and indexed_files[filename] == file_hash:
            continue
            
        # File is new or changed! Process it.
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        doc_id, title, body = extract_frontmatter(content)
        
        # Skip empty documents
        if not body.strip():
            indexed_files[filename] = file_hash
            continue
            
        chunks = chunk_text(body)
        
        # Prepare batch for Chroma
        ids = [f"{doc_id}_chunk_{j}" for j in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "title": title, "chunk_idx": j} for j in range(len(chunks))]
        
        try:
            collection.upsert(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            new_chunks += len(chunks)
        except Exception as e:
            print(f"Error indexing {filename}: {e}")
            continue
            
        # Update state
        indexed_files[filename] = file_hash
        
        # Save state every 100 documents
        if (i + 1) % 100 == 0:
            with open(status_file, 'w') as f:
                json.dump(indexed_files, f)
            print(f"Processed {i + 1} / {len(files)} files. Indexed {new_chunks} new chunks...")

    # Final save
    with open(status_file, 'w') as f:
        json.dump(indexed_files, f)
        
    print(f"\nVector Indexing Complete! Successfully indexed {new_chunks} new text chunks into Local ChromaDB.")

if __name__ == "__main__":
    main()
