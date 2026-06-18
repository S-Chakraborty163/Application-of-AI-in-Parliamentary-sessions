import os
import time
import requests
from pathlib import Path
from markitdown import MarkItDown
import duckdb
import pandas as pd

# We will import the existing scraper logic to get the URLs
import sansad_scraper

DB_PATH = "parliament_v2.duckdb"
md_parser = MarkItDown()

def process_and_ingest(records: list[dict], house: str):
    """Downloads PDFs, converts to Markdown, inserts to DuckDB, deletes PDF."""
    print(f"\n[Ingestion] Processing {len(records)} records for {house}...")
    
    conn = duckdb.connect(DB_PATH)
    http = requests.Session()
    
    # Temporary directory for PDFs
    temp_dir = Path("temp_ingest_pdfs")
    temp_dir.mkdir(exist_ok=True)
    
    success_count = 0
    
    for count, rec in enumerate(records, 1):
        # Check if already exists to prevent duplicates
        safe_title = str(rec.get("title", "")).strip()
        exists = conn.execute("SELECT 1 FROM parliamentary_documents WHERE title=?", (safe_title,)).fetchone()
        if exists:
            print(f"[{count}/{len(records)}] Skipping {safe_title[:30]} (Already in DB)")
            continue

        url = rec.get("pdf_url")
        if not url:
            continue
            
        # Download PDF
        title_tag = sansad_scraper._sanitize(str(rec.get("title", ""))[:60])
        fname = f"temp_{house}_{count}.pdf"
        fpath = temp_dir / fname
        
        print(f"[{count}/{len(records)}] Downloading {title_tag}...")
        try:
            hdr = sansad_scraper.RS_HEADERS if "rsdoc.nic.in" in url else sansad_scraper.LS_HEADERS
            r = sansad_scraper.safe_get(http, url, headers=hdr, stream=True)
            if not r:
                continue
                
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    
            # Parse with MarkItDown
            print(f"[{count}/{len(records)}] Parsing PDF to Markdown with MarkItDown...")
            try:
                md_result = md_parser.convert(str(fpath))
                markdown_text = md_result.text_content
            except Exception as e:
                print(f"MarkItDown Error: {e}")
                markdown_text = ""
                
            if markdown_text.strip():
                # Insert into DuckDB using parameterized queries to avoid SQL quoting errors
                safe_title = str(rec.get("title", "")).strip()
                date_str = str(rec.get("date", "")).strip()
                
                # Fix Date Format: Convert DD.MM.YYYY to YYYY-MM-DD
                if "." in date_str and len(date_str) == 10:
                    try:
                        d, m, y = date_str.split(".")
                        date_str = f"{y}-{m}-{d}"
                    except:
                        pass
                
                # Create a simple JSON metadata string
                import json
                meta = json.dumps({"original_url": url, "house": house})
                
                conn.execute("""
                    INSERT INTO parliamentary_documents (source_type, date, title, raw_markdown, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"{house}_Document", date_str, safe_title, markdown_text, meta))
                success_count += 1
                
        except Exception as e:
            print(f"Pipeline error on record {count}: {e}")
            
        finally:
            # Delete the PDF to save disk space
            if fpath.exists():
                os.remove(fpath)
                
        time.sleep(1) # Be polite to Sansad servers
        
    conn.close()
    print(f"\nSUCCESS: Ingestion Complete: {success_count} documents successfully saved to DuckDB!")


def run_bulk_ingest():
    """
    Full Production Run: Downloads ALL data for the 17th Lok Sabha (recent 5 years).
    Loops through every session and fetches all available documents.
    """
    print("Initiating V2 Data Ingestion Pipeline for ALL data...")
    http = requests.Session()
    
    # Get all Lok Sabha and Session pairs
    print("Fetching session lists...")
    all_pairs = sansad_scraper.ls_get_all_sessions(http)
    
    # Filter for Lok Sabha 17
    ls17_pairs = [(lk, s) for lk, s in all_pairs if lk == 17]
    print(f"Found {len(ls17_pairs)} sessions for the 17th Lok Sabha.")
    
    for lk, ses in ls17_pairs:
        print(f"\n--- Fetching Lok Sabha {lk}, Session {ses} ---")
        try:
            raw_data = sansad_scraper.ls_fetch_session_all(http, lk, ses)
            # Flatten the data to match the format process_and_ingest expects
            records = [sansad_scraper._flatten_ls(q, lk, ses) for q in raw_data]
            print(f"Found {len(records)} records for Session {ses}. Processing...")
            
            # Send the entire session's records to the ingestor
            process_and_ingest(records, "Lok Sabha")
        except Exception as e:
            print(f"Failed to process Lok Sabha {lk} Session {ses}: {e}")

    # 2. Fetch Rajya Sabha (RS) Sessions
    print("\nFetching Rajya Sabha session lists...")
    try:
        rs_sessions = sansad_scraper.rs_get_sessions(http)
        # RS sessions are numbered sequentially. For recent 5 years, grab the latest 15.
        rs_sessions = sorted(rs_sessions, reverse=True)[:15]
        print(f"Found {len(rs_sessions)} recent sessions for Rajya Sabha.")
        
        for ses in rs_sessions:
            print(f"\n--- Fetching Rajya Sabha Session {ses} ---")
            try:
                raw_data = sansad_scraper.rs_fetch_session(http, ses)
                records = [sansad_scraper._flatten_rs(q) for q in raw_data]
                print(f"Found {len(records)} records for RS Session {ses}. Processing...")
                process_and_ingest(records, "Rajya Sabha")
            except Exception as e:
                print(f"Failed to process Rajya Sabha Session {ses}: {e}")
    except Exception as e:
        print(f"Failed to fetch RS sessions: {e}")
            
    print("\nSUCCESS: All sessions for 17th Lok Sabha and recent Rajya Sabha have been processed and saved to DuckDB!")

if __name__ == "__main__":
    run_bulk_ingest()
