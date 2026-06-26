"""
Sansad Parliament Q&A Scraper
==============================
LOK SABHA  — uses eparlib.sansad.in (real full-text search engine)
RAJYA SABHA — uses rsdoc.nic.in

LS Architecture (discovered from network tab):
  - eparlib.sansad.in is the Solr-based search backend
  - browse endpoint: lists all Lok Sabhas (1–18) with question counts
  - search endpoint: full-text keyword search across ALL Lok Sabhas/sessions
  - qetFilteredQuestionsAns: used by the UI per-session (we don't need this now)

RS Architecture:
  - rsdoc.nic.in hosts the RS question database
  - Search_Questions?whereclause=ses_no=<n> returns entire session at once
"""

import re
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
import concurrent.futures

import requests
import pandas as pd

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sansad_scraper.log", encoding="utf-8"),
    ],
)
# Fixing Windows console encoding (cp1252 can't handle arrows/unicode)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
log = logging.getLogger(__name__)

# ─── API Endpoints ────────────────────────────────────────────────────────────

# Lok Sabha — eparlib.sansad.in is the Solr search backend
LS_SEARCH_API    = "http://eparlib.sansad.in/restv3/search"
LS_BROWSE_API    = "http://eparlib.sansad.in/restv3/field/browse"
# Per-session API (used for PDF paths if search results lack them)
LS_QUESTIONS_API = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"
LS_SESSIONS_API  = "https://sansad.in/api_ls/question/getAllLoksabhaAndSession"

# Rajya Sabha
RS_QUESTIONS_API = "https://rsdoc.nic.in/Question/Search_Questions"
RS_SESSIONS_API  = "https://rsdoc.nic.in/Question/Get_sessionforQuestionSearch"

# collectionId=3 is the Questions & Answers collection on eparlib
LS_COLLECTION_ID = 3

PAGE_SIZE     = 100
REQUEST_DELAY = 0.8
SESSION_DELAY = 1.0
MAX_RETRIES   = 3

# ─── Headers ──────────────────────────────────────────────────────────────────
LS_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en-IN;q=0.9,en;q=0.8",
    "Referer":         "https://sansad.in/ls/questions/questions-and-answers",
}

EPARLIB_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en-IN;q=0.9,en;q=0.8",
    "Origin":          "https://sansad.in",
    "Referer":         "https://sansad.in/",
}

RS_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en-IN;q=0.9,en;q=0.8",
    "Origin":          "https://sansad.in",
    "Referer":         "https://sansad.in/",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Site":  "cross-site",
}


# ─── HTTP Helper ──────────────────────────────────────────────────────────────
def safe_get(session: requests.Session, url: str, params: dict = None,
             headers: dict = None, stream: bool = False, attempt: int = 1):
    try:
        r = session.get(url, params=params, headers=headers,
                        stream=stream, timeout=60)
        r.raise_for_status()
        return r
    except requests.RequestException as exc:
        if attempt < MAX_RETRIES:
            wait = 2 ** attempt
            log.warning(f"Retry {attempt}/{MAX_RETRIES} in {wait}s — {exc}")
            time.sleep(wait)
            return safe_get(session, url, params, headers, stream, attempt + 1)
        log.error(f"Failed after {MAX_RETRIES} retries: {url} — {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  LOK SABHA
# ══════════════════════════════════════════════════════════════════════════════

def ls_get_all_sessions(http: requests.Session) -> list[tuple[int, int]]:
    r = safe_get(http, LS_SESSIONS_API, params={"locale": "en"}, headers=LS_HEADERS)
    if r:
        try:
            data = r.json()
            pairs = []
            for block in data:
                lk = int(block.get("loksabha") or 0)
                for ses in block.get("sessions") or []:
                    sno = int(ses.get("sessionNo") or 0)
                    if lk and sno:
                        pairs.append((lk, sno))
            if pairs:
                pairs.sort()
                log.info(f"[LS] Found {len(pairs)} sessions from API")
                return pairs
        except Exception:
            pass

    # Fallback: 
    log.info("[LS] Using hardcoded session list (API returned 404)")
    known = {
        13: 14, 14: 15, 15: 15, 16: 17, 17: 15, 18: 7
    }
    pairs = []
    for lk, max_ses in known.items():
        for s in range(1, max_ses + 1):
            pairs.append((lk, s))
    return sorted(pairs)


def ls_fetch_session_all(http: requests.Session, loksabha_no: int,
                         session_no: int) -> list[dict]:
    """
    Fetching ALL questions for one LS session via qetFilteredQuestionsAns,
    then filtering by keyword client-side.
    pageSize=500 to minimise requests.
    """
    records = []
    page = 1

    while True:
        params = {
            "loksabhaNo":    loksabha_no,
            "sessionNumber": session_no,
            "pageNo":        page,
            "pageSize":      500,
            "locale":        "en",
        }
        r = safe_get(http, LS_QUESTIONS_API, params=params, headers=LS_HEADERS)
        if not r:
            break
        try:
            data = r.json()
        except Exception as e:
            log.error(f"[LS] JSON error lk={loksabha_no} s={session_no} p={page}: {e}")
            break

        # Response is a list:
        if isinstance(data, list) and data and isinstance(data[0], dict):
            wrapper = data[0]
        elif isinstance(data, dict):
            wrapper = data
        else:
            wrapper = {}

        # Debugging on very first call only
        if page == 1 and loksabha_no == 18 and session_no == 2:
            log.info(f"[LS DEBUG] wrapper keys: {list(wrapper.keys())}")

        questions = None
        for k in ("listOfQuestions", "questions", "lstQuestion", "results", "content"):
            v = wrapper.get(k)
            if isinstance(v, list):
                questions = v
                break

        if not questions:
            break

        # Real total field is totalRecordSize (from response)
        total = int(wrapper.get("totalRecordSize") or wrapper.get("totalCount")
                    or wrapper.get("total") or 0)

        records.extend(questions)
        log.info(f"[LS] LK{loksabha_no} S{session_no} p{page}: {len(questions)} questions (total:{total})")

        # Stopping if we have all records or got less than a full page
        if len(questions) < 500:
            break
        if total and len(records) >= total:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return records


def _process_single_ls_session(http, lk, ses, kw):
    """Helper function to scrape and filter a single session."""
    records = []
    raw = ls_fetch_session_all(http, lk, ses)
    
    for q in raw:
        title = str(q.get("subjects") or "").lower()
        members = q.get("member") or []
        if isinstance(members, str): members = [members]
        members_str = " ".join(str(m) for m in members).lower()

        if kw not in title: continue
        if kw in members_str: continue

        records.append(_flatten_ls(q, lk, ses))
        
    return records

def ls_scrape(http: requests.Session, keyword: str,
              all_loksabhas: bool = True, loksabha_no: int = 18) -> list[dict]:
    log.info(f"[LS] Scraping keyword='{keyword}' (all_loksabhas={all_loksabhas})")
    all_pairs = ls_get_all_sessions(http)

    if not all_loksabhas:
        all_pairs = [(lk, s) for lk, s in all_pairs if lk == loksabha_no]

    log.info(f"[LS] Searching across {len(all_pairs)} sessions CONCURRENTLY...")
    all_records = []
    kw = keyword.lower()

    # Launch 10 parallel workers to hit the API
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks to the pool
        future_to_session = {
            executor.submit(_process_single_ls_session, http, lk, ses, kw): (lk, ses) 
            for lk, ses in all_pairs
        }
        
        # Gather results as they finish
        for future in concurrent.futures.as_completed(future_to_session):
            lk, ses = future_to_session[future]
            try:
                session_records = future.result()
                if session_records:
                    log.info(f"[LS] LK{lk} S{ses}: {len(session_records)} matches found")
                all_records.extend(session_records)
            except Exception as exc:
                log.error(f"[LS] LK{lk} S{ses} generated an exception: {exc}")

    log.info(f"[LS] Total matching records (after filter): {len(all_records)}")
    return all_records


def _flatten_ls(q: dict, loksabha_no: int, session_no: int) -> dict:
    members = q.get("member") or []
    if isinstance(members, str):
        members = [members]

    pdf = q.get("questionsFilePath") or ""
    if pdf and not pdf.startswith("http"):
        pdf = "https://sansad.in" + pdf

    pdf_hi = q.get("questionsFilePathHindi") or ""
    if pdf_hi and not pdf_hi.startswith("http"):
        pdf_hi = "https://sansad.in" + pdf_hi

    return {
        "house":          "Lok Sabha",
        "lok_sabha_no":   q.get("lokNo") or loksabha_no,
        "session_no":     q.get("sessionNo") or session_no,
        "question_no":    q.get("quesNo") or "",
        "title":          q.get("subjects") or q.get("subject") or "",
        "type":           q.get("type") or "",
        "date":           q.get("date") or "",
        "ministry":       q.get("ministry") or "",
        "members":        "; ".join(str(m) for m in members),
        "pdf_url":        pdf,
        "pdf_url_hindi":  pdf_hi,
        "local_pdf_path": "",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RAJYA SABHA  (via rsdoc.nic.in)
# ══════════════════════════════════════════════════════════════════════════════

def rs_get_sessions(http: requests.Session) -> list[int]:
    """
    Real response shape (from logs):
      [{'ssn_no': 174, 'sessiondate': None}, {'ssn_no': 175, ...}, ...]
    """
    r = safe_get(http, RS_SESSIONS_API, headers=RS_HEADERS)
    if not r:
        log.warning("[RS] Could not fetch sessions — defaulting 250–270")
        return list(range(250, 271))
    try:
        data = r.json()
    except Exception:
        return list(range(250, 271))

    sessions = []
    items = data if isinstance(data, list) else (
        data.get("data") or data.get("result") or []
    )
    for item in items:
        if isinstance(item, dict):
            sno = (item.get("ssn_no") or item.get("ses_no")
                   or item.get("SESSION_NO") or item.get("sessionNo"))
        else:
            sno = item
        try:
            sessions.append(int(sno))
        except (TypeError, ValueError):
            pass

    if not sessions:
        log.warning(f"[RS] Could not parse sessions. Raw: {str(data)[:200]}")
        return list(range(250, 271))

    log.info(f"[RS] Found {len(sessions)} sessions: {sorted(sessions)}")
    return sorted(sessions)


def rs_fetch_session(http: requests.Session, ses_no: int) -> list[dict]:
    """Fetching all RS questions for one session (entire session in one response)."""

    params = {"whereclause": f"ses_no={ses_no}"}
    r = safe_get(http, RS_QUESTIONS_API, params=params, headers=RS_HEADERS)
    if not r:
        return []
    try:
        data = r.json()
    except Exception as e:
        log.error(f"[RS] JSON error session={ses_no}: {e}")
        return []

    if isinstance(data, list):
        return data
    for k in ("questions", "data", "result", "records"):
        v = data.get(k)
        if isinstance(v, list):
            return v
    return []


def _process_single_rs_session(http, ses, kw):
    """Helper function to scrape and filter a single RS session concurrently."""
    records = []
    raw = rs_fetch_session(http, ses)
    
    for q in raw:
        title   = str(q.get("qtitle") or "").lower()
        qn_text = str(q.get("qn_text") or "").lower()
        name    = str(q.get("name") or "").lower()

        if kw not in title and kw not in qn_text:
            continue
        if kw in name:
            continue

        records.append(_flatten_rs(q))
        
    return records


def rs_scrape(http: requests.Session, keyword: str) -> list[dict]:
    """
    Scraping all RS sessions concurrently.
    Keep: keyword in qtitle or qn_text.
    Remove: member name (name field) contains keyword.
    """
    log.info(f"[RS] Scraping keyword='{keyword}'")
    sessions    = rs_get_sessions(http)
    all_records = []
    kw          = keyword.lower()

    log.info(f"[RS] Searching across {len(sessions)} sessions CONCURRENTLY...")

    # Launch parallel workers to hit the RS API
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks to the pool
        future_to_session = {
            executor.submit(_process_single_rs_session, http, ses, kw): ses 
            for ses in sessions
        }
        
        # Gather results as they finish
        for future in concurrent.futures.as_completed(future_to_session):
            ses = future_to_session[future]
            try:
                session_records = future.result()
                if session_records:
                    log.info(f"[RS] Session {ses}: {len(session_records)} matches found")
                all_records.extend(session_records)
            except Exception as exc:
                log.error(f"[RS] Session {ses} generated an exception: {exc}")

    log.info(f"[RS] Total matching records (after filter): {len(all_records)}")
    return all_records


def _flatten_rs(q: dict) -> dict:
    return {
        "house":          "Rajya Sabha",
        "session_no":     q.get("ses_no") or "",
        "question_no":    q.get("qno") or "",
        "qslno":          q.get("qslno") or "",
        "title":          q.get("qtitle") or "",
        "type":           str(q.get("qtype") or "").strip(),
        "date":           q.get("ans_date") or "",
        "ministry":       str(q.get("min_name") or "").strip(),
        "member_name":    q.get("name") or "",
        "status":         str(q.get("status") or "").strip(),
        "question_text":  _strip_html(str(q.get("qn_text") or "")),
        "answer_text":    _strip_html(str(q.get("ans_text") or "")),
        "pdf_url":        q.get("files") or "",
        "pdf_url_hindi":  q.get("hindifiles") or "",
        "local_pdf_path": "",
    }


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  PDF DOWNLOADER
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize(name: str, max_len: int = 100) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:max_len]


def download_pdfs(records: list[dict], pdf_dir: Path,
                  http: requests.Session) -> list[dict]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    to_dl = [(i, r) for i, r in enumerate(records) if r.get("pdf_url")]
    log.info(f"Downloading {len(to_dl)} PDFs → {pdf_dir}")

    for count, (_, rec) in enumerate(to_dl, 1):
        url       = rec["pdf_url"]
        house_tag = "LS" if rec["house"] == "Lok Sabha" else "RS"
        date_tag  = str(rec.get("date", ""))[:10].replace(".", "").replace("-", "")
        qno       = str(rec.get("question_no", ""))
        title_tag = _sanitize(str(rec.get("title", ""))[:60])
        fname     = f"{house_tag}_{date_tag}_Q{qno}_{title_tag}.pdf"
        fpath     = pdf_dir / fname

        if fpath.exists():
            rec["local_pdf_path"] = str(fpath)
            continue

        hdr = RS_HEADERS if "rsdoc.nic.in" in url else LS_HEADERS
        r   = safe_get(http, url, headers=hdr, stream=True)
        if not r:
            rec["local_pdf_path"] = "DOWNLOAD_FAILED"
            continue

        try:
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            rec["local_pdf_path"] = str(fpath)
            log.info(f"[PDF {count}/{len(to_dl)}] {fname}")
        except Exception as exc:
            rec["local_pdf_path"] = f"WRITE_ERROR: {exc}"

        time.sleep(REQUEST_DELAY * 0.5)

    ok = sum(1 for r in records if r.get("local_pdf_path") and
             not str(r["local_pdf_path"]).startswith(("DOWNLOAD", "WRITE", "NO")))
    log.info(f"PDFs: {ok}/{len(to_dl)} downloaded.")
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_excel(ls_records: list[dict], rs_records: list[dict],
               keyword: str, out_dir: Path) -> Path:
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out_dir / f"sansad_{keyword}_{ts}.xlsx"

    with pd.ExcelWriter(fname, engine="openpyxl") as writer:
        if ls_records:
            # Drop debug column before saving
            df = pd.DataFrame(ls_records).drop(columns=["_raw_keys"], errors="ignore")
            df.to_excel(writer, sheet_name="Lok Sabha", index=False)
            _col_width(writer, "Lok Sabha", df)

        if rs_records:
            df = pd.DataFrame(rs_records)
            df.to_excel(writer, sheet_name="Rajya Sabha", index=False)
            _col_width(writer, "Rajya Sabha", df)

        pd.DataFrame({
            "Item":  ["Keyword", "LS records", "RS records",
                      "LS PDFs found", "RS PDFs found", "Generated at"],
            "Value": [keyword, len(ls_records), len(rs_records),
                      sum(1 for r in ls_records if r.get("pdf_url")),
                      sum(1 for r in rs_records if r.get("pdf_url")),
                      ts],
        }).to_excel(writer, sheet_name="Summary", index=False)

    log.info(f"Excel saved: {fname}")
    return fname


def _col_width(writer, sheet: str, df: pd.DataFrame):
    ws = writer.sheets[sheet]
    for col in ws.columns:
        w = max((len(str(c.value or "")) for c in col), default=10)
        ws.column_dimensions[col[0].column_letter].width = min(w + 4, 60)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Sansad Parliament Q&A Scraper")
    p.add_argument("--keyword",        default="damodar")
    p.add_argument("--source",         choices=["ls", "rs", "both"], default="both")
    p.add_argument("--loksabha",       type=int, default=18,
                   help="Specific Lok Sabha number (default: 18). Ignored if --all-loksabhas is set.")
    p.add_argument("--all-loksabhas",  action="store_true",
                   help="Scrape ALL Lok Sabhas (1–18). Default: only --loksabha value.")
    p.add_argument("--no-pdf",         action="store_true")
    p.add_argument("--out-dir",        default="sansad_output")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    http       = requests.Session()
    keyword    = args.keyword.strip()
    ls_records = []
    rs_records = []

    if args.source in ("ls", "both"):
        ls_records = ls_scrape(http, keyword,
                               all_loksabhas=args.all_loksabhas,
                               loksabha_no=args.loksabha)
        if ls_records and not args.no_pdf:
            download_pdfs(ls_records, out_dir / "pdfs" / "ls", http)

    if args.source in ("rs", "both"):
        rs_records = rs_scrape(http, keyword)
        if rs_records and not args.no_pdf:
            download_pdfs(rs_records, out_dir / "pdfs" / "rs", http)

    if ls_records or rs_records:
        excel = save_excel(ls_records, rs_records, keyword, out_dir)
        print(f"\n✅  Done!")
        print(f"   Records : LS={len(ls_records)}, RS={len(rs_records)}")
        print(f"   Excel   : {excel.resolve()}")
        if not args.no_pdf:
            print(f"   PDFs    : {(out_dir / 'pdfs').resolve()}")
    else:
        print("\n⚠️  No records found.")
        print("   Check sansad_scraper.log for details.")


if __name__ == "__main__":
    main()
