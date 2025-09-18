#!/usr/bin/env python3
"""
Author Bibliography Status — Open Library + public OverDrive probe

Behavior-first version:
- Processes titles in order (deterministic), probes libraries concurrently.
- Robust de-duplication of OL titles (collapse editions/subtitles/&-vs-and).
- Language filtering to keep English; optional --strict-english to force OL checks.
- Availability preserved (Open Library + OverDrive HTML probe, 🎧 > 📱).
- Works with just authors.txt if no StoryGraph/Goodreads CSV is present.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------- HTTP ---------------------

def tuned_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.2, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retry)
    s.headers["User-Agent"] = "Mozilla/5.0 (CLI; Author Bibliography Status)"
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# --------------------- CSV helpers ---------------------

def read_csv(path: Path) -> Tuple[List[dict], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r), [h.strip() for h in (r.fieldnames or [])]

def detect_source(headers: List[str]) -> str:
    lower = {h.strip().lower() for h in headers}
    has_author = ("author" in lower) or ("author(s)" in lower) or ("authors" in lower)
    is_gr = ("exclusive shelf" in lower) and has_author
    is_sg = ("read status" in lower) and has_author
    return "Goodreads" if is_gr else ("StoryGraph" if is_sg else "Unknown")

def get_author_field(row: dict) -> str:
    for k in ("Author", "Primary Author", "author", "Authors", "Author(s)"):
        v = row.get(k)
        if v:
            return str(v)
    return ""

def get_title_field(row: dict) -> str:
    for k in ("Title", "Book Title", "title", "Work Title", "Name"):
        v = row.get(k)
        if v:
            return str(v)
    return ""

def compute_status(row: dict, source: str) -> str:
    """Map to one of: Unread, Read, Currently Reading, To-Read."""
    if source == "StoryGraph":
        v = (row.get("Read Status") or row.get("read status") or "").strip().lower()
        if v in {"currently reading", "currently-reading"}:
            return "Currently Reading"
        if v in {"read"}:
            return "Read"
        if v in {"to read", "to-read"}:
            return "To-Read"
        return "Unread"
    elif source == "Goodreads":
        shelf = (row.get("Exclusive Shelf") or row.get("exclusive shelf") or "").lower()
        if "currently" in shelf:
            return "Currently Reading"
        if "to-read" in shelf or "to read" in shelf:
            return "To-Read"
        if "read" in shelf:
            return "Read"
        return "Unread"
    else:
        return "Unread"

def build_library_index(export_paths: List[Path], author: str) -> Dict[str, dict]:
    """Return {normalized_title: {row, source}} for a given author across all exports."""
    idx: Dict[str, dict] = {}
    for p in export_paths:
        rows, headers = read_csv(p)
        src = detect_source(headers)
        for row in rows:
            a = get_author_field(row)
            if not a or author.lower() not in a.lower():
                continue
            t = get_title_field(row)
            if not t:
                continue
            idx[normalize(t)] = {"row": row, "source": src}
    return idx

# ------------------------ Open Library ------------------------

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).strip().lower()

def fetch_bibliography_openlibrary(author: str) -> List[str]:
    """Pull title list for author from OL (works-level, best effort)."""
    url = "https://openlibrary.org/search.json"
    params = {"author": author, "fields": "title", "limit": 500}
    s = tuned_session()
    try:
        r = s.get(url, params=params, timeout=12)
        r.raise_for_status()
    except requests.RequestException:
        return []
    data = r.json()
    titles: List[str] = []
    for d in data.get("docs", []):
        t = d.get("title")
        if t and t not in titles:
            titles.append(t)
    return titles

def ol_is_english(title: str, author: str) -> Optional[bool]:
    """Best-effort language check via OL search; returns True/False/None."""
    s = tuned_session()
    try:
        r = s.get("https://openlibrary.org/search.json",
                  params={"title": title, "author": author, "limit": 5, "fields": "language"},
                  timeout=8)
        if r.status_code != 200:
            return None
        j = r.json()
        langs = []
        for d in j.get("docs", []):
            for lang in (d.get("language") or []):
                langs.append(str(lang).lower())
        if not langs:
            return None
        return any(l.startswith("eng") or l == "en" for l in langs)
    except requests.RequestException:
        return None

def ol_search_identifiers(title: str, author: str) -> Tuple[List[str], List[str]]:
    """Return (ISBN list, Internet Archive IDs) for a title."""
    s = tuned_session()
    try:
        r = s.get("https://openlibrary.org/search.json",
                  params={"title": title, "author": author, "limit": 10, "fields": "isbn,ia"},
                  timeout=8)
        if r.status_code != 200:
            return [], []
        j = r.json()
        isbns: List[str] = []
        ia: List[str] = []
        for d in j.get("docs", []):
            for x in (d.get("isbn") or []):
                if x and x not in isbns:
                    isbns.append(x)
            for x in (d.get("ia") or []):
                if x and x not in ia:
                    ia.append(x)
        return isbns, ia
    except requests.RequestException:
        return [], []

def ol_availability_for_ia(ia_ids: List[str]) -> bool:
    """True if any IA id is borrowable/open."""
    if not ia_ids:
        return False
    s = tuned_session()
    try:
        r = s.get("https://openlibrary.org/api/volumes/brief/json/" + ",".join(ia_ids), timeout=8)
        if r.status_code != 200:
            return False
        j = r.json()
        for _k, v in j.items():
            status = (v.get("status") or "").lower()
            if status in {"open", "borrow_available"}:
                return True
        return False
    except requests.RequestException:
        return False

# ------------------------ OverDrive / Libby (public) --------------------

@dataclass
class Library:
    name: str
    libby_key: Optional[str] = None
    overdrive_site: Optional[str] = None  # e.g., "sno-isle.overdrive.com"

def parse_libraries_file(path: Optional[Path]) -> List[Library]:
    libs: List[Library] = []
    if not path or not path.exists():
        return libs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 1:
                libs.append(Library(name=parts[0]))
            elif len(parts) == 2:
                libs.append(Library(name=parts[0], libby_key=parts[1], overdrive_site=f"{parts[1]}.overdrive.com"))
            else:
                libs.append(Library(name=parts[0], libby_key=parts[1] or None, overdrive_site=parts[2] or None))
    return libs

def ensure_overdrive_site(lib: Library) -> Optional[str]:
    if lib.overdrive_site:
        return lib.overdrive_site
    if lib.libby_key:
        return f"{lib.libby_key}.overdrive.com"
    return None

def _ascii_englishish(title: str) -> bool:
    return not re.search(r"[^\x00-\x7F]", title)

def looks_non_english_ascii(title: str) -> bool:
    """Heuristic for ASCII titles that *look* non-English (common ES/DE/FR articles)."""
    s = f" {title.lower()} "
    tokens = [
        " el ", " la ", " los ", " las ", " del ", " de ", " y ",
        " der ", " die ", " das ", " und ",
        " le ", " les ", " des ", " du ", " et ",
        " una ", " uno ", " un "
    ]
    return any(tok in s for tok in tokens)

def _title_present_in_html(title: str, html: str) -> bool:
    words = [w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 2]
    if not words:
        return False
    h = html.lower()
    hits = sum(1 for w in words if w in h)
    return hits >= max(2, min(5, len(words)))

def _extract_formats(html: str) -> Tuple[bool, bool]:
    """Return (has_audio, has_ebook)."""
    h = html.lower()
    has_audio = any(x in h for x in ["audiobook", "audio-book", "icon-audiobook", "format-audiobook", "media-audiobook"])
    has_ebook = any(x in h for x in ["ebook", "e-book", "icon-ebook", "format-ebook", "media-ebook"])
    # JSON-LD sniff
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.I|re.S):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue
        blob = json.dumps(data).lower()
        if "audiobook" in blob: has_audio = True
        if "ebook" in blob or "e-book" in blob: has_ebook = True
    return has_audio, has_ebook

def probe_overdrive_presence(site: str, title: str, author: str, timeout_sec: int = 5, debug: bool = False) -> Optional[str]:
    """
    Return icon: 🎧 if audiobook present, 📱 if ebook present, else None.
    Strategy:
      1) /search/title?query=<title audiobook>, then <title>
      2) /search?query=<title audiobook>, then <title>
    """
    s = tuned_session()
    base_paths = ["search/title", "search"]
    q_variants = [f"{title} audiobook", title]
    best = None
    for path in base_paths:
        for q in q_variants:
            url = f"https://{site}/{path}?query={requests.utils.quote(q)}"
            try:
                r = s.get(url, timeout=timeout_sec)
                if debug:
                    print(f"[probe] {url} -> {r.status_code}")
                if r.status_code != 200:
                    continue
                html = r.text
                if not _title_present_in_html(title, html):
                    continue
                has_audio, has_ebook = _extract_formats(html)
                if has_audio:
                    return "🎧"
                if has_ebook:
                    best = best or "📱"
            except requests.RequestException:
                continue
    return best

# --------------------- Dedupe ---------------------

def dedupe_titles(titles: List[str], author: str) -> List[str]:
    """
    Collapse duplicates/variants:
      - strip subtitle after a colon
      - fold accents; normalize to ASCII
      - & -> and
      - drop edition/format words: collector(s), edition, limited, hardcover, paperback, ebook,
        sneak/peek/preview/sample, vol/volume/book/part + number
      - remove leading 'the/a/an'
      - remove punctuation, collapse whitespace
    """
    noise_words = [
        "collectors", "collector", "edition", "limited", "regular",
        "hardcover", "paperback", "ebook", "sneak", "peek", "preview", "sample",
        "volume", "vol", "book", "part",
    ]
    def norm_one(t: str) -> str:
        base = t.split(":", 1)[0]
        s = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
        s = s.replace("&", " and ").lower()
        s = re.sub(r"\b(" + "|".join(noise_words) + r")\b", " ", s)
        s = re.sub(r"\b(?:part|book|volume|vol)\.?\s*[ivx\d]+\b", " ", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"^(the|a|an)\s+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    seen = set()
    out: List[str] = []
    for t in titles:
        key = norm_one(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out

# --------------------- Core evaluation ---------------------

def evaluate_title(
    t: str,
    author_name: str,
    titles_index: Dict[str, dict],
    libraries: List[Library],
    show_libby: bool,
    english_only: bool,
    probe_timeout: int,
    lib_workers: int,
    status_only: bool,
    debug: bool,
    strict_english: bool = False,
) -> Optional[dict]:
    # Language gate
    if english_only:
        if strict_english:
            maybe = ol_is_english(t, author_name)
            if maybe is not True:
                return None
        else:
            if not _ascii_englishish(t):
                maybe = ol_is_english(t, author_name)
                if maybe is not True:
                    return None
            else:
                if looks_non_english_ascii(t):
                    maybe = ol_is_english(t, author_name)
                    if maybe is not True:
                        return None

    # Status
    rec = titles_index.get(normalize(t), {})
    status = compute_status(rec.get("row", {}), rec.get("source", "-"))

    if status_only:
        return {"Title": t, "Status": status}

    # Availability
    availability_items: List[str] = []

    # Open Library eBook presence
    isbns, ia_ids = ol_search_identifiers(t, author_name)
    if ia_ids and ol_availability_for_ia(ia_ids):
        availability_items.append("Open Library 📱")

    # OverDrive per-library probes (concurrent)
    def task(lib: Library):
        site = ensure_overdrive_site(lib)
        if not site:
            return None
        icon = probe_overdrive_presence(site, t, author_name, timeout_sec=probe_timeout, debug=debug)
        if icon:
            piece = f"{lib.name} {icon}"
            if show_libby and lib.libby_key:
                piece += f" — https://libbyapp.com/library/{lib.libby_key}"
            return piece
        return None

    if libraries:
        with ThreadPoolExecutor(max_workers=max(1, lib_workers)) as ex:
            futs = [ex.submit(task, lib) for lib in libraries]
            for fut in as_completed(futs):
                piece = fut.result()
                if piece:
                    availability_items.append(piece)

    return {"Title": t, "Status": status, "Availability": " \n ".join(availability_items)}

def evaluate_bibliography(
    bibliography_titles: List[str],
    titles_index: Dict[str, dict],
    libraries: List[Library],
    author_name: str,
    show_libby: bool = False,
    english_only: bool = True,
    debug: bool = False,
    lib_workers: int = 12,
    probe_timeout: int = 5,
    status_only: bool = False,
    strict_english: bool = False,
) -> List[dict]:
    rows: List[dict] = []
    for t in bibliography_titles:  # keep deterministic order; speed only via lib_workers
        res = evaluate_title(
            t, author_name, titles_index, libraries, show_libby,
            english_only, probe_timeout, lib_workers, status_only, debug,
            strict_english=strict_english
        )
        if res:
            rows.append(res)
    return rows

# ----------------------- Printing -----------------------

def print_table(rows: List[dict], status_only: bool = False) -> None:
    if not rows:
        print("No rows.")
        return
    cols = ["Title", "Status"] if status_only else ["Title", "Status", "Availability"]
    colw = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            cell_lines = (r.get(c, "") or "").split("\n")
            for line in cell_lines:
                colw[c] = max(colw[c], len(line))
    sep = "  "
    header = sep.join(c.ljust(colw[c]) for c in cols)
    rule = sep.join("-"*colw[c] for c in cols)
    print(header)
    print(rule)
    for r in rows:
        line_parts = []
        for c in cols:
            v = (r.get(c, "") or "").split("\n")
            first = v[0] if v else ""
            line_parts.append(first.ljust(colw[c]))
        print(sep.join(line_parts))
        extra = max((len((r.get(c, "") or "").split("\n")) for c in cols), default=1) - 1
        for i in range(extra):
            line_parts = []
            for c in cols:
                v = (r.get(c, "") or "").split("\n")
                cell = v[i+1] if i+1 < len(v) else ""
                line_parts.append(cell.ljust(colw[c]))
            print(sep.join(line_parts))

# -------------------------- CLI --------------------------

def load_authors_file(path: Path) -> List[str]:
    authors: List[str] = []
    if not path.exists():
        return authors
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            authors.append(line)
    return authors

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Author bibliography vs your library (Open Library + public OverDrive probing)")
    p.add_argument("author", nargs="?", help="Single author name (e.g., 'T. Kingfisher')")
    p.add_argument("exports", nargs="*", help="CSV exports (StoryGraph and/or Goodreads). Defaults to StoryGraph.csv if present.")
    p.add_argument("-a", "--authors", "--authors-file", dest="authors_file", help="Path to authors.txt (one author per line)", default="authors.txt")
    p.add_argument("-l", "--libs", "--libraries", dest="libraries", help="Path to libraries file", default="libraries.txt")
    p.add_argument("-L", "--libby", action="store_true", default=False, help="Include Libby Search links in availability")
    p.add_argument("-o", "--only", choices=["unread", "read", "to-read", "currently-reading", "available"], help="Filter rows before printing", default=None)
    p.add_argument("--debug", action="store_true", default=False, help="Print debug info about site probing")
    p.add_argument("--lib-workers", type=int, default=12, help="Concurrent library probes per title (default 12)")
    p.add_argument("--probe-timeout", type=int, default=5, help="Timeout per OverDrive request in seconds (default 5)")
    p.add_argument("--no-lang-check", action="store_true", default=False, help="Skip language filtering entirely")
    p.add_argument("--strict-english", action="store_true", default=False, help="Require OL to confirm English for every title (slower, cleanest)")
    p.add_argument("--status-only", action="store_true", default=False, help="Only print Title + Status (skip availability)")
    args = p.parse_args(argv)

    # Default exports: use StoryGraph.csv if it exists; otherwise allow empty (status-only or availability with only OL data)
    if not args.exports:
        if Path("StoryGraph.csv").exists():
            args.exports = ["StoryGraph.csv"]
        else:
            args.exports = []  # OK; we'll proceed without personal status mapping

    return args

def run_for_author(
    author: str,
    export_paths: List[Path],
    libraries_path: Optional[Path],
    only: Optional[str] = None,
    show_libby: bool = False,
    debug: bool = False,
    lib_workers: int = 12,
    probe_timeout: int = 5,
    no_lang_check: bool = False,
    strict_english: bool = False,
    status_only: bool = False,
) -> None:
    print("\n" + "=" * 80)
    print(f"Author: {author}")
    print("=" * 80)

    titles_index = build_library_index(export_paths, author) if export_paths else {}
    bibliography = fetch_bibliography_openlibrary(author)
    bibliography = dedupe_titles(bibliography, author)
    if not bibliography:
        print("No titles found via Open Library.")
        return

    libraries: List[Library] = parse_libraries_file(Path(libraries_path)) if libraries_path else parse_libraries_file(Path("libraries.txt"))

    # 1. map status only (fast)
    rows = evaluate_bibliography(
        bibliography,
        titles_index,
        libraries=[],
        author_name=author,
        status_only=True,
        english_only=not no_lang_check,
        strict_english=strict_english,
    )

    # 2. filter rows by --only
    if only:
        o = only.lower()
        if o == "available":
            # keep all; availability still needs to be checked later
            pass
        else:
            rows = [r for r in rows if r.get("Status", "").lower().replace(" ", "-") == o]

    # 3. if not status-only mode, upgrade rows with availability
    if not status_only:
        filtered_titles = [r["Title"] for r in rows]
        rows = evaluate_bibliography(
            filtered_titles,
            titles_index,
            libraries,
            author,
            show_libby=show_libby,
            english_only=not no_lang_check,
            debug=debug,
            lib_workers=lib_workers,
            probe_timeout=probe_timeout,
            status_only=False,
            strict_english=strict_english,
        )

    print_table(rows, status_only=status_only)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    export_paths = [Path(x) for x in args.exports if Path(x).exists()]
    # NOTE: we do NOT exit when no exports; we proceed (status defaults to Unread)

    # figure authors (arg wins, else authors.txt)
    if args.author:
        authors = [args.author]
    else:
        authors = load_authors_file(Path(args.authors_file))

    if not authors:
        print("No authors provided (pass an author or populate authors.txt).")
        return 2

    libraries_path = Path(args.libraries) if args.libraries else None

    for author in authors:
        run_for_author(
            author,
            export_paths,
            libraries_path,
            args.only,
            show_libby=args.libby,
            debug=args.debug,
            lib_workers=args.lib_workers,
            probe_timeout=args.probe_timeout,
            no_lang_check=args.no_lang_check,
            strict_english=args.strict_english,
            status_only=args.status_only,
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

