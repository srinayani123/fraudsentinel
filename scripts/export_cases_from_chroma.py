"""One-time script: export all cases from ChromaDB back to individual JSON files
so the Pattern Library page can browse them without querying the vector DB."""

import json
from pathlib import Path

import chromadb

from src.utils.config import FRAUD_CASES_DIR

CHROMA_PATH = "models/chroma_db"
COLLECTION_NAME = "fraud_cases"


def main():
    print(f"Loading ChromaDB from {CHROMA_PATH}…")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"Found {total} cases in collection '{COLLECTION_NAME}'")

    # Pull everything (in chunks if huge, but 410 fits fine)
    result = collection.get(include=["documents", "metadatas"])

    ids = result.get("ids", [])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])

    FRAUD_CASES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing JSON files to {FRAUD_CASES_DIR}…")

    written = 0
    skipped = 0

    for i, (case_id, doc, meta) in enumerate(zip(ids, docs, metas)):
        meta = meta or {}

        # Reconstruct the case shape the page expects
        # Try to get fields from metadata first, fall back to parsing the document
        case = {
            "id": meta.get("id") or case_id or f"case_{i+1:03d}",
            "title": meta.get("title", "Untitled pattern"),
            "pattern": meta.get("pattern", "unknown"),
            "narrative": meta.get("narrative") or doc or "",
            "indicators": [],
        }

        # Indicators may be stored as a comma-joined string in metadata
        ind_raw = meta.get("indicators", "")
        if isinstance(ind_raw, str) and ind_raw:
            case["indicators"] = [s.strip() for s in ind_raw.split(",") if s.strip()]
        elif isinstance(ind_raw, list):
            case["indicators"] = ind_raw

        # If narrative is empty but we have the document, use it
        if not case["narrative"] and doc:
            case["narrative"] = doc

        # Filename: case_001.json, case_002.json, etc.
        # Use existing case ID if it looks like case_NNN, otherwise generate
        if case["id"].startswith("case_"):
            filename = f"{case['id']}.json"
        else:
            filename = f"case_{i+1:03d}.json"

        out_path = FRAUD_CASES_DIR / filename

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(case, f, indent=2, ensure_ascii=False)
            written += 1
        except Exception as e:
            print(f"  Failed {filename}: {e}")
            skipped += 1

    print(f"\nDone: {written} written, {skipped} skipped")
    print(f"Pattern Library should now show {written} cases.")


if __name__ == "__main__":
    main()
    