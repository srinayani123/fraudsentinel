"""Build the ChromaDB knowledge base from fraud case JSON files.

Embeds each pattern's title + narrative + indicators using BAAI/bge-base-en-v1.5
and indexes it with metadata (including LLM-generated summary + reasoning + the
new plain-English indicator_explanations field) so the UI can display clean,
human-readable content alongside the technical thresholds.

Run after:
  - generating new patterns (generate_patterns.py / generate_new_patterns.py)
  - summarizing (summarize_patterns.py)
  - backfilling explanations (backfill_indicator_explanations.py)

    python -m src.agentic.build_knowledge_base
"""

from __future__ import annotations

import json

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from src.utils.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    FRAUD_CASES_DIR,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)
load_dotenv()

# Must match tools.py — embeddings from different models live in different
# vector spaces and cannot be compared.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def main():
    if not FRAUD_CASES_DIR.exists():
        raise RuntimeError(f"FRAUD_CASES_DIR does not exist: {FRAUD_CASES_DIR}")

    case_paths = sorted(FRAUD_CASES_DIR.glob("case_*.json"))
    if not case_paths:
        raise RuntimeError("No case files found in FRAUD_CASES_DIR")

    logger.info(f"Loading {len(case_paths)} fraud-pattern files...")

    cases = []
    for path in case_paths:
        try:
            # IMPORTANT: explicit utf-8 encoding. On Windows, the default open()
            # encoding is cp1252 which chokes on curly quotes (U+201D = byte 0x9d
            # in cp1252 = the right double quotation mark). The LLM that wrote
            # the narrative fields uses curly quotes, so this file must be read
            # as utf-8 to handle them correctly.
            with open(path, encoding="utf-8") as f:
                case = json.load(f)
            cases.append(case)
        except Exception as e:
            logger.warning(f"Skipping {path.name}: {e}")

    logger.info(f"Loaded {len(cases)} valid cases")

    # Stats: how many have each enrichment field?
    n_with_summary = sum(1 for c in cases if c.get("summary"))
    n_with_reasoning = sum(1 for c in cases if c.get("reasoning"))
    n_with_explanations = sum(
        1 for c in cases
        if isinstance(c.get("indicator_explanations"), list) and c["indicator_explanations"]
    )
    logger.info(f"  with `summary`:                {n_with_summary}/{len(cases)}")
    logger.info(f"  with `reasoning`:              {n_with_reasoning}/{len(cases)}")
    logger.info(f"  with `indicator_explanations`: {n_with_explanations}/{len(cases)}")

    if n_with_summary < len(cases):
        logger.warning(
            "Some patterns are missing `summary`. Run "
            "`python -m src.agentic.summarize_patterns` first for best UX."
        )
    if n_with_explanations < len(cases):
        logger.warning(
            "Some patterns are missing `indicator_explanations`. Run "
            "`python -m src.agentic.backfill_indicator_explanations` first for best UX."
        )

    # Pattern distribution stats
    pattern_counts: dict[str, int] = {}
    for c in cases:
        p = c.get("pattern", "unknown")
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
    logger.info("Pattern distribution:")
    for p in sorted(pattern_counts.keys()):
        logger.info(f"  {p}: {pattern_counts[p]}")

    # Setup ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Wipe and recreate
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    for case in cases:
        case_id = case.get("id", "")
        title = case.get("title", "")
        narrative = case.get("narrative", "")
        pattern = case.get("pattern", "")
        indicators = case.get("indicators", []) or []
        explanations = case.get("indicator_explanations", []) or []
        summary = case.get("summary", "")
        reasoning = case.get("reasoning", "")

        # Document text for embedding — embed BOTH technical indicators and
        # plain-English explanations so retrieval works for either kind of query
        indicators_str = " ".join(indicators)
        explanations_str = " ".join(explanations)
        doc_text = f"{title}. {narrative} {indicators_str} {explanations_str}".strip()

        ids.append(case_id)
        documents.append(doc_text)
        metadatas.append({
            "title": title,
            "pattern": pattern,
            "indicators": json.dumps(indicators),
            "indicator_explanations": json.dumps(explanations),
            "summary": summary,
            "reasoning": reasoning,
        })

    logger.info(f"Indexing {len(ids)} patterns into ChromaDB...")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(f"Indexed {collection.count()} patterns")
    logger.info(f"ChromaDB location: {CHROMA_DIR}")
    logger.info("")
    logger.info("Next: restart Streamlit and verify Pattern Library renders correctly.")


if __name__ == "__main__":
    main()
    