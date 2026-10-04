"""
src/data_prep.py
Build FIM (fill-in-the-middle) training triples from Python source files.

Usage (local, no GPU needed):
    python src/data_prep.py
    python src/data_prep.py --source-dirs src local_tests --output data/fim_dataset.jsonl
"""

import argparse
import ast
import json
import random
from pathlib import Path

# ── Qwen2.5-Coder native FIM tokens ─────────────────────────────────────────
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_valid_python(source: str) -> bool:
    """Return True only if source is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def make_fim_triples(
    source: str,
    n_samples: int = 5,
    min_middle_lines: int = 1,
    max_middle_lines: int = 15,
) -> list[dict]:
    """
    Given a Python source string, produce up to n_samples FIM triples by
    randomly masking contiguous line spans.

    Span variety matters: mix line-level, block-level, and function-level
    spans to match the realistic FIM training distribution (Bavarian et al.).
    """
    lines = source.splitlines()
    if len(lines) < 10:
        return []  # too short — not enough context above/below the span

    triples: list[dict] = []

    # Try 3× as many iterations as needed to collect n_samples valid triples
    for _ in range(n_samples * 3):
        if len(triples) >= n_samples:
            break

        span_len = random.randint(
            min_middle_lines, min(max_middle_lines, len(lines) - 2)
        )
        start = random.randint(1, len(lines) - span_len - 1)
        end = start + span_len

        prefix = "\n".join(lines[:start])
        middle = "\n".join(lines[start:end])
        suffix = "\n".join(lines[end:])

        # Skip trivially empty middles (blank lines only)
        if not middle.strip():
            continue

        triples.append(
            {
                "prefix": prefix,
                "suffix": suffix,
                "middle": middle,
                # The full formatted text the model trains on
                "text": f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}",
            }
        )

    return triples


# ── Main dataset builder ──────────────────────────────────────────────────────


def build_dataset(
    source_dirs: list[str],
    output_path: str,
    max_files: int = 5000,
    samples_per_file: int = 5,
    seed: int = 42,
) -> None:
    """
    Walk source_dirs, collect .py files, generate FIM triples, deduplicate,
    and write a JSONL file to output_path.

    Args:
        source_dirs:     List of directories to search recursively for .py files.
        output_path:     Destination JSONL file (created if missing).
        max_files:       Cap on total files processed (shuffle + take first N).
        samples_per_file: FIM triples to extract per source file.
        seed:            Random seed for reproducibility.
    """
    random.seed(seed)

    # Collect all .py files from all source dirs
    python_files: list[Path] = []
    for src_dir in source_dirs:
        found = list(Path(src_dir).rglob("*.py"))
        python_files.extend(found)
        print(f"  {src_dir}: {len(found)} .py files found")

    random.shuffle(python_files)
    python_files = python_files[:max_files]
    print(f"\nProcessing {len(python_files)} files (capped at {max_files})...")

    records: list[dict] = []
    skipped_invalid = 0
    skipped_unreadable = 0

    for fpath in python_files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            skipped_unreadable += 1
            continue

        if not is_valid_python(source):
            skipped_invalid += 1
            continue

        triples = make_fim_triples(source, n_samples=samples_per_file)
        records.extend(triples)

    # Simple deduplication on the full formatted text
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in records:
        if rec["text"] not in seen:
            seen.add(rec["text"])
            deduped.append(rec)

    print(f"\n{'─' * 50}")
    print(
        f"Files processed :  {len(python_files) - skipped_invalid - skipped_unreadable}"
    )
    print(f"Skipped (invalid syntax): {skipped_invalid}")
    print(f"Skipped (unreadable):     {skipped_unreadable}")
    print(f"Raw FIM triples :  {len(records)}")
    print(f"After dedup     :  {len(deduped)}")
    print(f"{'─' * 50}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✓ Dataset saved → {output}  ({output.stat().st_size / 1024:.1f} KB)")


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FIM training triples from Python source files."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=["src"],
        help="Directories to search for .py files (default: src)",
    )
    parser.add_argument(
        "--output",
        default="data/fim_dataset.jsonl",
        help="Output JSONL path (default: data/fim_dataset.jsonl)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5000,
        help="Maximum number of .py files to process (default: 5000)",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=5,
        help="FIM triples to extract per file (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("=" * 50)
    print("FIM Dataset Builder")
    print("=" * 50)
    build_dataset(
        source_dirs=args.source_dirs,
        output_path=args.output,
        max_files=args.max_files,
        samples_per_file=args.samples_per_file,
        seed=args.seed,
    )
