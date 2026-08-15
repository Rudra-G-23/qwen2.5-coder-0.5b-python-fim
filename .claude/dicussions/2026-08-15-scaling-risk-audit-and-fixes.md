# 2026-08-15 — Scaling risk audit + fixes: chunk_exists, seen_hashes, comparison-chart flakiness

Rudra ran the Stage 1 notebooks for real and hit a live bug (checkpoint
`chunk_file` pointing at the old flat-repo filename — already fixed by a
concurrent session, `fix_migrated_checkpoint_paths.py`). He asked: what
*other* small resource issues exist that work fine at 10,000 files but
would silently degrade or break once this pipeline is pointed at millions,
and what's causing the generation notebook's end-of-run comparison chart to
be inconsistent between runs ("swing the result"). First pass was analysis
only (no code); this pass implements the fixes worth doing now.

## What was already fixed by a concurrent session (checked before touching anything)

- `load_curated_files` in `scripts/generate_fim_variants.py` was rewritten
  (commit `4214d9f`) to fix a schema-cast crash, switching from
  `datasets.load_dataset("parquet", ...)` to per-file `pyarrow` reads. As a
  side effect, the file list is now explicitly `sorted(...)` before being
  turned into FIM candidates — this **already fixes** the audit's "unsorted
  file order feeding a deterministic seed" finding (the more likely cause
  of "swing the result"). No action needed here.

## Fixed this pass

1. **`chunk_exists()` full-repo listing on every chunk upload**
   (`src/curation/checkpoint.py`, `scripts/build_sample.py`). Was
   `api.list_repo_files(repo_id)` once per chunk — cost grows with total
   repo size, so each successive chunk in a long run got slower to upload
   than the last. Now `run_session` lists the repo once per session into an
   `existing_files` set, threaded into `_write_and_upload_chunk`, updated
   locally after each upload. `ckpt.chunk_exists()` itself is untouched
   (still used/tested standalone) — just no longer called in the per-chunk
   hot path.

2. **`rebuild_seen_hashes()` re-downloading the entire corpus on every
   resume** — the audit's biggest "millions of files" finding, and
   `data-stage-2.md` §4's already-documented open risk. Added
   `save_seen_hashes`/`load_seen_hashes` to `checkpoint.py`: the hash set is
   cached as `{path_prefix}/checkpoints/seen_hashes.json`, tagged with the
   `(shard_index, row_offset)` of the last checkpoint it accounts for.
   `load_seen_hashes` is now the fast path in `build_sample.py`'s
   `_resume_state` — falls back to a full `rebuild_seen_hashes` only when no
   cache exists yet (fresh repo, or history predating this file, e.g.
   original Stage 1 checkpoints), and merges in just the missing tail when
   the cache is behind the resume point instead of full history.

   **Deliberately deviates from §4's original "flush per chunk" fix
   direction** — see the updated §4 in `data-stage-2.md` for the full
   reasoning: flushing every chunk means re-uploading the whole (growing)
   hash set every chunk, which at full-corpus scale could move *more* total
   bytes than the every-resume rebuild it replaces. Flushing once per
   session (same call sites as `write_filter_report`/`write_metadata_json`)
   and having the next resume's delta-merge re-derive any gap from
   checkpoint history (always authoritative) gets the same "never re-admit
   a duplicate" guarantee for a fraction of the network cost. `ExactDedup`
   gained a `seen_hashes` property (`src/dedup/exact_dedup.py`) as the clean
   public accessor for this, instead of reaching into `dedup._seen`.

3. **`fetch_other_variant_counts()` swallowing every exception as "sibling
   not generated yet"** (`scripts/generate_fim_variants.py`) — narrowed
   `except Exception` to `except (EntryNotFoundError, RepositoryNotFoundError)`.
   A transient network/auth/rate-limit error now propagates as a real error
   instead of silently making the comparison chart say "deferred" on a
   rerun of an already-complete state. Same narrowing applied to
   `checkpoint.load_seen_hashes`'s cache-miss detection, for the same
   reason.

4. **`load_curated_files` OOM risk at full-corpus scale** — not fixed (a
   real fix means a streaming/reservoir-sampling rewrite of
   `distribution_random.py`/`distribution_planned.py`'s "load everything,
   sample after" approach, which is a bigger, separate design decision, not
   a small careful fix). Added a loud warning instead: if the source folder
   has more chunk files than the 10k pilot's ~10, it now prints a warning
   before attempting to load everything into memory, so a future OOM at
   full-corpus scale is an explained, expected failure instead of a
   confusing crash partway through a run.

## Not touched (still just noted, per the original audit)

- `ExactDedup`'s in-memory hash set has no upper bound — fine into the low
  millions of files, would need an on-disk index only far beyond current
  scale.
- Local `hf_hub_download` cache growth on Kaggle's disk — not urgent at
  current scale; revisit if disk quota actually gets hit.
- `write_filter_report`/`write_metadata_json` re-aggregate from all
  checkpoints on every call (once per session, not once per chunk) — same
  root cause as #1/#2 above, smaller cost, not addressed this pass.

## Tests

`local_tests/test_checkpoint.py` gained a `TestSeenHashesPersistence` class
(5 cases: no-cache fallback, up-to-date fast path — proven by asserting no
chunk file is downloaded, stale-cache delta merge, and two `save_seen_hashes`
round-trip/path-scoping checks). Full suite: 124/124 passing (was 119).
