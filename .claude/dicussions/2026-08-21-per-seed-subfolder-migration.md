# 2026-08-21 — SAFIM pilot result review, per-seed HF subfolders, and migrating the pre-fix seed 42 adapter

Session goal: review the first real pilot SAFIM number, fix a shared-HF-subfolder overwrite bug in
`scripts/run_pilot_experiment.py` flagged by Rudra, and then untangle the follow-on migration
problem that fix created for data already pushed under the old scheme.

## 1. SAFIM pass@1 review — Base 0.000 / LoRA-FT 0.090 (`random`, seed 42, config=`block`)

Reviewed via `artifacts/experiments/random/output.log` and `.../media/table/eval/safim/results_table_*.table.json`
(downloaded with `scripts/download_wandb_media.py` from W&B run
`qwen05b-lora-r16-e1-dsrandom-20260816-a42`).

- **Base = 0.000 is real, not a bug** — the running pass@1 stayed exactly `0.000` through all 100
  samples in the log, not one lucky pass. `config="block"` (the only config this run tested) is
  SAFIM's hardest Python slice (multi-line block completions), and pass@1 here is
  *execution-based*: greedy-decoded output has to produce code whose stdout exactly matches every
  unit test. A 0.5B un-tuned base model failing every one of 100 hard, strict-grading samples is
  plausible, not evidence of a broken harness.
- **LoRA-FT = 0.090 is a real, non-trivial lift** — it climbed 0.000 → 0.050 → 0.100 → settled
  ~0.075–0.100 through the run (not a single-sample fluke), showing the 1-epoch fine-tune teaches
  the model to produce *some* working block completions where base produces none.
- **Caveat, not a red flag**: n=100 → binomial SE ≈ ±3 points on 9%, so treat 9% as "somewhere in
  the ~4–15% range," not a precise number. This is exactly why the pilot already runs 3 seeds per
  variant (`configs/training/pilot.yaml`'s `[42, 123, 7]`) — `aggregate_pilot_results()` in
  `src/training/safim_eval.py` is what turns per-seed numbers into a real random-vs-planned
  comparison, not any single seed's score. Only `block` config was run here too;
  `run_safim_evaluation_by_type` (same file) would show whether the gain concentrates in one span
  type — closer to what the random-vs-planned hypothesis is actually about.

**Verdict:** treat this as a working pilot signal (fine-tune clearly does *something* over base),
not yet a basis for picking a winner — wait for all 3 seeds × 2 variants before judging.

## 2. Why the pilot notebooks don't let you pick a seed manually

Read `notebooks/experiments/random.ipynb` (and by the same pattern, `planned.ipynb`). Cell 4 is:

```python
subprocess.run([sys.executable, "scripts/run_pilot_experiment.py", "--variant", "random"], check=True)
```

That's it — no seed argument. `run()` (`scripts/run_pilot_experiment.py`) loops over **every**
seed in `configs/training/pilot.yaml` (`[42, 123, 7]`) internally, training all 3 in one notebook
run. This isn't an oversight: `pilot.yaml` is the single source of truth for which seeds the pilot
uses, not the notebook — keeping the notebook a "thin controller" is `CLAUDE.md`'s core rule
("notebooks contain no real logic"). Changing seeds means editing `pilot.yaml`, not the notebook.

## 3. Decision: per-seed HF subfolders, and migrating the one pre-fix push

### Problem found

`scripts/run_pilot_experiment.py` originally pushed every seed of a variant to the **same**
subfolder (`output_subfolder`, e.g. `experiment/random_model`) — each new seed's push landed as a
new commit on top of the last, so only the most recently trained seed's adapter survived on the
default branch; earlier seeds were only recoverable by hunting up their exact commit SHA. This was
called out in the code's own docstring as an intentional, left-for-later naming decision. Rudra
asked to fix it now.

### Fix, part 1 — per-seed subfolders

`overrides["output"]["subfolder"]` (and the matching resume-fetch path) now compose
`f"{variant['output_subfolder']}/seed{seed}"` (e.g. `experiment/random_model/seed42`) instead of
just `variant['output_subfolder']`. Every seed gets its own never-shared path.

### Problem found *from* the fix — a stale historical commit reference

Checking the live HF repo (`Rudra-G-23/qwen2.5-coder-0.5b-python-fim`, read-only) turned up:
`experiment/random_model/` already held a fully-pushed adapter from **before** this fix — seed 42's
pilot run (`metadata.json`: `run_id: qwen05b-lora-r16-e1-dsrandom-20260816-a42`, `seed: 42`), still
at the old un-suffixed path. `experiment/distributed_model/` (planned variant) had nothing pushed
yet, so this was an isolated, single-adapter problem.

Rudra asked directly: does this need to be downloaded and moved manually, or left as-is? Neither,
as it turned out — the honest answer required fixing the *resume logic itself*, not just the data:

The old resume-skip path (`_find_existing_run_commit`) found "already done" runs by matching a
commit **title** prefix against repo history, then pinned `snapshot_download(...,
revision=<that commit's SHA>, ...)` to fetch from that **exact historical commit** — necessary under
the old shared-subfolder scheme specifically to grab a seed's data before a later seed's push
overwrote the same folder. After the subfolder fix, re-running `random` would: match the old
pre-fix commit by title (unchanged run_id) → skip training (looks done) → try to fetch
`experiment/random_model/seed42/*` pinned to that *old* commit SHA → find nothing there (that
commit's tree predates any `seed42/` path) → crash on a missing adapter. **Just copying files to
the new HF path would not have fixed this alone** — the SHA-pinning is what's fundamentally
incompatible with any later migration, since a historical commit's tree never changes.

### Fix, part 2 — stop pinning to a historical commit

Since every seed's subfolder is now unique and never shared, the entire reason for SHA-pinning
(dodging an overwrite from a different seed's push) no longer applies. Replaced
`_find_existing_run_commit` (commit-history title matching) with `_subfolder_already_pushed`
(`scripts/run_pilot_experiment.py`) — a direct `HfApi.file_exists()` check against the seed's
subfolder on the **current default branch**. The resume-fetch `snapshot_download()` call no longer
takes a `revision=` pin either — it always reads whatever is currently on `main` at that (now
stable) path. Simpler, no commit-history API calls, and correctly picks up migrated data.

### Migration itself

New reusable script, `scripts/migrate_hf_subfolder.py` — server-side move (via
`CommitOperationCopy` + `CommitOperationDelete` in one atomic commit; no re-uploading the ~44 MB
adapter's bytes) of `experiment/random_model/*` → `experiment/random_model/seed42/*`, run once:

```bash
python scripts/migrate_hf_subfolder.py \
  --repo Rudra-G-23/qwen2.5-coder-0.5b-python-fim \
  --old-prefix experiment/random_model \
  --new-prefix experiment/random_model/seed42
```

Confirmed after running: the repo now has only `experiment/random_model/seed42/*` (no stray
top-level copies), and `_subfolder_already_pushed(...)` resolves `True` for it — a re-run of the
`random` variant will now correctly skip re-training seed 42 and fetch its migrated adapter,
instead of crashing.

### Side fix — `.env` had CRLF line endings

While running the migration, `HfApi(token=...)` failed with `httpx.LocalProtocolError: Illegal
header value ...\r` — `.env` (gitignored, local-only) had Windows CRLF line endings, so
`HF_TOKEN` picked up a trailing `\r` after `source .env`. Fixed once with `sed -i 's/\r$//' .env`;
this had gone unnoticed until now because prior local scripts only used `WANDB_API_KEY` via
`wandb.Api()`, which doesn't hit this code path the same way.

## What changed

1. `scripts/run_pilot_experiment.py` — `_find_existing_run_commit` → `_subfolder_already_pushed`;
   resume-fetch no longer pins `revision=`; docstring updated (module-level "Resume across a
   Kaggle-session interruption" section, and the per-seed-subfolder paragraph).
2. `local_tests/test_run_pilot_experiment.py` — rewritten for `_subfolder_already_pushed` (file
   existence, not commit-title matching).
3. New `scripts/migrate_hf_subfolder.py` — generic, reusable server-side subfolder mover.
4. Ran the migration once for `experiment/random_model` → `experiment/random_model/seed42` on the
   live HF repo.
5. `.env` — fixed CRLF line endings (local-only, not committed).
6. `CLAUDE.md` — "One shared HF model repo, subfolder-separated" section updated to describe the
   per-seed subfolder scheme instead of the old shared-subfolder limitation.

Not touched: `experiment/distributed_model/` (planned variant) — confirmed empty, no migration
needed there. `configs/training/pilot.yaml`'s `output_subfolder` values are unchanged; they're now
the variant-level prefix, with the seed suffix composed in `run()`.
