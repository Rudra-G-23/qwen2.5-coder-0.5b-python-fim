"""
src/fim/ast_bucketer.py
Stage 1 §6: tag every valid FIM span in a Python source file with a bucket
type, required by both distribution_random.py and distribution_planned.py.

Bucket types (per .claude/stages/data/data-stage-1.md §6):
    line / expression / statement / block / function-body / method-body /
    class-level / api-call

Two heuristics worth calling out (AST alone can't resolve either exactly):
  - api-call: a `Call` node is bucketed `api-call` if its `func` is an
    `ast.Attribute` (an `x.y(...)` shape — method/module calls); a bare-name
    call like `len(x)` falls back to `expression`.
  - block: the full `body` of a compound statement (If/For/While/With/Try)
    is tagged `block` only when it has 2+ statements — a single-statement
    body is already covered by `statement`, and treating it as a block too
    would just duplicate the same span under two labels.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

# expr node types substantial enough to be a useful FIM "expression" span —
# deliberately excludes atomic Name/Constant, which would flood the pool
_EXPRESSION_NODE_TYPES = (
    ast.Call,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.Subscript,
)

_COMPOUND_STMT_TYPES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)


@dataclass
class Span:
    span_type: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    node_type: str
    nesting_depth: int
    enclosing_function: str | None
    enclosing_class: str | None
    structural_position: str


@dataclass
class _Context:
    depth: int = 0
    function_stack: list[str] = field(default_factory=list)
    class_stack: list[str] = field(default_factory=list)

    @property
    def enclosing_function(self) -> str | None:
        return self.function_stack[-1] if self.function_stack else None

    @property
    def enclosing_class(self) -> str | None:
        return self.class_stack[-1] if self.class_stack else None

    @property
    def structural_position(self) -> str:
        if self.function_stack:
            return "function_body"
        if self.class_stack:
            return "class_body"
        return "module_level"


def _node_span(node: ast.AST) -> tuple[int, int] | None:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is None or end_lineno is None:
        return None
    return lineno, end_lineno


def _make_span(span_type: str, node: ast.AST, ctx: _Context) -> Span | None:
    lines = _node_span(node)
    if lines is None:
        return None
    start, end = lines
    return Span(
        span_type=span_type,
        start_line=start,
        end_line=end,
        node_type=type(node).__name__,
        nesting_depth=ctx.depth,
        enclosing_function=ctx.enclosing_function,
        enclosing_class=ctx.enclosing_class,
        structural_position=ctx.structural_position,
    )


def _is_api_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute)


def _walk_expr(node: ast.AST, ctx: _Context, spans: list[Span]) -> None:
    """Tag expression-type spans reachable from `node` without crossing into
    any nested statement — nested statement bodies (an If's `body`, a
    FunctionDef's `body`, ...) are walked separately by `_walk_stmt`/
    `_walk_body`, so descending into them here would re-tag the same
    expressions twice under the wrong (pre-descent) context."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            continue
        if isinstance(child, ast.Call):
            span_type = "api-call" if _is_api_call(child) else "expression"
            span = _make_span(span_type, child, ctx)
            if span:
                spans.append(span)
        elif isinstance(child, _EXPRESSION_NODE_TYPES):
            span = _make_span("expression", child, ctx)
            if span:
                spans.append(span)
        _walk_expr(child, ctx, spans)


def _walk_body(body: list[ast.stmt], ctx: _Context, spans: list[Span]) -> None:
    for stmt in body:
        _walk_stmt(stmt, ctx, spans)


def _walk_stmt(node: ast.stmt, ctx: _Context, spans: list[Span]) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        span_type = "method-body" if ctx.class_stack else "function-body"
        body_span = _body_span(node.body)
        if body_span:
            start, end = body_span
            spans.append(
                Span(
                    span_type=span_type,
                    start_line=start,
                    end_line=end,
                    node_type=type(node).__name__,
                    nesting_depth=ctx.depth,
                    enclosing_function=ctx.enclosing_function,
                    enclosing_class=ctx.enclosing_class,
                    structural_position=ctx.structural_position,
                )
            )
        child_ctx = _Context(
            depth=ctx.depth + 1,
            function_stack=[*ctx.function_stack, node.name],
            class_stack=list(ctx.class_stack),
        )
        _walk_expr(node, ctx, spans)  # decorators/defaults/return annotation, at outer depth
        _walk_body(node.body, child_ctx, spans)
        return

    if isinstance(node, ast.ClassDef):
        child_ctx = _Context(
            depth=ctx.depth + 1,
            function_stack=list(ctx.function_stack),
            class_stack=[*ctx.class_stack, node.name],
        )
        _walk_expr(node, ctx, spans)  # decorators/base classes, at outer depth
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _walk_stmt(stmt, child_ctx, spans)
            else:
                span = _make_span("class-level", stmt, child_ctx)
                if span:
                    spans.append(span)
                _walk_expr(stmt, child_ctx, spans)
                _walk_stmt_children(stmt, child_ctx, spans)
        return

    # Ordinary statement — tag it, then recurse into its expressions/sub-blocks.
    span = _make_span("statement", node, ctx)
    if span:
        spans.append(span)
    _walk_expr(node, ctx, spans)
    _walk_stmt_children(node, ctx, spans)


def _walk_stmt_children(node: ast.stmt, ctx: _Context, spans: list[Span]) -> None:
    """Recurse into a compound statement's nested bodies, tagging each as a
    `block` (if 2+ statements) before descending one nesting level."""
    if not isinstance(node, _COMPOUND_STMT_TYPES):
        return

    child_ctx = _Context(
        depth=ctx.depth + 1,
        function_stack=list(ctx.function_stack),
        class_stack=list(ctx.class_stack),
    )
    for body_attr in ("body", "orelse", "finalbody", "handlers"):
        body = getattr(node, body_attr, None)
        if not body:
            continue
        if body_attr == "handlers":
            for handler in body:
                _walk_stmt_children_body(handler.body, child_ctx, spans)
            continue
        _walk_stmt_children_body(body, child_ctx, spans)


def _walk_stmt_children_body(body: list[ast.stmt], ctx: _Context, spans: list[Span]) -> None:
    if len(body) >= 2:
        block_span = _body_span(body)
        if block_span:
            start, end = block_span
            spans.append(
                Span(
                    span_type="block",
                    start_line=start,
                    end_line=end,
                    node_type="block",
                    nesting_depth=ctx.depth,
                    enclosing_function=ctx.enclosing_function,
                    enclosing_class=ctx.enclosing_class,
                    structural_position=ctx.structural_position,
                )
            )
    _walk_body(body, ctx, spans)


def _body_span(body: list[ast.stmt]) -> tuple[int, int] | None:
    if not body:
        return None
    first, last = _node_span(body[0]), _node_span(body[-1])
    if first is None or last is None:
        return None
    return first[0], last[1]


def _line_spans(source: str) -> list[Span]:
    """Every interior line is a candidate 1-line span — needs at least one
    line of prefix and suffix context above/below."""
    lines = source.splitlines()
    spans: list[Span] = []
    for i in range(2, len(lines)):  # 1-indexed line i, skip first/last line
        spans.append(
            Span(
                span_type="line",
                start_line=i,
                end_line=i,
                node_type="line",
                nesting_depth=0,
                enclosing_function=None,
                enclosing_class=None,
                structural_position="module_level",
            )
        )
    return spans


def bucket_spans(source: str) -> list[Span]:
    """
    Parse `source` and return every tagged span. Raises SyntaxError on
    unparseable source — callers are expected to have already run this
    through src.curation.quality_filters.is_ast_parseable.
    """
    tree = ast.parse(source)
    ctx = _Context()
    spans: list[Span] = []
    _walk_body(tree.body, ctx, spans)
    spans.extend(_line_spans(source))
    return spans


def span_to_prefix_suffix_middle(source: str, span: Span) -> tuple[str, str, str] | None:
    """
    Slice `source` around `span` into (prefix, suffix, middle) line-wise.
    Returns None for a degenerate span with no prefix or no suffix context,
    or a middle that's blank once masked out.
    """
    lines = source.splitlines()
    start_idx = span.start_line - 1  # 0-indexed, inclusive
    end_idx = span.end_line  # 0-indexed, exclusive

    if start_idx <= 0 or end_idx >= len(lines):
        return None  # no prefix or no suffix context available

    middle = "\n".join(lines[start_idx:end_idx])
    if not middle.strip():
        return None

    prefix = "\n".join(lines[:start_idx])
    suffix = "\n".join(lines[end_idx:])
    return prefix, suffix, middle


def iter_bucketed_files(
    files: Iterable[dict[str, str]]
) -> Iterator[tuple[str, str, Span]]:
    """
    Bucket every file's spans, yielding (content_id, content, span). Files
    that fail to parse are skipped — callers are expected to be feeding the
    already-curated (ast-parseable) sample, so this is a defensive guard,
    not the primary quality gate.
    """
    for file in files:
        content = file["content"]
        try:
            spans = bucket_spans(content)
        except SyntaxError:
            continue
        for span in spans:
            yield file["content_id"], content, span


def build_fim_record(
    content_id: str, span: Span, prefix: str, suffix: str, middle: str
) -> dict[str, Any]:
    """Assemble one §6 FIM example record from a sliced span."""
    return {
        "fim_type": span.span_type,
        "content_id": content_id,
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle,
        "ast_depth": span.nesting_depth,
        "nesting_depth": span.nesting_depth,
        "node_type": span.node_type,
        "enclosing_function": span.enclosing_function,
        "enclosing_class": span.enclosing_class,
        "structural_position": span.structural_position,
        "span_line_count": span.end_line - span.start_line + 1,
    }


def measure_natural_distribution(files: Iterable[dict[str, str]]) -> dict[str, int]:
    """Span-type counts across the full bucketed pool, unweighted by any
    sampling target — the corpus's natural span-type frequency. Used to
    compare against distribution_planned.py's hand-set percentages, which are
    a starting hypothesis rather than a derived value (see that module's
    docstring)."""
    counts: dict[str, int] = {}
    for _content_id, _content, span in iter_bucketed_files(files):
        counts[span.span_type] = counts.get(span.span_type, 0) + 1
    return counts


def render_natural_distribution_report(
    natural_counts: dict[str, int], planned_distribution: dict[str, float]
) -> dict[str, Any]:
    """Pair the measured natural distribution with the planned targets, so
    the two can be compared directly rather than one being asserted without
    the other."""
    total = sum(natural_counts.values()) or 1
    natural_percent = {
        span_type: round(100 * count / total, 2) for span_type, count in natural_counts.items()
    }
    return {
        "natural_counts": natural_counts,
        "natural_percent": natural_percent,
        "planned_distribution": planned_distribution,
    }


def write_natural_distribution_report(
    files: Iterable[dict[str, str]],
    planned_distribution: dict[str, float],
    out_path: str = "reports/natural_span_distribution.json",
) -> Path:
    natural_counts = measure_natural_distribution(files)
    report = render_natural_distribution_report(natural_counts, planned_distribution)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


# ── Experiment-stage reporting (data-stage-2.md §3, §7) ─────────────────────
# scripts/generate_fim_variants.py calls these once a variant's FIM records
# are sampled — separate from the natural-distribution report above, which
# runs before sampling.


def count_by_fim_type(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Span-type counts across a variant's already-sampled FIM records —
    what actually ended up in the output, not the pre-sampling pool."""
    counts: dict[str, int] = {}
    for record in records:
        span_type = record["fim_type"]
        counts[span_type] = counts.get(span_type, 0) + 1
    return counts


def plot_bucket_distribution(
    random_counts: dict[str, int],
    planned_counts: dict[str, int],
    output_dir: str = "reports/plots",
) -> Path:
    """
    Grouped bar chart, random vs planned/distributed span-type counts side
    by side — data-stage-2.md §7's explicit chart choice (bar, not pie, for
    comparing several categories at a glance). Plain matplotlib only, no
    seaborn/pandas — this runs from the lightweight CPU-only experiment
    notebooks, not the GPU training ones that already depend on those.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span_types = sorted(set(random_counts) | set(planned_counts))
    random_vals = [random_counts.get(t, 0) for t in span_types]
    planned_vals = [planned_counts.get(t, 0) for t in span_types]

    x = range(len(span_types))
    width = 0.38

    _fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], random_vals, width, label="random")
    ax.bar([i + width / 2 for i in x], planned_vals, width, label="distributed (planned)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(span_types, rotation=30, ha="right")
    ax.set_ylabel("FIM examples")
    ax.set_title("Span-type distribution — random vs distributed variant")
    ax.legend()
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bucket_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


_DEFAULT_STAGE_LABEL = "FIM variant generation (10k pilot sample)"
_DEFAULT_CHECKPOINTS_NOTE = (
    "empty — generation is a single in-memory pass over the 10k-file "
    "sample (minutes, not hours), nothing to resume"
)


def render_experiment_metadata(
    folder_name: str,
    records: list[dict[str, Any]],
    sampling_cfg: dict[str, Any],
    shortfall: dict[str, int] | None = None,
    stage_label: str = _DEFAULT_STAGE_LABEL,
    checkpoints_note: str = _DEFAULT_CHECKPOINTS_NOTE,
) -> dict[str, Any]:
    """Aggregate metadata.json for one experiment/{folder_name}/ folder
    (`folder_name` is e.g. "random_data" or "distributed_data" — the HF
    folder name, not the internal random/planned code) — counts + config,
    never a per-row dump (data-stage-2.md §3; per-row fim_type/content_id/
    span metadata already lives on every record in the parquet file
    itself).

    `stage_label`/`checkpoints_note` default to the original 10k-pilot
    wording (every existing call site keeps that text unchanged); pass
    overrides for a folder generated under different circumstances — e.g.
    experiment/safim_eval_1000's fixed, never-retrained eval set."""
    return {
        "folder": f"experiment/{folder_name}",
        "stage": stage_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": folder_name,
        "sampling_config": dict(sampling_cfg),
        "counts": {
            "total_examples": len(records),
            "by_fim_type": count_by_fim_type(records),
        },
        "shortfall_vs_target": dict(shortfall or {}),
        "checkpoints": checkpoints_note,
    }


def write_experiment_metadata_json(
    folder_name: str,
    records: list[dict[str, Any]],
    sampling_cfg: dict[str, Any],
    shortfall: dict[str, int] | None = None,
    out_path: str | None = None,
    stage_label: str = _DEFAULT_STAGE_LABEL,
    checkpoints_note: str = _DEFAULT_CHECKPOINTS_NOTE,
) -> Path:
    metadata = render_experiment_metadata(
        folder_name, records, sampling_cfg, shortfall, stage_label, checkpoints_note
    )
    path = Path(out_path or f"reports/experiment_{folder_name}_metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
