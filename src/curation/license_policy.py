"""
src/curation/license_policy.py
Stage 1 filter stage 3: license policy.

License fields are recorded unconditionally for every file that reaches this
stage (see .claude/data-v1.md §0) — this module only decides whether a file
is allowed to continue toward the curated sample. Filtering by license
happens here at storage-decision time; nothing about license is ever silently
dropped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LicenseDecision:
    keep: bool
    reason: str  # "keep" | "flagged_ambiguous"
    license_type: str
    detected_licenses: list[str]


def decide_license(
    license_type: str | None,
    detected_licenses: list[str] | None,
    keep_types: list[str],
) -> LicenseDecision:
    """
    Decide whether a file's license allows it into the curated sample.

    Args:
        license_type:      raw `files[].license_type` field (may be missing/None).
        detected_licenses: raw `files[].detected_licenses` field (may be missing/None).
        keep_types:        license_type values that pass unconditionally, from
                            configs/data/stack_v3_filter.yaml `license.keep_types`.

    Files whose license_type is not in keep_types are not dropped from the
    record — the caller is responsible for logging their full license fields
    for manual review — but they are excluded from the curated count and
    should be counted under filter_stats.license_reject.
    """
    normalized_type = license_type or "no_license"
    normalized_licenses = list(detected_licenses) if detected_licenses else []

    if normalized_type in keep_types:
        return LicenseDecision(
            keep=True,
            reason="keep",
            license_type=normalized_type,
            detected_licenses=normalized_licenses,
        )

    return LicenseDecision(
        keep=False,
        reason="flagged_ambiguous",
        license_type=normalized_type,
        detected_licenses=normalized_licenses,
    )
