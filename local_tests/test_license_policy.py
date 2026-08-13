"""
local_tests/test_license_policy.py
Unit tests for src/curation/license_policy.py.

Usage:
    pytest local_tests/test_license_policy.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.curation.license_policy import decide_license

KEEP_TYPES = ["permissive", "no_license"]


class TestDecideLicense:
    def test_permissive_is_kept(self):
        decision = decide_license("permissive", ["MIT"], KEEP_TYPES)
        assert decision.keep is True
        assert decision.reason == "keep"

    def test_no_license_is_kept(self):
        decision = decide_license("no_license", [], KEEP_TYPES)
        assert decision.keep is True

    def test_ambiguous_type_is_flagged_not_kept(self):
        decision = decide_license("restricted", ["GPL-3.0"], KEEP_TYPES)
        assert decision.keep is False
        assert decision.reason == "flagged_ambiguous"

    def test_missing_license_type_defaults_to_no_license(self):
        decision = decide_license(None, None, KEEP_TYPES)
        assert decision.keep is True
        assert decision.license_type == "no_license"

    def test_license_fields_always_recorded_even_when_flagged(self):
        """§0: license fields are saved unconditionally, never omitted."""
        decision = decide_license("restricted", ["Custom-EULA"], KEEP_TYPES)
        assert decision.license_type == "restricted"
        assert decision.detected_licenses == ["Custom-EULA"]

    def test_missing_detected_licenses_becomes_empty_list_not_none(self):
        decision = decide_license("permissive", None, KEEP_TYPES)
        assert decision.detected_licenses == []
