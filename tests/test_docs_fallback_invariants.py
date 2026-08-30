"""
Unit test enforcing documentation invariants for RAM Coffers fallback behavior.
"""

import re
import unittest
from pathlib import Path


class TestDocsFallbackInvariants(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent.parent
        self.readme = self.root / "README.md"
        self.canonical_doc = self.root / "FALLBACK_BEHAVIOR.md"

    def test_readme_has_exactly_one_fallback_heading(self):
        """README.md must contain exactly one top-level Fallback Behavior heading."""
        content = self.readme.read_text(encoding="utf-8")
        matches = re.findall(r"^##\s+Fallback Behavior", content, re.MULTILINE | re.IGNORECASE)
        self.assertEqual(
            len(matches),
            1,
            f"Expected exactly 1 '## Fallback Behavior' heading in README.md, found {len(matches)}: {matches}",
        )

    def test_canonical_fallback_file_exists_and_contains_matrix(self):
        """FALLBACK_BEHAVIOR.md must be the canonical source containing the per-header matrix."""
        self.assertTrue(self.canonical_doc.exists(), "FALLBACK_BEHAVIOR.md must exist")
        content = self.canonical_doc.read_text(encoding="utf-8")

        required_headers = [
            "ggml-ram-coffers.h",
            "ggml-ram-coffer.h",
            "ggml-coffer-mmap.h",
            "ggml-vcipher-collapse.h",
            "ggml-intelligent-collapse.h",
            "apple-silicon/unified-memory-coffers.h",
        ]
        for header in required_headers:
            self.assertIn(header, content, f"Canonical doc missing header matrix entry: {header}")

    def test_readme_links_to_canonical_fallback_doc(self):
        """README.md must explicitly reference FALLBACK_BEHAVIOR.md."""
        content = self.readme.read_text(encoding="utf-8")
        self.assertIn("FALLBACK_BEHAVIOR.md", content)


if __name__ == "__main__":
    unittest.main()
