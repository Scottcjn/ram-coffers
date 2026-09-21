"""Every relative link and in-page anchor in README.md must resolve.

External URLs are not fetched (CI should not depend on third-party uptime);
this guards the things the repo itself controls: files it links to and
headings it points at. A README that links to a file that was moved, or to a
section that was renamed, is the docs-equivalent of a dangling pointer.
"""

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")

LINK = re.compile(r"\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)


def github_slug(heading: str) -> str:
  """GitHub's heading -> anchor rule: lowercase, drop punctuation except
  hyphens, spaces to hyphens. Markdown emphasis and code spans are stripped."""
  text = re.sub(r"[`*_]", "", heading)
  text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)   # [text](url) -> text
  text = text.strip().lower()
  text = re.sub(r"[^\w\- ]", "", text)
  return text.replace(" ", "-")


class TestReadmeLinks(unittest.TestCase):
  def setUp(self):
    with open(README, encoding="utf-8") as f:
      self.text = f.read()
    self.anchors = {github_slug(h) for h in HEADING.findall(self.text)}
    self.links = LINK.findall(self.text)

  def test_readme_has_links_to_check(self):
    self.assertGreater(len(self.links), 10)

  def test_relative_file_links_exist(self):
    missing = []
    for link in self.links:
      if link.startswith(("http://", "https://", "#", "mailto:")):
        continue
      target = link.split("#", 1)[0]
      if not os.path.exists(os.path.join(ROOT, target)):
        missing.append(link)
    self.assertEqual(missing, [])

  def test_in_page_anchors_point_at_real_headings(self):
    dangling = [link for link in self.links
                if link.startswith("#") and link[1:] not in self.anchors]
    self.assertEqual(dangling, [], f"known headings: {sorted(self.anchors)}")

  def test_x86_support_is_described_consistently(self):
    """The README once said x86 was both a compile error and a working
    scalar fallback with an AES-NI port. Keep the two summary tables from
    drifting apart again: neither may call x86 a compile error while the
    x86-64 port section exists."""
    self.assertIn("## x86-64 Port", self.text)
    row = re.search(r"^\|\s*x86[^\n]*❌ Compile error[^\n]*$", self.text,
                    re.MULTILINE)
    self.assertIsNone(row, f"x86 called a compile error: {row and row.group(0)}")
    self.assertNotIn("Compilation will fail", self.text,
                     "x86 subsection still says compilation fails")


if __name__ == "__main__":
  unittest.main()
