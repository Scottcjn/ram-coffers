# SPDX-License-Identifier: MIT
"""Unit tests for scripts/topology_preflight.py."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Import preflight module
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.topology_preflight import (
    evaluate_compatibility,
    get_expected_compiler_flags,
    run_preflight,
)


class TestTopologyPreflight(unittest.TestCase):
    def test_json_cli_invocation(self):
        """Executing scripts/topology_preflight.py --json prints valid JSON schema."""
        script_path = repo_root / "scripts" / "topology_preflight.py"
        res = subprocess.run(
            [sys.executable, str(script_path), "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        self.assertIn("architecture", data)
        self.assertIn("cpu_info", data)
        self.assertIn("numa_topology", data)
        self.assertIn("compiler_flags", data)
        self.assertIn("verdict", data)
        self.assertEqual(data["preflight_version"], "1.0.0")

    def test_expected_compiler_flags_present(self):
        """Compiler flags reference contains POWER8, MASS, and fallback flags."""
        flags = get_expected_compiler_flags()
        self.assertIn("power8_release", flags)
        self.assertIn("power8_mass", flags)
        self.assertIn("-mcpu=power8", flags["power8_release"])
        self.assertIn("-maltivec", flags["power8_release"])
        self.assertIn("-mvsx", flags["power8_release"])
        self.assertIn("x86_64_fallback", flags)

    def test_power8_optimal_compatibility(self):
        """POWER8 host with VSX and 4 NUMA nodes yields OPTIMAL verdict."""
        arch = {"machine": "ppc64le", "system": "linux", "is_powerpc": True}
        cpu = {"has_vsx": True, "has_altivec": True, "has_crypto": True}
        numa = {"num_nodes": 4}

        verdict = evaluate_compatibility(arch, cpu, numa)
        self.assertEqual(verdict["status"], "OPTIMAL")
        self.assertEqual(verdict["benchmark_mode"], "native_power8_4coffer")
        self.assertTrue(verdict["can_reproduce_canonical_147_ts"])

    def test_power8_single_node_compatibility(self):
        """POWER8 host with VSX but <4 NUMA nodes yields COMPATIBLE verdict."""
        arch = {"machine": "ppc64le", "system": "linux", "is_powerpc": True}
        cpu = {"has_vsx": True, "has_altivec": True, "has_crypto": False}
        numa = {"num_nodes": 1}

        verdict = evaluate_compatibility(arch, cpu, numa)
        self.assertEqual(verdict["status"], "COMPATIBLE")
        self.assertEqual(verdict["benchmark_mode"], "native_power8_sub_topology")
        self.assertFalse(verdict["can_reproduce_canonical_147_ts"])

    def test_x86_fallback_compatibility(self):
        """x86 host yields FALLBACK_ONLY verdict."""
        arch = {"machine": "x86_64", "system": "linux", "is_powerpc": False}
        cpu = {"has_vsx": False, "has_altivec": False, "has_crypto": False}
        numa = {"num_nodes": 2}

        verdict = evaluate_compatibility(arch, cpu, numa)
        self.assertEqual(verdict["status"], "FALLBACK_ONLY")
        self.assertEqual(verdict["benchmark_mode"], "foreign_architecture_emulation")
        self.assertFalse(verdict["can_reproduce_canonical_147_ts"])


if __name__ == "__main__":
    unittest.main()
