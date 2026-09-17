#!/usr/bin/env python3
"""
Unit tests for scripts/topology_preflight.py.
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.topology_preflight import (
    evaluate_isa,
    evaluate_topology_preflight,
    get_cpu_info,
)


class TestTopologyPreflight(unittest.TestCase):
    def test_evaluate_isa_power8_detection(self):
        cpu_info = {
            "arch": "ppc64le",
            "system": "Linux",
            "model": "POWER8 (architected), altivec supported",
            "flags": ["vsx", "altivec", "vcipher"],
            "cores": 16,
        }
        isa = evaluate_isa(cpu_info)
        self.assertTrue(isa["is_power"])
        self.assertTrue(isa["has_power8"])
        self.assertTrue(isa["has_vsx"])
        self.assertTrue(isa["has_altivec"])
        self.assertIn("-mcpu=power8", isa["recommended_compiler_flags"])
        self.assertIn("-mvsx", isa["recommended_compiler_flags"])

    def test_evaluate_isa_x86_64_detection(self):
        cpu_info = {
            "arch": "x86_64",
            "system": "Linux",
            "model": "AMD Ryzen 5 7600X",
            "flags": ["avx2", "fma", "sse4_2"],
            "cores": 12,
        }
        isa = evaluate_isa(cpu_info)
        self.assertFalse(isa["is_power"])
        self.assertTrue(isa["is_x86_64"])
        self.assertTrue(isa["has_avx2"])
        self.assertIn("-mavx2", isa["recommended_compiler_flags"])

    def test_verdict_power8_multi_numa_reference(self):
        mock_cpu = {
            "arch": "ppc64le",
            "system": "Linux",
            "model": "POWER8 (raw), altivec supported",
            "flags": ["vsx", "altivec"],
            "cores": 128,
        }
        mock_numa = {
            "system": "linux",
            "num_nodes": 4,
            "nodes": {
                0: {"cpus": list(range(32)), "size_mb": 131072, "free_mb": 100000},
                1: {"cpus": list(range(32, 64)), "size_mb": 131072, "free_mb": 100000},
                2: {"cpus": list(range(64, 96)), "size_mb": 131072, "free_mb": 100000},
                3: {"cpus": list(range(96, 128)), "size_mb": 131072, "free_mb": 100000},
            },
        }
        with patch("scripts.topology_preflight.get_cpu_info", return_value=mock_cpu), \
             patch("scripts.topology_preflight.detect_numa", return_value=mock_numa):
            report = evaluate_topology_preflight()
            self.assertTrue(report["verdict"]["compatible"])
            self.assertEqual(report["verdict"]["benchmark_mode"], "power8_s824_reference")
            self.assertTrue(report["verdict"]["meets_reference_power8_spec"])
            self.assertEqual(report["numa_topology"]["num_nodes"], 4)

    def test_verdict_single_node_fallback(self):
        mock_cpu = {
            "arch": "x86_64",
            "system": "Linux",
            "model": "Intel Core i7",
            "flags": ["avx2"],
            "cores": 8,
        }
        mock_numa = {
            "system": "linux",
            "num_nodes": 1,
            "nodes": {
                0: {"cpus": list(range(8)), "size_mb": 16384, "free_mb": 8000},
            },
        }
        with patch("scripts.topology_preflight.get_cpu_info", return_value=mock_cpu), \
             patch("scripts.topology_preflight.detect_numa", return_value=mock_numa):
            report = evaluate_topology_preflight()
            self.assertEqual(report["verdict"]["benchmark_mode"], "single_node_fallback")
            self.assertFalse(report["verdict"]["meets_reference_power8_spec"])


if __name__ == "__main__":
    unittest.main()
