import os
import tempfile
import unittest

from ram_coffers_topology import (
  _parse_numactl_output,
  build_placement_plan,
  calculate_weights,
)


class TestRamCoffersTopology(unittest.TestCase):
  def test_parse_numactl_output_expands_cpu_ranges_and_memory_fields(self):
    output = """
available: 2 nodes (0-1)
node 0 cpus: 0-3 8 9
node 0 size: 32768 MB
node 0 free: 12000 MB
node 1 cpus: 4 5 6 7
node 1 size: 65536 MB
node 1 free: 50000 MB
node distances:
"""

    topology = _parse_numactl_output(output)

    self.assertEqual(topology["system"], "linux")
    self.assertEqual(topology["num_nodes"], 2)
    self.assertEqual(topology["nodes"][0]["cpus"], [0, 1, 2, 3, 8, 9])
    self.assertEqual(topology["nodes"][0]["size_mb"], 32768)
    self.assertEqual(topology["nodes"][0]["free_mb"], 12000)
    self.assertEqual(topology["nodes"][1]["cpus"], [4, 5, 6, 7])
    self.assertEqual(topology["nodes"][1]["size_mb"], 65536)
    self.assertEqual(topology["nodes"][1]["free_mb"], 50000)

  def test_calculate_weights_uses_memory_proportions(self):
    topology = {
      "nodes": {
        0: {"size_mb": 32768},
        1: {"size_mb": 65536},
      }
    }

    weights = calculate_weights(topology)

    self.assertAlmostEqual(weights[0], 33.3333333333)
    self.assertAlmostEqual(weights[1], 66.6666666667)

  def test_calculate_weights_splits_evenly_when_memory_is_unknown(self):
    topology = {
      "nodes": {
        0: {"size_mb": 0},
        1: {"size_mb": 0},
        2: {"size_mb": 0},
        3: {"size_mb": 0},
      }
    }

    weights = calculate_weights(topology)

    self.assertEqual(weights, {0: 25.0, 1: 25.0, 2: 25.0, 3: 25.0})

  def test_build_placement_plan_estimates_model_shards(self):
    topology = {
      "num_nodes": 2,
      "nodes": {
        0: {"cpus": [0, 1], "size_mb": 1024, "free_mb": 512},
        1: {"cpus": [2, 3], "size_mb": 3072, "free_mb": 2048},
      }
    }
    weights = calculate_weights(topology)

    with tempfile.NamedTemporaryFile(delete=False) as model:
      model.write(b"x" * 4096)
      model_path = model.name

    try:
      plan = build_placement_plan(topology, weights, model_path)
    finally:
      os.unlink(model_path)

    self.assertEqual(plan["mode"], "dry-run")
    self.assertEqual(plan["num_nodes"], 2)
    self.assertIsNone(plan["fallback_reason"])
    self.assertAlmostEqual(plan["model_size_mb"], 4096 / (1024 * 1024))
    self.assertAlmostEqual(
      plan["nodes"]["0"]["planned_shard_mb"],
      plan["model_size_mb"] * 0.25,
    )
    self.assertAlmostEqual(
      plan["nodes"]["1"]["planned_shard_mb"],
      plan["model_size_mb"] * 0.75,
    )

  def test_build_placement_plan_reports_missing_model_fallback(self):
    topology = {
      "num_nodes": 1,
      "nodes": {
        0: {"cpus": [0], "size_mb": 0, "free_mb": 0},
      }
    }

    plan = build_placement_plan(topology, {0: 100.0}, "missing.gguf")

    self.assertEqual(plan["model_path"], "missing.gguf")
    self.assertEqual(plan["model_size_mb"], None)
    self.assertIn("model path not found", plan["fallback_reason"])


if __name__ == "__main__":
  unittest.main()
