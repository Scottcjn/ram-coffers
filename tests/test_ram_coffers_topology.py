import contextlib
import io
import os
import tempfile
import unittest

from ram_coffers_topology import (
  _parse_numactl_output,
  _parse_sysfs_numa,
  build_placement_plan,
  calculate_weights,
  display_topology_text,
  format_cpu_list,
  parse_cpu_list,
)


class TestParseCpuList(unittest.TestCase):
  def test_single_cpu(self):
    self.assertEqual(parse_cpu_list("5"), [5])

  def test_single_range(self):
    self.assertEqual(parse_cpu_list("0-31"), list(range(32)))

  def test_mixed_ranges_and_singles_comma_separated(self):
    # sysfs cpulist spelling; the old parser split "0-31,64-95" on "-" and
    # got three pieces, so a POWER8 SMT node raised ValueError.
    self.assertEqual(parse_cpu_list("0-31,64-95"),
                     list(range(0, 32)) + list(range(64, 96)))
    self.assertEqual(parse_cpu_list("0-3,8,9"), [0, 1, 2, 3, 8, 9])

  def test_mixed_space_separated_numactl_spelling(self):
    self.assertEqual(parse_cpu_list("0-3 8 9"), [0, 1, 2, 3, 8, 9])

  def test_whitespace_is_tolerated(self):
    self.assertEqual(parse_cpu_list("  0-3 , 8 ,\t9  \n"), [0, 1, 2, 3, 8, 9])

  def test_empty_means_memory_only_node(self):
    self.assertEqual(parse_cpu_list(""), [])
    self.assertEqual(parse_cpu_list("\n"), [])

  def test_result_is_sorted_and_deduplicated(self):
    self.assertEqual(parse_cpu_list("9,0-3,3,8"), [0, 1, 2, 3, 8, 9])

  def test_garbage_is_rejected_not_guessed(self):
    for bad in ("a-b", "3-1", "1--2", "x", "1-"):
      with self.assertRaises(ValueError, msg=bad):
        parse_cpu_list(bad)


class TestFormatCpuList(unittest.TestCase):
  def test_gaps_stay_visible(self):
    # The old display printed f"{cpus[0]}-{cpus[-1]}", so an SMT node's
    # 0-31,64-95 showed as 0-95: 32 CPUs that belong to another node.
    self.assertEqual(format_cpu_list(list(range(32)) + list(range(64, 96))),
                     "0-31,64-95")

  def test_singles_and_pairs(self):
    self.assertEqual(format_cpu_list([0, 1, 2, 3, 8, 9]), "0-3,8-9")
    self.assertEqual(format_cpu_list([5]), "5")
    self.assertEqual(format_cpu_list([1, 3, 5]), "1,3,5")

  def test_empty_is_na(self):
    self.assertEqual(format_cpu_list([]), "N/A")

  def test_round_trips_through_the_parser(self):
    cpus = [0, 1, 2, 3, 8, 9, 64, 65, 66]
    self.assertEqual(parse_cpu_list(format_cpu_list(cpus)), cpus)


class TestRamCoffersTopology(unittest.TestCase):
  def test_parse_numactl_output_handles_smt_sibling_ranges(self):
    output = "node 0 cpus: 0-31,64-95\nnode 0 size: 1 MB\n"
    topology = _parse_numactl_output(output)
    self.assertEqual(topology["nodes"][0]["cpus"],
                     list(range(0, 32)) + list(range(64, 96)))

  def test_parse_sysfs_numa_accepts_ranges_and_memory_only_nodes(self):
    with tempfile.TemporaryDirectory() as base:
      for node_id, cpulist in ((0, "0-31,64-95\n"), (1, "\n")):
        node_path = os.path.join(base, f"node{node_id}")
        os.makedirs(node_path)
        with open(os.path.join(node_path, "cpulist"), "w") as f:
          f.write(cpulist)
      topology = _parse_sysfs_numa(base)
    self.assertEqual(topology["nodes"][0]["cpus"],
                     list(range(0, 32)) + list(range(64, 96)))
    self.assertEqual(topology["nodes"][1]["cpus"], [])

  def test_display_text_does_not_flatten_cpu_gaps(self):
    topology = {
      "num_nodes": 2,
      "nodes": {
        0: {"cpus": list(range(32)) + list(range(64, 96)), "size_mb": 1024},
        1: {"cpus": [], "size_mb": 1024},
      },
    }
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      display_topology_text(topology, {0: 50.0, 1: 50.0})
    self.assertIn("CPUs 0-31,64-95", out.getvalue())
    self.assertIn("CPUs N/A", out.getvalue())
    self.assertNotIn("0-95", out.getvalue())

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

  def test_parse_sysfs_numa_reads_memory_value_not_node_id(self):
    # Reproduces the kernel sysfs layout: "Node <id> MemTotal: <kB> kB".
    # The memory value is the 4th field; a naive parser grabs the node id
    # (field 2) instead and reports 0 MB for every node.
    with tempfile.TemporaryDirectory() as base:
      layout = {
        0: {"cpulist": "0-3", "MemTotal": 33554432, "MemFree": 12582912},
        1: {"cpulist": "4-7", "MemTotal": 67108864, "MemFree": 50331648},
      }
      for node_id, spec in layout.items():
        node_path = os.path.join(base, f"node{node_id}")
        os.makedirs(node_path)
        with open(os.path.join(node_path, "cpulist"), "w") as f:
          f.write(spec["cpulist"] + "\n")
        with open(os.path.join(node_path, "meminfo"), "w") as f:
          f.write(f"Node {node_id} MemTotal:       {spec['MemTotal']} kB\n")
          f.write(f"Node {node_id} MemFree:        {spec['MemFree']} kB\n")
          f.write(f"Node {node_id} MemUsed:        1000 kB\n")

      topology = _parse_sysfs_numa(base)

    self.assertEqual(topology["num_nodes"], 2)
    self.assertEqual(topology["nodes"][0]["cpus"], [0, 1, 2, 3])
    self.assertEqual(topology["nodes"][0]["size_mb"], 33554432 // 1024)
    self.assertEqual(topology["nodes"][0]["free_mb"], 12582912 // 1024)
    self.assertEqual(topology["nodes"][1]["size_mb"], 67108864 // 1024)
    self.assertEqual(topology["nodes"][1]["free_mb"], 50331648 // 1024)

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
