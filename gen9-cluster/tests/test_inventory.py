"""Inventory files and the CLI that consumes them."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from gen9_cluster.cli import main
from gen9_cluster.hardware import GB, ComputeBackend, Runtime
from gen9_cluster.inventory import (addresses, deployment_config,
                                    entry_from_dict, load_fleet, save_fleet,
                                    synthetic_fleet, units)
from gen9_cluster.model import DEEPSEEK_TINY
from gen9_cluster.planner import plan_split

FLEET = {
    "fleet": [
        {"unit_id": "ps5-01", "sku": "ps5", "runtime": "ps5-linux",
         "host": "10.0.0.11", "port": 9713},
        {"unit_id": "ps5-02", "sku": "ps5", "runtime": "ps5-linux",
         "host": "10.0.0.12",
         "downbin": {"cu_disabled": 4, "gpu_ghz_cap": 1.8,
                     "tier_losses": {"gddr6": 2147483648},
                     "reasons": ["dead GDDR6 package", "fan replaced"]}},
        {"unit_id": "bc-01", "sku": "bc-250", "runtime": "salvage-linux",
         "host": "10.0.0.31", "cu_enabled_override": 40, "backend": "rocm",
         "measured_gemv_gflops": 940.0},
        {"unit_id": "xsx-01", "sku": "xbox-series-x", "runtime": "xbox-devmode",
         "host": "10.0.0.21", "labels": {"rack": "b", "slot": "4"}},
    ]
}


class InventoryFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "fleet.json"
        self.path.write_text(json.dumps(FLEET))


class TestLoading(InventoryFile):
    def test_every_entry_is_read(self):
        self.assertEqual(len(load_fleet(self.path)), 4)

    def test_downbin_survives_the_file(self):
        fleet = {e.unit.unit_id: e for e in load_fleet(self.path)}
        damaged = fleet["ps5-02"].unit
        self.assertEqual(damaged.downbin.cu_disabled, 4)
        self.assertEqual(damaged.downbin.gpu_ghz_cap, 1.8)
        self.assertEqual(damaged.downbin.tier_losses["gddr6"], 2 * GB)
        self.assertIn("dead GDDR6 package", damaged.downbin.reasons)
        self.assertEqual(damaged.effective().cu_active, 32)

    def test_a_declared_rocm_node_keeps_its_measurement(self):
        fleet = {e.unit.unit_id: e for e in load_fleet(self.path)}
        board = fleet["bc-01"].unit
        self.assertEqual(board.backend, ComputeBackend.ROCM)
        self.assertEqual(board.effective().gemv_gflops, 940.0)
        self.assertTrue(board.effective().throughput_measured)
        self.assertEqual(board.effective().cu_active, 40)

    def test_addresses_are_extracted_for_the_dispatcher(self):
        table = addresses(load_fleet(self.path))
        self.assertEqual(table["ps5-01"].host, "10.0.0.11")
        self.assertEqual(table["ps5-01"].port, 9713)
        self.assertEqual(table["bc-01"].port, 9713)     # default

    def test_a_duplicate_unit_id_is_rejected(self):
        """Two consoles with the same id means one of them silently never gets
        its shards, and the plan will not say so."""
        path = Path(self.dir.name) / "dupe.json"
        path.write_text(json.dumps({"fleet": [FLEET["fleet"][0],
                                              FLEET["fleet"][0]]}))
        with self.assertRaises(ValueError):
            load_fleet(path)

    def test_a_missing_field_names_itself(self):
        with self.assertRaises(ValueError) as caught:
            entry_from_dict({"sku": "ps5"})
        self.assertIn("unit_id", str(caught.exception))

    def test_round_trip_through_save(self):
        original = load_fleet(self.path)
        out = Path(self.dir.name) / "out.json"
        save_fleet(out, original)
        again = {e.unit.unit_id: e.unit for e in load_fleet(out)}
        self.assertEqual(again["ps5-02"].downbin.reasons,
                         ("dead GDDR6 package", "fan replaced"))
        self.assertEqual(again["xsx-01"].labels, {"rack": "b", "slot": "4"})
        self.assertEqual(again["xsx-01"].runtime, Runtime.XBOX_DEVMODE)


class TestSyntheticFleet(unittest.TestCase):
    def test_counts_are_honoured_and_ids_are_unique(self):
        fleet = synthetic_fleet({"ps5": 3, "xbox-series-s": 2})
        self.assertEqual(len(fleet), 5)
        self.assertEqual(len({e.unit.unit_id for e in fleet}), 5)
        self.assertEqual(len({e.port for e in fleet}), 5)


class TestDeploymentConfig(InventoryFile):
    def test_each_console_is_told_only_its_own_shards(self):
        fleet = load_fleet(self.path)
        plan = plan_split(DEEPSEEK_TINY, units(fleet), context_tokens=512)
        config = deployment_config(plan, fleet)
        self.assertEqual(set(config["nodes"]), set(plan.units))
        for unit_id, node in config["nodes"].items():
            self.assertEqual(node["host"],
                             {e.unit.unit_id: e.host for e in fleet}[unit_id])
            for shard in node["shards"]:
                self.assertEqual(shard["unit_id"], unit_id)

    def test_the_config_is_json_serialisable(self):
        fleet = load_fleet(self.path)
        plan = plan_split(DEEPSEEK_TINY, units(fleet), context_tokens=512)
        json.dumps(deployment_config(plan, fleet))

    def test_an_assumed_model_is_marked_in_the_deployment(self):
        fleet = load_fleet(self.path)
        plan = plan_split(DEEPSEEK_TINY, units(fleet), context_tokens=512)
        self.assertIn("model_assumed", deployment_config(plan, fleet))


class TestCLI(InventoryFile):
    def run_cli(self, *argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(list(argv))
        return code, buffer.getvalue()

    def test_model_sizes_a_profile(self):
        code, out = self.run_cli("model", "--model", "deepseek-v3")
        self.assertEqual(code, 0)
        self.assertIn("deepseek-v3", out)

    def test_plan_places_a_model_on_the_inventory(self):
        code, out = self.run_cli("plan", str(self.path), "--model",
                                 "deepseek-tiny", "--context", "512")
        self.assertEqual(code, 0)
        self.assertIn("shelves", out)
        self.assertIn("bc-01", out)

    def test_plan_can_write_a_deployment_config(self):
        target = Path(self.dir.name) / "cluster.json"
        code, _ = self.run_cli("plan", str(self.path), "--model",
                               "deepseek-tiny", "--config", str(target))
        self.assertEqual(code, 0)
        self.assertIn("nodes", json.loads(target.read_text()))

    def test_plan_reports_an_impossible_fleet_without_a_traceback(self):
        small = Path(self.dir.name) / "one.json"
        small.write_text(json.dumps({"fleet": [FLEET["fleet"][0]]}))
        code, _ = self.run_cli("plan", str(small), "--model",
                               "deepseek-v4-pro", "--no-ssd")
        self.assertEqual(code, 1)

    def test_size_answers_how_many_consoles(self):
        code, out = self.run_cli("size", "--model", "deepseek-tiny", "--ps5",
                                 "4")
        self.assertEqual(code, 0)
        self.assertIn("4 x ps5", out)

    def test_size_without_counts_is_a_usage_error(self):
        code, _ = self.run_cli("size", "--model", "deepseek-tiny")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
