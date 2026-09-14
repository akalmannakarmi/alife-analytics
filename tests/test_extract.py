import contextlib
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extract

STATS_MAGIC = 0x54415453
ACTIONS_MAGIC = 0x4C544341


def stats_bin(rows):
    out = struct.pack("<II", STATS_MAGIC, 2)
    out += b"".join(struct.pack("<QIIQQQII", *r) for r in rows)
    return out


def actions_bin(records):
    out = struct.pack("<II", ACTIONS_MAGIC, 1)
    for tick, actions in records:
        out += struct.pack("<QI", tick, len(actions))
        for aid, tag, tail in actions:
            out += struct.pack("<IB", aid, tag) + bytes(tail)
    return out


def run_extract(saves, out):
    args = []
    for s in (saves if isinstance(saves, list) else [saves]):
        args += ["--saves", s]
    args += ["--out", out]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        extract.main(args)


def run_extract_stdout(saves, out, extra=None):
    args = []
    for s in (saves if isinstance(saves, list) else [saves]):
        args += ["--saves", s]
    args += ["--out", out]
    if extra:
        args += extra
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        extract.main(args)
    return buf.getvalue(), err.getvalue()


def run_extract_defaults(out, alife_dir=None, script_file=None):
    """Extract with no --saves, optionally under a fake ALIFE_DIR and/or a
    relocated extract.py (script-dir resolves the legacy defaults)."""
    args = ["--out", out]
    env = {"ALIFE_DIR": alife_dir} if alife_dir is not None else {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with mock.patch.dict(os.environ, env, clear=True):
            if script_file is not None:
                with mock.patch.object(extract, "__file__", script_file):
                    extract.main(args)
            else:
                extract.main(args)


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def make_world(self, root, name, files):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        for fname, payload in files.items():
            mode = "wb" if isinstance(payload, bytes) else "w"
            with open(os.path.join(d, fname), mode, encoding=None if isinstance(payload, bytes) else "utf-8") as fh:
                fh.write(payload)
        return d

    def out_dir(self):
        return os.path.join(self._tmp, "out")

    def load_index(self):
        with open(os.path.join(self.out_dir(), "index.json")) as fh:
            return json.load(fh)

    def test_full_save(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "golden world", {
            "manifest.json": json.dumps({
                "version": 5,
                "tick_count": 2,
                "confs": {
                    "seed": 42, "width": 10, "height": 10,
                    "starting_agent_count": 4, "starting_energy_cell_count": 2,
                    "enable_logging": True, "log_dir_path": None, "checkpoint_interval": 0,
                },
                "agent_count": 3, "energy_cell_count": 5,
            }),
            "settings.json": json.dumps({
                "target_tick_rate": 10, "map_mode": 0,
                "brain_name": "Ollama gpt-oss:120b-cloud", "brain_kind": "llm",
            }),
            "stats.bin": stats_bin([
                (0, 4, 2, 100, 40, 60, 1, 2),
                (1, 3, 5, 90, 30, 60, 0, 1),
            ]),
            "actions.bin": actions_bin([
                (0, [(0, 1, [0]), (1, 2, [3, 50]), (2, 3, [3])]),
                (1, [(0, 1, [1])]),
            ]),
        })
        run_extract(saves, self.out_dir())
        index = self.load_index()
        self.assertEqual(index["count"], 1)
        w = index["worlds"][0]
        self.assertEqual(w["id"], "golden_world")
        self.assertEqual(w["name"], "golden world")
        self.assertEqual(w["brain"], {"name": "Ollama gpt-oss:120b-cloud", "kind": "llm"})
        self.assertEqual(w["settings"]["target_tick_rate"], 10)
        self.assertEqual(w["series_length"], 2)
        self.assertEqual(w["final"]["agent_count"], 3)
        self.assertEqual(w["final"]["tick"], 1)
        self.assertEqual(w["file"], "worlds/golden_world.json")
        self.assertEqual(w["flags"], [])

        with open(os.path.join(self.out_dir(), w["file"])) as fh:
            world = json.load(fh)
        for key, expected in {
            "tick": [0, 1],
            "agent_count": [4, 3],
            "energy_cell_count": [2, 5],
            "grid_energy": [100, 90],
            "agent_energy": [40, 30],
            "unplaced_energy": [60, 60],
            "births": [1, 0],
            "deaths": [2, 1],
            "move": [1, 1],
            "split": [1, 0],
            "absorb": [1, 0],
        }.items():
            self.assertEqual(world["series"][key], expected, key)

    def test_settings_absent(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "plain", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {"seed": 1, "width": 8, "height": 8}}),
            "stats.bin": stats_bin([(0, 2, 1, 10, 20, 30, 0, 0)]),
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertIsNone(w["brain"]["kind"])
        self.assertIsNone(w["brain"]["name"])
        self.assertIsNone(w["settings"]["target_tick_rate"])
        self.assertEqual(w["series_length"], 1)

    def test_partial_settings(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "partial", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "settings.json": json.dumps({"brain_kind": "random"}),
            "stats.bin": stats_bin([(0, 2, 1, 10, 20, 30, 0, 0)]),
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertEqual(w["brain"]["kind"], "random")
        self.assertIsNone(w["brain"]["name"])

    def test_empty_stats(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "emptyworld", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 5, "confs": {}}),
            "stats.bin": struct.pack("<II", STATS_MAGIC, 2),
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertEqual(w["series_length"], 0)
        self.assertIn("empty", w["flags"])
        self.assertIsNotNone(w["file"])
        with open(os.path.join(self.out_dir(), w["file"])) as fh:
            world = json.load(fh)
        self.assertEqual(world["series"]["tick"], [])

    def test_no_stats(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "notlogged", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 3, "confs": {}, "agent_count": 9, "energy_cell_count": 1}),
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertIn("no_data", w["flags"])
        self.assertIsNone(w["file"])
        self.assertFalse(os.path.exists(os.path.join(self.out_dir(), "worlds", "notlogged.json")))

    def test_bad_stats_magic(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "bad", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": struct.pack("<II", 0xDEADBEEF, 2) + b"\x00" * 48,
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertIn("stats_bad_magic", w["flags"])
        self.assertEqual(w["series_length"], 0)

    def test_actions_bad_header(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "badact", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
            "actions.bin": struct.pack("<II", 0x12345678, 1) + b"\x00" * 12,
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertIn("actions_bad_header", w["flags"])

    def test_actions_truncated(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        body = struct.pack("<QI", 0, 1) + struct.pack("<IB", 0, 2) + b"\x00"
        self.make_world(saves, "trunc", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
            "actions.bin": struct.pack("<II", ACTIONS_MAGIC, 1) + body,
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertIn("actions_truncated", w["flags"])

    def test_union_ticks_carry_forward(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "union", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 4, 2, 100, 40, 60, 1, 2)]),
            "actions.bin": actions_bin([
                (0, [(0, 1, [0])]),
                (3, [(0, 2, [0, 50]), (1, 2, [0, 50])]),
            ]),
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertEqual(w["series_length"], 2)
        with open(os.path.join(self.out_dir(), w["file"])) as fh:
            world = json.load(fh)
        s = world["series"]
        self.assertEqual(s["tick"], [0, 3])
        self.assertEqual(s["agent_count"], [4, 4])
        self.assertEqual(s["move"], [1, 0])
        self.assertEqual(s["split"], [0, 2])
        self.assertEqual(s["absorb"], [0, 0])

    def test_duplicate_names_suffixed(self):
        root_a = os.path.join(self._tmp, "a")
        root_b = os.path.join(self._tmp, "b")
        for root in (root_a, root_b):
            self.make_world(root, "same", {
                "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
                "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
            })
        run_extract([root_a, root_b], self.out_dir())
        ids = [w["id"] for w in self.load_index()["worlds"]]
        self.assertEqual(sorted(ids), ["same", "same_2"])

    def test_collector_output_naming(self):
        saves = os.path.join(self._tmp, "collector")
        self.make_world(os.path.join(saves, "brain-server deepseekFree"), "output", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
        })
        run_extract(saves, self.out_dir())
        w = self.load_index()["worlds"][0]
        self.assertEqual(w["name"], "brain-server deepseekFree")
        self.assertEqual(w["id"], "brain-server_deepseekFree")

    def test_default_roots_use_alife_dir_when_set(self):
        alife_dir = os.path.join(self._tmp, "rt")
        self.make_world(os.path.join(alife_dir, "saves"), "got", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
        })
        self.make_world(os.path.join(alife_dir, "collector", "brain-a", "output"), "output", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
        })
        run_extract_defaults(self.out_dir(), alife_dir=alife_dir)
        targets = self.load_index()["worlds"]
        gens = {(w["source"], w["name"]) for w in targets}
        self.assertIn((os.path.join(alife_dir, "saves"), "got"), gens)
        self.assertIn((os.path.join(alife_dir, "collector"), "brain-a"), gens)
        file_count = sum(1 for w in targets if w["file"])
        self.assertEqual(file_count, 2)

    def test_default_roots_legacy_when_alife_unset(self):
        script_dir = os.path.join(self._tmp, "analytics")
        os.makedirs(script_dir, exist_ok=True)
        self.make_world(os.path.join(self._tmp, "runtime", "saves"), "w", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
        })
        self.make_world(os.path.join(self._tmp, "alife-data-collector", "runtime-collector", "brain-b"), "output", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
        })
        fake_extract = os.path.join(script_dir, "extract.py")
        run_extract_defaults(self.out_dir(), script_file=fake_extract)
        self.assertEqual(self.load_index()["count"], 2)

    def test_obs_bin_never_touched(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_world(saves, "big", {
            "manifest.json": json.dumps({"version": 5, "tick_count": 1, "confs": {}}),
            "stats.bin": stats_bin([(0, 2, 1, 1, 1, 1, 0, 0)]),
            "obs.bin": b"\xff" * 1024 * 1024,
        })
        run_extract(saves, self.out_dir())
        self.assertEqual(self.load_index()["count"], 1)

    def cache_path(self):
        return os.path.join(self.out_dir(), ".extract-cache.json")

    def load_cache_json(self):
        with open(self.cache_path()) as fh:
            return json.load(fh)

    def make_full_world(self, saves, name, extra_files=None, stats_rows=None, manifest=None):
        files = {
            "manifest.json": json.dumps(manifest or {
                "version": 5, "tick_count": 2,
                "confs": {"seed": 42, "width": 10, "height": 10},
                "agent_count": 3, "energy_cell_count": 5,
            }),
            "stats.bin": stats_bin(stats_rows or [(0, 4, 2, 100, 40, 60, 1, 2)]),
        }
        if extra_files:
            files.update(extra_files)
        return self.make_world(saves, name, files)

    def test_cache_first_run_extracts_all_and_writes_cache(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_full_world(saves, "a")
        self.make_full_world(saves, "b")
        out, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("2 worlds (2 re-extracted, 0 cached)", out)
        self.assertTrue(os.path.isfile(self.cache_path()))
        cache = self.load_cache_json()
        self.assertEqual(cache["cache_format"], extract.CACHE_FORMAT_VERSION)
        self.assertEqual(len(cache["worlds"]), 2)
        self.assertEqual(self.load_index()["count"], 2)

    def test_cache_second_run_unchanged_reuses_files(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        d = self.make_full_world(saves, "world")
        run_extract(saves, self.out_dir())
        world_json = os.path.join(self.out_dir(), "worlds", "world.json")
        with open(world_json, "rb") as fh:
            content1 = fh.read()
        mtime1 = os.stat(world_json).st_mtime_ns
        time.sleep(0.02)
        out2, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("1 worlds (0 re-extracted, 1 cached)", out2)
        with open(world_json, "rb") as fh:
            content2 = fh.read()
        self.assertEqual(content1, content2)
        self.assertEqual(os.stat(world_json).st_mtime_ns, mtime1)

    def test_cache_index_identical_across_runs(self):
        root_a = os.path.join(self._tmp, "a")
        root_b = os.path.join(self._tmp, "b")
        for root in (root_a, root_b):
            self.make_full_world(root, "same")
        run_extract([root_a, root_b], self.out_dir())
        with open(os.path.join(self.out_dir(), "index.json")) as fh:
            index1 = json.load(fh)
        run_extract([root_a, root_b], self.out_dir())
        with open(os.path.join(self.out_dir(), "index.json")) as fh:
            index2 = json.load(fh)
        self.assertEqual(index1, index2)
        self.assertEqual([w["id"] for w in index2["worlds"]], ["same", "same_2"])

    def test_cache_touch_stats_bin_re_extracts(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_full_world(saves, "w")
        run_extract(saves, self.out_dir())
        with open(os.path.join(saves, "w", "stats.bin"), "wb") as fh:
            fh.write(stats_bin([(0, 9, 9, 999, 999, 999, 0, 0)]))
        out2, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("1 worlds (1 re-extracted, 0 cached)", out2)
        with open(os.path.join(self.out_dir(), "worlds", "w.json")) as fh:
            world = json.load(fh)
        self.assertEqual(world["series"]["agent_count"], [9])

    def test_cache_add_settings_re_extracts(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_full_world(saves, "plain")
        run_extract(saves, self.out_dir())
        self.assertIsNone(self.load_index()["worlds"][0]["brain"]["kind"])
        self.make_world(saves, "plain", {
            "settings.json": json.dumps({"brain_kind": "random", "target_tick_rate": 10}),
        })
        out2, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("1 worlds (1 re-extracted, 0 cached)", out2)
        w = self.load_index()["worlds"][0]
        self.assertEqual(w["brain"]["kind"], "random")
        self.assertEqual(w["settings"]["target_tick_rate"], 10)

    def test_cache_delete_stats_bin_re_extracts(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_full_world(saves, "w")
        run_extract(saves, self.out_dir())
        self.assertIsNotNone(self.load_index()["worlds"][0]["file"])
        os.remove(os.path.join(saves, "w", "stats.bin"))
        out2, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("1 worlds (1 re-extracted, 0 cached)", out2)
        w = self.load_index()["worlds"][0]
        self.assertIn("no_data", w["flags"])
        self.assertIsNone(w["file"])

    def test_cache_no_cache_re_extracts(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_full_world(saves, "w")
        run_extract(saves, self.out_dir())
        world_json = os.path.join(self.out_dir(), "worlds", "w.json")
        content1 = open(world_json, "rb").read()
        time.sleep(0.02)
        out2, _err = run_extract_stdout(saves, self.out_dir(), extra=["--no-cache"])
        self.assertIn("1 worlds (1 re-extracted, 0 cached)", out2)
        content2 = open(world_json, "rb").read()
        self.assertEqual(content1, content2)
        self.assertTrue(os.path.isfile(self.cache_path()))

    def test_cache_obs_bin_change_does_not_invalidate(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_full_world(saves, "w", extra_files={"obs.bin": b"\xff"})
        run_extract(saves, self.out_dir())
        with open(os.path.join(saves, "w", "obs.bin"), "wb") as fh:
            fh.write(b"\x00" * 1024)
        out2, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("1 worlds (0 re-extracted, 1 cached)", out2)

    def test_cache_prunes_stale_world(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        d = self.make_full_world(saves, "gone")
        run_extract(saves, self.out_dir())
        self.assertEqual(len(self.load_cache_json()["worlds"]), 1)
        shutil.rmtree(d)
        out2, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("0 worlds (0 re-extracted, 0 cached)", out2)
        self.assertEqual(self.load_cache_json()["worlds"], {})
        self.assertEqual(self.load_index()["count"], 0)

    def test_cache_invalidated_when_code_changes(self):
        saves = os.path.join(self._tmp, "saves")
        os.makedirs(saves)
        self.make_full_world(saves, "w")
        run_extract(saves, self.out_dir())
        with mock.patch.object(extract, "CACHE_FORMAT_VERSION", extract.CACHE_FORMAT_VERSION + 1):
            out2, _err = run_extract_stdout(saves, self.out_dir())
        self.assertIn("1 worlds (1 re-extracted, 0 cached)", out2)


if __name__ == "__main__":
    unittest.main()