import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment
from experiment import (
    DEFAULT_SAVES,
    RC_FAILED,
    RC_INTERRUPTED,
    RC_OK,
    ConfigError,
    Runner,
    RunOutcome,
    build_argv,
    cmd_clean,
    cmd_list,
    cmd_run,
    default_saves_dir,
    expand_matrix,
    load_config,
    main,
    summarize,
)


def make_cfg(tmp, name="exp", brains=None, presets=None, seeds=None,
             defaults=None, max_parallel=None, binary=None):
    cfg = {
        "name": name,
        "brains": brains or [{"name": "Random", "kind": "random"}],
    }
    if presets is not None:
        cfg["presets"] = presets
    if seeds is not None:
        cfg["seeds"] = seeds
    if defaults is not None:
        cfg["defaults"] = defaults
    if max_parallel is not None:
        cfg["max_parallel"] = max_parallel
    if binary is not None:
        cfg["alife_binary"] = binary
    path = os.path.join(tmp, "cfg.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return path


class FakeProc:
    next_pid = 5000

    def __init__(self, cmd, rc=0, delay=0, stdout=None, stderr=None, env=None,
                 on_done=None):
        self.cmd = cmd
        self.rc = rc
        self.delay = delay
        self.pid = FakeProc.next_pid
        FakeProc.next_pid += 1
        self._polls = 0
        self._terminated = False
        self._reported_done = False
        self.on_done = on_done
        self.launched = {"stdout": stdout, "stderr": stderr, "env": env}

    def poll(self):
        if self._terminated:
            return -15
        self._polls += 1
        if self._polls > self.delay:
            if not self._reported_done and self.on_done:
                self.on_done()
            self._reported_done = True
            return self.rc
        return None

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True

    def wait(self, timeout=None):
        return self.rc


class ExperimentCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def saves(self):
        return os.path.join(self._tmp, "saves")

    # -- argv construction --------------------------------------------------

    def test_build_argv_random(self):
        cfg = load_config(make_cfg(self._tmp, presets=[{"name": "hard", "cfg": {"move_cost_same_dir": 2}}]))
        spec = expand_matrix(cfg, Path(self.saves()))[0]
        argv = build_argv("bin", spec, Path(self._tmp))
        self.assertEqual(argv, [
            "bin", "--headless", "--brain-kind", "random",
            "--ticks", "200", "--seed", "0", "--width", "50",
            "--height", "50", "--agents", "20", "--energy-cells", "80",
            "--output-dir", str(spec.output_dir),
            "--cfg", "move_cost_same_dir=2",
        ])

    def test_build_argv_checkpoint_every(self):
        cfg = load_config(make_cfg(self._tmp, defaults={"checkpoint_every": 50}))
        spec = expand_matrix(cfg, Path(self.saves()))[0]
        argv = build_argv("bin", spec, Path(self._tmp))
        self.assertIn("--checkpoint-every", argv)
        self.assertEqual(argv[argv.index("--checkpoint-every") + 1], "50")

    def test_build_argv_llm(self):
        cfg = load_config(make_cfg(self._tmp, brains=[{
            "name": "Sonnet", "kind": "llm",
            "llm_endpoint": "http://127.0.0.1:8000/v1",
            "llm_model": "sonnet5", "llm_api_key": "dummy",
        }]))
        spec = expand_matrix(cfg, Path(self.saves()))[0]
        argv = build_argv("bin", spec, Path(self._tmp))
        self.assertEqual(argv[argv.index("--llm-endpoint") + 1], "http://127.0.0.1:8000/v1")
        self.assertEqual(argv[argv.index("--llm-model") + 1], "sonnet5")
        self.assertEqual(argv[argv.index("--llm-api-key") + 1], "dummy")
        self.assertNotIn("--nn-path", argv)

    def test_build_argv_nn_relative_resolved_against_cfg_dir(self):
        cfg = load_config(make_cfg(self._tmp, brains=[{
            "name": "Net", "kind": "neural_net", "nn_path": "weights.bin",
        }]))
        spec = expand_matrix(cfg, Path(self.saves()))[0]
        argv = build_argv("bin", spec, Path(self._tmp))
        self.assertEqual(argv[argv.index("--nn-path") + 1],
                         str(Path(self._tmp) / "weights.bin"))

    def test_build_argv_nn_absolute_passthrough(self):
        cfg = load_config(make_cfg(self._tmp, brains=[{
            "name": "Net", "kind": "neural_net",
            "nn_path": "/abs/weights.bin",
        }]))
        spec = expand_matrix(cfg, Path(self.saves()))[0]
        argv = build_argv("bin", spec, Path(self._tmp))
        self.assertEqual(argv[argv.index("--nn-path") + 1], "/abs/weights.bin")

    def test_build_argv_uses_output_dir_under_saves_experiment(self):
        cfg = load_config(make_cfg(self._tmp, name="myexp", seeds=[1, 2]))
        specs = expand_matrix(cfg, Path(self.saves()))
        self.assertTrue(str(specs[0].output_dir).endswith(
            os.path.join("saves", "myexp", "Random-baseline-s1")))

    # -- config validation ---------------------------------------------------

    def test_config_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config(os.path.join(self._tmp, "nope.json"))

    def test_config_bad_json(self):
        path = os.path.join(self._tmp, "bad.json")
        with open(path, "w") as fh:
            fh.write("{not json")
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_config_requires_name(self):
        path = os.path.join(self._tmp, "c.json")
        with open(path, "w") as fh:
            json.dump({"brains": [{"name": "R", "kind": "random"}]}, fh)
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_config_requires_brains(self):
        path = os.path.join(self._tmp, "c.json")
        with open(path, "w") as fh:
            json.dump({"name": "e", "brains": []}, fh)
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_config_bad_kind(self):
        path = make_cfg(self._tmp, brains=[{"name": "R", "kind": "hologram"}])
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_config_duplicate_brain_names(self):
        path = make_cfg(self._tmp, brains=[
            {"name": "R", "kind": "random"},
            {"name": "R", "kind": "random"},
        ])
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_config_bad_seeds(self):
        path = make_cfg(self._tmp, seeds=["a"])
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_config_defaults_presets_and_seeds(self):
        path = make_cfg(self._tmp)
        cfg = load_config(path)
        self.assertEqual(cfg["presets"], [{"name": "baseline", "cfg": {}}])
        self.assertEqual(cfg["seeds"], [0])

    # -- matrix expansion -----------------------------------------------------

    def test_expand_matrix_cross_product(self):
        cfg = load_config(make_cfg(
            self._tmp,
            brains=[
                {"name": "Random", "kind": "random"},
                {"name": "Sonnet", "kind": "llm"},
            ],
            presets=[
                {"name": "base", "cfg": {}},
                {"name": "spendy", "cfg": {"move_cost_same_dir": 9}},
            ],
            seeds=[1, 2],
        ))
        specs = expand_matrix(cfg, Path(self.saves()))
        names = [s.name for s in specs]
        self.assertEqual(names, [
            "Random-base-s1", "Random-base-s2",
            "Random-spendy-s1", "Random-spendy-s2",
            "Sonnet-base-s1", "Sonnet-base-s2",
            "Sonnet-spendy-s1", "Sonnet-spendy-s2",
        ])
        self.assertEqual({s.seed for s in specs}, {1, 2})
        self.assertEqual({s.brain["kind"] for s in specs}, {"random", "llm"})

    def test_expand_matrix_brain_name_sanitized(self):
        cfg = load_config(make_cfg(self._tmp, brains=[{"name": "brain server x", "kind": "random"}]))
        specs = expand_matrix(cfg, Path(self.saves()))
        self.assertEqual(specs[0].name, "brain_server_x-baseline-s0")

    # -- runner lifecycle -----------------------------------------------------

    def test_runner_respects_max_parallel_and_reaps_all(self):
        cfg = load_config(make_cfg(self._tmp, seeds=[1, 2, 3, 4]))
        specs = expand_matrix(cfg, Path(self.saves()))
        procs = []
        active = set()
        max_active = [0]

        def maker(cmd, stdout=None, stderr=None, env=None, **kw):
            out = cmd[cmd.index("--output-dir") + 1]
            name = os.path.basename(out)
            p = FakeProc(cmd, rc=0, delay=1, stdout=stdout, stderr=stderr,
                         env=env, on_done=lambda: active.discard(name))
            procs.append(p)
            active.add(name)
            max_active[0] = max(max_active[0], len(active))
            return p

        runner = Runner("bin", 2, popen=maker, poll_interval=0)
        outcomes, interrupted = runner.run(specs, Path(self._tmp))
        self.assertFalse(interrupted)
        self.assertEqual(len(procs), 4)
        self.assertEqual(len(outcomes), 4)
        # cap 2: job 3 cannot start until a slot frees
        self.assertEqual(max_active[0], 2)
        for o in outcomes.values():
            self.assertEqual(o.rc, 0)

    def test_runner_failure_nonzero_exit(self):
        cfg = load_config(make_cfg(self._tmp, brains=[
            {"name": "A", "kind": "random"},
            {"name": "B", "kind": "random"},
        ]))

        def maker(cmd, stdout=None, stderr=None, env=None, **kw):
            name = os.path.basename(cmd[cmd.index("--output-dir") + 1])
            rc = 3 if name.endswith("B-baseline-s0") else 0
            return FakeProc(cmd, rc=rc)

        with mock.patch.object(experiment, "logger"):
            runner = Runner("bin", 2, popen=maker, poll_interval=0)
            specs = expand_matrix(cfg, Path(self._tmp))
            outcomes, _ = runner.run(specs, Path(self._tmp))
        ok, exhausted, failed = summarize(outcomes)
        self.assertEqual(ok, 1)
        self.assertEqual(failed, 1)
        self.assertIn("B-baseline-s0", outcomes)
        self.assertTrue(outcomes["B-baseline-s0"].failed)

    def test_runner_exhausted_rc2_not_failed(self):
        def maker(cmd, stdout=None, stderr=None, env=None, **kw):
            return FakeProc(cmd, rc=2)

        with mock.patch.object(experiment, "logger"):
            runner = Runner("bin", 2, popen=maker, poll_interval=0)
            cfg = load_config(make_cfg(self._tmp))
            specs = expand_matrix(cfg, Path(self._tmp))
            outcomes, _ = runner.run(specs, Path(self._tmp))
        o = outcomes[specs[0].name]
        self.assertTrue(o.exhausted)
        self.assertFalse(o.failed)
        ok, exhausted, failed = summarize(outcomes)
        self.assertEqual((ok, exhausted, failed), (0, 1, 0))

    def test_runner_spawn_error_marks_failed(self):
        def maker(cmd, stdout=None, stderr=None, env=None, **kw):
            raise OSError("boom")

        with mock.patch.object(experiment, "logger"):
            runner = Runner("bin", 2, popen=maker, poll_interval=0)
            cfg = load_config(make_cfg(self._tmp))
            specs = expand_matrix(cfg, Path(self._tmp))
            outcomes, _ = runner.run(specs, Path(self._tmp))
        self.assertIsNotNone(outcomes[specs[0].name].error)

    # -- cmd_run --------------------------------------------------------------

    def test_cmd_run_dry_run_prints_argv_spawns_nothing(self):
        path = make_cfg(self._tmp)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = cmd_run(path, dry_run=True, max_parallel=None,
                         saves_dir=self.saves(), binary=None)
        self.assertEqual(rc, RC_OK)
        self.assertIn("--headless", buf.getvalue())
        self.assertIn("--brain-kind random", buf.getvalue())
        self.assertFalse(os.path.exists(self.saves()))

    def test_cmd_run_nonzero_exit_on_failed_run(self):
        path = make_cfg(self._tmp, seeds=[1, 2])

        def maker(cmd, stdout=None, stderr=None, env=None, **kw):
            name = os.path.basename(cmd[cmd.index("--output-dir") + 1])
            rc = 9 if name.endswith("-s2") else 0
            return FakeProc(cmd, rc=rc)

        with mock.patch.object(experiment, "Runner") as runner_cls:
            runner_cls.return_value.run = lambda specs, cfd: self._fake_run(specs, maker)
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = cmd_run(path, dry_run=False, max_parallel=None,
                             saves_dir=self.saves(), binary=None)
        self.assertEqual(rc, RC_FAILED)
        self.assertIn("1 failed", buf.getvalue())
        report = json.loads(Path(self.saves(), "exp", "report.json").read_text())
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["ok"], 1)

    def test_cmd_run_interrupted_returns_rc_interrupted(self):
        path = make_cfg(self._tmp)

        def fake_run(specs, cfd):
            outcomes = {s.name: RunOutcome(name=s.name, output_dir=s.output_dir)
                        for s in specs}
            for o in outcomes.values():
                o.interrupted = True
            return outcomes, True

        with mock.patch.object(experiment, "Runner") as runner_cls:
            runner_cls.return_value.run = fake_run
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = cmd_run(path, dry_run=False, max_parallel=None,
                             saves_dir=self.saves(), binary=None)
        self.assertEqual(rc, RC_INTERRUPTED)

    def test_cmd_run_config_error(self):
        bad = os.path.join(self._tmp, "bad.json")
        with open(bad, "w") as fh:
            fh.write("{}")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = cmd_run(bad, dry_run=False, max_parallel=None,
                         saves_dir=self.saves(), binary=None)
        self.assertEqual(rc, RC_FAILED)

    # -- list / clean ---------------------------------------------------------

    def _make_world(self, root, name):
        d = os.path.join(root, name)
        os.makedirs(d)
        with open(os.path.join(d, "manifest.json"), "w") as fh:
            json.dump({"version": 5}, fh)

    def test_list_and_clean(self):
        saves = self.saves()
        exp = os.path.join(saves, "exp1")
        self._make_world(exp, "run1")
        self._make_world(exp, "run2")
        self._make_world(saves, "solo")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = cmd_list(saves_dir=saves)
        self.assertEqual(rc, RC_OK)
        out = buf.getvalue()
        self.assertIn("exp1/ (experiment, 2 runs)", out)
        self.assertIn("solo/", out)

        with contextlib.redirect_stdout(io.StringIO()):
            rc = cmd_clean("exp1", saves_dir=saves, yes=True, dry=False)
        self.assertEqual(rc, RC_OK)
        self.assertFalse(os.path.exists(exp))
        self.assertTrue(os.path.exists(os.path.join(saves, "solo")))

    def test_clean_dry_run_does_not_delete(self):
        saves = self.saves()
        exp = os.path.join(saves, "exp")
        self._make_world(os.path.join(exp, "run"), "placeholder")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cmd_clean("exp", saves_dir=saves, yes=True, dry=True)
        self.assertEqual(rc, RC_OK)
        self.assertTrue(os.path.exists(exp))

    def test_clean_missing_dir_fails(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = cmd_clean("ghost", saves_dir=self.saves(), yes=True, dry=False)
        self.assertEqual(rc, RC_FAILED)

    def _fake_run(self, specs, maker):
        got = {}
        for spec in specs:
            name = spec.name
            p = maker(build_argv("bin", spec, Path(self._tmp)))
            got[name] = RunOutcome(name=name, output_dir=spec.output_dir, rc=p.rc)
        return got, False

    # -- cli dispatch ---------------------------------------------------------

    # -- default saves root (ALIFE_DIR-aware) ----------------------------------

    def test_default_saves_dir_uses_alife_dir(self):
        with mock.patch.dict(os.environ, {"ALIFE_DIR": "/rt"}, clear=True):
            self.assertEqual(default_saves_dir(), Path("/rt") / "saves")

    def test_default_saves_dir_legacy_when_alife_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_saves_dir(), Path(DEFAULT_SAVES))

    def test_cmd_list_uses_alife_dir_default(self):
        with mock.patch.dict(os.environ, {"ALIFE_DIR": self._tmp}, clear=True):
            self._make_world(os.path.join(self._tmp, "saves"), "w")
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = cmd_list(saves_dir=None)
        self.assertEqual(rc, RC_OK)
        self.assertIn("w/", buf.getvalue())

    def test_cmd_clean_uses_alife_dir_default(self):
        with mock.patch.dict(os.environ, {"ALIFE_DIR": self._tmp}, clear=True):
            self._make_world(os.path.join(self._tmp, "saves"), "gone")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cmd_clean("gone", saves_dir=None, yes=True, dry=False)
        self.assertEqual(rc, RC_OK)
        self.assertFalse(os.path.exists(os.path.join(self._tmp, "saves", "gone")))

    def test_cmd_run_saves_under_alife_dir_default(self):
        path = make_cfg(self._tmp)

        def maker(cmd, stdout=None, stderr=None, env=None, **kw):
            return FakeProc(cmd, rc=0)

        with mock.patch.dict(os.environ, {"ALIFE_DIR": self._tmp}, clear=True):
            with mock.patch.object(experiment, "Runner") as runner_cls:
                runner_cls.return_value.run = lambda specs, cfd: self._fake_run(specs, maker)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = cmd_run(path, dry_run=False, max_parallel=None,
                                 saves_dir=None, binary=None)
        self.assertEqual(rc, RC_OK)
        report = json.loads(Path(self._tmp, "saves", "exp", "report.json").read_text())
        self.assertEqual(report["summary"]["ok"], 1)

    def test_main_dispatch(self):
        missing = os.path.join(self._tmp, "nope")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["list", "--saves-dir", missing]), RC_FAILED)
            self.assertEqual(main(["clean", "x", "--saves-dir", missing]), RC_FAILED)
            with self.assertRaises(SystemExit):
                main([])


if __name__ == "__main__":
    unittest.main()