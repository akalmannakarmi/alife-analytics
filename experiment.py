#!/usr/bin/env python3
"""One-shot experiment runner for alife.

Glues a matrix of brains x presets x seeds to the real headless CLI, produces
world saves under the analytics save roots, and can extract + serve the
dashboard over the results.

Python stdlib only — no third-party dependencies.

Commands:
  experiment run <cfg.json> [--dry-run]   run the matrix (parallel, capped)
  experiment chart [dir] [--port N]       extract + serve the dashboard
  experiment list [--saves-dir DIR]       list save dirs under the root
  experiment clean <name> [--yes]         remove an experiment's save dir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import extract

DEFAULT_BINARY = "../alife/zig-out/bin/alife"
DEFAULT_SAVES = "../runtime/saves"
DEFAULT_PORT = 8765

KINDS = ("random", "neural_net", "llm")

# Exit codes: 0 = all runs ok, 1 = one or more runs failed (or config error),
# 2 = interrupted by signal. Child exit code 2 (brain keys exhausted) counts
# as a completed run (the world is still saved) and is reported separately.
RC_OK = 0
RC_FAILED = 1
RC_INTERRUPTED = 2

logger = logging.getLogger("experiment")


# ---------------------------------------------------------------------------
# Naming / sanitising
# ---------------------------------------------------------------------------

def sanitize(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return clean or "run"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    pass


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    cfg_path = Path(path)
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {cfg_path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file is not valid JSON: {cfg_path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    if not isinstance(raw.get("name"), str) or not raw["name"].strip():
        raise ConfigError('config requires a string "name" (experiment name)')
    brains = raw.get("brains")
    if not isinstance(brains, list) or not brains:
        raise ConfigError('config requires a non-empty "brains" list')
    seen = set()
    for b in brains:
        if not isinstance(b, dict):
            raise ConfigError("each brain entry must be an object")
        if not isinstance(b.get("name"), str) or not b["name"].strip():
            raise ConfigError("each brain entry requires a string name")
        if b["name"] in seen:
            raise ConfigError(f"duplicate brain name: {b['name']}")
        seen.add(b["name"])
        kind = b.get("kind")
        if kind not in KINDS:
            raise ConfigError(
                f"brain {b['name']!r}: kind must be one of {', '.join(KINDS)} (got {kind!r})"
            )
    presets = raw.get("presets", [{"name": "baseline", "cfg": {}}])
    if not isinstance(presets, list) or not presets:
        raise ConfigError('config requires a non-empty "presets" list')
    pseen = set()
    for p in presets:
        if not isinstance(p, dict):
            raise ConfigError("each preset entry must be an object")
        if not isinstance(p.get("name"), str) or not p["name"].strip():
            raise ConfigError("each preset requires a string name")
        if p["name"] in pseen:
            raise ConfigError(f"duplicate preset name: {p['name']}")
        pseen.add(p["name"])
        cfg = p.get("cfg", {})
        if not isinstance(cfg, dict):
            raise ConfigError(f"preset {p['name']!r}: cfg must be an object of field=value pairs")
    seeds = raw.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds or not all(
        isinstance(s, int) and not isinstance(s, bool) for s in seeds
    ):
        raise ConfigError('config requires a non-empty "seeds" list of integers')
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError('"defaults" must be an object')
    raw["presets"] = presets
    raw["seeds"] = seeds
    raw["defaults"] = defaults
    return raw


# ---------------------------------------------------------------------------
# Matrix expansion
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    name: str
    brain: dict[str, Any]
    preset: dict[str, Any]
    seed: int
    ticks: int
    width: int
    height: int
    agents: int
    energy_cells: int
    checkpoint_every: int
    output_dir: Path


def expand_matrix(cfg: dict[str, Any], saves_base: Path) -> list[RunSpec]:
    """Expand brains x presets x seeds into concrete run specs."""
    exp_dir = saves_base / cfg["name"]
    defaults = cfg.get("defaults", {})
    ticks = defaults.get("ticks", 200)
    width = defaults.get("width", 50)
    height = defaults.get("height", 50)
    agents = defaults.get("agents", 20)
    energy_cells = defaults.get("energy_cells", 80)
    checkpoint_every = defaults.get("checkpoint_every", 0)

    specs = []
    for brain in cfg["brains"]:
        for preset in cfg["presets"]:
            for seed in cfg["seeds"]:
                run_name = f"{sanitize(brain['name'])}-{sanitize(preset['name'])}-s{seed}"
                specs.append(
                    RunSpec(
                        name=run_name,
                        brain=brain,
                        preset=preset,
                        seed=seed,
                        ticks=ticks,
                        width=width,
                        height=height,
                        agents=agents,
                        energy_cells=energy_cells,
                        checkpoint_every=checkpoint_every,
                        output_dir=exp_dir / run_name,
                    )
                )
    return specs


# ---------------------------------------------------------------------------
# argv construction (the ONLY thing that talks to the real CLI)
# ---------------------------------------------------------------------------

def resolve_nn_path(brain: dict[str, Any], cfg_file_dir: Path) -> str | None:
    """Relative --nn-path resolves against the config file's directory."""
    nn_path = brain.get("nn_path")
    if not nn_path:
        return None
    p = Path(nn_path)
    if not p.is_absolute():
        p = cfg_file_dir / p
    return str(p)


def build_argv(binary: str | Path, spec: RunSpec, cfg_file_dir: Path) -> list[str]:
    """Build the real headless argv for one run. Never invents flags."""
    brain = spec.brain
    kind = brain["kind"]
    argv = [
        str(binary),
        "--headless",
        "--brain-kind", kind,
        "--ticks", str(spec.ticks),
        "--seed", str(spec.seed),
        "--width", str(spec.width),
        "--height", str(spec.height),
        "--agents", str(spec.agents),
        "--energy-cells", str(spec.energy_cells),
        "--output-dir", str(spec.output_dir),
    ]
    if spec.checkpoint_every > 0:
        argv += ["--checkpoint-every", str(spec.checkpoint_every)]
    for field_name, value in spec.preset.get("cfg", {}).items():
        argv += ["--cfg", f"{field_name}={value}"]
    if kind == "llm":
        if brain.get("llm_endpoint"):
            argv += ["--llm-endpoint", str(brain["llm_endpoint"])]
        if brain.get("llm_model"):
            argv += ["--llm-model", str(brain["llm_model"])]
        if brain.get("llm_api_key"):
            argv += ["--llm-api-key", str(brain["llm_api_key"])]
    elif kind == "neural_net":
        nn_path = resolve_nn_path(brain, cfg_file_dir)
        if nn_path:
            argv += ["--nn-path", nn_path]
    return argv


# ---------------------------------------------------------------------------
# Runner (parallel, capped — pattern from alife-data-collector)
# ---------------------------------------------------------------------------

@dataclass
class RunOutcome:
    name: str
    output_dir: Path
    rc: int | None = None
    error: str | None = None
    interrupted: bool = False

    @property
    def failed(self) -> bool:
        if self.interrupted:
            return True
        if self.error is not None:
            return True
        return self.rc not in (0, 2)

    @property
    def exhausted(self) -> bool:
        return self.rc == 2 and not self.interrupted


class Runner:
    def __init__(
        self,
        binary: str | Path,
        max_parallel: int = 3,
        popen: Callable[..., Any] | None = None,
        poll_interval: float = 0.5,
    ) -> None:
        self.binary = binary
        self.max_parallel = max(1, max_parallel)
        self.popen = popen or subprocess.Popen
        self.poll_interval = poll_interval
        self._shutdown = False

    def _handle_shutdown(self, _signum: int, _frame: Any) -> None:
        self._shutdown = True

    def _install_signals(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def run(self, specs: list[RunSpec], cfg_file_dir: Path) -> tuple[dict[str, RunOutcome], bool]:
        """Run all specs (up to max_parallel at once), returning name->outcome."""
        self._install_signals()
        pending = list(specs)
        running: dict[str, tuple[Any, Any]] = {}
        outcomes: dict[str, RunOutcome] = {
            s.name: RunOutcome(name=s.name, output_dir=s.output_dir) for s in specs
        }

        def start(spec: RunSpec) -> None:
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            log_fh = open(spec.output_dir / "run.log", "w", encoding="utf-8")
            cmd = build_argv(self.binary, spec, cfg_file_dir)
            try:
                proc = self.popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            except Exception as e:
                log_fh.close()
                outcomes[spec.name].error = str(e)
                logger.error("[%s] failed to start: %s", spec.name, e)
                return
            running[spec.name] = (proc, log_fh)
            logger.info("[%s] started (pid %d)", spec.name, proc.pid)

        while (pending or running) and not self._shutdown:
            if pending:
                for _ in range(self.max_parallel - len(running)):
                    if not pending or self._shutdown:
                        break
                    spec = pending.pop(0)
                    start(spec)
            for name in list(running):
                proc, log_fh = running[name]
                rc = proc.poll()
                if rc is None:
                    continue
                log_fh.close()
                outcomes[name].rc = rc
                level = logging.WARNING if rc == 2 else (logging.INFO if rc == 0 else logging.ERROR)
                logger.log(level, "[%s] exited rc=%d", name, rc)
                del running[name]
            if not running and not pending:
                break
            time.sleep(self.poll_interval)

        interrupted = self._shutdown
        if interrupted:
            procs = [(name, proc) for name, (proc, _fh) in running.items()]
            for _name, proc in procs:
                try:
                    proc.terminate()
                except Exception:
                    pass
            for _name, proc in procs:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
            for name, (_proc, log_fh) in running.items():
                log_fh.close()
                outcomes[name].interrupted = True
            for name, outcome in outcomes.items():
                if outcome.rc is None and not outcome.error:
                    outcome.interrupted = True
        return outcomes, interrupted


def summarize(outcomes: dict[str, RunOutcome]) -> tuple[int, int, int]:
    ok = sum(1 for o in outcomes.values() if o.rc == 0 and not o.interrupted)
    exhausted = sum(1 for o in outcomes.values() if o.exhausted)
    failed = sum(1 for o in outcomes.values() if o.failed)
    return ok, exhausted, failed


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def report_outcomes(outcomes: dict[str, RunOutcome]) -> None:
    for outcome in sorted(outcomes.values(), key=lambda o: o.name):
        if outcome.interrupted:
            status = "interrupted"
        elif outcome.error:
            status = f"FAILED ({outcome.error})"
        elif outcome.exhausted:
            status = "exhausted (rc=2, world saved)"
        elif outcome.rc == 0:
            status = "ok"
        else:
            status = f"FAILED (rc={outcome.rc})"
        print(f"  {outcome.name:<40} {status:<45} {outcome.output_dir}")


def cmd_run(cfg_path: str, dry_run: bool, max_parallel: int | None,
            saves_dir: str | None, binary: str | None) -> int:
    try:
        cfg = load_config(cfg_path)
    except ConfigError as e:
        print(f"experiment: {e}", file=sys.stderr)
        return RC_FAILED

    cfg_file_dir = Path(cfg_path).resolve().parent
    saves_base = Path(saves_dir) if saves_dir else Path(DEFAULT_SAVES)
    if not saves_base.is_absolute():
        saves_base = Path.cwd() / saves_base
    bin_path = binary or cfg.get("alife_binary", DEFAULT_BINARY)
    parallel = max_parallel or int(cfg.get("max_parallel", 3))

    specs = expand_matrix(cfg, saves_base)
    exp_dir = specs[0].output_dir.parent
    print(f"experiment: {cfg['name']}: {len(specs)} runs "
          f"({len(cfg['brains'])} brains x {len(cfg['presets'])} presets "
          f"x {len(cfg['seeds'])} seeds)")
    print(f"experiment: output -> {exp_dir} (max_parallel={parallel})")

    if dry_run:
        for spec in specs:
            cmd = build_argv(bin_path, spec, cfg_file_dir)
            print("  " + " ".join(cmd))
        return RC_OK

    runner = Runner(bin_path, parallel)
    outcomes, interrupted = runner.run(specs, cfg_file_dir)

    ok, exhausted, failed = summarize(outcomes)
    print(f"experiment: results ({cfg['name']}):")
    report_outcomes(outcomes)
    print(f"experiment: summary: {ok} ok, {exhausted} exhausted, {failed} failed")

    report = {
        "experiment": cfg["name"],
        "saves_dir": str(saves_base),
        "count": len(specs),
        "summary": {"ok": ok, "exhausted": exhausted, "failed": failed},
        "runs": [
            {
                "name": o.name,
                "output_dir": str(o.output_dir),
                "rc": o.rc,
                "error": o.error,
                "exhausted": o.exhausted,
                "interrupted": o.interrupted,
            }
            for o in sorted(outcomes.values(), key=lambda o: o.name)
        ],
    }
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"experiment: could not write report.json: {e}", file=sys.stderr)

    if interrupted:
        print("experiment: interrupted; partial results only", file=sys.stderr)
        return RC_INTERRUPTED
    return RC_OK if failed == 0 else RC_FAILED


def cmd_chart(exp_dir_arg: str | None, port: int, out: str) -> int:
    """Extract over the produced worlds, then serve the dashboard."""
    extract_args = ["--out", out]
    if exp_dir_arg:
        extract_args += ["--saves", exp_dir_arg]
    extract.main(extract_args)

    root = Path(__file__).resolve().parent
    os.chdir(str(root))  # serve from the repo root so /dashboard/ and /analytics/ resolve
    quiet_handler = type(
        "QuietHandler",
        (SimpleHTTPRequestHandler,),
        {"log_message": lambda self, fmt, *args: None},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), quiet_handler)
    print(f"experiment: dashboard: http://localhost:{port}/dashboard/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("experiment: server stopped")
    finally:
        httpd.server_close()
    return RC_OK


def cmd_list(saves_dir: str | None) -> int:
    saves_base = Path(saves_dir) if saves_dir else Path(DEFAULT_SAVES)
    if not saves_base.is_absolute():
        saves_base = Path.cwd() / saves_base
    if not saves_base.is_dir():
        print(f"experiment: no saves dir: {saves_base}", file=sys.stderr)
        return RC_FAILED
    found = []
    for entry in sorted(saves_base.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        if (entry / "manifest.json").is_file():
            found.append(("run", entry.name))
        else:
            runs = [c for c in entry.iterdir() if c.is_dir() and (c / "manifest.json").is_file()]
            if runs:
                found.append(("experiment", entry.name, len(runs)))
    if not found:
        print(f"experiment: nothing under {saves_base}")
    for item in found:
        if item[0] == "experiment":
            print(f"  {item[1]}/ (experiment, {item[2]} runs)")
        else:
            print(f"  {item[1]}/")
    return RC_OK


def cmd_clean(name: str, saves_dir: str | None, yes: bool, dry: bool) -> int:
    saves_base = Path(saves_dir) if saves_dir else Path(DEFAULT_SAVES)
    if not saves_base.is_absolute():
        saves_base = Path.cwd() / saves_base
    target = saves_base / name
    if not target.is_dir():
        print(f"experiment: no such save dir: {target}", file=sys.stderr)
        return RC_FAILED
    if dry:
        print(f"experiment: would remove {target}")
        return RC_OK
    if not yes:
        reply = input(f"experiment: remove {target}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("experiment: aborted")
            return RC_OK
    shutil.rmtree(target)
    print(f"experiment: removed {target}")
    return RC_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment",
        description="Run one-shot alife experiments: brains x presets x seeds -> "
                    "world saves -> analytics dashboard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a matrix from a JSON config")
    run_p.add_argument("cfg", help="path to experiment config JSON")
    run_p.add_argument("--dry-run", action="store_true",
                       help="print the constructed CLI argv for each run without running")
    run_p.add_argument("--max-parallel", type=int, default=None,
                       help="override config max_parallel")
    run_p.add_argument("--saves-dir", default=None,
                       help="override the save root (default: ../runtime/saves)")
    run_p.add_argument("--binary", default=None,
                       help="override the alife binary path")

    chart_p = sub.add_parser("chart", help="extract + serve the dashboard over produced worlds")
    chart_p.add_argument("dir", nargs="?", default=None,
                         help="experiment/save dir to extract (default: extract.py defaults)")
    chart_p.add_argument("--port", type=int, default=DEFAULT_PORT,
                         help=f"dashboard port (default: {DEFAULT_PORT})")
    chart_p.add_argument("--out", default="analytics",
                         help="extract output dir (default: analytics)")

    list_p = sub.add_parser("list", help="list save dirs under the save root")
    list_p.add_argument("--saves-dir", default=None)

    clean_p = sub.add_parser("clean", help="remove an experiment's save dir")
    clean_p.add_argument("name", help="save dir name under the save root")
    clean_p.add_argument("--saves-dir", default=None)
    clean_p.add_argument("--yes", action="store_true",
                         help="skip the confirmation prompt")
    clean_p.add_argument("--dry", action="store_true",
                         help="list what would be removed without deleting")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args.cfg, args.dry_run, args.max_parallel,
                       args.saves_dir, args.binary)
    if args.command == "chart":
        return cmd_chart(args.dir, args.port, args.out)
    if args.command == "list":
        return cmd_list(args.saves_dir)
    if args.command == "clean":
        return cmd_clean(args.name, args.saves_dir, args.yes, args.dry)
    return RC_FAILED


if __name__ == "__main__":
    sys.exit(main())