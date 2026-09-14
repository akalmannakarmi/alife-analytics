# alife-analytics

Extract per-tick world metrics from saved alife-world runs and render them in a
static line-chart dashboard for side-by-side brain comparison.

Two pieces, no build step:

- `extract.py` — Python 3.12 stdlib-only extractor. Reads save dirs and emits
  JSON: `analytics/index.json` (world list + metadata) and
  `analytics/worlds/<id>.json` (per-world time-series arrays).
- `dashboard/` — static site (uPlot.js vendored, no CDN) served from any local
  static file server.

## Requirements

- Python 3.12+ (stdlib only)
- A static file server for the dashboard (`python3 -m http.server` is enough)

## Extract

```sh
python3 extract.py
```

Default input: `../runtime/saves` and
`../alife-data-collector/runtime-collector` (both relative to this repo).
Output: `analytics/` in the current directory.

```sh
python3 extract.py --saves ../other/saves --out /tmp/out
```

- `--saves <dir>` — repeatable; every directory tree is walked and any dir
  containing `manifest.json` is treated as a world.
- `--out <dir>` — default `analytics`.

A world is emitted only if it has `stats.bin`. Save dirs with only
`manifest.json` (logging disabled) are listed in the index with a
`no_data` flag and no series file. A header-only `stats.bin` (8 bytes) yields a
zero-length series flagged `empty`. `obs.bin` is never read.

## Serve the dashboard

```sh
python3 -m http.server 8000 --directory ..   # from this repo: serves repo root
# open http://localhost:8000/dashboard/
```

The dashboard reads `../analytics/` relative to its URL by default. If the
data lives somewhere else, override with a `data` query param:

```sh
# http://localhost:8000/dashboard/?data=../analytics
```

## Test

```sh
python3 -m unittest discover -s tests
```

## Data format

### `analytics/index.json`

```jsonc
{
  "count": 2,
  "worlds": [
    {
      "id": "scar-run",
      "name": "scar-run",
      "file": "worlds/scar-run.json",
      "source": "../runtime/saves",
      "dir": "../runtime/saves/scar-run",
      "brain": { "name": null, "kind": "llm" },
      "settings": { "target_tick_rate": 10, "map_mode": null },
      "config": { "seed": 42, "width": 512, "height": 512,
                  "starting_agent_count": 200, "starting_energy_cell_count": 40,
                  "enable_logging": true, "log_dir_path": "...", "checkpoint_interval": 1000 },
      "final": { "tick": 50, "agent_count": 200, "energy_cell_count": 190,
                 "manifest_tick_count": 50 },
      "series_length": 50,
      "flags": []
    }
  ]
}
```

- `brain`/`settings` come from `settings.json` (absent fields → `null`;
  pre-metadata saves show as unknown brain). Brain kinds:
  `none | random | neural_net | llm`.
- `flags`: `empty` (header-only stats), `no_data` (no stats.bin),
  `stats_truncated`, `actions_truncated`, `stats_bad_magic`,
  `stats_unsupported_version`, `actions_bad_header`.
- Duplicate world names become suffixed ids (`MyWorld`, `MyWorld_2`).

### `analytics/worlds/<id>.json`

```jsonc
{
  "name": "scar-run",
  "tick_count": 50,
  "flags": [],
  "series": {
    "tick": [1, 2, ...],
    "agent_count": [...],
    "energy_cell_count": [...],
    "grid_energy": [...],
    "agent_energy": [...],
    "unplaced_energy": [...],
    "births": [...],
    "deaths": [...],
    "move": [...],
    "split": [...],
    "absorb": [...]
  }
}
```

Columnar arrays, aligned on `tick`. `move`/`split`/`absorb` are per-tick counts
aggregated from `actions.bin` (individual actions never materialized).
`births`/`deaths` are per-tick rates (reset every tick by the engine).

## Source formats

Both binary files are header-then-records, little-endian:

- `stats.bin` — magic `STAT`, version 2; records of 48 bytes:
  `u64 tick, u32 agent_count, u32 energy_cell_count, u64 grid_energy,
  u64 agent_energy, u64 unplaced_energy, u32 births, u32 deaths`.
- `actions.bin` — magic `ACTL`, version 1; records of `u64 tick, u32 count`
  followed by `count ×` 7-byte action entries (`u32 agent_id, u8 tag[, payload]`
  — tag 1 move, 2 split, 3 absorb). Only tags are counted.