#!/usr/bin/env python3
import argparse
import json
import os
import re
import struct
import sys

STATS_MAGIC = 0x54415453
STATS_VERSION = 2
STATS_REC_SIZE = 48
ACTIONS_MAGIC = 0x4C544341
ACTIONS_VERSION = 1
STATS_FORMAT = struct.Struct("<QIIQQQII")
ACTION_REC = struct.Struct("<QI")
ACTION_TAIL = struct.Struct("<IB")

CACHE_FORMAT_VERSION = 1
WORLD_FILES = ("manifest.json", "settings.json", "stats.bin", "actions.bin")
CACHE_META_KEYS = ("name", "source", "dir", "brain", "settings", "config", "final", "series_length", "flags")


def warn(msg):
    print(f"extract: {msg}", file=sys.stderr)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def parse_stats(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 8:
        return [], ["stats_truncated"]
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != STATS_MAGIC:
        return [], ["stats_bad_magic"]
    if version != STATS_VERSION:
        return [], ["stats_unsupported_version"]
    body = data[8:]
    rows = []
    flags = []
    pos = 0
    total = len(body)
    while pos + STATS_REC_SIZE <= total:
        rows.append(STATS_FORMAT.unpack_from(body, pos))
        pos += STATS_REC_SIZE
    if pos != total:
        flags.append("stats_truncated")
    return rows, flags


def parse_actions(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 8:
        return {}, ["actions_truncated"]
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != ACTIONS_MAGIC or version != ACTIONS_VERSION:
        return {}, ["actions_bad_header"]
    body = data[8:]
    counts = {}
    flags = []
    pos = 0
    total = len(body)
    while pos + 12 <= total:
        tick, count = ACTION_REC.unpack_from(body, pos)
        pos += 12
        if pos + 6 * count > total:
            break
        move = 0
        split = 0
        absorb = 0
        ok = True
        for _ in range(count):
            _, tag = ACTION_TAIL.unpack_from(body, pos)
            pos += 7 if tag == 2 else 6
            if pos > total:
                ok = False
                break
            if tag == 1:
                move += 1
            elif tag == 2:
                split += 1
            elif tag == 3:
                absorb += 1
        if not ok:
            break
        counts[tick] = (move, split, absorb)
    if pos != total:
        flags.append("actions_truncated")
    return counts, flags


def build_series(stats_rows, action_counts):
    ticks = sorted({r[0] for r in stats_rows} | set(action_counts))
    agent_count = []
    energy_cell_count = []
    grid_energy = []
    agent_energy = []
    unplaced_energy = []
    births = []
    deaths = []
    move = []
    split = []
    absorb = []
    idx = 0
    total = len(stats_rows)
    last = None
    for tick in ticks:
        while idx < total and stats_rows[idx][0] < tick:
            last = stats_rows[idx]
            idx += 1
        row = None
        if idx < total and stats_rows[idx][0] == tick:
            row = stats_rows[idx]
            last = row
            idx += 1
        if row is None and last is None:
            agent_count.append(0)
            energy_cell_count.append(0)
            grid_energy.append(0)
            agent_energy.append(0)
            unplaced_energy.append(0)
            births.append(0)
            deaths.append(0)
        else:
            _, ac, ec, ge, ae, ue, b, d = row if row is not None else last
            agent_count.append(ac)
            energy_cell_count.append(ec)
            grid_energy.append(ge)
            agent_energy.append(ae)
            unplaced_energy.append(ue)
            births.append(b)
            deaths.append(d)
        mv, sp, ab = action_counts.get(tick, (0, 0, 0))
        move.append(mv)
        split.append(sp)
        absorb.append(ab)
    return {
        "tick": ticks,
        "agent_count": agent_count,
        "energy_cell_count": energy_cell_count,
        "grid_energy": grid_energy,
        "agent_energy": agent_energy,
        "unplaced_energy": unplaced_energy,
        "births": births,
        "deaths": deaths,
        "move": move,
        "split": split,
        "absorb": absorb,
    }


def sanitize(name):
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return clean or "world"


GENERIC_NAMES = {"output", "out", "world", "saves"}


def display_name(sub):
    parts = sub.replace(os.sep, "/").split("/")
    while parts and parts[-1] in GENERIC_NAMES:
        parts.pop()
    if parts:
        return parts[-1]
    return sub.replace(os.sep, "/").split("/")[-1]


def load_world(root, sub):
    world_dir = os.path.join(root, sub)
    manifest = read_json(os.path.join(world_dir, "manifest.json"))
    if not isinstance(manifest, dict):
        return None
    confs = manifest.get("confs") if isinstance(manifest.get("confs"), dict) else {}
    raw_settings = read_json(os.path.join(world_dir, "settings.json"))
    settings = raw_settings if isinstance(raw_settings, dict) else {}

    stats_path = os.path.join(world_dir, "stats.bin")
    actions_path = os.path.join(world_dir, "actions.bin")
    flags = []
    has_stats_file = os.path.isfile(stats_path)
    if has_stats_file:
        stats_rows, stats_flags = parse_stats(stats_path)
        flags.extend(stats_flags)
    else:
        stats_rows = []
        flags.append("no_data")
    if os.path.isfile(actions_path):
        action_counts, action_flags = parse_actions(actions_path)
        flags.extend(action_flags)
    else:
        action_counts = {}

    series = build_series(stats_rows, action_counts)
    series_length = len(series["tick"])
    if series_length == 0 and has_stats_file:
        flags.append("empty")

    manifest_count = manifest.get("agent_count")
    manifest_energy = manifest.get("energy_cell_count")
    if series_length:
        final = {
            "tick": series["tick"][-1],
            "agent_count": series["agent_count"][-1],
            "energy_cell_count": series["energy_cell_count"][-1],
            "manifest_tick_count": manifest.get("tick_count"),
        }
    else:
        final = {
            "tick": None,
            "agent_count": manifest_count,
            "energy_cell_count": manifest_energy,
            "manifest_tick_count": manifest.get("tick_count"),
        }

    meta = {
        "name": display_name(sub),
        "source": root,
        "dir": world_dir,
        "brain": {
            "name": settings.get("brain_name"),
            "kind": settings.get("brain_kind"),
        },
        "settings": {
            "target_tick_rate": settings.get("target_tick_rate"),
            "map_mode": settings.get("map_mode"),
        },
        "config": confs,
        "final": final,
        "series_length": series_length,
        "flags": flags,
    }
    return meta, series


def write_world(out_dir, world_id, meta, series):
    payload = {
        "name": meta["name"],
        "tick_count": meta["series_length"],
        "flags": meta["flags"],
        "series": series,
    }
    path = os.path.join(out_dir, "worlds", f"{world_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return f"worlds/{world_id}.json"


def file_mark(path):
    try:
        st = os.stat(path)
    except OSError:
        return [0, 0]
    return [st.st_size, st.st_mtime_ns]


def fingerprint_world(world_dir):
    return [file_mark(os.path.join(world_dir, f)) for f in WORLD_FILES]


def fingerprint_self():
    return file_mark(os.path.abspath(__file__))


def load_cache(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    if raw.get("cache_format") != CACHE_FORMAT_VERSION:
        return {}
    if raw.get("self_fingerprint") != fingerprint_self():
        return {}
    entries = raw.get("worlds")
    if not isinstance(entries, dict):
        return {}
    return entries


def make_cache(path, entries):
    return {
        "cache_format": CACHE_FORMAT_VERSION,
        "self_fingerprint": fingerprint_self(),
        "worlds": entries,
    }


def save_cache(path, entries):
    payload = make_cache(path, entries)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="extract",
        description="Extract per-tick world metrics from save dirs into JSON for the analytics dashboard.",
    )
    parser.add_argument(
        "--saves",
        action="append",
        metavar="DIR",
        help="directory tree containing world save dirs (root of each world holds manifest.json); repeatable",
    )
    parser.add_argument(
        "--out",
        default="analytics",
        metavar="DIR",
        help="output directory (default: analytics)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the extraction cache and re-extract every world (cache is still refreshed)",
    )
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.saves:
        roots = [os.path.normpath(os.path.join(os.getcwd(), r)) for r in args.saves]
    else:
        alife_dir = os.environ.get("ALIFE_DIR")
        if alife_dir:
            roots = [
                os.path.normpath(os.path.join(alife_dir, "saves")),
                os.path.normpath(os.path.join(alife_dir, "collector")),
            ]
        else:
            roots = [
                os.path.normpath(os.path.join(script_dir, "..", "runtime", "saves")),
                os.path.normpath(os.path.join(script_dir, "..", "alife-data-collector", "runtime-collector")),
            ]

    existing = []
    for root in roots:
        if os.path.isdir(root):
            existing.append(root)
        else:
            warn(f"skipping missing saves dir: {root}")

    found = []
    for root in existing:
        for base, dirs, files in os.walk(root):
            if "manifest.json" in files:
                found.append((root, os.path.relpath(base, root)))
                dirs[:] = []
    found.sort(key=lambda item: (item[0], item[1]))

    out_dir = args.out if os.path.isabs(args.out) else os.path.normpath(os.path.join(os.getcwd(), args.out))
    worlds_dir = os.path.join(out_dir, "worlds")
    os.makedirs(worlds_dir, exist_ok=True)

    cache_path = os.path.join(out_dir, ".extract-cache.json")
    cache = {} if args.no_cache else load_cache(cache_path)
    if cache is None:
        cache = {}
    fresh_cache = {}

    used_ids = {}
    index = []
    had_stats = 0
    skipped_data = 0
    re_extracted = 0
    cached_entries = 0
    fresh_cache = {}
    for root, sub in found:
        world_dir = os.path.normpath(os.path.join(root, sub))
        fp = fingerprint_world(world_dir)
        existing = cache.get(world_dir)
        cached_valid = False
        if existing is not None and existing.get("fingerprint") == fp:
            f = existing.get("file")
            if f is None or os.path.isfile(os.path.join(out_dir, f)):
                meta = existing.get("meta")
                if (isinstance(meta, dict)
                        and meta.get("dir") == world_dir
                        and all(k in meta for k in CACHE_META_KEYS)):
                    cached_valid = True

        if cached_valid:
            entry = dict(existing["meta"])
            cached_entries += 1
        else:
            loaded = load_world(root, sub)
            if loaded is None:
                warn(f"skipping unreadable world dir: {world_dir}")
                continue
            meta, series = loaded
            entry = dict(meta)
            re_extracted += 1

        base_id = sanitize(entry["name"])
        used_ids[base_id] = used_ids.get(base_id, 0) + 1
        entry["id"] = base_id if used_ids[base_id] == 1 else f"{base_id}_{used_ids[base_id]}"

        if "no_data" in entry["flags"]:
            entry["file"] = None
            skipped_data += 1
        else:
            had_stats += 1
            if cached_valid:
                entry["file"] = existing.get("file")
            else:
                entry["file"] = write_world(out_dir, entry["id"], entry, series)

        fresh_cache[world_dir] = {
            "fingerprint": fp,
            "meta": {k: entry[k] for k in CACHE_META_KEYS},
            "file": entry["file"],
        }
        index.append(entry)

    save_cache(cache_path, fresh_cache)

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump({"count": len(index), "worlds": index}, fh, indent=1, separators=(",", ": "))

    print(f"extract: {len(index)} worlds ({re_extracted} re-extracted, {cached_entries} cached)")
    print(f"extract: {had_stats} with timeseries, {skipped_data} without stats.bin (skipped)")
    print(f"extract: wrote {index_path}")


if __name__ == "__main__":
    main()