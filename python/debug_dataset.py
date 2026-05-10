#!/usr/bin/env python3
# =============================================================================
# debug_dataset.py -- Diagnose why a finished simulation run produced fewer
# dataset rows than expected.
#
# Three artifact locations:
#   - logs/     : SLURM per-task stdout/stderr (sim-<JID>_<task>.{out,err})
#                 plus the finalize job's logs. One file pair per array task.
#   - sims/     : per-run metadata (metadata_<design>_<run>.json), per-run
#                 SPICE logs (run_<design>_<run>.log), and the aggregate
#                 metadata_<design>.json written by run_sims.py --finalize.
#   - dataset/  : the finalized dataset_*.json row list.
#
# The default analysis is cheap: it reads the ~991 SLURM logs and the
# aggregate metadata file, then compares the recorded-run count against
# the expected total. That alone usually identifies the cause -- e.g. N
# array tasks crashed and their JOBS_PER_TASK chunks never wrote metadata.
#
# Pass --parse-logs to also re-run parse_results.parse_spice_log on every
# per-run log and categorise parse failures. This is slow (minutes on
# network storage) but tells you whether surviving simulations were dropped
# at the parse stage rather than at the SLURM stage.
#
# Usage:
#   python debug_dataset.py --design 2inv --dataset dataset_22nm_LP --expected 100000
#   python debug_dataset.py --design 2inv --dataset dataset_22nm_LP --parse-logs
# =============================================================================

import argparse
import json
import os
import re
import sys
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, THIS_DIR)

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SIMS_DIR = os.path.join(PROJECT_ROOT, "sims")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

FAIL_MARKERS = (
    "Traceback", "error:", "Error:", "Killed", "CANCELLED",
    "DUE TO TIME LIMIT", "oom-kill", "Segmentation fault",
)


def _log(msg):
    print(msg, flush=True)


def _section(title):
    _log("")
    _log(title)
    _log("-" * len(title))


# ---------------------------------------------------------------------------
# SLURM task log scan
# ---------------------------------------------------------------------------

def _scan_slurm_task_logs(show):
    task_re = re.compile(r"^sim-(\d+)_(\d+)\.(out|err)$")
    finalize_re = re.compile(r"^finalize-(\d+)\.(out|err)$")
    by_jid = {}
    finalize_entries = []

    # Use scandir so we get sizes without a second stat() syscall per file.
    # This matters on network-mounted filesystems where stat latency dominates.
    try:
        scan_entries = list(os.scandir(LOGS_DIR))
    except FileNotFoundError:
        return by_jid, finalize_entries

    for de in scan_entries:
        name = de.name
        m = task_re.match(name)
        if m:
            jid, task, kind = m.group(1), int(m.group(2)), m.group(3)
            info = by_jid.setdefault(
                jid,
                {"tasks": {}, "failed_tasks": [],
                 "missing_err": [],
                 "err_snippets": {}},
            )
            size = de.stat().st_size
            info["tasks"].setdefault(task, {})[kind] = (de.path, size)
            continue
        m = finalize_re.match(name)
        if m:
            finalize_entries.append(de.path)

    for jid, info in by_jid.items():
        tasks = info["tasks"]
        info["total"] = len(tasks)

        for task, kinds in sorted(tasks.items()):
            err = kinds.get("err")
            if err is None:
                info["missing_err"].append(task)
                continue

            err_path, err_size = err
            if err_size == 0:
                continue  # empty stderr -> task OK

            flagged = False
            first_err_line = None
            try:
                with open(err_path, "r", errors="replace") as f:
                    err_text = f.read(4000)
                for line in err_text.splitlines():
                    if any(mk in line for mk in FAIL_MARKERS):
                        flagged = True
                        first_err_line = line.strip()
                        break
                if not flagged and err_text.strip():
                    flagged = True
                    first_err_line = err_text.strip().splitlines()[0]
            except OSError:
                pass

            if flagged:
                info["failed_tasks"].append(task)
                if first_err_line:
                    info["err_snippets"][task] = first_err_line[:200]

    if not by_jid:
        _log("  none found")
    else:
        for jid, info in sorted(by_jid.items()):
            _log(f"  job {jid}: {info['total']} tasks, "
                 f"{len(info['failed_tasks'])} flagged, "
                 f"{len(info['missing_err'])} missing .err")
            if info["failed_tasks"]:
                sample = info["failed_tasks"][:show]
                _log(f"    first flagged tasks : {sample}")
                common = Counter(info["err_snippets"].values()).most_common(show)
                for snippet, count in common:
                    _log(f"      {count:>4}x  {snippet}")

    return by_jid, finalize_entries


def _print_finalize(paths):
    if not paths:
        _log("  none")
        return
    for path in paths:
        try:
            size = os.path.getsize(path)
            head = ""
            if size:
                with open(path, "r", errors="replace") as f:
                    head = f.read(400)
        except OSError:
            size, head = -1, ""
        first_line = head.splitlines()[0] if head else ""
        _log(f"  {os.path.basename(path)}  ({size} bytes)  {first_line}")


# ---------------------------------------------------------------------------
# Aggregate-metadata analysis (fast path)
# ---------------------------------------------------------------------------

def _load_aggregate(design):
    path = os.path.join(SIMS_DIR, f"metadata_{design}.json")
    if not os.path.exists(path):
        return path, None
    try:
        with open(path, "r") as f:
            return path, json.load(f)
    except Exception as e:
        _log(f"  [warn] could not load {path}: {e}")
        return path, None


# ---------------------------------------------------------------------------
# Optional: parse every per-run log
# ---------------------------------------------------------------------------

def _parse_all_logs(entries, show):
    from parse_results import parse_spice_log

    buckets = {
        "OK_PARSE": [],
        "MISSING_LOG": [],
        "PARSE_NO_SUCCESS": [],
        "PARSE_EXCEPTION": [],
    }
    error_messages = Counter()
    total = len(entries)

    for i, entry in enumerate(entries):
        if (i + 1) % 5000 == 0:
            _log(f"    ...parsed {i + 1}/{total}")

        run_id = entry.get("run")
        log_path = entry.get("output", "")
        if log_path and not os.path.isabs(log_path):
            log_path = os.path.join(PROJECT_ROOT, log_path)

        if not log_path or not os.path.exists(log_path):
            buckets["MISSING_LOG"].append(run_id)
            continue

        try:
            parsed = parse_spice_log(log_path)
        except Exception as e:
            buckets["PARSE_EXCEPTION"].append((run_id, str(e)))
            error_messages[f"exception: {type(e).__name__}"] += 1
            continue

        if parsed.get("simulation_success"):
            buckets["OK_PARSE"].append(run_id)
        else:
            msg = parsed.get("error_message") or "simulation_success=False"
            buckets["PARSE_NO_SUCCESS"].append((run_id, msg))
            error_messages[msg] += 1

    _log(f"  analyzed      : {sum(len(v) for v in buckets.values())}")
    for key in ("OK_PARSE", "MISSING_LOG", "PARSE_NO_SUCCESS", "PARSE_EXCEPTION"):
        vals = buckets[key]
        _log(f"  {key:<18}: {len(vals)}")
        if vals and key != "OK_PARSE":
            sample = [v if isinstance(v, int) else v[0] for v in vals[:show]]
            _log(f"    sample run ids  : {sample}")

    if error_messages:
        _log("  most common failure messages:")
        for msg, count in error_messages.most_common(show):
            _log(f"    {count:>5}x  {msg[:120]}")

    return buckets


# ---------------------------------------------------------------------------
# Dataset comparison
# ---------------------------------------------------------------------------

def _load_dataset(dataset_name):
    base = dataset_name if dataset_name.endswith(".json") else dataset_name + ".json"
    path = base if os.path.isabs(base) else os.path.join(DATASET_DIR, base)
    if not os.path.exists(path):
        return path, None
    try:
        with open(path, "r") as f:
            return path, json.load(f)
    except Exception as e:
        _log(f"  [warn] could not load {path}: {e}")
        return path, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose missing dataset rows after a finished run."
    )
    parser.add_argument("--design", default="2inv",
                        help="Design name in metadata_<design>*.json")
    parser.add_argument("--dataset", default=None,
                        help="Dataset base name (e.g. dataset_22nm_LP) to "
                             "compare row count against.")
    parser.add_argument("--expected", type=int, default=None,
                        help="Expected total number of simulations.")
    parser.add_argument("--show", type=int, default=5,
                        help="Per-bucket sample size to print.")
    parser.add_argument("--parse-logs", action="store_true",
                        help="Re-run parse_spice_log on every per-run log "
                             "(slow).")
    args = parser.parse_args()

    _section("Workspace paths")
    _log(f"  project root : {PROJECT_ROOT}")
    _log(f"  logs dir     : {LOGS_DIR}")
    _log(f"  sims dir     : {SIMS_DIR}")
    _log(f"  dataset dir  : {DATASET_DIR}")

    _section("SLURM task logs (logs/sim-*)")
    task_info, finalize_paths = _scan_slurm_task_logs(args.show)

    _section("Finalize logs (logs/finalize-*)")
    _print_finalize(finalize_paths)

    _section(f"Aggregate metadata (sims/metadata_{args.design}.json)")
    agg_path, agg = _load_aggregate(args.design)
    if agg is None:
        _log(f"  NOT FOUND at {agg_path}")
        recorded = 0
    else:
        recorded = len(agg)
        _log(f"  path         : {agg_path}")
        _log(f"  recorded runs: {recorded}")
        run_ids = [e.get("run") for e in agg if "run" in e]
        if run_ids:
            _log(f"  run id range : {min(run_ids)} .. {max(run_ids)}")
            span = max(run_ids) + 1
            gaps = sorted(set(range(span)) - set(run_ids))
            _log(f"  gaps in ids  : {len(gaps)}")
            if gaps:
                _log(f"    first gaps : {gaps[:args.show]}")

    dataset_rows = None
    if args.dataset:
        _section(f"Dataset (dataset/{args.dataset})")
        ds_path, rows = _load_dataset(args.dataset)
        if rows is None:
            _log(f"  NOT FOUND at {ds_path}")
        else:
            dataset_rows = len(rows)
            _log(f"  path         : {ds_path}")
            _log(f"  rows         : {dataset_rows}")
            if rows:
                ids = [r.get("ID") for r in rows if "ID" in r]
                _log(f"  unique IDs   : {len(set(ids))}")
            if args.expected is not None:
                _log(f"  expected     : {args.expected}")
                _log(f"  shortfall    : {args.expected - dataset_rows}")

    parse_buckets = None
    if args.parse_logs and agg:
        _section("Parsing every per-run log (slow)")
        parse_buckets = _parse_all_logs(agg, args.show)

    _section("Diagnosis")
    total_tasks = sum(info["total"] for info in task_info.values())
    failed_tasks = sum(len(info["failed_tasks"]) for info in task_info.values())
    _log(f"  SLURM tasks total   : {total_tasks}")
    _log(f"  SLURM tasks flagged : {failed_tasks}")
    _log(f"  recorded runs       : {recorded}")
    if dataset_rows is not None:
        _log(f"  dataset rows        : {dataset_rows}")
    if args.expected is not None:
        miss_sim = args.expected - recorded
        _log(f"  expected            : {args.expected}")
        _log(f"  simulations missed  : {miss_sim}")
        if dataset_rows is not None:
            dropped_at_parse = recorded - dataset_rows
            _log(f"  dropped at parse    : {dropped_at_parse}")
        if failed_tasks and miss_sim > 0 and total_tasks:
            avg_chunk = args.expected / total_tasks
            _log(
                f"\n  With {failed_tasks} flagged SLURM tasks and an average "
                f"chunk of ~{avg_chunk:.1f} sims/task, expect up to "
                f"~{int(failed_tasks * avg_chunk)} missed simulations -- "
                f"compare with the gap of {miss_sim}."
            )

    if parse_buckets is not None:
        no_success = len(parse_buckets["PARSE_NO_SUCCESS"])
        missing_log = len(parse_buckets["MISSING_LOG"])
        exc = len(parse_buckets["PARSE_EXCEPTION"])
        _log(
            f"  parse outcomes: ok={len(parse_buckets['OK_PARSE'])}, "
            f"missing_log={missing_log}, no_success={no_success}, exc={exc}"
        )


if __name__ == "__main__":
    main()
