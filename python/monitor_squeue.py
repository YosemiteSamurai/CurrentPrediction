import subprocess
import time
import smtplib
from email.mime.text import MIMEText
import argparse
import os
from datetime import datetime

USER = "jonesm25"
EMAIL = "jonesm25@oregonstate.edu"
CHECK_INTERVAL = 3600  # 1 hour in seconds
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))


def _normalize_run_tag(run_tag):
    value = (run_tag or '').strip()
    return value if value else 'baseline'


def rename_job_out_log(job_id, model, run_tag):
    """Rename logs/slurm-<JOBID>.out to slurm-<model>_<run_tag>.out."""
    if not job_id:
        return None

    model_name = (model or '').strip()
    if not model_name:
        print("[monitor] --model not provided; skipping output log rename")
        return None

    normalized_tag = _normalize_run_tag(run_tag)
    src = os.path.join(LOGS_DIR, f"slurm-{job_id}.out")
    dst = os.path.join(LOGS_DIR, f"slurm-{model_name}_{normalized_tag}.out")

    if not os.path.exists(src):
        print(f"[monitor] source log not found; skipping rename: {src}")
        return None

    if os.path.exists(dst):
        os.remove(dst)
    os.replace(src, dst)
    print(f"[monitor] renamed {src} -> {dst}")
    return dst


def rename_job_err_log(job_id, model, run_tag):
    """Rename logs/slurm-<JOBID>.err to slurm-<model>_<run_tag>.err."""
    if not job_id:
        return None

    model_name = (model or '').strip()
    if not model_name:
        print("[monitor] --model not provided; skipping error log copy")
        return None

    normalized_tag = _normalize_run_tag(run_tag)
    src = os.path.join(LOGS_DIR, f"slurm-{job_id}.err")
    dst = os.path.join(LOGS_DIR, f"slurm-{model_name}_{normalized_tag}.err")

    if not os.path.exists(src):
        print(f"[monitor] source error log not found; skipping rename: {src}")
        return None

    if os.path.exists(dst):
        os.remove(dst)
    os.replace(src, dst)
    print(f"[monitor] renamed {src} -> {dst}")
    return dst

def filter_squeue_output(squeue_output):
    """Remove jobs in CG (Completing) state from squeue text output."""
    lines = squeue_output.strip().splitlines()
    if not lines:
        return ""

    header = lines[0].split()
    if "ST" not in header:
        return squeue_output.strip()

    st_idx = header.index("ST")
    kept = [lines[0]]
    for line in lines[1:]:
        parts = line.split(None, len(header) - 1)
        if len(parts) <= st_idx:
            continue
        if parts[st_idx] == "CG":
            continue
        kept.append(line)

    return "\n".join(kept).strip()


def get_squeue_output(job_id=None):
    cmd = ["squeue", "-u", USER]
    if job_id:
        cmd.extend(["-j", str(job_id)])
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return filter_squeue_output(result.stdout)

def format_pretty_squeue(squeue_output):
    lines = squeue_output.strip().splitlines()
    if not lines or len(lines) < 2:
        return "No jobs found."
    header = lines[0].split()
    jobs = lines[1:]
    blocks = []
    for idx, line in enumerate(jobs):
        # Handle lines that may have spaces in the last column (NODELIST(REASON))
        parts = line.split(None, len(header) - 1)
        job_info = dict(zip(header, parts))
        block_lines = []
        # Print in requested format
        for key in header:
            block_lines.append(f"{key}: {job_info.get(key, '')}")
        blocks.append("\n".join(block_lines))
        if idx != len(jobs) - 1:
            blocks.append("")
    return "\n".join(blocks)

def extract_job_ids(squeue_output):
    lines = squeue_output.strip().splitlines()
    if not lines or len(lines) < 2:
        return []

    header = lines[0].split()
    if "JOBID" not in header:
        return []

    jobid_idx = header.index("JOBID")
    job_ids = []
    for line in lines[1:]:
        parts = line.split(None, len(header) - 1)
        if len(parts) <= jobid_idx:
            continue
        job_ids.append(parts[jobid_idx])
    return job_ids

def get_log_file_for_job(job_id, suffix):
    path = os.path.join(LOGS_DIR, f"slurm-{job_id}.{suffix}")
    if os.path.exists(path):
        return path
    return None

def get_tail_nonblank_lines(file_path, limit=10):
    if not file_path:
        return []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.rstrip() for line in f if line.strip()]
        return lines[-limit:]
    except Exception as e:
        return [f"[ERROR] Could not read file: {e}"]

def format_log_tail_section(file_path, empty_label):
    if not file_path:
        return f"{empty_label}:\n<file not found>"
    name = os.path.basename(file_path)
    tail_lines = get_tail_nonblank_lines(file_path, limit=10)
    if not tail_lines:
        return f"{name}:\n(no non-blank lines)"
    return f"{name}:\n" + "\n".join(tail_lines)

def build_email_body(squeue_output, fallback_job_ids=None):
    pretty = format_pretty_squeue(squeue_output)
    job_ids = extract_job_ids(squeue_output)
    if not job_ids:
        job_ids = fallback_job_ids or []

    log_sections = []
    if not job_ids:
        log_sections.append(format_log_tail_section(None, 'slurm-<JOBID>.out'))
        log_sections.append(format_log_tail_section(None, 'slurm-<JOBID>.err'))
    else:
        for job_id in job_ids:
            out_file = get_log_file_for_job(job_id, 'out')
            err_file = get_log_file_for_job(job_id, 'err')
            log_sections.append(format_log_tail_section(out_file, f'slurm-{job_id}.out'))
            log_sections.append(format_log_tail_section(err_file, f'slurm-{job_id}.err'))

    return f"{pretty}\n\n" + "\n\n".join(log_sections)

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = EMAIL

    # Adjust SMTP server as needed (here: localhost)
    with smtplib.SMTP("localhost") as server:
        server.sendmail(EMAIL, [EMAIL], msg.as_string())

def main():
    parser = argparse.ArgumentParser(description="Monitor squeue and email on change.")
    parser.add_argument('--test', action='store_true', help='Send current squeue output and exit')
    parser.add_argument('--monitor', action='store_true', help='After first change, keep emailing every check interval until no jobs remain, then exit')
    parser.add_argument('--job-id', default='', help='Optional specific SLURM JOBID to monitor')
    parser.add_argument('--model', default='', help='Model/process label used for completion log rename (e.g., 22nm_LP)')
    parser.add_argument('--run-tag', default='baseline', help='Run tag used for completion log rename (default: baseline)')
    args = parser.parse_args()

    target_job_id = args.job_id.strip()
    scope_label = f"job {target_job_id}" if target_job_id else f"user {USER}"

    if args.test:
        current_output = get_squeue_output(target_job_id or None)
        subject = f"SLURM squeue --test output ({scope_label})"
        body = build_email_body(current_output)
        send_email(subject, body)
        print(body)
        print(f"Finished monitor_squeue at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return

    last_output = get_squeue_output(target_job_id or None)
    last_job_ids = extract_job_ids(last_output)
    if target_job_id and target_job_id not in last_job_ids:
        last_job_ids = [target_job_id]
    pretty = format_pretty_squeue(last_output)
    print(pretty)
    subject = f"SLURM squeue initial output ({scope_label})"
    body = build_email_body(last_output)
    send_email(subject, body)

    while True:
        time.sleep(CHECK_INTERVAL)
        current_output = get_squeue_output(target_job_id or None)

        if args.monitor:
            if current_output.strip() == '' or len(current_output.strip().splitlines()) < 2:
                if target_job_id:
                    subject = f"SLURM squeue: job {target_job_id} finished"
                else:
                    subject = "SLURM squeue: all jobs finished"
                body = build_email_body(current_output, fallback_job_ids=last_job_ids)
                send_email(subject, body)
                if target_job_id:
                    rename_job_out_log(target_job_id, args.model, args.run_tag)
                    rename_job_err_log(target_job_id, args.model, args.run_tag)
                print(f"Finished monitor_squeue at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                break

            subject = f"SLURM squeue periodic update ({scope_label})"
            body = build_email_body(current_output)
            send_email(subject, body)
            last_output = current_output
            last_job_ids = extract_job_ids(current_output)
            if target_job_id and target_job_id not in last_job_ids:
                last_job_ids = [target_job_id]
            continue

        if current_output != last_output:
            subject = f"SLURM squeue change detected ({scope_label})"
            body = build_email_body(current_output)
            send_email(subject, body)
            last_output = current_output
            last_job_ids = extract_job_ids(current_output)
            if target_job_id and target_job_id not in last_job_ids:
                last_job_ids = [target_job_id]
            if target_job_id:
                rename_job_out_log(target_job_id, args.model, args.run_tag)
                rename_job_err_log(target_job_id, args.model, args.run_tag)
            print(f"Finished monitor_squeue at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            break

if __name__ == "__main__":
    main()