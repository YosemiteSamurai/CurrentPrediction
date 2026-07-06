"""
privacy_attack.py
Runs all three privacy attacks (embedding inversion, edge reconstruction, membership inference)
on the specified process. Attack artifacts live in --privacy_dir (attacks/inputs/); the general
inference outputs live in --results_dir (results/).
"""
import os
import subprocess
import argparse
import sys

ATTACKS = [
    ("embedding_inversion.py", "Embedding Inversion"),
    ("edge_reconstruction.py", "Edge Reconstruction"),
    ("membership_inference.py", "Membership Inference"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', required=True, help='Process name (e.g., 22nm_LP)')
    parser.add_argument('--run-tag', default='baseline', help='Run tag (e.g., baseline, dp, sl, both)')
    parser.add_argument('--privacy_dir', default='.', help='Directory with attack .npz artifacts')  # original: parser.add_argument('--privacy_dir', default='.', help='Directory with .npz outputs and scripts')
    parser.add_argument('--results_dir', default=None, help='Directory with general inference outputs (defaults to --privacy_dir)')
    args = parser.parse_args()

    run_tag = args.run_tag.strip().strip('_') or 'baseline'
    process_with_tag = f"{args.process}_{run_tag}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or args.privacy_dir

    for script, desc in ATTACKS:
        print(f"\n=== Running {desc} ===")
        cmd = [sys.executable, os.path.join(script_dir, script),  # original: cmd = [sys.executable, script,
               '--process', process_with_tag,
               '--privacy_dir', args.privacy_dir,
               '--results_dir', results_dir]
        result = subprocess.run(cmd)  # original: result = subprocess.run(cmd, cwd=args.privacy_dir)
        if result.returncode != 0:
            print(f"[ERROR] {desc} failed with exit code {result.returncode}")
        else:
            print(f"[OK] {desc} completed.")

if __name__ == '__main__':
    main()
