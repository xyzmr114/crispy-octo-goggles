"""
Master One-Click Execution Pipeline for CMPSC 448 Midterm Project
Executes all data processing, models, research questions, extra credits, and figure generation.
"""

import sys
import subprocess
import time


def run_step(step_name, command_args):
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    t0 = time.time()
    res = subprocess.run([sys.executable] + command_args, capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Step {step_name} failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"--> Completed {step_name} in {elapsed:.2f} seconds.")


def main():
    print("======================================================================")
    print("  CMPSC 448 MIDTERM PROJECT: WHO WROTE IT? IDENTIFYING LLM AUTHORSHIP")
    print("  Full Execution Pipeline (RQ1, RQ2, RQ3, RQ4)")
    print("======================================================================")
    
    # Step 1: Preprocess Data
    run_step("1. Data Preprocessing & Balancing", ["src/preprocess.py"])
    
    # Step 2: RQ1 Core Experiments (Baseline, CNN, BiLSTM)
    run_step("2. RQ1 Core Experiments (CNN vs BiLSTM)", ["run_rq1.py"])
    
    # Step 3: RQ2 Prompt Synergy Experiments
    run_step("3. RQ2 Prompt Synergy (Input vs Output vs Both)", ["run_rq2.py"])
    
    # Step 4: RQ3 Cross-Domain Generalization (Extra Credit +5)
    run_step("4. RQ3 Domain Transfer & Generalization", ["run_rq3.py"])
    
    # Step 5: RQ4 Stylometrics & Feature Ablations (Extra Credit +5)
    run_step("5. RQ4 Stylometric Profiling & Ablations", ["run_rq4.py"])
    
    # Step 6: Generate High-Resolution Publication Figures
    run_step("6. High-Resolution Figure Generation", ["generate_figures.py"])
    
    print("\n======================================================================")
    print("  ALL EXPERIMENTAL PIPELINES COMPLETED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    main()
