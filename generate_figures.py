"""
Publication-Quality Figure Generation Engine for CMPSC 448 Midterm Report
Generates high-resolution (300 DPI) figures matching IEEE/ACM standards.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set clean aesthetic typography
mpl.rcParams["font.sans-serif"] = "DejaVu Sans"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["grid.linestyle"] = "--"

FIGURES_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_rq1():
    rq1_path = os.path.join(RESULTS_DIR, "rq1_results.json")
    if not os.path.exists(rq1_path):
        print(f"Skipping RQ1 plots: {rq1_path} not found")
        return
        
    with open(rq1_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    classes = data["classes"]
    
    # 1. Loss & Accuracy Convergence Curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Loss curves
    cnn_hist = data["cnn"]["history"]
    rnn_hist = data["rnn"]["history"]
    epochs_cnn = range(1, len(cnn_hist["train_loss"]) + 1)
    epochs_rnn = range(1, len(rnn_hist["train_loss"]) + 1)
    
    axes[0].plot(epochs_cnn, cnn_hist["train_loss"], "b--", label="CNN Train Loss", lw=1.8)
    axes[0].plot(epochs_cnn, cnn_hist["val_loss"], "b-", label="CNN Val Loss", lw=2.2)
    axes[0].plot(epochs_rnn, rnn_hist["train_loss"], "r--", label="BiLSTM Train Loss", lw=1.8)
    axes[0].plot(epochs_rnn, rnn_hist["val_loss"], "r-", label="BiLSTM Val Loss", lw=2.2)
    axes[0].set_title("(a) Training & Validation Cross-Entropy Loss", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(fontsize=9, frameon=True)
    
    # Accuracy curves
    axes[1].plot(epochs_cnn, cnn_hist["train_acc"], "b--", label="CNN Train Acc", lw=1.8)
    axes[1].plot(epochs_cnn, cnn_hist["val_acc"], "b-", label="CNN Val Acc", lw=2.2)
    axes[1].plot(epochs_rnn, rnn_hist["train_acc"], "r--", label="BiLSTM Train Acc", lw=1.8)
    axes[1].plot(epochs_rnn, rnn_hist["val_acc"], "r-", label="BiLSTM Val Acc", lw=2.2)
    axes[1].set_title("(b) Classification Accuracy Progression", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend(fontsize=9, frameon=True)
    
    plt.tight_layout()
    loss_path = os.path.join(FIGURES_DIR, "rq1_loss_curves.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved: {loss_path}")
    
    # 2. Confusion Matrices (CNN vs BiLSTM)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    def plot_cm(ax, cm_raw, title):
        cm = np.array(cm_raw, dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums!=0)
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_ylabel("True LLM Family", fontsize=10)
        ax.set_xlabel("Predicted LLM Family", fontsize=10)
        
        # Numbers inside cells
        thresh = cm_norm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})"
                ax.text(j, i, val, ha="center", va="center", fontsize=7.5,
                        color="white" if cm_norm[i, j] > thresh else "black")
        return im

    im1 = plot_cm(axes[0], data["cnn"]["metrics"]["confusion_matrix"], "1D-CNN Normalized Confusion Matrix")
    im2 = plot_cm(axes[1], data["rnn"]["metrics"]["confusion_matrix"], "BiLSTM Normalized Confusion Matrix")
    
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "rq1_confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path}")


def plot_rq2():
    rq2_path = os.path.join(RESULTS_DIR, "rq2_results.json")
    if not os.path.exists(rq2_path):
        return
    with open(rq2_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    bench = data.get("benchmark", {})
    arena = data.get("arena", {})
    
    modes = ["input", "output", "both"]
    labels = ["Input Only\n(Prompt)", "Output Only\n(Response)", "Input + Output\n(Synergistic)"]
    
    bench_accs = [bench[m]["cnn_accuracy"] * 100 for m in modes if m in bench]
    arena_accs = [arena[m]["cnn_accuracy"] * 100 for m in modes if m in arena]
    
    x = np.arange(len(modes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - width/2, bench_accs, width, label="Controlled Benchmark (6 Models)", color="#2b5c8f")
    if arena_accs:
        rects2 = ax.bar(x + width/2, arena_accs, width, label="Wild Arena Logs (4 Models)", color="#d95f02")
        
    # Chance levels
    ax.axhline(100.0 / 6.0, color="#2b5c8f", linestyle=":", alpha=0.7, label="Controlled Random Chance (16.7%)")
    if arena_accs:
        ax.axhline(100.0 / 4.0, color="#d95f02", linestyle=":", alpha=0.7, label="Arena Random Chance (25.0%)")
        
    ax.set_ylabel("CNN Accuracy (%)", fontsize=10)
    ax.set_title("RQ2: Impact of Prompt Informativeness across Regimes", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=True)
    ax.set_ylim(0, 100)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    if arena_accs:
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
                        
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "rq2_prompt_synergy.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_rq3():
    rq3_path = os.path.join(RESULTS_DIR, "rq3_results.json")
    if not os.path.exists(rq3_path):
        return
    with open(rq3_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    t2c = data["technical_to_creative"]
    c2t = data["creative_to_technical"]
    
    pairs = ["Trained: Technical\nTested: In-Domain", "Trained: Technical\nTested: Creative (OOD)",
             "Trained: Creative\nTested: In-Domain", "Trained: Creative\nTested: Technical (OOD)"]
    accs = [t2c["in_domain_accuracy"] * 100, t2c["ood_accuracy"] * 100,
            c2t["in_domain_accuracy"] * 100, c2t["ood_accuracy"] * 100]
    colors = ["#2b5c8f", "#75aadb", "#2ca02c", "#98df8a"]
    
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(pairs, accs, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("RQ3: Cross-Domain Generalization & Transfer Degradation", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    # Annotate drop
    ax.annotate(f"Transfer Gap: -{t2c['generalization_gap']*100:.1f}%",
                xy=(1, t2c["ood_accuracy"] * 100), xytext=(0.5, 75),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=9, color="darkred", fontweight="bold")
                
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "rq3_domain_transfer.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_rq4():
    rq4_path = os.path.join(RESULTS_DIR, "rq4_results.json")
    if not os.path.exists(rq4_path):
        return
    with open(rq4_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    styl = data.get("stylometric_summary", {})
    abl = data.get("ablation_results", {})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 1. Stylometric Profiling (Words & Burstiness)
    families = list(styl.keys())
    burstiness = [styl[f]["mean_burstiness"] for f in families]
    ttr = [styl[f]["mean_ttr"] * 100 for f in families]
    
    x = np.arange(len(families))
    width = 0.35
    axes[0].bar(x - width/2, burstiness, width, label="Sentence Burstiness (std dev)", color="#1f77b4")
    axes[0].bar(x + width/2, ttr, width, label="Lexical TTR (%)", color="#ff7f0e")
    axes[0].set_title("(a) Stylometrics: Burstiness & Lexical Richness", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(families, rotation=30, ha="right", fontsize=9)
    axes[0].legend(fontsize=8.5, frameon=True)
    
    # 2. Ablation Degradation Waterfall
    abl_names = ["Raw Baseline", "Strip Structure\n(Markdown/Bullets)", "Truncate Length\n(150 Words)", "Strip Punctuation\n(Case Folded)"]
    abl_accs = [
        abl["raw_text"]["accuracy"] * 100,
        abl["structure_stripped"]["accuracy"] * 100,
        abl["length_truncated_150w"]["accuracy"] * 100,
        abl["punctuation_stripped"]["accuracy"] * 100
    ]
    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bars = axes[1].bar(abl_names, abl_accs, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    axes[1].set_ylabel("Classification Accuracy (%)", fontsize=10)
    axes[1].set_title("(b) Ablation Analysis: Performance Degradation", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 100)
    
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                         
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "rq4_stylometrics_ablation.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_all_plots():
    print("\n--- Generating All Research Publication Figures ---")
    plot_rq1()
    plot_rq2()
    plot_rq3()
    plot_rq4()
    print("[SUCCESS] All available figures generated.")


if __name__ == "__main__":
    generate_all_plots()
