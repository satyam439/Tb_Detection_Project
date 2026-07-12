import os, sys, shutil, csv, time, random, argparse
import numpy as np

# ── CLI ARGS ──────────────────────────────────────────────────────────────────
# Lets you run a bigger, randomized test without editing the file each time:
#   python3 batch_test.py                      -> default larger sample (50/50)
#   python3 batch_test.py --n_tb 100 --n_normal 100
#   python3 batch_test.py --all                -> use every available image
#   python3 batch_test.py --seed 7              -> change the random sample

parser = argparse.ArgumentParser(description="TB batch consistency test")
parser.add_argument("--n_tb", type=int, default=50, help="Number of TB images to test")
parser.add_argument("--n_normal", type=int, default=50, help="Number of normal images to test")
parser.add_argument("--all", action="store_true", help="Use every available image instead of a sample")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
args = parser.parse_args()

random.seed(args.seed)

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TB_DIR      = os.path.join(BASE_DIR, "dataset", "val", "tb")
NORMAL_DIR  = os.path.join(BASE_DIR, "dataset", "val", "normal")
OUTPUT_ROOT = os.path.join(BASE_DIR, "batch_results")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

N_TB     = args.n_tb
N_NORMAL = args.n_normal

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "tb"),     exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "normal"), exist_ok=True)

# ── COLLECT IMAGES ────────────────────────────────────────────────────────────

all_tb_images     = sorted([f for f in os.listdir(TB_DIR)     if f.endswith('.png')])
all_normal_images = sorted([f for f in os.listdir(NORMAL_DIR) if f.endswith('.png')])

if args.all:
    tb_images     = all_tb_images
    normal_images = all_normal_images
else:
    # Random (but reproducible, via --seed) sample instead of just the first
    # N alphabetically — avoids any bias from filename/patient-ID ordering.
    tb_images     = random.sample(all_tb_images,     min(N_TB, len(all_tb_images)))
    normal_images = random.sample(all_normal_images, min(N_NORMAL, len(all_normal_images)))

print(f"\n{'='*65}")
print(f"  AI TB ANALYSIS — BATCH CONSISTENCY TEST")
print(f"{'='*65}")
print(f"  TB images available    : {len(all_tb_images)}   (testing {len(tb_images)})")
print(f"  Normal images available: {len(all_normal_images)}   (testing {len(normal_images)})")
print(f"  Total tested this run  : {len(tb_images)+len(normal_images)} scans")
print(f"  Random seed            : {args.seed}")
print(f"  Output dir              : {OUTPUT_ROOT}")
print(f"{'='*65}\n")

est_seconds = (len(tb_images) + len(normal_images)) * 3.1
print(f"[INFO] Estimated run time: ~{est_seconds/60:.1f} minutes at ~3.1s/scan\n")

# ── RESULT COLLECTOR ──────────────────────────────────────────────────────────

results = []   # list of dicts, one per image

def read_result_from_outputs():
    """
    Read prediction results directly from the .npy files that predict_tb.py
    saves to outputs/ — this is more reliable than parsing stdout.
    Returns a dict with all fields, or None if files are missing.
    """
    tb_center_path = os.path.join(OUTPUTS_DIR, "tb_center.npy")
    gradcam_path   = os.path.join(OUTPUTS_DIR, "gradcam.npy")

    result = {
        "prediction":   "UNKNOWN",
        "confidence_tb": 0.0,
        "confidence_hl": 0.0,
        "lung":         "N/A",
        "lobe":         "N/A",
        "hotspot_px":   "N/A",
        "depth_mm":     "N/A",
        "heatmap_peak": 0.0,
    }

    if os.path.exists(tb_center_path):
        try:
            arr = np.load(tb_center_path)
            cx, cy = int(arr[0]), int(arr[1])
            result["hotspot_px"] = f"({cx},{cy})"
            result["prediction"] = "TB"
            result["lung"]       = "RIGHT" if cx < 112 else "LEFT"

            # Estimate lobe from cy
            if cy < 75:
                lobe_suffix = "UPPER LOBE"
            elif cy < 150:
                lobe_suffix = "MIDDLE/UPPER LOBE"
            else:
                lobe_suffix = "LOWER LOBE"
            result["lobe"] = f"{result['lung']} {lobe_suffix}"

        except Exception as e:
            print(f"  [WARN] Could not read tb_center.npy: {e}")
    else:
        result["prediction"] = "HEALTHY"

    if os.path.exists(gradcam_path):
        try:
            cam = np.load(gradcam_path).astype(np.float32)
            mn, mx = cam.min(), cam.max()
            cam_norm = (cam - mn) / (mx - mn + 1e-9)
            result["heatmap_peak"] = float(cam_norm.max())
            result["confidence_tb"] = float(cam_norm.max()) * 100.0
            result["confidence_hl"] = 100.0 - result["confidence_tb"]
        except Exception as e:
            print(f"  [WARN] Could not read gradcam.npy: {e}")

    return result


# ── RUN PIPELINE ON EACH IMAGE ────────────────────────────────────────────────

def run_one(img_path, true_label, save_subdir, idx, total):
    """
    Run predict_tb.py on one image, collect results, copy output files.
    true_label: 'TB' or 'NORMAL'
    """
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]

    print(f"  [{idx}/{total}] [{true_label}] {img_name} ...", end="", flush=True)
    t0 = time.time()

    # Run the full pipeline (no 3D viewer — headless)
    exit_code = os.system(
        f'python3 "{os.path.join(BASE_DIR, "predict_tb.py")}" "{img_path}" '
        f'2>/dev/null'
    )

    elapsed = time.time() - t0

    # Read results from saved .npy files
    r = read_result_from_outputs()
    r["image"]      = img_name
    r["true_label"] = true_label
    r["elapsed_s"]  = round(elapsed, 1)

    # Correct flag
    pred = r["prediction"]
    if true_label == "TB"     and pred == "TB":      r["correct"] = True
    elif true_label == "NORMAL" and pred == "HEALTHY": r["correct"] = True
    else:                                               r["correct"] = False

    print(f"  → {pred}  {'✅' if r['correct'] else '❌'}  ({elapsed:.1f}s)")

    # Copy output files into batch_results/
    save_dir = os.path.join(OUTPUT_ROOT, save_subdir)

    for src_name, dst_suffix in [
        ("heatmap_overlay.png",          "_heatmap.png"),
        ("lung_annotated_diagram.png",   "_diagram.png"),
        ("coord_verification.png",       "_coords.png"),
    ]:
        src = os.path.join(OUTPUTS_DIR, src_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(save_dir, base_name + dst_suffix))

    return r


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

total_scans = len(tb_images) + len(normal_images)
scan_num = 0

print("── TB CASES ──────────────────────────────────────────────────────")
for img_name in tb_images:
    scan_num += 1
    r = run_one(os.path.join(TB_DIR, img_name), "TB", "tb", scan_num, total_scans)
    results.append(r)

print("\n── NORMAL CASES ──────────────────────────────────────────────────")
for img_name in normal_images:
    scan_num += 1
    r = run_one(os.path.join(NORMAL_DIR, img_name), "NORMAL", "normal", scan_num, total_scans)
    results.append(r)

# ── COMPUTE METRICS ───────────────────────────────────────────────────────────

total   = len(results)
correct = sum(1 for r in results if r["correct"])
tb_res  = [r for r in results if r["true_label"] == "TB"]
nl_res  = [r for r in results if r["true_label"] == "NORMAL"]

tp = sum(1 for r in tb_res if r["prediction"] == "TB")
fn = sum(1 for r in tb_res if r["prediction"] != "TB")
tn = sum(1 for r in nl_res if r["prediction"] == "HEALTHY")
fp = sum(1 for r in nl_res if r["prediction"] != "HEALTHY")

sensitivity = tp / max(tp+fn, 1) * 100   # recall for TB
specificity = tn / max(tn+fp, 1) * 100
accuracy    = correct / total * 100

avg_time = np.mean([r["elapsed_s"] for r in results])

print(f"\n{'='*65}")
print(f"  BATCH TEST RESULTS SUMMARY")
print(f"{'='*65}")
print(f"  Total scans tested  : {total}")
print(f"  Correct predictions : {correct}/{total}  ({accuracy:.0f}%)")
print(f"  TB sensitivity      : {tp}/{tp+fn}  ({sensitivity:.0f}%)  — correctly identified TB")
print(f"  Specificity         : {tn}/{tn+fp}  ({specificity:.0f}%)  — correctly identified Normal")
print(f"  Avg inference time  : {avg_time:.1f} seconds per scan")
print(f"{'='*65}")
print()

# ── SAVE CSV ──────────────────────────────────────────────────────────────────

csv_path = os.path.join(OUTPUT_ROOT, "batch_results.csv")
fieldnames = ["image","true_label","prediction","correct",
              "lung","lobe","hotspot_px","confidence_tb",
              "confidence_hl","heatmap_peak","depth_mm","elapsed_s"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k,"") for k in fieldnames})

print(f"[INFO] CSV saved → {csv_path}")

# ── VISUAL SUMMARY REPORT FOR MANAGER ─────────────────────────────────────────

print("[INFO] Generating manager summary report ...")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import matplotlib.patheffects as pe

    fig = plt.figure(figsize=(22, 16), facecolor="#04060A")
    gs  = GridSpec(3, 4, figure=fig,
                   hspace=0.45, wspace=0.35,
                   left=0.05, right=0.97, top=0.92, bottom=0.05)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "AI TB ANALYSIS PLATFORM  —  Batch Consistency Test Report\n"
        f"Shenzhen Dataset (CHNCXR)  |  {len(tb_images)} TB + {len(nl_res)} Normal  "
        f"|  Model: DenseNet121 + GradCAM++",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    # ── Panel 1: Accuracy bar chart ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0A0C10")
    metrics     = ["Accuracy", "Sensitivity\n(TB Recall)", "Specificity\n(Normal)"]
    values      = [accuracy, sensitivity, specificity]
    bar_colors  = ["#4AAED8", "#FF6644", "#44CC88"]
    bars = ax1.bar(metrics, values, color=bar_colors, width=0.55, zorder=3)
    ax1.set_ylim(0, 115)
    ax1.axhline(100, color="#444", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x()+bar.get_width()/2, val+2,
                 f"{val:.0f}%", ha="center", va="bottom",
                 color="white", fontsize=13, fontweight="bold")
    ax1.set_title("Model Performance", color="white", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="white", labelsize=9)
    ax1.set_facecolor("#0A0C10")
    for sp in ax1.spines.values(): sp.set_edgecolor("#334455")
    ax1.yaxis.label.set_color("white")
    ax1.set_ylabel("Score (%)", color="#AAAAAA", fontsize=9)
    ax1.grid(axis="y", color="#1A2A3A", zorder=0)

    # ── Panel 2: Confusion matrix ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0A0C10")
    cm_data  = np.array([[tp, fn], [fp, tn]])
    cm_labels= [["TP", "FN"], ["FP", "TN"]]
    cm_colors= [[0.2, 0.7], [0.7, 0.2]]
    im = ax2.imshow(cm_colors, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm_labels[i][j]}\n{cm_data[i][j]}",
                     ha="center", va="center", fontsize=14,
                     fontweight="bold", color="white")
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(["Pred: TB","Pred: Normal"], color="white", fontsize=8)
    ax2.set_yticklabels(["True: TB","True: Normal"], color="white", fontsize=8)
    ax2.set_title("Confusion Matrix", color="white", fontsize=11, fontweight="bold")
    for sp in ax2.spines.values(): sp.set_edgecolor("#334455")

    # ── Panel 3: Pie chart correct/wrong ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#0A0C10")
    pie_vals   = [correct, total-correct]
    pie_labels = [f"Correct\n({correct})", f"Wrong\n({total-correct})"]
    pie_cols   = ["#44CC88", "#FF4444"]
    wedges, texts, autotexts = ax3.pie(
        pie_vals, labels=pie_labels, colors=pie_cols,
        autopct="%1.0f%%", startangle=90,
        textprops={"color":"white","fontsize":10},
        wedgeprops={"edgecolor":"#0A0C10","linewidth":2}
    )
    for at in autotexts: at.set_fontsize(12); at.set_fontweight("bold")
    ax3.set_title("Overall Accuracy", color="white", fontsize=11, fontweight="bold")

    # ── Panel 4: Timing bar ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor("#0A0C10")
    times = [r["elapsed_s"] for r in results]
    names = [r["image"][:14] for r in results]
    cols  = ["#FF6644" if r["true_label"]=="TB" else "#4AAED8" for r in results]
    ax4.barh(range(len(times)), times, color=cols, height=0.7, zorder=3)
    ax4.set_yticks(range(len(names))); ax4.set_yticklabels(names, fontsize=6, color="white")
    ax4.set_xlabel("Seconds", color="#AAAAAA", fontsize=8)
    ax4.set_title(f"Inference Time\n(avg {avg_time:.1f}s)", color="white", fontsize=11, fontweight="bold")
    ax4.tick_params(colors="white")
    for sp in ax4.spines.values(): sp.set_edgecolor("#334455")
    ax4.grid(axis="x", color="#1A2A3A", zorder=0)
    ax4.legend(handles=[
        mpatches.Patch(color="#FF6644", label="TB"),
        mpatches.Patch(color="#4AAED8", label="Normal")
    ], loc="lower right", fontsize=7, facecolor="#0A0C10",
       edgecolor="#334455", labelcolor="white")

    # ── Panel 5+: Per-case result table (rows 1-2 of grid) ───────────────────
    ax_tbl = fig.add_subplot(gs[1:, :])
    ax_tbl.set_facecolor("#04060A")
    ax_tbl.axis("off")

    col_headers = ["#", "Image File", "True\nLabel", "Prediction", "✓/✗",
                   "Lung", "Lobe", "Hotspot\n(px)", "Conf\nTB%", "Time\n(s)"]
    col_widths  = [0.03, 0.16, 0.07, 0.09, 0.04, 0.07, 0.16, 0.09, 0.07, 0.06]
    n_cols = len(col_headers)

    # Table header
    y = 0.97; x_starts = []
    x = 0.0
    for w in col_widths:
        x_starts.append(x); x += w

    for xi, (hdr, xs, w) in enumerate(zip(col_headers, x_starts, col_widths)):
        ax_tbl.text(xs+w/2, y, hdr, ha="center", va="top",
                    color="#AAEEFF", fontsize=9, fontweight="bold",
                    fontfamily="monospace", transform=ax_tbl.transAxes)

    # Separator line (FIX: axhline() doesn't accept a transform kwarg — it
    # always builds its own. Use plot() with explicit x/y in axes-fraction
    # coordinates instead, since ax_tbl.axis("off") means we're drawing
    # everything relative to transAxes.)
    y -= 0.045
    ax_tbl.plot([0, 1], [y, y], color="#334455", linewidth=1, transform=ax_tbl.transAxes)
    y -= 0.005

    row_h = 0.062
    # Cap how many individual rows we render in the table — with a large
    # batch (e.g. 100 images) the per-row table becomes unreadable/overflows
    # the page. Show up to MAX_TABLE_ROWS, sorted so misclassifications
    # appear first (most useful to review), and note the rest in the footer.
    MAX_TABLE_ROWS = 40
    results_for_table = sorted(results, key=lambda r: r["correct"])  # wrong first
    truncated = len(results_for_table) > MAX_TABLE_ROWS
    results_for_table = results_for_table[:MAX_TABLE_ROWS]

    for idx, r in enumerate(results_for_table):
        bg_col = "#0D1520" if idx % 2 == 0 else "#080E18"
        # Row background
        ax_tbl.add_patch(mpatches.FancyBboxPatch(
            (0, y - row_h + 0.01), 1.0, row_h - 0.005,
            boxstyle="round,pad=0.002",
            facecolor=bg_col, edgecolor="none",
            transform=ax_tbl.transAxes, zorder=0
        ))

        correct_sym = "✅" if r["correct"] else "❌"
        pred_col    = "#FF6644" if r["prediction"] == "TB" else "#44CC88"
        label_col   = "#FF9966" if r["true_label"] == "TB"  else "#66DDAA"

        row_vals = [
            str(idx+1),
            r["image"][:22],
            r["true_label"],
            r["prediction"],
            correct_sym,
            r.get("lung","N/A"),
            r.get("lobe","N/A")[:20],
            r.get("hotspot_px","N/A"),
            f"{r.get('confidence_tb',0):.0f}%",
            str(r.get("elapsed_s","?")),
        ]
        row_colors = [
            "white","#CCDDEE",label_col,pred_col,
            "#44FF88" if r["correct"] else "#FF4444",
            "#AACCFF","#AACCFF","#FFDD88","#FFAA44","#AAAAAA"
        ]

        for xi2, (val, xs, w, rc) in enumerate(zip(row_vals, x_starts, col_widths, row_colors)):
            ax_tbl.text(xs+w/2, y - row_h/2 + 0.005, val,
                        ha="center", va="center",
                        color=rc, fontsize=8.5, fontfamily="monospace",
                        transform=ax_tbl.transAxes, zorder=2)

        y -= row_h

    # Summary footer
    y -= 0.01
    ax_tbl.plot([0, 1], [y, y], color="#334455", linewidth=1, transform=ax_tbl.transAxes)
    y -= 0.04
    footer_note = (
        f" (showing {len(results_for_table)} of {total} — misclassifications listed first)"
        if truncated else ""
    )
    summary_text = (
        f"SUMMARY:  {total} scans tested{footer_note}  |  "
        f"Accuracy: {accuracy:.0f}%  |  "
        f"TB Sensitivity: {sensitivity:.0f}%  |  "
        f"Specificity: {specificity:.0f}%  |  "
        f"Avg time: {avg_time:.1f}s/scan  |  "
        f"Dataset: Shenzhen CHNCXR (Chinese Hospital dataset — comparable to Indian population TB patterns)"
    )
    ax_tbl.text(0.5, y, summary_text, ha="center", va="top",
                color="#88BBCC", fontsize=9, fontfamily="monospace",
                transform=ax_tbl.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0A1828",
                          edgecolor="#335566", linewidth=1))

    report_path = os.path.join(OUTPUT_ROOT, "batch_summary_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches="tight", facecolor="#04060A")
    plt.close()
    print(f"[INFO] ✅ Summary report saved → {report_path}")

except Exception as e:
    import traceback
    print(f"[WARN] Report generation failed: {e}")
    traceback.print_exc()

# ── FINAL CONSOLE SUMMARY ─────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  BATCH TEST COMPLETE")
print(f"{'='*65}")
print(f"  Files saved to: {OUTPUT_ROOT}/")
print(f"    batch_summary_report.png  ← show this to your manager")
print(f"    batch_results.csv         ← open in Excel for full table")
print(f"    tb/      ← heatmap + diagram for each TB case")
print(f"    normal/  ← heatmap for each normal case")
print(f"{'='*65}")
print(f"\n  RESULTS:")
print(f"    Total tested  : {total} scans")
print(f"    Correct       : {correct}/{total} ({accuracy:.0f}% accuracy)")
print(f"    TB detected   : {tp}/{len(tb_images)} TB cases correctly identified")
print(f"    Normal correct: {tn}/{len(nl_res)} normal cases correctly identified")
print(f"    Sensitivity   : {sensitivity:.0f}%")
print(f"    Specificity   : {specificity:.0f}%")
print(f"    Avg time      : {avg_time:.1f}s per scan")
print(f"{'='*65}\n")
print("Done")