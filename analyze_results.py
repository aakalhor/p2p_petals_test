#!/usr/bin/env python3
"""
Phase 6: Analysis & Plot Generation

Generates all plots from experiment results:
  1. Latency distribution (histogram): baseline vs each fault type
  2. Latency over time (scatter): per request with fault event markers
  3. p50/p95/p99 comparison (bar chart)
  4. Failure rate (bar chart)
  5. Recovery time analysis

Usage:
    conda activate petals
    python analyze_results.py
"""

import json
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Ordered list of experiments with display names
EXPERIMENT_ORDER = [
    ("baseline_4srv.json", "Baseline\n(4srv)"),
    ("sigterm_4srv_60s.json", "SIGTERM\n4srv 60s"),
    ("sigkill_4srv_60s.json", "SIGKILL\n4srv 60s"),
    ("sigkill_4srv_30s.json", "SIGKILL\n4srv 30s"),
    ("sigkill_6srv_60s.json", "SIGKILL\n6srv 60s"),
    ("sigkill_6srv_repl.json", "SIGKILL\n6srv repl"),
]

COLORS = plt.cm.Set2(np.linspace(0, 1, len(EXPERIMENT_ORDER)))


def load_experiment(filepath):
    """Load experiment data from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def load_all_experiments():
    """Load all available experiment results."""
    experiments = {}
    for filename, label in EXPERIMENT_ORDER:
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            experiments[label] = load_experiment(filepath)
        else:
            print(f"  Warning: {filepath} not found, skipping.")
    return experiments


def compute_stats(data):
    """Compute summary statistics for an experiment."""
    results = data.get("results", [])
    successful = [r for r in results if r.get("success", True)]
    failed = [r for r in results if not r.get("success", True)]

    if not successful:
        return {
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "failure_rate": 1.0,
            "total_requests": len(results),
            "successful_requests": 0,
        }

    latencies = [r["latency_sec"] if "latency_sec" in r else r.get("elapsed_sec", 0) for r in successful]
    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "failure_rate": len(failed) / len(results) if results else 0,
        "total_requests": len(results),
        "successful_requests": len(successful),
    }


# =====================================================
# Plot 1: Latency Distribution (Histogram)
# =====================================================
def plot_latency_distribution(experiments):
    """Histogram of per-request latency for baseline vs each fault type."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    for idx, (label, data) in enumerate(experiments.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        results = data.get("results", [])
        successful = [r for r in results if r.get("success", True)]
        latencies = [r.get("latency_sec", r.get("elapsed_sec", 0)) for r in successful]

        if latencies:
            ax.hist(latencies, bins=30, color=COLORS[idx], alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.axvline(np.median(latencies), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(latencies):.2f}s")
            ax.legend(fontsize=8)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Latency (s)")
        if idx % 3 == 0:
            ax.set_ylabel("Count")

    # Hide unused axes
    for idx in range(len(experiments), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Latency Distribution by Experiment", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_latency_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_latency_distribution.png")


# =====================================================
# Plot 2: Latency Over Time (Scatter)
# =====================================================
def plot_latency_over_time(experiments):
    """Scatter plot of latency per request with fault injection markers."""
    n = len(experiments)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for idx, (label, data) in enumerate(experiments.items()):
        ax = axes[idx]
        results = data.get("results", [])
        faults = data.get("fault_events", [])

        times = [r.get("time", i) for i, r in enumerate(results)]
        latencies = [r.get("latency_sec", r.get("elapsed_sec", 0)) for r in results]
        successes = [r.get("success", True) for r in results]

        # Plot successful and failed requests
        success_times = [t for t, s in zip(times, successes) if s]
        success_lats = [l for l, s in zip(latencies, successes) if s]
        fail_times = [t for t, s in zip(times, successes) if not s]
        fail_lats = [l for l, s in zip(latencies, successes) if not s]

        ax.scatter(success_times, success_lats, c="green", alpha=0.5, s=15, label="Success")
        if fail_times:
            ax.scatter(fail_times, fail_lats, c="red", alpha=0.7, s=25, marker="x", label="Failure")

        # Mark fault events
        for f in faults:
            ax.axvline(x=f["time"], color="red", linestyle="--", alpha=0.4, linewidth=1)

        ax.set_ylabel("Latency (s)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Latency Over Time (with Fault Injection Markers)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_latency_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_latency_over_time.png")


# =====================================================
# Plot 3: p50/p95/p99 Comparison (Grouped Bar Chart)
# =====================================================
def plot_percentile_comparison(experiments):
    """Bar chart comparing p50, p95, p99 latency across all experiments."""
    labels = list(experiments.keys())
    stats = [compute_stats(data) for data in experiments.values()]

    p50s = [s["p50"] for s in stats]
    p95s = [s["p95"] for s in stats]
    p99s = [s["p99"] for s in stats]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, p50s, width, label="p50", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x, p95s, width, label="p95", color="#FF9800", alpha=0.8)
    bars3 = ax.bar(x + width, p99s, width, label="p99", color="#F44336", alpha=0.8)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Percentiles: p50 / p95 / p99", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_percentile_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_percentile_comparison.png")


# =====================================================
# Plot 4: Failure Rate (Bar Chart)
# =====================================================
def plot_failure_rate(experiments):
    """Bar chart of % failed requests per experiment."""
    labels = list(experiments.keys())
    stats = [compute_stats(data) for data in experiments.values()]
    failure_rates = [s["failure_rate"] * 100 for s in stats]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), failure_rates, color=COLORS[:len(labels)], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("Request Failure Rate by Experiment", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(max(failure_rates) * 1.2, 5))

    # Add value labels
    for bar, rate in zip(bars, failure_rates):
        ax.annotate(f"{rate:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_failure_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_failure_rate.png")


# =====================================================
# Plot 5: Recovery Time Analysis
# =====================================================
def plot_recovery_time(experiments):
    """
    Recovery time: time from fault injection to first successful request
    that is within 2x the baseline median latency.
    """
    # First, compute baseline median
    baseline_data = None
    for label, data in experiments.items():
        if "baseline" in label.lower():
            baseline_data = data
            break

    if baseline_data is None:
        print("  ⚠ No baseline data found, skipping recovery time plot.")
        return

    baseline_stats = compute_stats(baseline_data)
    baseline_median = baseline_stats["p50"]
    threshold = baseline_median * 2  # "recovered" = within 2x baseline median

    recovery_times = {}
    for label, data in experiments.items():
        if "baseline" in label.lower():
            continue

        faults = data.get("fault_events", [])
        results = data.get("results", [])

        if not faults:
            continue

        fault_recovery = []
        for fault in faults:
            fault_time = fault["time"]
            # Find first successful request after fault that's below threshold
            recovered = False
            for r in results:
                req_time = r.get("time", 0)
                if req_time > fault_time and r.get("success", True):
                    latency = r.get("latency_sec", r.get("elapsed_sec", 0))
                    if latency <= threshold:
                        recovery = req_time - fault_time
                        fault_recovery.append(recovery)
                        recovered = True
                        break
            if not recovered:
                fault_recovery.append(float("nan"))

        if fault_recovery:
            valid = [r for r in fault_recovery if not np.isnan(r)]
            recovery_times[label] = {
                "mean": np.mean(valid) if valid else float("nan"),
                "values": fault_recovery,
                "num_recovered": len(valid),
                "num_faults": len(fault_recovery),
            }

    if not recovery_times:
        print("  ⚠ No recovery time data, skipping plot.")
        return

    labels = list(recovery_times.keys())
    means = [recovery_times[l]["mean"] for l in labels]
    recovery_pct = [recovery_times[l]["num_recovered"] / recovery_times[l]["num_faults"] * 100
                    for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Mean recovery time
    bars1 = ax1.bar(range(len(labels)), means, color=COLORS[1:len(labels)+1],
                    edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("Mean Recovery Time (s)")
    ax1.set_title("Mean Recovery Time After Fault", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    for bar, val in zip(bars1, means):
        if not np.isnan(val):
            ax1.annotate(f"{val:.1f}s", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # Recovery percentage
    bars2 = ax2.bar(range(len(labels)), recovery_pct, color=COLORS[1:len(labels)+1],
                    edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Experiment")
    ax2.set_ylabel("Recovery Rate (%)")
    ax2.set_title("% of Faults Recovered (within 2× baseline latency)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 110)
    for bar, val in zip(bars2, recovery_pct):
        ax2.annotate(f"{val:.0f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_recovery_time.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_recovery_time.png")


# =====================================================
# Summary Table
# =====================================================
def print_summary_table(experiments):
    """Print a formatted summary table of all experiments."""
    print(f"\n{'='*90}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*90}")
    header = f"{'Experiment':<20} {'Requests':>8} {'Success':>8} {'Fail%':>6} {'p50(s)':>8} {'p95(s)':>8} {'p99(s)':>8} {'Mean(s)':>8}"
    print(header)
    print("-" * 90)

    for label, data in experiments.items():
        stats = compute_stats(data)
        total = stats["total_requests"]
        success = stats["successful_requests"]
        fail_pct = stats["failure_rate"] * 100
        print(f"{label:<20} {total:>8} {success:>8} {fail_pct:>5.1f}% {stats['p50']:>8.3f} "
              f"{stats['p95']:>8.3f} {stats['p99']:>8.3f} {stats['mean']:>8.3f}")

    print(f"{'='*90}\n")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading experiment results...")
    experiments = load_all_experiments()

    if not experiments:
        print(f"ERROR: No experiment results found in {RESULTS_DIR}/")
        print("Run 'python run_experiments.py' first.")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiment(s).\n")

    print_summary_table(experiments)

    print("Generating plots...")
    plot_latency_distribution(experiments)
    plot_latency_over_time(experiments)
    plot_percentile_comparison(experiments)
    plot_failure_rate(experiments)
    plot_recovery_time(experiments)

    print(f"\nAll plots saved to {PLOTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
