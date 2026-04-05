from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "turboquant-benchmark.json"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"


def load_rows() -> list[dict]:
    data = json.loads(INPUT.read_text())
    rows = []
    for case in data["results"]:
        case_meta = case["case"]
        dynamic = case["dynamic"]
        for sweep in case["sweepResults"]:
            turbo = sweep["turboquant"]
            comparison = sweep["comparison"]
            cfg = sweep["cacheConfig"]
            rows.append(
                {
                    "case_id": case_meta["id"],
                    "case_label": case_meta["label"],
                    "case_description": case_meta["description"],
                    "max_new_tokens": case_meta["maxNewTokens"],
                    "config_id": cfg["id"],
                    "config_label": cfg["label"],
                    "b_key": cfg["b_key"],
                    "b_value": cfg["b_value"],
                    "residual_length": cfg["residual_length"],
                    "dynamic_avg_ms": dynamic["averageMs"],
                    "dynamic_ttft_ms": dynamic["averageTtftMs"],
                    "dynamic_tps": dynamic["decodeTokensPerSecond"],
                    "dynamic_packed_bytes": dynamic["cacheStats"]["packed_bytes"],
                    "dynamic_dense_bytes": dynamic["cacheStats"]["dense_bytes"],
                    "turbo_avg_ms": turbo["averageMs"],
                    "turbo_ttft_ms": turbo["averageTtftMs"],
                    "turbo_tps": turbo["decodeTokensPerSecond"],
                    "turbo_packed_bytes": turbo["cacheStats"]["packed_bytes"],
                    "turbo_dense_bytes": turbo["cacheStats"]["dense_bytes"],
                    "speed_ratio": comparison["speedRatio"],
                    "compression_ratio": comparison["compressionRatio"],
                    "prefix_agreement_ratio": comparison["prefixAgreementRatio"],
                    "prefix_agreement_pct": comparison["prefixAgreementRatio"] * 100.0,
                    "exact_match": int(comparison["exactMatch"]),
                }
            )
    return rows


def ensure_dirs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)


def write_csv(df: pd.DataFrame) -> None:
    df.to_csv(TABLE_DIR / "benchmark_rows.csv", index=False)


def write_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["config_label", "b_key", "b_value", "residual_length"], as_index=False)
        .agg(
            avg_speed_ratio=("speed_ratio", "mean"),
            avg_compression_ratio=("compression_ratio", "mean"),
            avg_prefix_agreement_pct=("prefix_agreement_pct", "mean"),
            avg_dynamic_ms=("dynamic_avg_ms", "mean"),
            avg_turbo_ms=("turbo_avg_ms", "mean"),
            avg_dynamic_ttft_ms=("dynamic_ttft_ms", "mean"),
            avg_turbo_ttft_ms=("turbo_ttft_ms", "mean"),
            avg_dynamic_tps=("dynamic_tps", "mean"),
            avg_turbo_tps=("turbo_tps", "mean"),
            exact_match_count=("exact_match", "sum"),
            cases=("case_id", "count"),
        )
        .sort_values("avg_prefix_agreement_pct", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "config_summary.csv", index=False)

    case_summary = (
        df.groupby(["case_label", "config_label"], as_index=False)
        .agg(
            speed_ratio=("speed_ratio", "mean"),
            compression_ratio=("compression_ratio", "mean"),
            prefix_agreement_pct=("prefix_agreement_pct", "mean"),
            dynamic_ms=("dynamic_avg_ms", "mean"),
            turbo_ms=("turbo_avg_ms", "mean"),
        )
        .sort_values(["case_label", "config_label"])
    )
    case_summary.to_csv(TABLE_DIR / "case_summary.csv", index=False)


def plot_speed_vs_quality(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("config_label", as_index=False)
        .agg(
            avg_speed_ratio=("speed_ratio", "mean"),
            avg_prefix_agreement_pct=("prefix_agreement_pct", "mean"),
            avg_compression_ratio=("compression_ratio", "mean"),
        )
        .sort_values("avg_prefix_agreement_pct", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=summary,
        x="avg_speed_ratio",
        y="avg_prefix_agreement_pct",
        size="avg_compression_ratio",
        hue="config_label",
        sizes=(200, 700),
        legend=False,
    )
    for _, row in summary.iterrows():
        plt.text(row["avg_speed_ratio"] + 0.003, row["avg_prefix_agreement_pct"] + 0.1, row["config_label"], fontsize=9)
    plt.axvline(1.0, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Average speed ratio (dynamic / turboquant)")
    plt.ylabel("Average prefix agreement (%)")
    plt.title("Speed-quality frontier across TurboQuant configurations")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "speed_quality_frontier.png", dpi=200)
    plt.close()


def plot_case_heatmaps(df: pd.DataFrame) -> None:
    pivot_quality = df.pivot(index="case_label", columns="config_label", values="prefix_agreement_pct")
    pivot_speed = df.pivot(index="case_label", columns="config_label", values="speed_ratio")
    pivot_comp = df.pivot(index="case_label", columns="config_label", values="compression_ratio")

    for matrix, title, filename, fmt in [
        (pivot_quality, "Prefix agreement by benchmark case (%)", "heatmap_prefix_agreement.png", ".1f"),
        (pivot_speed, "Speed ratio by benchmark case", "heatmap_speed_ratio.png", ".3f"),
        (pivot_comp, "Compression ratio by benchmark case", "heatmap_compression_ratio.png", ".3f"),
    ]:
        plt.figure(figsize=(8, 4.2))
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="YlGnBu")
        plt.title(title)
        plt.xlabel("Configuration")
        plt.ylabel("Benchmark case")
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=200)
        plt.close()


def plot_latency_bars(df: pd.DataFrame) -> None:
    plot_df = (
        df.groupby("config_label", as_index=False)
        .agg(dynamic_ms=("dynamic_avg_ms", "mean"), turbo_ms=("turbo_avg_ms", "mean"))
        .melt(id_vars="config_label", var_name="path", value_name="ms")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="config_label", y="ms", hue="path")
    plt.ylabel("Average latency (ms)")
    plt.xlabel("Configuration")
    plt.title("Average end-to-end latency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "latency_bars.png", dpi=200)
    plt.close()


def plot_ttft_tps(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ttft = (
        df.groupby("config_label", as_index=False)
        .agg(dynamic_ttft_ms=("dynamic_ttft_ms", "mean"), turbo_ttft_ms=("turbo_ttft_ms", "mean"))
        .melt(id_vars="config_label", var_name="path", value_name="ttft_ms")
    )
    sns.barplot(data=ttft, x="config_label", y="ttft_ms", hue="path", ax=axes[0])
    axes[0].set_title("Average TTFT")
    axes[0].set_ylabel("TTFT (ms)")
    axes[0].set_xlabel("Configuration")

    tps = (
        df.groupby("config_label", as_index=False)
        .agg(dynamic_tps=("dynamic_tps", "mean"), turbo_tps=("turbo_tps", "mean"))
        .melt(id_vars="config_label", var_name="path", value_name="tokens_per_second")
    )
    sns.barplot(data=tps, x="config_label", y="tokens_per_second", hue="path", ax=axes[1])
    axes[1].set_title("Decode throughput")
    axes[1].set_ylabel("Tokens / second")
    axes[1].set_xlabel("Configuration")

    for ax in axes:
        ax.legend_.set_title("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ttft_tps_bars.png", dpi=200)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ensure_dirs()
    df = pd.DataFrame(load_rows())
    write_csv(df)
    write_summary(df)
    plot_speed_vs_quality(df)
    plot_case_heatmaps(df)
    plot_latency_bars(df)
    plot_ttft_tps(df)
    print("wrote assets to", ROOT)


if __name__ == "__main__":
    main()
