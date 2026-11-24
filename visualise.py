import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



def flatten(values):
    return np.asarray(values, dtype=float).reshape(-1)


def summarise_log(log_path):
    log_path = Path(log_path)
    with log_path.open("r", encoding="utf-8") as log_file:
        log = json.load(log_file)

    summary = log
    for data_name, data in summary.items():
        if not data_name.endswith(" data") or not isinstance(data, dict):
            continue

        for trial_name, trial in data.items():
            if isinstance(trial, dict):
                for key in ("y", "noises", "x"):
                    trial.pop(key, None)

    summary_path = log_path.with_name("summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")

    print(f"Created log summary at {summary_path}")


def fitted_parameter(model, name):
    values = flatten(model[name])
    if values.size != 1:
        raise ValueError(f"Expected one {name} value, got {values.size}")
    return float(values[0])


def plot_trial(log, distribution_name, data_name, trial_name, trial, output_path):
    x = flatten(trial["x"])
    y = flatten(trial["y"])

    model = log["causal_model"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=18, alpha=0.7, label="Observed samples")

    if "x_f" in trial and "y_f" in trial:
        x_f = flatten(trial["x_f"])
        y_f = flatten(trial["y_f"])
        ax.scatter(x_f, y_f, s=90, color="green", marker="*", zorder=5,
                   label="Interventional sample")

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    if model["type"] == "LinearAdditiveModel":
        line_x = np.linspace(x_limits[0], x_limits[1], 300)
        line_y = model["slope"] * line_x + model["intercept"]
        ax.plot(line_x, line_y, color="crimson", linewidth=2, label="Real causal line")

    if "x_to_y_model" in trial:
        fitted = trial["x_to_y_model"]
        fitted_slope = fitted_parameter(fitted, "slope")
        fitted_intercept = fitted_parameter(fitted, "intercept")
        fitted_line_x = np.linspace(x_limits[0], x_limits[1], 300)
        fitted_line_y = fitted_slope * fitted_line_x + fitted_intercept
        ax.plot(fitted_line_x, fitted_line_y, color="dodgerblue", linewidth=2,
                linestyle="--", label="Trained x → y line")

    if "y_to_x_model" in trial:
        fitted = trial["y_to_x_model"]
        fitted_slope = fitted_parameter(fitted, "slope")
        fitted_intercept = fitted_parameter(fitted, "intercept")

        fitted_line_y = np.linspace(y_limits[0], y_limits[1], 300)
        fitted_line_x = fitted_slope * fitted_line_y + fitted_intercept
        ax.plot(fitted_line_x, fitted_line_y, color="darkorange", linewidth=2,
                linestyle="--", label="Trained y → x line")

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    is_correct = bool(trial.get("correct", False))
    p_x_to_y = trial.get("pval_x_to_y")
    p_y_to_x = trial.get("pval_y_to_x")
    # print(p_x_to_y)

    p_x_to_y_text = f"{p_x_to_y:.3f}" 
    p_y_to_x_text = f"{p_y_to_x:.3f}" 

    outcome = "Correct" if is_correct else "Incorrect"
    ax.set_title(f"{distribution_name} — trial {trial_name} (P(x -> y): {p_x_to_y_text}) (P(y -> x): {p_y_to_x_text})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def visualise(log_path):
    log_path = Path(log_path)
    summarise_log(log_path)
    with log_path.open("r", encoding="utf-8") as log_file:
        log = json.load(log_file)

    output_root = log_path.parent / "plots"
    created = 0

    for data_name, data in log.items():
        if not data_name.endswith(" data") or not isinstance(data, dict):
            continue
        
        distribution_folder = data.get("distr_name", data_name.removesuffix(" data"))

        trials_by_outcome = {"correct": [], "incorrect": []}
        for trial_name, trial in data.items():
            if trial_name == "distr_name" or not isinstance(trial, dict):
                continue

            # TODO Make it so that outcomes with pval 0 are somehow indicated
            outcome = "correct" if trial.get("correct") is True else "incorrect"



            trials_by_outcome[outcome].append((trial_name, trial))

        for outcome, trials in trials_by_outcome.items():
            for trial_name, trial in trials[:10]:
                output_path = (
                    output_root
                    / distribution_folder
                    / outcome
                    / f"trial_{trial_name}.png"
                )
                plot_trial(log, distribution_folder, data_name, trial_name, trial, output_path)
                created += 1

    print(f"Created {created} plots in {output_root}")


def plot_algorithm_comparison(results_root):
    results_root = Path(results_root)
    algorithms = {}

    for log_path in sorted(results_root.glob("*/log.json")):
        with log_path.open("r", encoding="utf-8") as log_file:
            log = json.load(log_file)

        trials = log.get("trials", {})
        if not isinstance(trials, dict):
            continue

        algorithm_results = {}
        for key, value in trials.items():
            distribution_name = re.sub(r"\s+sucess$", "", str(key))
            algorithm_results[distribution_name] = float(value)

        if algorithm_results:
            algorithms[log_path.parent.name] = algorithm_results

    if not algorithms:
        print("No trial results found to compare")
        return

    json_output_path = results_root / "algorithm_comparison.json"
    with json_output_path.open("w", encoding="utf-8") as json_file:
        json.dump(algorithms, json_file, indent=2)
        json_file.write("\n")
    print(f"Created extracted trial results at {json_output_path}")

    distribution_names = sorted({name for values in algorithms.values() for name in values})
    algorithm_names = sorted(algorithms)

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(distribution_names))
    width = 0.8 / len(algorithm_names)

    for index, algorithm_name in enumerate(algorithm_names):
        scores = [algorithms[algorithm_name].get(name, 0.0) for name in distribution_names]
        offset = (index - (len(algorithm_names) - 1) / 2) * width
        ax.bar(x + offset, scores, width, label=algorithm_name)

    ax.set_xlabel("Distribution")
    ax.set_ylabel("Success rate")
    ax.set_title("Algorithm comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(distribution_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Algorithm")
    fig.tight_layout()

    output_path = results_root / "algorithm_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Created algorithm comparison bar diagram at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark trials for one or more result folders")
    parser.add_argument(
        "folders",
        nargs="*",
        help="Folder names under results/, or paths to result folders or log.json files",
    )
    args = parser.parse_args()

    folders = args.folders
    for folder in folders:
        folder_path = Path(folder)
        if folder_path.is_file():
            log_path = folder_path
        else:
            if not folder_path.exists():
                folder_path = Path("results") / folder
            log_path = folder_path if folder_path.name == "log.json" else folder_path / "log.json"

        if not log_path.is_file():
            parser.error(f"No log.json found in {folder_path}")
        visualise(log_path)

    plot_algorithm_comparison(Path("results"))


if __name__ == "__main__":
    main()
