import argparse
import copy
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



def flatten(values):
    return np.asarray(values, dtype=float).reshape(-1)

def write_json(log, folder, name):
    with (folder / name).open("w", encoding="utf-8") as file:
        json.dump(log, file, indent=2)
        file.write("\n")


def summarise_statistic(nums):
    return {
        "mean": np.mean(nums),
        "var": np.var(nums)
    }

def _summarise_optional(values):
    """Summarise a list that may contain None; returns None if nothing to summarise."""
    present = [v for v in values if v is not None]
    return summarise_statistic(present) if present else None


def _summarise_model(models):
    """models: list of {"slope": ..., "intercept": ...} or None entries."""
    present = [m for m in models if m is not None]
    if not present:
        return None
    return {
        "slope": summarise_statistic([m["slope"] for m in present]),
        "intercept": summarise_statistic([m["intercept"] for m in present]),
    }


def summarise_direction(trials, direction):
    results = [t["results"][direction] for t in trials]
    return {
        "expected_reject": results[0]["expected_reject"],
        "rejection_rate": summarise_statistic([r["rejected_null"] for r in results]),
        "correct": summarise_statistic([r["correct"] for r in results]),
        "p_null": summarise_statistic([r["p_null"] for r in results]),
        "p_alt": _summarise_optional([r["p_alt"] for r in results]),
        "model_null": _summarise_model([r["model_null"] for r in results]),
        "model_alt": _summarise_model([r["model_alt"] for r in results]),
    }


def summarise_log(log, folder_path):
    summary = copy.deepcopy(log)  # don't destroy the caller's log
    for setting in summary["settings"]:
        trials = setting.pop("trials")
        setting["results"] = {
            direction: summarise_direction(trials, direction)
            for direction in ("x->y", "y->x")
        }

    write_json(summary, folder_path, "summary.json")
    return summary


def fitted_parameter(model, name):
    values = flatten(model[name])
    if values.size != 1:
        raise ValueError(f"Expected one {name} value, got {values.size}")
    return float(values[0])

def plot_regression(
        ax, 
        xs, ys, 
        xy_slope, xy_intercept, 
        yx_slope, yx_intercept,
        real_slope, real_intercept
    ):

    x_line = np.linspace(xs.min(), xs.max(), 200)
    ax.scatter(xs, ys, s=10, alpha=0.4, color="gray", label="data")

    # x -> y model: y = a*x + b
    ax.plot(x_line, xy_slope * x_line + xy_intercept, linewidth=2, label="x → y fit", linestyle="--")

    ax.plot(x_line, real_slope * x_line + real_intercept, linewidth=2, label="real model")

    # y -> x model: x = c*y + d, inverted to y = (x - d) / c so it lives on the same axes
    if yx_slope != 0:
        ax.plot(x_line, (x_line - yx_intercept) / yx_slope, linewidth=2, linestyle="--", label="y → x fit")
    else:
        # slope 0 means x is constant, i.e. a vertical line at x = d
        ax.axvline(yx_intercept, linewidth=2, linestyle="--", label="y → x fit")

    # keep the view on the data, since steep inverted lines can blow up the y-range
    pad = 0.05 * (ys.max() - ys.min() or 1.0)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()


import numpy as np
from scipy.stats import gaussian_kde, norm


def plot_residuals(ax, xs, ys, slope, intercept, name, pval=None, color="tab:blue"):
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    # sklearn's coef_ / intercept_ may be arrays, so reduce them to scalars
    slope = float(np.ravel(slope)[0])
    intercept = float(np.ravel(intercept)[0])

    residuals = (slope * xs + intercept - ys)

    # histogram normalized so its area is 1, i.e. an empirical density
    ax.hist(residuals, bins="auto", density=True,
            color=color, alpha=0.35, edgecolor="white", linewidth=0.5)

    grid = np.linspace(residuals.min(), residuals.max(), 300)

    # smooth density estimate of the residuals
    if residuals.std() > 0:
        ax.plot(grid, gaussian_kde(residuals)(grid),
                color=color, linewidth=2, label="KDE")

    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.7)  # zero-residual reference

    mu, sigma = residuals.mean(), residuals.std()
    
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu, ),
        r'$\sigma=%.2f$' % (sigma, )))

    if pval:
        textstr += "\n" + r"p-value=%.3f" % (pval, )

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')

    ax.set_title(name, fontsize=11)
    ax.set_xlabel("residual")
    ax.set_ylabel("density")
    ax.set_yticks([])  # absolute density values rarely matter here
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

def plot_trial(xs, ys, xy_slope, xy_intercept, yx_slope, yx_intercept,
               real_slope, real_intercept,
               p_null, p_alt,
               output_folder, filename="trial.png"):
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()

    # sklearn's coef_ / intercept_ may be arrays, so reduce them to scalars
    xy_slope, xy_intercept = float(np.ravel(xy_slope)[0]), float(np.ravel(xy_intercept)[0])
    yx_slope, yx_intercept = float(np.ravel(yx_slope)[0]), float(np.ravel(yx_intercept)[0])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 11))
    plot_regression(ax[0, 0], xs, ys, xy_slope, xy_intercept, yx_slope, yx_intercept, real_slope, real_intercept)
    plot_residuals(ax[0, 1], xs, ys, real_slope, real_intercept, "Real residuals")
    plot_residuals(ax[1, 0], xs, ys, xy_slope, xy_intercept, "null residuals", p_null)
    plot_residuals(ax[1, 1], xs, ys, yx_slope, yx_intercept, "alt residuals", p_alt)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # avoids piling up open figures when called in a loop
    return path


def visualise(log, folder, max_per_class=10):
    folder = Path(folder)
    output_root = folder / "plots"
    created = 0

    real_slope = log["meta"]["causal_model"]["slope"]
    real_intercept = log["meta"]["causal_model"]["intercept"]

    for setting in log["settings"]:
        setting_name = f"{setting['x_distribution']}_{setting['noise_distribution']}"
        setting_folder = output_root / setting_name

        correct = 0
        incorrect = 0

        for trial in setting["trials"]:
            if correct >= max_per_class and incorrect >= max_per_class:
                break

            if trial["correct"]:
                if correct >= max_per_class:
                    continue
                correct += 1
                label = "correct"
            else:
                if incorrect >= max_per_class:
                    continue
                incorrect += 1
                label = "incorrect"

            models = trial["models"]
            # if models["x_to_y"] is None or models["y_to_x"] is None:
            #     continue  # plot_trial needs both fitted lines

            data = trial["data"]
            plot_trial(
                data["x"], data["y"],
                models["model_null"]["slope"], models["model_null"]["intercept"],
                models["model_alt"]["slope"], models["model_alt"]["intercept"],
                real_slope, real_intercept,
                trial["p_null"], trial["p_alt"],
                setting_folder,
                filename=f"{label}_trial_{trial['trial']}.png",
            )
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

def read_log(folder):
    folder_path = Path(folder)
    if folder_path.is_file():
        log_path = folder_path
    else:
        if not folder_path.exists():
            folder_path = Path("results") / folder
        log_path = folder_path if folder_path.name == "log.json" else folder_path / "log.json"

    log_path = Path(log_path)
    
    with log_path.open("r", encoding="utf-8") as log_file:
        log = json.load(log_file)

    return log, folder_path
        
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
        log, folder_path = read_log(folder)
        
        # visualise(log, folder_path)

        summarise_log(log, folder_path)
    # plot_algorithm_comparison(Path("results"))


if __name__ == "__main__":
    main()
 