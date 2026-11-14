"""Charts that compare the modules on the same conversation: input tokens and cost per question.

Each module saves what it measured with save_runs("module_1", runs), and a later module loads it
with load_runs, so Module 2 can chart itself against Module 1, and Module 4 against both.
The numbers live in data/.runs/, next to the index cache, and are never committed.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from .settings import ROOT

RUNS = ROOT / "data" / ".runs"
COLORS = {"Module 1": "#9e9e9e", "Module 2": "#4c78a8", "Module 4": "#2e9e6b"}


def save_runs(module: str, runs: list[dict]) -> None:
    """Keep this module's question, prompt_tokens and cost_usd per run, for the modules after it."""
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{module}.json").write_text(json.dumps(runs, indent=1))


def load_runs(module: str) -> list[dict] | None:
    """What an earlier module saved, or None if it hasn't been run yet."""
    path = RUNS / f"{module}.json"
    return json.loads(path.read_text()) if path.exists() else None


def compare(results: dict[str, list[dict]], title: str) -> None:
    """Two bar charts side by side, input tokens and cost per question, one bar per module.

    Questions are matched by their text, in the first module's order.
    """
    labels = [r["question"] for r in next(iter(results.values()))]
    results = {m: [r for r in runs if r["question"] in labels] for m, runs in results.items()}
    x = np.arange(len(labels))
    width = 0.8 / len(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    panels = [("prompt_tokens", "Input tokens per question", "{:,.0f}"),
              ("cost_usd", "Cost per question (USD)", "${:.4f}")]
    for ax, (field, heading, fmt) in zip(axes, panels):
        for n, (module, runs) in enumerate(results.items()):
            by_question = {r["question"]: r.get(field) or 0 for r in runs}
            values = [by_question.get(q, 0) for q in labels]
            bars = ax.bar(x + (n - (len(results) - 1) / 2) * width, values, width,
                          label=module, color=COLORS.get(module))
            ax.bar_label(bars, labels=[fmt.format(v).replace("$", r"\$") for v in values], fontsize=8, padding=2)
        ax.set_title(heading)
        ax.set_xticks(x, labels, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.15)
        ax.set_yticks([])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(results), frameon=False)
    totals = "   ".join(f"{m}: {sum(r['prompt_tokens'] for r in runs):,} tokens, "
                       f"${sum(r.get('cost_usd') or 0 for r in runs):.3f}" for m, runs in results.items())
    fig.suptitle(f"{title}\n{totals}".replace("$", r"\$"), fontsize=11) # a plain $, not matplotlib math
    fig.tight_layout(rect=(0, 0.07, 1, 1)) # room for the legend underneath
    plt.show()
