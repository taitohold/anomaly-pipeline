"""Rewrite the generated number blocks in README.md from the files in results/."""

from pathlib import Path

from anomaly.report import fill, render

README = Path("README.md")
README.write_text(
    fill(README.read_text(), render(Path("results"), Path("checkpoints/thresholds.json")))
)
print("README.md numbers updated from results/")
