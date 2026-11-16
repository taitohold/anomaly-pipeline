"""Every number in the README must come from a script. This fails if the
README's generated blocks differ from what the results files say: rerun
`make readme` (or `make reproduce`) and commit the README."""

from pathlib import Path

from anomaly.report import fill, render

ROOT = Path(__file__).resolve().parents[1]


def test_readme_numbers_match_the_results_files():
    readme = (ROOT / "README.md").read_text()
    blocks = render(ROOT / "results", ROOT / "checkpoints/thresholds.json")
    assert fill(readme, blocks) == readme, "README is out of date: run `make readme`"
