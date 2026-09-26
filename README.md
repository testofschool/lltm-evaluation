# Explaining Benchmark Difficulty: Linear Logistic Test Models for Feature-Based AI Evaluation

Jung Min Kang, Independent Researcher, Seoul, South Korea

## Reproducing the experiments

```bash
pip install -r requirements.txt
python experiment.py
```

**Note:** Full runtime depends on CPU speed and may exceed 10 minutes due to repeated SciPy optimization loops. Precomputed results are included in `results.json`.

## Files

- `main.tex` — LaTeX source
- `main.bbl` — Compiled bibliography
- `references.bib` — BibTeX source
- `experiment.py` — Complete experiment and figure generation
- `requirements.txt` — Python dependencies
- `results.json` — Results written by `python experiment.py` (regenerated from the script as committed; an earlier version of this file contained hand-added fields — a `_schema` key, `"se": null` for Experiments 2–4, and rounded Experiment 1 standard errors — that the script does not emit)
- `figures/` — PDF figures

## Citation

```bibtex
@misc{kang2026lltm,
  title={Explaining Benchmark Difficulty: Linear Logistic Test Models for Feature-Based AI Evaluation},
  author={Kang, Jung Min},
  note={Manuscript},
  year={2026}
}
```
