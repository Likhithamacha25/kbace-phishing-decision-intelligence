# KB-ACE: knowledge-based, cost-aware evidence acquisition for phishing triage

Code and results for the Short Communication *"Knowledge-based AI and data intelligence for cybersecurity decision making:
forgery-resistant, cost-aware evidence acquisition for phishing triage"* (L. Macha, submitted to *Knowledge-Based Systems*).

A phishing screen has to decide, for every URL, whether to **release** it, **hold** it for verification, or **buy more evidence**.
KB-ACE gives that decision to an analyst-owned knowledge base. For each of the 30 cues in the dataset, the knowledge base records
the evidence tier and its cost, whether an attacker can forge the cue, and its documented direction of risk. Two rules follow:

* **Trust anchoring.** Cheap, forgeable evidence (URL string, page content) may trigger a hold but never a release. A release
  requires tier-3 evidence that the attacker cannot fake.
* **Threat-informed augmentation.** Each training phishing site gets a copy whose forgeable cues are spliced in from a
  legitimate site.

Thresholds that minimise expected cost (misses, false holds and look-up spend) are learned from out-of-fold training scores only.

## Headline results (20 group-aware 70/30 splits, UCI Phishing Websites)

| | Gradient boosting, all cues | Cost-aware cascade | **KB-ACE** |
|---|---|---|---|
| ROC-AUC (clean) | 0.993 | — | — |
| Recall when every forgeable cue is faked | 0.0% | 0.0% | **92.2%** |
| Evidence spend per URL | 0.120 | 0.037 | 0.072 |
| Expected cost per URL, clean / forged | 0.237 / 4.631 | 0.161 / 4.533 | 0.220 / **0.524** |

KB-ACE becomes the cheapest policy once 1.5% of phishing sites forge their cues. A knowledge–data audit also finds
5 cues whose observed direction contradicts their documentation. 5,270 of the 11,055 rows are exact
duplicates, so every split keeps identical feature vectors on the same side.

## Reproduce

```bash
pip install -r requirements.txt
python kbace.py        # ~17 min on 2 CPU cores; downloads the UCI data on first run
python figures.py      # Figures 1-2, Tables 1-2 and the graphical abstract -> figures/
```

`python kbace.py --splits 3` runs a quick check. Seeds are fixed (`--seed 2026`).

## Files

| Path | Content |
|---|---|
| `kbace.py` | knowledge base, audit, learners, cascade decision layer, forgery stress test, statistics |
| `figures.py` | figures, tables and graphical abstract built from `results/` |
| `results/per_split.csv` | every metric for every policy and split |
| `results/attack_curve.csv` | recall for k = 0…22 forged cues, per policy and split |
| `results/sensitivity.csv` | costs over the 4 × 5 cost-ratio × acquisition-cost grid |
| `results/paired_tests.csv` | Wilcoxon + Holm and corrected resampled t-tests |
| `results/knowledge_audit.csv` | documented polarity versus data, per cue |
| `results/summary.csv`, `run_summary.json`, `breakeven.json` | aggregates used in the paper |

## Data

R. Mohammad, L. McCluskey, *Phishing Websites*, UCI Machine Learning Repository, 2015,
[doi:10.24432/C51W2X](https://doi.org/10.24432/C51W2X), CC BY 4.0. `kbace.py` downloads the data and converts it to
`data/phishing.csv` on first run.

## Citation

L. Macha, KB-ACE: knowledge-based, cost-aware evidence acquisition for phishing triage, version 1.0 [software], GitHub, 2026.

Code under the MIT licence.
