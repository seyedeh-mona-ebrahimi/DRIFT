<h1 align="center">🌊 DRIFT</h1>

<p align="center">
  <b>Constrained Dynamic Topic Models for Weak-Signal Trend Discovery</b>
</p>

<p align="center">
  Seyedeh Fatemeh Ebrahimi · Jaakko Peltonen<br>
  <i>Transactions on Machine Learning Research (TMLR), 2026</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TMLR-2026-1f6feb" alt="TMLR 2026">
  <img src="https://img.shields.io/badge/topic%20models-dynamic%20%2B%20constrained-6f42c1" alt="Dynamic and constrained topic models">
  <img src="https://img.shields.io/badge/inference-variational%20EM-2ea44f" alt="Variational EM">
</p>

---

## 🔍 Why DRIFT?

Some meaningful themes in large text collections are also rare. Think of mental-health talk in everyday YouTube comments: it is a small part of the conversation, it comes and goes, and the words people use for it change over time.

Standard topic models recover the dominant topics, and existing extensions add either temporal dynamics or domain guidance, but rarely both. DRIFT brings the two together.

## ✨ Key ideas

- 🌱 **One seed list for the whole domain.** You give a single list of seed words, without saying which word belongs to which topic.
- ⚖️ **Seed-mass floor.** Each minority topic keeps a minimum share of its probability on the seed words.
- 🔽 **Less weight where it does not belong.** Minority topics get a lower weight in documents without seed words.
- ⏳ **Change over time.** Topic prevalence changes through document covariates, and topic content through a random walk.
- ✅ **Constraints by construction.** Both constraints are built into the model instead of being added as penalties, so every distribution stays a valid probability distribution.
- 🧮 **Variational EM.** The model is fitted with variational EM and reparameterized Monte Carlo estimates.

## 🗺️ DRIFT at a glance

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="drift_overview_dark.svg">
    <img src="drift_overview_light.svg" width="100%" alt="DRIFT at a glance: inputs, the four parts of the model, and outputs">
  </picture>
</p>


## 🔒 Data and privacy

The real-data case study uses YouTube comments from Finnish mental-health-related vlog channels. We do not share the dateset.

> [!NOTE]
> This repository is still being updated. The remaining experiment scripts will be added soon.

## ⚙️ Requirements

Python 3.11 or newer, with the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Generate the synthetic benchmark

```bash
python generate_synthetic_data.py --out_csv synthetic_data.csv --out_params_json synthetic_params.json
```


## 🧪 Try DRIFT on your own data

To fit DRIFT to your own corpus, you need:

- **Documents with time stamps**, tokenized and grouped into time bins (for example, months).
- **Document covariates**, such as features of time, so that topic prevalence can change over time.
- **One seed list** for the domain you are interested in. The seed words are not assigned to topics.
- **The number of topics K**, and how many of them are minority topics (K_min).

The model is fitted with `fit_constrained_stm_vi` in `drift_model.py`. The two constraints follow the paper.


## 📄 Citation

please cite the work if it is inetersting to you:

```bibtex
@article{ebrahimi2026drift,
  title   = {{DRIFT}: Constrained Dynamic Topic Models for Weak-Signal Trend Discovery},
  author  = {Ebrahimi, Seyedeh Fatemeh and Peltonen, Jaakko},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026}
}
```

## 📬 Contact

Questions and feedback are welcome. Email Mona Ebrahimi (seyedeh.ebrahimi@tuni.fi).
