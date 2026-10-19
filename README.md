# DRIFT

Code for the paper DRIFT: Constrained Dynamic Topic Models for Weak-Signal Trend Discovery by Seyedeh Fatemeh Ebrahimi and Jaakko Peltonen, Transactions on Machine Learning Research, 2026.

## Overview

DRIFT is a dynamic topic model for finding rare topics that change over time and are easily hidden by a large background of everyday content. The user gives one list of seed words for the domain of interest, without assigning the words to topics. DRIFT models how both topic prevalence and topic content change over time, and adds two constraints that are built into the model. The model is fitted with variational EM.

## Data

The real-data case study uses public YouTube comments from Finnish mental-health-related vlog channels. We do not share the comments, user names or any other individual-level information.

## Citation

```bibtex
@article{ebrahimi2026drift,
  title   = {{DRIFT}: Constrained Dynamic Topic Models for Weak-Signal Trend Discovery},
  author  = {Ebrahimi, Seyedeh Fatemeh and Peltonen, Jaakko},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026}
}
```

## Contact

Mona Ebrahimi (seyedeh.ebrahimi@tuni.fi)
