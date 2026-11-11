# Machine Learning-Based Inverse Alloy Design for Bainitic Steels Using Bayesian Optimization and Surrogate Modeling

**Mini-Thesis** · Namitha Rajeev · M.Sc. Materials Engineering, RWTH Aachen University

---

## Overview

Designing bainitic steels means finding alloy compositions that produce the right microstructure: bainitic ferrite with retained austenite, while avoiding unwanted phases such as cementite, pearlite or martensite. Searching this space with experiments or full thermodynamic calculations alone is slow and expensive.


## Repository Structure

```
Mini-thesis/
├── Equilibrium phase constitution_128_All Phases.xls   # Thermodynamic dataset (equilibrium phase fractions)
├── phase_fraction_ml.py                                # Surrogate model training & evaluation
├── ml_results/                                         # Model outputs: metrics, plots, predictions
├── launch.json                                         # VS Code run/debug configuration
└── README.md
```

| File / Folder | Description |
|---|---|
| `Equilibrium phase constitution_128_All Phases.xls` | Dataset of equilibrium phase constitution for 128 alloy compositions (all phases). It was generated with <!-- TODO: Thermo-Calc / MatCalc / other, database e.g. TCFE --> |
| `phase_fraction_ml.py` | Main script. It loads the dataset, preprocesses features and targets, then trains, evaluates and saves the ML surrogate models for phase fraction prediction. |
| `ml_results/` | Generated results, such as parity plots, error metrics and feature importances. <!-- TODO: list key files --> |
| `launch.json` | VS Code debug configuration. It normally belongs in a `.vscode/` folder. |

---

## Methodology


## Getting Started

### Requirements
- Python 3.9+
- Dependencies (adjust to match your imports):

```bash
pip install numpy pandas scikit-learn matplotlib xlrd openpyxl
# for Bayesian optimization, e.g.:
pip install scikit-optimize   # or bayesian-optimization / botorch
```

> **Note:** Reading legacy `.xls` files with pandas requires `xlrd`.

### Usage

```bash
git clone https://github.com/namitharajeev08-tech/Mini-thesis.git
cd Mini-thesis
python phase_fraction_ml.py
```

Results are written to `ml_results/`.

---

## Results

<!-- TODO: Summarize key findings, e.g.:
- Best surrogate model and its test R² / MAE per phase
- Parity plot (embed: ![Parity plot](ml_results/parity_plot.png))
- Optimized composition(s) found by BO and their predicted phase fractions
-->

---

## Roadmap


---

## References

<!-- TODO: add your key references, e.g.:
- H.K.D.H. Bhadeshia, *Bainite in Steels*, 3rd ed., CRC Press, 2015.
- B. Shahriari et al., "Taking the Human Out of the Loop: A Review of Bayesian Optimization," *Proc. IEEE*, 104(1), 2016.
-->

---

## License

<!-- TODO: e.g. MIT, or "All rights reserved – academic work" -->

---

## Contact

**Namitha Rajeev**
M.Sc. Materials Engineering, RWTH Aachen University
📧 [namitharajeev08@gmail.com](mailto:namitharajeev08@gmail.com)
