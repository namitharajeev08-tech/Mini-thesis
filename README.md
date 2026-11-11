# Machine Learning-Based Inverse Alloy Design for Bainitic Steels Using Bayesian Optimization and Surrogate Modeling

**Mini-Thesis** · Namitha Rajeev · M.Sc. Materials Engineering, RWTH Aachen University

---

## Overview

Designing bainitic steels means finding alloy compositions that produce the right microstructure: bainitic ferrite with retained austenite, while avoiding unwanted phases such as cementite, pearlite or martensite. Searching this space with experiments or full thermodynamic calculations alone is slow and expensive.

This project takes an **inverse design** approach:

1. **Forward problem (surrogate model):** Train machine-learning models on CALPHAD-based equilibrium phase data. The models predict phase constitution (phase fractions) from alloy composition (and temperature).
2. **Inverse problem (Bayesian optimization):** Use the trained surrogate as a fast, cheap objective function. Bayesian optimization then searches composition space for alloys that meet target phase-fraction criteria for bainitic microstructures.

Using the surrogate in place of repeated thermodynamic calculations makes the search much faster. The probabilistic optimizer also balances exploring new regions of composition space against refining compositions that already look promising.

---

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

### 1. Data Generation
- Equilibrium phase fractions were computed with CALPHAD for 128 compositions. <!-- TODO: software, database -->
- **Input features:** alloying element contents in wt.% <!-- TODO: e.g. C, Si, Mn, Cr, Mo, Ni, Al … --> and temperature <!-- TODO: range -->
- **Targets:** equilibrium fractions of the relevant phases <!-- TODO: e.g. FCC_A1 (austenite), BCC_A2 (ferrite), cementite, carbides … -->

### 2. Surrogate Modeling
- Preprocessing: <!-- TODO: e.g. scaling, train/test split ratio, cross-validation -->
- Models evaluated: <!-- TODO: e.g. Gaussian Process Regression, Random Forest, Gradient Boosting, Neural Network -->
- Evaluation metrics: R², MAE, RMSE <!-- TODO: confirm -->

### 3. Bayesian Optimization (Inverse Design)
- **Objective:** find compositions that maximize/minimize <!-- TODO: e.g. retained austenite fraction, suppression of cementite --> subject to composition constraints
- **Surrogate within BO:** <!-- TODO: e.g. Gaussian Process -->
- **Acquisition function:** <!-- TODO: e.g. Expected Improvement / UCB -->
- **Design space bounds:** <!-- TODO: element ranges -->

### 4. Validation
- The proposed compositions are checked against CALPHAD calculations and metallurgical knowledge of bainitic steels, such as the T₀ concept and how Si suppresses cementite. <!-- TODO -->

---

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

- [x] Generate equilibrium phase dataset
- [x] Train surrogate models for phase fraction prediction
- [ ] Implement Bayesian optimization loop for inverse design
- [ ] Validate optimized compositions with CALPHAD
- [ ] Extend to kinetic descriptors (e.g. Bs/Ms temperatures, T₀)

<!-- TODO: tick off what's done -->

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
