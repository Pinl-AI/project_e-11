# Project E-11

#### Description:

This project predicts the reaction order (zero, first, second, or third) of chemical reactions using ODE simulations and machine learning

## Requirements

- Python 3.10

- TensorFlow 2.15

- SciPy

- NumPy, Pandas, Matplotlib

- Scikit-learn

- Gradio

- Localtunnel

- Streamlit

## Key Features

- Reaction Order Prediction: Predicts whether the chemical reaction follows a zero, first, second, or third-order kinetics.

- Customizable Input: Adjust reaction conditions such as temperature, activation energy, initial concentrations, catalyst, and more.

- Real-time Simulation: Simulates the concentration of reactants and products over time based on predicted reaction order.

- Interactive Visualization: Displays concentration vs. time plots of reactants and products for easy analysis.

## Key Functions

### `compute_k`

Calculates the rate constant k using the Arrhenius equation.

### `ode1`

Simulates the reaction based on the initial concentrations, temperature, activation energy, and pre-exponential factor. Randomly selects a reaction order and returns concentration profiles.

### `ode2`

Simulates the reaction based on the predicted reaction order and other inputs, and returns concentration vs. time data.

### `predict_order`

Predicts the reaction order using the trained model.

## Dataset

| order  | temp | pH    | Ea  | A_factor     | pressure | log_pressure | weight | structure | catalyst | ... | C1       | C2       | C3       | C4       | C5       | C6       | C7       | C8       | C9       | C10      |
| ------ | ---- | ----- | --- | ------------ | -------- | ------------ | ------ | --------- | -------- | --- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| zero   | 273  | 6.93  | 93  | 2.241372e+17 | 2.44     | 0.892        | 151.6  | Linear    | Base     | ... | 5.037570 | 5.325140 | 5.612709 | 5.900279 | 6.187849 | 6.475419 | 6.762988 | 7.050558 | 7.338128 | 7.625698 |
| third  | 278  | 7.13  | 96  | 1.412727e+17 | 3.90     | 1.361        | 47.2   | Unknown   | None     | ... | 1.653542 | 1.706803 | 1.726140 | 1.733800 | 1.736689 | 1.737977 | 1.738369 | 1.738675 | 1.738670 | 1.738693 |
| first  | 278  | 8.61  | 96  | 4.880562e+16 | 3.92     | 1.366        | 73.5   | Unknown   | Acid     | ... | 4.728580 | 4.746029 | 4.762415 | 4.777803 | 4.792254 | 4.805825 | 4.818569 | 4.830536 | 4.841775 | 4.852329 |
| second | 279  | 4.22  | 94  | 2.441632e+17 | 4.84     | 1.577        | 57.4   | Branched  | Enzyme   | ... | 5.964510 | 6.085491 | 6.095228 | 6.096048 | 6.095868 | 6.095985 | 6.096414 | 6.095743 | 6.095272 | 6.095830 |
| first  | 273  | 13.05 | 90  | 4.989334e+17 | 2.04     | 0.713        | 129.7  | Linear    | Enzyme   | ... | 1.130000 | 1.130000 | 1.130000 | 1.130000 | 1.130000 | 1.130000 | 1.130000 | 1.130000 | 1.130000 | 1.130000 |

### Data

| **Parameter** | **Description**                                        |
| ------------- | ------------------------------------------------------ |
| temp          | Temperature in Kelvin (K)                              |
| pH            | Acidity/basicity of the solution                       |
| Ea            | Activation energy (kJ/mol)                             |
| A_factor      | Pre-exponential factor                                 |
| pressure      | Pressure in atmospheres (atm)                          |
| log_pressure  | Natural logarithm of the pressure                      |
| weight        | Mass of the sample (g)                                 |
| structure     | Type of molecular structure (e.g., Linear, Branched)   |
| catalyst      | Type of catalyst used (e.g., Base, Acid, Enzyme, None) |
| A0 to C10     | Concentration of chemical over time                    |

## Machine Learning

### Deep Neural Network (DNN)

Features: 50+ (concentrations, kinetic constants, and reaction conditions)

Architecture: [50, 40] hidden layers

Optimizer: RMSprop

Feature Columns

- Categorical: structure, catalyst

- Numerical: Temperature, pressure, rate constants, concentrations

<a href="https://ibb.co/1YRqMpFn"><img src="https://i.ibb.co/9myThfQv/Untitled.png" alt="Untitled" border="0"></a>

## Results

| Model               | Accuracy | Best Parameters                                               |
| ------------------- | -------- | ------------------------------------------------------------- |
| Logistic Regression | 80.1%    | `{'C': 10, 'penalty': 'l2'}`                                  |
| SVC (RBF)           | 87.9%    | `{'C': 10, 'kernel': 'rbf'}`                                  |
| KNN                 | 56.7%    | `{'n_neighbors': 7, 'weights': 'uniform'}`                    |
| Random Forest       | 83.6%    | `{'max_depth': None, 'n_estimators': 200}`                    |
| Gradient Boosting   | 89.4%    | `{'max_depth': 5, 'n_estimators': 200}`                       |
| XGBoost             | 89.5%    | `{'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}` |
| DNN Classifier      | 99.98%   | `(5000 steps)`                                                |

#### Best model:

- DNN Classifier

## Chemistry

### Arrhenius Equation:

$$
k = A \cdot e^{-\frac{E_a}{RT}}
$$

Where:

- `k`: Rate constant
- `A`: Pre-exponential factor
- `Ea`: Activation energy (kJ/mol)
- `R`: Universal gas constant = 8.314 J/mol·K
- `T`: Temperature in Kelvin

---

### ODEs:

#### Zero order:

`A → Product`

Rate Law:

$$
\frac{d[A]}{dt} = -k
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/6RcgvQmY/Untitled-1.png" alt="Untitled-1" border="0" style="width: 400px;">
    </a>
</div>

#### First order:

**Decay:**  
`A → Product`

Rate Law:

$$
\frac{d[A]}{dt} = -k[A]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/Xf5tJJBY/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**First:**  
`A → C`

Rate Laws:

$$
\frac{d[A]}{dt} = -k[A] \quad , \quad \frac{d[C]}{dt} = k[A]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/RTyyVH1z/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**Reversible:**  
`A ⇌ B`

Rate Laws:

$$
\frac{d[A]}{dt} = -k_1[A] + k_{-1}[B]
$$

$$
\frac{d[B]}{dt} = k_1[A] - k_{-1}[B]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/V02GDwS8/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

---

#### Second order:

**Type 1:**  
`2A → Product`

Rate Law:

$$
\frac{d[A]}{dt} = -k[A]^2
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/qYF1CNwM/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**Type 2:**  
`A + B → Product`

Rate Law:

$$
\frac{d[A]}{dt} = -k[A][B]
$$

$$
\frac{d[B]}{dt} = -k[A][B]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/60w8rfMF/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**Reversible Type 1:**  
`2A ⇌ C`

Rate Laws:

$$
\frac{d[A]}{dt} = -2k[A]^2 + 2k_{-1}[C]
$$

$$
\frac{d[C]}{dt} = k[A]^2 - k_{-1}[C]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/Z1T4Yp4T/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**Reversible Type 2:**  
`A + B ⇌ C`

Rate Laws:

$$
\frac{d[A]}{dt} = -k[A][B] + k_{-1}[C]
$$

$$
\frac{d[B]}{dt} = -k[A][B] + k_{-1}[C]
$$

$$
\frac{d[C]}{dt} = k[A][B] - k_{-1}[C]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/LzWfdVPX/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

---

#### Third order:

**Type 1 (3A):**  
`3A → Product`

**Rate Law:**

$$
\frac{d[A]}{dt} = -k[A]^3
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/8tpCd3Y/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**Type 2 (2A + B):**  
`2A + B → Product`

Rate Laws:

$$
\frac{d[A]}{dt} = -2k[A]^2[B]
$$

$$
\frac{d[B]}{dt} = -k[A]^2[B]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/60DMCT3x/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**Reversible Type 1:**  
`3A ⇌ C`

Rate Laws:

$$
\frac{d[A]}{dt} = -3k[A]^3 + 3k_{-1}[C]
$$

$$
\frac{d[C]}{dt} = k[A]^3 - k_{-1}[C]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/gFzf8w87/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

**Reversible Type 2:**  
`2A + B ⇌ C`

Rate Laws:

$$
\frac{d[A]}{dt} = -2k[A]^2[B] + 2k_{-1}[C]
$$

$$
\frac{d[B]}{dt} = -k[A]^2[B] + k_{-1}[C]
$$

$$
\frac{d[C]}{dt} = k[A]^2[B] - k_{-1}[C]
$$

<div style="text-align: center;">
    <a href="https://imgbb.com/">
        <img src="https://i.ibb.co/gq8sbdj/Untitled.png" alt="Untitled" border="0" style="width: 400px;">
    </a>
</div>

Where:

- `A`: Concentrations of chemical A (mol/L)
- `B`: Concentrations of chemical B (mol/L)
- `C`: Concentrations of chemical C (mol/L)
- `k`: Forward rate constant
- `k_1`: Backward rate constant

---

## Preview

### Gradio

<p align="center">
  <a href="https://huggingface.co/spaces/PinlAI/ProjectE-11">Project e-11 huggingface</a>
</p>

<p align="center">
  <a href="https://ibb.co/9mf3TSTL">
    <img src="https://i.ibb.co/358YNQNq/Untitled-design.jpg" alt="Untitled-design" width="550" border="0">
  </a>
  <a href="https://ibb.co/RGcH1PKR">
    <img src="https://i.ibb.co/LD1pM6yV/e-11.gif" alt="e-11" width="550" border="0">
</a>
</p>

### Team Members:

- Mujtaba-4T4
- TurboGlitch
- Hydroxymeth
- muzammil-max
