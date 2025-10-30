# Financial Mathematics Library - Complete Documentation

## Table of Contents
1. [Core Functionality Overview](#core-functionality-overview)
2. [Mathematical Models](#mathematical-models)
3. [Numerical Methods](#numerical-methods)
4. [Option Pricing Framework](#option-pricing-framework)
5. [Characteristic Functions](#characteristic-functions)
6. [Simulation Methods](#simulation-methods)
7. [Utility Functions](#utility-functions)
8. [Error Handling & Calibration](#error-handling--calibration)
9. [Dependencies](#dependencies)
10. [Usage Examples](#usage-examples)

## Core Functionality Overview

This C++ library provides a comprehensive implementation of advanced financial mathematics models, specializing in:

- Stochastic volatility modeling
- Jump-diffusion processes
- Fourier-based option pricing
- Monte Carlo simulations
- Implied volatility calculations

The code leverages Eigen for high-performance linear algebra and Boost for advanced mathematical functions.

## Mathematical Models

### CGMY Model Components
Here's the complete README file content that documents all functions in your financial mathematics library:

## Core Functionality Overview

This C++ library provides a comprehensive implementation of advanced financial mathematics models, specializing in:

- Stochastic volatility modeling
- Jump-diffusion processes
- Fourier-based option pricing
- Monte Carlo simulations
- Implied volatility calculations

The code leverages Eigen for high-performance linear algebra and Boost for advanced mathematical functions.

## Mathematical Models

### CGMY Model Components
cpp

VectorXcd char_CGMY(const VectorXcd& u, double C, double G, double M, double Y)
- Implements the characteristic function for the CGMY process
- Parameters: Complex vector `u`, CGMY parameters (C, G, M, Y)
- Returns: Characteristic function values
cpp

VectorXcd char_CGMY_neg(const VectorXcd& u, double C, double G, double Y)
- Characteristic function for negative jumps only in CGMY model

### Heston Model Components
cpp

VectorXcd char_func_Heston(const VectorXcd& u, const VectorXd& para, const VectorXd& v0, double T)
- Computes Heston model characteristic function
- Parameters: Complex vector `u`, model parameters, initial variance `v0`, maturity `T`

## Numerical Methods

### Runge-Kutta Integrators
cpp

void rk4(...)

void rk4_T(...)

void rk4_2(...)
- Multiple implementations of 4th-order Runge-Kutta method
- Specialized versions for:
  - General ODEs
  - Time-dependent characteristic functions
  - Financial applications

### Root Finding & Interpolation
Here's the complete README file content that documents all functions in your financial mathematics library:

markdown
复制
# Financial Mathematics Library - Complete Documentation

## Table of Contents
1. [Core Functionality Overview](#core-functionality-overview)
2. [Mathematical Models](#mathematical-models)
3. [Numerical Methods](#numerical-methods)
4. [Option Pricing Framework](#option-pricing-framework)
5. [Characteristic Functions](#characteristic-functions)
6. [Simulation Methods](#simulation-methods)
7. [Utility Functions](#utility-functions)
8. [Error Handling & Calibration](#error-handling--calibration)
9. [Dependencies](#dependencies)
10. [Usage Examples](#usage-examples)

## Core Functionality Overview

This C++ library provides a comprehensive implementation of advanced financial mathematics models, specializing in:

- Stochastic volatility modeling
- Jump-diffusion processes
- Fourier-based option pricing
- Monte Carlo simulations
- Implied volatility calculations

The code leverages Eigen for high-performance linear algebra and Boost for advanced mathematical functions.

## Mathematical Models

### CGMY Model Components
cpp

VectorXcd char_CGMY(const VectorXcd& u, double C, double G, double M, double Y)

复制
- Implements the characteristic function for the CGMY process
- Parameters: Complex vector `u`, CGMY parameters (C, G, M, Y)
- Returns: Characteristic function values
cpp

VectorXcd char_CGMY_neg(const VectorXcd& u, double C, double G, double Y)

- Characteristic function for negative jumps only in CGMY model

### Heston Model Components
cpp

VectorXcd char_func_Heston(const VectorXcd& u, const VectorXd& para, const VectorXd& v0, double T)


- Computes Heston model characteristic function
- Parameters: Complex vector `u`, model parameters, initial variance `v0`, maturity `T`

## Numerical Methods

### Runge-Kutta Integrators
cpp

void rk4(...)

void rk4_T(...)

void rk4_2(...)

- Multiple implementations of 4th-order Runge-Kutta method
- Specialized versions for:
  - General ODEs
  - Time-dependent characteristic functions
  - Financial applications

### Root Finding & Interpolation
cpp

int bisect_left(const Ref<const VectorXd>& a, const double& x)
- Binary search implementation (bisect_left equivalent)
- Used for efficient interpolation in pricing formulas

## Option Pricing Framework

### SPX Options Pricing
cpp

tuple<VectorXd, VectorXd, VectorXd, VectorXd, double>

spx_options_pricing_formula_cos_CTCL(...)
# CTCM-JD Option Pricing Module (C++)

This module implements a **hybrid stochastic volatility and jump-diffusion model** for joint **SPX–VIX option pricing**, using a combination of:
- **COS Fourier-Cosine expansion** for option valuation,
- **Heston-type stochastic volatility dynamics,**
- **CTCM jump-diffusion process** with non-central chi-squared sampling,
- **Monte Carlo simulation** for conditional variance paths.

The model aims to reproduce the **volatility surface (vol-surf)** of both SPX and VIX options under a consistent joint dynamic.

---

## 📁 File Overview

This code contains the following key functional areas:

| Section | Purpose |
|----------|----------|
| [1. Utility Functions](#1-utility-functions) | Basic tools such as vector sampling, interpolation, and Bessel/Gamma/Possion draws. |
| [2. COS Expansion Methods](#2-cos-expansion-methods) | Computes Fourier-Cosine coefficients and characteristic functions for option pricing. |
| [3. Conditional Variance Simulation](#3-conditional-variance-simulation) | Generates variance paths under the CTCM process. |
| [4. Heston Characteristic Functions](#4-heston-characteristic-functions) | Analytical solutions for characteristic functions in Heston-type models. |
| [5. Composite Spot Dynamics](#5-composite-spot-dynamics) | Combines jump-diffusion (JD) and Heston volatility components to approximate VIX process. |
| [6. Pricing Formula](#6-pricing-formula) | Core simulation and valuation engine for VIX futures and options. |
| [7. Error Evaluation](#7-error-evaluation) | Calibration objective functions for optimization. |

---

## 1. Utility Functions

### `bisect_left(const VectorXd& a, double x)`
Performs a **binary search** to find the index of the first element in ascending vector `a` that is not less than `x`.  
Equivalent to Python's `bisect_left`.

**Usage:** Used in sampling and numerical interpolation.

---

### `calculate_x_simu(const VectorXd& rand, const MatrixXd& F, const MatrixXd& x)`
Generates simulated samples from a given cumulative distribution `F`.  
Each random number `rand(i)` is mapped to the corresponding segment in `F` and linearly interpolated to produce the simulated variable.

**Inputs:**
- `rand`: random uniform samples `[0,1]`
- `F`: cumulative probability matrix
- `x`: quantile grid

**Output:** A `VectorXd` of simulated values.

---

### `besseli_ratio(double v, double z)`  
Computes the ratio \( I_{v+1}(z) / I_v(z) \) of modified Bessel functions, required for CIR or Bessel-type processes.

---

### `sample_bessel(double v, double z)`  
Samples from a **Bessel distribution** using the above ratio and normal approximation.  
Used to generate paths of conditional variance under CTCM or CIR processes.

---

## 2. COS Expansion Methods

### `calculate_Vk_spx(double L, double A, int N)`
Computes the **Fourier-Cosine coefficients \( V_k \)** for a given payoff function, required by the COS pricing method.

### `calculate_chi(double k, double a, double b)`
Computes the cosine integral part of the Fourier projection.

### `calculate_psi(double k, double a, double b)`
Computes the sine integral part of the Fourier projection.

---

## 3. Conditional Variance Simulation

### `get_quants(const VectorXd& para, double T)`
Derives **quantitative parameters** for the conditional distribution of volatility under a **CTCM** or **CIR** process.

**Outputs:**
- `quant`: vector of quantile levels,
- `gamma_n`, `lambda_n`: shape and non-centrality parameters for chi-square approximation.

---

### `get_conditional_V(tuple<VectorXd, VectorXd, VectorXd> quants, double v0, const VectorXd& v_T, double T)`
Generates the **conditional volatility path \( V_T \)** given the initial variance \( v_0 \) and terminal variance \( v_T \), using a Gamma–Poisson–Bessel decomposition.

---

### `get_X1`, `get_X2`, `get_X3`
Helper components for `get_conditional_V`, producing:
- `X1`: Poisson-Gamma mixture term,
- `X2`: central Gamma term,
- `X3`: Bessel correction term.

---

## 4. Heston Characteristic Functions

### `char_func_Heston(const VectorXcd& u, const VectorXd& para, const VectorXd& v0, double T)`
Computes the **Heston model characteristic function** for a complex vector `u`.

\[
\phi(u) = \exp(A(u,T) + B(u,T)v_0)
\]

Used for Fourier-based pricing and VIX dynamics.

---

### `char_func_Heston_number(const complex<double>& u, const VectorXd& para, const VectorXd& v0, double T)`
Same as above but for a single complex input \( u \).  
Used in analytical components like `spot_Composite_JD`.

---

## 5. Composite Spot Dynamics

### `spot_Composite_JD(const VectorXd& para, const VectorXd& v)`
Builds the **jump-diffusion (JD) component** of the spot variance model, capturing the effect of CGMY-type jumps.

**Outputs:**
- \( a(v), b, c(v) \): coefficients of linearized VIX approximation
\[
VIX = a(v) \cdot u_T + b \cdot v_T + c(v)
\]

---

### `spot_Composite_Heston(const VectorXd& para, const VectorXd& v)`
Analogous to `spot_Composite_JD`, but for the **pure Heston** stochastic volatility component.

---

## 6. Pricing Formula

### `pricing_CTCM_JD(const list<VectorXd>& TK_V, const VectorXd& vol_VIX, const VectorXd& futures, const VectorXd& para)`
The **core simulation and pricing routine**.

#### Key Steps:
1. Loop over maturities `t` in `TK_V`.
2. For each:
   - Compute variance parameters via `get_quants()`.
   - Sample variance paths `v_T`, `V_T`, and auxiliary processes `u_T`, `u2_T`.
   - Compute coefficients using `spot_Composite_JD` and `spot_Composite_Heston`.
   - Combine both processes:
     \[
     VIX = (a u_T + b v_T + c) + (a_2 u_{2T} + b_2 v_T + c_2)
     \]
   - Derive VIX futures price:
     \[
     F = 100 \times \mathbb{E}[\sqrt{VIX}]
     \]
   - Compute option prices via Monte Carlo simulation:
     \[
     C(K) = \mathbb{E}[\max(VIX / F - K/F, 0)]
     \]
   - Infer implied volatility via `implied_volatility`.

#### Outputs:
- `vol_surf`: modeled implied volatility surface of VIX options.
- `VIX_model`: option prices.
- `mean_fut_error`: mean absolute deviation between model and futures data.

---

## 7. Error Evaluation

### `error_function(...)`
Computes the **scalar calibration loss function** used to fit parameters to observed market data.

**Procedure:**
1. Construct full parameter vector.
2. Call `spx_options_pricing_formula_cos_CTCL(...)`.
3. Compute RMSE and relative errors for both SPX and VIX implied volatilities.

**Output:** single scalar objective value (used in optimization).

---

### `error_function2(...)`
Variant of `error_function` returning **a detailed vector of calibration diagnostics**, including:
- errors for SPX/VIX,
- absolute, relative, and mean errors,
- calibration date,
- parameter values.

Used for logging and convergence analysis.

---

## ⚙️ Dependencies

- **Eigen**: matrix and vector operations  
- **Boost::math**: root solving and special functions  
- **Boost::random**: non-central chi-squared, Poisson, Gamma distributions  
- **csv2**: CSV file reading for external precomputed simulation data  
- **C++17 STL**: tuple, list, random engines, complex numbers  

---

## 🧩 Typical Workflow

1. **Parameter Initialization:** set `para` vector (model parameters).
2. **Data Loading:** read option maturities `TK_S`, `TK_V`, and implied vol surfaces.
3. **Run Pricing:**  
   ```cpp
   auto [vol_surf, VIX_model, fut_error] = pricing_CTCM_JD(TK_V, vol_VIX, futures, para);

