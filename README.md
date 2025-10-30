Here's a detailed **README** for your code. I’ve structured it to explain the purpose, usage, and mathematical context of each major function and module in your code.

---

# README

## Overview

This C++ code provides a high-performance numerical framework for computing characteristic functions, Runge-Kutta-based solutions of ODEs, and simulations for stochastic volatility and jump models in quantitative finance. It is particularly designed for option pricing and risk management using CGMY processes, leverage-neutral measures, and related processes.

The code extensively uses **Eigen** for linear algebra and complex arithmetic, along with standard C++ random number generation and Boost for normal distribution functions.

---

## Key Components

### 1. **Runge-Kutta ODE Solvers**

The code provides several inline implementations of the 4th-order Runge-Kutta method (`RK4`) to numerically solve ODEs. These are essential for evolving characteristic functions or other time-dependent state variables.

#### `rk4`

```cpp
inline void rk4(void(*f)(const VectorXcd&, const VectorXd&, VectorXcd&), 
                double t0, const VectorXcd& y0, const VectorXd& para, 
                double h, double tf, MatrixXcd& y)
```

* Solves ODE: `dy/dt = f(y, para)` over `[t0, tf]`.
* `y0`: initial state.
* `h`: time step size.
* `y`: output matrix storing solution at each time step.
* Internally uses **4th-order Runge-Kutta**:

  * ( k_1 = f(y_i) )
  * ( k_2 = f(y_i + h/2 * k_1) )
  * ( k_3 = f(y_i + h/2 * k_2) )
  * ( k_4 = f(y_i + h * k_3) )
  * ( y_{i+1} = y_i + h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4) )

#### `rk4_2`

```cpp
inline void rk4_2(double t0, const VectorXcd& y0, const VectorXd& para, 
                  double h, double tf, const VectorXd& u, MatrixXcd& y)
```

* Specialized version using a precomputed characteristic function (`kx = char_L1(para, u)`).
* Useful for solving systems with an external control vector `u`.

#### `rk4_T`

```cpp
inline void rk4_T(void(*f)(const VectorXcd&, const double&, const VectorXcd&, const VectorXd&, const VectorXd&, const VectorXcd&, VectorXcd&),
                  double t0, const VectorXcd& y0, double h, double tf, const VectorXd& u, const VectorXd& para, MatrixXcd& y)
```

* Extended Runge-Kutta solver for **time-dependent characteristic functions** with multiple parameters.
* Handles `psi_u`, `kappa_lnm`, `char_u` for advanced models like CGMY or jump-diffusions.

---

### 2. **Characteristic Functions**

#### CGMY Characteristic Function

```cpp
inline VectorXcd char_CGMY(const VectorXcd& u, double C, double G, double M, double Y)
```

* Computes the characteristic function for a CGMY process.
* Parameters:

  * `C, G, M, Y`: CGMY parameters controlling jump intensity, scale, and tail behavior.
* Used for modeling Lévy processes with jumps.

#### Derived Characteristic Functions

* `char_phi_1`, `char_L1`, `char_CGMY_neg`, `char_J1_neg`, `char_J1_lnm`
* Provide variants of CGMY characteristic exponents for **negative jumps**, **leverage-neutral measure**, or controlled processes.
* Example:

```cpp
inline VectorXcd char_L1(const VectorXd& para, const VectorXcd& u)
```

* Combines positive and negative jump contributions for a generalized process.

---

### 3. **Coefficient Computations (`A` and `B`)**

The code implements routines to compute coefficients for analytical or semi-analytical pricing formulas.

* `coef_B` / `coef_B_T` / `coef_B_T2`
  Computes B-coefficients via Runge-Kutta integration of derivative functions (`derivative_B`, `derivative_B_T`, `derivative_B_T2`).

* `coef_A_alltime` / `coef_A_notalltime` / `coef_A_T` / `coef_A_T2`
  Computes cumulative sums of derivatives to form matrix `A` representing evolution of the system over time.

* `derivative_A` / `derivative_A_T2`
  Defines the time derivative of A-coefficients: often proportional to the current state times a parameter.

---

### 4. **Simulation of Stochastic Processes**

#### Log-Stock Price Simulations

* Functions `get_log_char`, `get_char`, `get_U` generate **Monte Carlo simulations** or characteristic function-based moments for stochastic volatility and jump-diffusion models.
* Uses **Bessel functions** and **gamma distributions** for conditional sampling.
* Key helper functions:

  * `sample_bessel`
  * `besseli_ratio`
  * `calculate_x_simu`
  * `bisect_left` for interpolation.

#### Quantile-Based Approximations

* Functions `get_quants`, `get_conditional_V`, `get_X1`, `get_X2`, `get_X3`
* Approximate stochastic integrals or conditional distributions for variance or jumps using **Poisson and Gamma distributions**.

---

### 5. **Option Pricing Helpers**

* `calculate_Vk_spx`, `calculate_chi`, `calculate_psi`
  Computes Fourier-cosine coefficients for **SPX option pricing**.
* `implied_volatility`
  Computes implied volatility using **root-finding on the Black-Scholes formula** (Boost normal CDF used).

---

### 6. **Random Number Integration**

* CSV-based random number loading:

```cpp
csv2::Reader<...> csv;
if (csv.mmap("rand.csv")) { ... }
```

* Enables **reproducible simulations** by reading pre-generated uniform random numbers.

---

## Dependencies

* **Eigen**: For `MatrixXcd`, `VectorXd`, linear algebra operations.
* **Boost**: For `boost::math::normal_distribution` (CDF and quantile).
* **C++17** or higher (for `auto`, lambda functions, `tie`, `tuple`).
* **csv2** library: For reading CSV files of random numbers.
* **Eigen::Rand**: For gamma and normal sampling.
* Standard `<random>`, `<chrono>`, `<tuple>`, `<cmath>`.

---

## Usage

1. Include the header:

```cpp
#include "cgmy_rk4.hpp"
```

2. Initialize parameters:

```cpp
VectorXd para(15); // Model parameters
VectorXcd y0 = VectorXcd::Zero(n); // Initial state
MatrixXcd y(steps, n);
```

3. Solve ODE using RK4:

```cpp
rk4(my_derivative_function, t0, y0, para, h, tf, y);
```

4. Compute characteristic functions:

```cpp
VectorXcd u = VectorXcd::LinSpaced(n, 0.0, 1.0);
VectorXcd phi = char_CGMY(u, C, G, M, Y);
```

5. Simulate stochastic paths:

```cpp
VectorXd V_T = get_U(para, V_T, h, N_simu);
```

6. Calculate option-related coefficients:

```cpp
VectorXd Vk = calculate_Vk_spx(K, L, A, N);
```

---

## Notes on Numerical Stability

* All RK4 methods assume **small step size `h`** for accuracy.
* Interpolation in `calculate_x_simu` prevents out-of-bounds sampling.
* `besseli_ratio` and `sample_bessel` handle extreme values for Bessel functions to avoid overflow.

---

## References

1. **Carr, Geman, Madan, Yor (2002)** – “The fine structure of asset returns: An empirical investigation.”
2. **CGMY Lévy Process** – Characteristic function formulas.
3. **Higham, N. J. (2002)** – “Accuracy and stability of numerical integration of ODEs.”
4. **Eigen documentation** – Matrix and complex arithmetic.
5. **Boost.Math** – Normal CDF and quantiles for implied volatility calculations.

---

This README should provide a full guide for using, understanding, and extending the code.

---

If you want, I can also **create a diagram showing the full workflow** from **input parameters → RK4 integration → characteristic function → Monte Carlo simulation → option pricing**. This helps new users quickly grasp the data flow.

Do you want me to do that?
