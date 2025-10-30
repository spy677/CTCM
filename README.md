# Time-changed Lévy Process and Option Pricing Framework

This repository implements a numerical framework for **option pricing under time-changed Lévy processes**, following the model of **Carr & Wu (2004, Journal of Financial Economics)**.  
The code simulates asset dynamics, computes characteristic functions, and prices European-style contingent claims via **Fourier-based (FFT) and Monte Carlo** methods.  
Key numerical solvers such as **Runge–Kutta 4 (RK4)** are included for solving ODEs in the Laplace transform of random time.

---

## 🧩 1. Theoretical Background

### 1.1 Time-changed Lévy Processes
A Lévy process \( X_t \) with triplet \( (m, S, P) \) is evaluated at a **stochastic time change** \( T_t \):
\[
Y_t = X_{T_t}, \quad T_t = \int_0^t v(s) ds
\]
where \( v(t) \) is the **instantaneous activity rate**.  
This framework allows:
- **Jumps** (via Lévy process \( X_t \))  
- **Stochastic volatility** (via random time \( T_t \))  
- **Leverage effect** (via correlation between \( X_t \) and \( T_t \))

The characteristic function under the leverage-neutral measure \( Q(y) \) is:
\[
\phi_Y(y) = E^Q[e^{-T_t C_X(y)}] = L_{T_t}^y(C_X(y))
\]
where \( L_{T_t}^y(\cdot) \) denotes the **Laplace transform** of the random time under a complex-valued measure.

---

## ⚙️ 2. Numerical Modules

### 2.1 Lévy Characteristic Exponent
Implements the **Lévy–Khintchine representation**:
\[
C_X(y) = i m y - \frac{1}{2} y^T S y + \int_{\mathbb{R}} (1 - e^{i y x} + i y x 1_{|x|<1}) P(dx)
\]
Supports both finite-activity (Merton, Kou) and infinite-activity (VG, NIG, CGMY) jump models.

### 2.2 Activity Rate Dynamics
The random time process \( T_t \) satisfies:
\[
\frac{dT_t}{dt} = v(t)
\]
For affine or quadratic specifications (e.g., CIR or OU), the **Laplace transform**:
\[
L_{T_t}(\lambda) = \exp[-b(t) v_0 - c(t)]
\]
is obtained by solving coupled ODEs for \( b(t) \) and \( c(t) \) using **Runge–Kutta 4 (RK4)** integration.

#### RK4 Implementation
RK4 numerically integrates:
\[
y' = f(t, y)
\]
via:
\[
y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\]
with intermediate slopes \( k_i = f(t_i, y_i) \).  
This solver ensures stability and high-order accuracy for the **b(t), c(t)** ODE system derived from affine or quadratic term structures (Eq. (27) and (28) in Carr & Wu, 2004).

### 2.3 Monte Carlo Simulation
Monte Carlo paths are generated for:
\[
Y_t = X_{T_t}
\]
using:
1. Simulation of Lévy increments \( X_{\Delta t} \)
2. Random time increments \( T_t \)
3. Optional correlation between \( X_t \) and \( T_t \)

Option payoffs (calls, puts, binaries) are computed under the **risk-neutral measure**.

### 2.4 FFT-based Option Pricing
Following **Carr & Madan (1999)** and **Carr & Wu (2004)**:
\[
G(k) = \frac{1}{2\pi} \int_{i z_i - \infty}^{i z_i + \infty} e^{-i z k} G(z; a,b,W,c) dz
\]
is efficiently evaluated using **Fast Fourier Transform (FFT)** for:
- European Calls and Puts
- Binaries
- Covered Calls

---

## 🧮 3. Implementation Overview
   
