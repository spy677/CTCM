# Financial Mathematics Library - Complete Mathematical Framework

## Mathematical Foundation

### Core Stochastic Processes

#### 1. CGMY Jump-Diffusion Process
The model combines a CGMY Lévy process with stochastic volatility:

**Characteristic Function:**
math
\phi_{CGMY}(u) = C\Gamma(-Y)\left[(M - iu)^Y - M^Y + (G + iu)^Y - G^Y\right]


**Negative Jumps Only:**
math
\phi_{CGMY}^{-}(u) = C\Gamma(-Y)G^Y\left[\left(1 + \frac{iu}{G}\right)^Y - 1\right]


#### 2. Heston Stochastic Volatility
The volatility follows a mean-reverting square-root process:
math
dv_t = \kappa(\theta - v_t)dt + \sigma\sqrt{v_t}dW_t^v


**Heston Characteristic Function:**
math
\phi_{Heston}(u) = \exp\left(A(T) + B(T)v_0\right)

where:
math
A(T) = \frac{\kappa\theta}{\sigma^2}\left[(\kappa - \delta)T - 2\log\left(1 - \frac{\delta - \kappa}{2\delta}(1 - e^{-\delta T})\right)\right]

math
B(T) = \frac{2iu(1 - e^{-\delta T})}{\delta + \kappa + (\delta - \kappa)e^{-\delta T}}

math
\delta = \sqrt{\kappa^2 - 2i u \sigma^2}


## Option Pricing Framework

### 1. Fourier Cosine Method (COS)

The COS method approximates the option price using Fourier cosine expansions:

**Call Option Price:**
math
C(t) \approx e^{-rT}\sum_{k=0}^{N-1} V_k \cdot \text{Re}\left[\phi\left(\frac{k\pi}{b-a}\right)e^{-ik\pi\frac{a}{b-a}}\right]


**Coefficients:**
- **V_k**: Fourier cosine coefficients for the payoff function
- **χ and ψ functions**: Trigonometric integrals for coefficient calculation

### 2. Characteristic Function Decomposition

The model decomposes the characteristic function into two factors:

**First Factor (Leverage-Neutral Measure):**
math
\phi_1(u) = \exp\left(A_1(T) + B_1(T)u_0\right)


**Second Factor (Volatility Process):**
math
\phi_2(u) = \exp\left(A_2(T) + B_2(T)u_{20}\right)


**Combined Characteristic Function:**
math
\phi(u) = \phi_1(u) \cdot \phi_2(u)


## Numerical Methods

### 1. Runge-Kutta ODE Solvers

**4th Order RK for ODEs:**
math
y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)

where:
math
k_1 = hf(t_n, y_n)

math
k_2 = hf(t_n + \frac{h}{2}, y_n + \frac{k_1}{2})

math
k_3 = hf(t_n + \frac{h}{2}, y_n + \frac{k_2}{2})

math
k_4 = hf(t_n + h, y_n + k_3)


### 2. Monte Carlo Simulation

**Variance Process Simulation:**
math
V_T = X_1 + X_2 + X_3


**Component Distributions:**
- **X₁**: Poisson-weighted Gamma distribution
- **X₂**: Gamma distribution for stationary component
- **X₃**: Bessel process component

### 3. Bessel Process Sampling

**Bessel Ratio Approximation:**
math
\frac{I_{\nu+1}(z)}{I_\nu(z)} \approx \frac{z}{\nu + \frac{1}{2} + \sqrt{(\nu + \frac{3}{2})^2 + z^2}}


**Normal Approximation:**
math
X \sim \mathcal{N}(\mu, \sigma^2)

where:
math
\mu = \frac{z}{2}\frac{I_{\nu+1}(z)}{I_\nu(z)}

math
\sigma^2 = \frac{z^2}{4}\left(\frac{I_{\nu+2}(z)}{I_\nu(z)} - \left(\frac{I_{\nu+1}(z)}{I_\nu(z)}\right)^2\right) + \mu - \mu^2


## VIX Modeling

### VIX Formula Derivation

**VIX² Calculation:**
math
VIX^2 = \frac{2}{T}\sum_{i=1}^N \frac{\Delta K_i}{K_i^2}e^{rT}Q(K_i) - \frac{1}{T}\left(\frac{F}{K_0} - 1\right)^2


**Model Implementation:**
math
VIX = \sqrt{(a \cdot u_T + b \cdot v_T + c) + (a_2 \cdot u_{2T} + b_2 \cdot v_T + c_2)}


## Implied Volatility Calculation

### Black-Scholes Implied Volatility

**Call Option Formula:**
math
C = S_0N(d_1) - Ke^{-rT}N(d_2)

where:
math
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}

math
d_2 = d_1 - \sigma\sqrt{T}


**Root Finding:**
math
f(\sigma) = BS(\sigma) - C_{\text{market}} = 0


## Error Metrics for Calibration

### 1. Mean Squared Error (MSE)
math
\text{MSE} = \frac{1}{N}\sum_{i=1}^N (\sigma_{\text{model}}^i - \sigma_{\text{market}}^i)^2


### 2. Absolute Error
math
\text{AbsError} = \frac{1}{N}\sum_{i=1}^N \sigma_{\text{model}}^i - \sigma_{\text{market}}^i



### 3. Relative Error
math
\text{RelError} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left(\frac{\sigma_{\text{model}}^i - \sigma_{\text{market}}^i}{\sigma_{\text{market}}^i}\right)^2}


### 4. Composite Error Function
math
\text{TotalError} = w_1\cdot\text{MSE}_{SPX} + w_2\cdot\text{MSE}_{VIX} + w_3\cdot\text{FuturesError}


## Mathematical Components

### 1. Leverage-Neutral Measure Transformation

**Measure Change:**
math
\frac{d\mathbb{Q}^L}{d\mathbb{P}} = \exp\left(\int_0^T \lambda_s dW_s - \frac{1}{2}\int_0^T \lambda_s^2 ds\right)


**Characteristic Exponent Adjustment:**
math
\psi^L(u) = \psi(u) - iu\phi_1'(-i)


### 2. Fourier Inversion Techniques

**Damping Factor:**
math
\hat{C}(\alpha) = e^{\alpha k}C(k)


**Inversion Formula:**
math
C(k) = \frac{e^{-\alpha k}}{\pi}\int_0^\infty e^{-ivk}\hat{C}(v)dv


### 3. Numerical Integration Schemes

**Simpson's Rule:**
math
\int_a^b f(x)dx \approx \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2) + \cdots + f(x_n)\right]


**Adaptive Grid:**
- Fine grid near at-the-money options
- Coarser grid for deep in/out-of-the-money options

## Computational Optimizations

### 1. Vectorized Operations
- **Eigen library** for linear algebra
- **Broadcasting** for batch computations
- **Memory-efficient** matrix operations

### 2. Parallel Processing
- **Multi-threading** for Monte Carlo simulations
- **Parallel ODE solving** for different strikes/maturities

### 3. Caching and Memoization
- **Pre-computed** characteristic functions
- **Cached** numerical integrals
- **Reusable** intermediate results

## Model Parameters

### Core Parameters (15-dimensional):
1. **κ₁**: Mean reversion speed (1st factor)
2. **θ₁**: Long-term mean (1st factor)
3. **σ₁**: Volatility of vol (1st factor)
4. **κ₂**: Mean reversion speed (2nd factor)
5. **θ₂**: Long-term mean (2nd factor)
6. **σ₂**: Volatility of vol (2nd factor)
7. **C**: CGMY intensity parameter
8. **G**: CGMY decay parameter (right tail)
9. **M**: CGMY decay parameter (left tail)
10. **Y**: CGMY stability parameter
11. **ρ**: Leverage correlation
12. **u₀**: Initial value (1st factor)
13. **u₂₀**: Initial value (2nd factor)
14. **v₀**: Initial variance
15. **λ**: Jump intensity

This mathematical framework provides a comprehensive foundation for pricing SPX and VIX options using advanced stochastic processes and Fourier-based methods.
