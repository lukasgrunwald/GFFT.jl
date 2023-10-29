
# Theoretical Background
This is part of an appendix of my master thesis in theoretical condensed matter physics, that explains the basic algorithm, and it's application to approximate continuous Fourier transforms. For an alternative and more complete description see the Fourier transform chapter of [Numerical Recipies](http://numerical.recipes/).

# The DFT Algorithm

The DFT (and IDFT) maps a sequence $\{s[n] \mid n \in [ 0, 1, \dots N-1] \}$ onto a new sequence $\{\hat{s}[k] \mid k \in [0, 1, \dots N-1] \}$ by implementing the sums
```math
\begin{align}
    \text{ DFT:} \quad \hat{s}[k] &= \sum_{n=0}^{N-1} e^{-i\frac{2\pi}{N} n \cdot k} \; s[n], \\
    \text{IDFT:} \quad s[n] &= \frac{1}{N}\sum_{n=0}^{N-1} e^{+i\frac{2\pi}{N} n \cdot k} \; \hat{s}[k].
\end{align}
```
Because of the Euler identity $e^{2\pi i k} = 1 \; \forall k \in \mathbb{Z}$, a DFT always acts on the periodically continued signal, i.e. it is implied that $\hat{s}[k+N]=\hat{s}[k]$ and in turn $s[n+N]=s[n]$, which can be seen by inspection of (1) and (2). The Euler identity also leads to the Nyquist theorem: The "frequencies" $k \in [N/2, \dots ,N-1]$ are the same as $k \in [-N/2, \dots, -1]$, meaning that the DFT does not actually calculate frequencies with $k \geq N/2$, but rather, it always generates an (asymmetric) frequency-array around the origin, i.e. $k \in [0, \dots, N/2-1, -N/2, \dots, -1]$. An array in this ordering is called FFT-ordered. This array can be ordered in $\mathcal{O}(N)$ by using a circular index shift (circshift).

A naive implementation of the DFT has $\mathcal{O}(N^2)$ complexity, but with a clever trick which was popularized by the [Cooley-Tukey FFT algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm), this can be improved to $\mathcal{O}(N \log(N))$ if $N = 2^m, \; m \in \mathbb{N}$. (Generalizations are possible to the case where $N$ is a product of low prime numbers, but these are extremely complicated.)
The main insight comes from splitting of the DFT (or IDFT) into even and odd terms
```math
\begin{align}
    \hat{s}[k] &= \sum_{n=0}^{N/2-1} e^{-i\frac{2\pi}{N/2} n \cdot k} \; s[2n] +
    e^{-i\frac{2\pi}{N}k}\sum_{n=0}^{N/2-1} e^{-i\frac{2\pi}{N/2} n \cdot k} \; s[2n+1] \\
    &= x_{even}[k] + e^{-i\frac{2\pi}{N}k} x_{odd}[k],
\end{align}
```
followed by the realization that $x_{odd}$ and $x_{even}$ are again DFT's with length $N/2$ that can be calculated recursively. The recursion depth being $x=\log_2(N)$, $\hat{s}[k]$ can be calculated in $\mathcal{O}\big( N^2/2^x + xN \big) \in \mathcal{O}(N \log(N))$, where the first term describes the recursion and the second term the recombination step.

Next to this basic idea, a plethora of tricks (cache locality, SIMD-instruction etc.) are used in modern DFT implementations leading to further $\mathcal{O}(N)$-speedups. This is why in this thesis we use the Julia bindings to the [FFTW ("Fastest Fourier Transform in the West")](https://www.fftw.org/) library for calculating DFT's, instead of relying on a custom version.

# Quadratures for Strongly Oscillatory Integrals

As a proxy for the continuous FT, we next discuss how to numerically calculate strongly oscillatory integrals for $f: \mathbb{R} \to \mathbb{C}, f \in L^2 $ of the form
```math
\begin{equation}
    \hat{f}(\omega) = \int_a^b \text{d} t \;  e^{i\omega t} f(t),
\end{equation}
```
where we are usually interested in many values of $\omega$. A naive, but illustrative approach is to choose a uniform discretization of the time axis $t=a + j\Delta, j\in[0, N]$ with spacing $\Delta = N^{-1}(b-a)$ and function values $f_j \equiv f(t_j)$, and to evaluate the integral using the trapezoidal-rule (or more generally Gregory Integration).[^1]

The sum in the resulting expression
```math
\begin{equation}
    \hat{f}(\omega) \approx \Delta e^{i\omega a} \sum_{n=0}^{N-1} f_j e^{i\omega \Delta j}
    - \frac{\Delta}{2} \Big( e^{i\omega a}h_0 - e^{i\omega b}h_N \Big)
\end{equation}
```
can be efficiently calculated using the DFT, when choosing $\omega_j = 2\pi (b-a)^{-1}  [-\frac{N}{2}, \dots, \frac{N}{2}-1]_j$. As noted before, the frequency array resulting from a DFT is asymmetric around the origin, $\omega_{-N/2}$ being the smallest and $\omega_{N/2-1}$ the largest value. To generate a symmetric array, which is usually needed for further calculations, we obtain $\omega_{N/2}$ using a third-order spline extrapolation, using the last three frequency points. In cases where the data is "noisy", i.e. where we have residual fluctuations from the DFT we sometimes also use the previous frequency point as an approximation for $\omega_{N/2}$.

Notice that the grid-spacing is $\delta \omega \sim (b-a)^{-1}$ and $\omega\_{max} \sim N$, meaning that while the latter is determined by our discretization, the first is fixed by the integral itself. It is possible to decrease $\delta \omega$ by using zero-padding, but this only corresponds to a sinc-interpolation of the original $\hat{f}(\omega)$ and thus does not contain new information, so that the frequency resolution in this approach is fundamentally limited by $\delta \omega = 2\pi (b-a)^{-1}$. [^2] Nonetheless, zero-padding is very useful, as it represents a cheap way to obtain a high order interpolation of $\hat{f}(\omega)$ which is e.g. needed when $\hat{f}(\omega)$ is to be used in further analysis.

While (6) can give accurate results if $\lim_{t \to a,b} f(t) \approx 0$ fast enough (e.g. exponentially fast), in general it will be vastly inaccurate. The issue is that due to the oscillatory nature of the integral, the small parameter appearing in error terms is not $\Delta / (b-a)$, but rather $\omega \Delta$, which can become as large as $\pi$. As a result (6) becomes systematically inaccurate as $\omega$ increases.

A more sophisticated approach can be formulated by approximating $f(t)$ by its polynomial interpolation of order $s$ with mesh-points $f_j \equiv f(t_j)$, with $t_j$ as defined above. This can be viewed as approximating $f(t)$ by a sum of kernel functions which only depend on the interpolation scheme used
```math
\begin{equation}
    f(t) \approx
     \sum_{j=0}^N f_j \psi \bigg(\frac{t-t_j}{\Delta} \bigg) 
    + \sum_{j \in \text{endpts}} f_j \phi_j \bigg(\frac{t-t_j}{\Delta}\bigg).
\end{equation}
```
Here $\psi(s)$ are the kernel function of an interior point (Lagrange-polynomials of order $s$) and $\phi_j(s)$ are the boundary corrections, which are needed since close to $a,b$ non-centered interpolation formulas have to be used. The number of endpoint functions is equal to $2s$: $s$ functions at every side of the interval. Inserting (7) into (5) and exchanging sum and integral, one can rewrite the expression as
```math
\begin{equation}
    \hat{f}(\omega_n) = \Delta e^{i\omega_n a} W(\theta_n) 
    \sum_{j=0}^{N-1} f_j e^{i\theta_n j}
    + \Delta \sum_{k=0}^{s-1} \Big( f_k \alpha_k(\theta_n) e^{i\omega_n a} 
    + f_{N-k} \tilde{\alpha}_{k}(\theta_n) e^{i\omega_n b} \Big).
\end{equation}
```
Here we introduced the shorthand $\theta_n = \omega_n \Delta$, with $\omega_n$ defined such that the DFT can be used for the first sum.[^3]
Further, we defined the two functions (the $\tilde{\alpha}_k$'s can be expressed in terms of $\alpha_k$'s, see below)
```math
\begin{align}
    W(\theta) = \int_{-\infty}^\infty \text{d} s \; e^{is\theta} \psi(s), \qquad
    \alpha_j(\theta) = \int_{-\infty}^{\infty} \text{d} s \; e^{i\theta} \phi_j(s-j).
\end{align}
```
For the cubic ($s=3$) interpolation scheme, the explicit expressions for $W, \alpha, \tilde{\alpha}$ read
```math
\begin{align}
    W(\theta) &= \left(\frac{6 + \theta^2}{3\theta^4}\right)
    \left[ 3 - 4 \cos\theta + \cos 2 \theta \right] \\
    \alpha_0(\theta) &= 
    \frac{(-42 + 5 \theta^2) (6 \theta^2) (8 \cos \theta - \cos 2\theta)}{6\theta^4}
    + i \frac{(-12\theta + 6\theta^3) + (6 + \theta^2) \sin 2\theta}{6\theta^4}, \\
    \alpha_1(\theta) &= \frac{14(3 - \theta^2)- 7(6 + \theta^2) \cos \theta}{6\theta^4} + 
    \frac{30\theta - 5(6 + \theta^2) \sin \theta}{6\theta^4}, \\
    \alpha_2(\theta) &= \frac{-4(3 - \theta^2) + 2(6 + \theta^2) \cos \theta}{3\theta^4} + 
    i \frac{-12\theta + 2(6 + \theta^2) \sin \theta}{3\theta^4}, \\
    \alpha_3(\theta) &= \frac{2(3 - \theta^2) - (6 + \theta^2) \cos \theta}{6 \theta^4} +
    i \frac{6\theta - (6 + \theta^2) \sin \theta}{6 \theta^4}, \\
    \tilde{\alpha}_{0}(\theta) &= \frac{(-6 + 11\theta^2) + (6 \theta^2)\cos 2\theta}{6\theta^4} - i \Im[\alpha_0(\theta)], \\
    \tilde{\alpha}_{k \geq 1}(\theta) &= \alpha_{k}^*(\theta) .
\end{align}
```
These formulas have cancelations to high orders in $\theta$, such that in a numerical implementation their 6th order Taylor-expansion has to be used for $|\theta| < 0.11$, instead of the full expression. The *crossover-value* of $\theta$ is determined experimentally as the value when Taylor expansion and analytical expression give the same result and dependents on details of the programming language and implementation (see `benchmark/optimization_eps.jl`). Note that the asymptotic complexity of (8) is the same as for the naive trapezoidal rule approach (6), i.e. $\mathcal{O}(N \log N)$.

[^1]: Since we want to evaluate the integral at many values of $\omega$ and because we often only know $f(t)$ at fixed, evenly spaced points, standard high-order rules such as Gauss-quadratures are infeasible since they scale as $\mathcal{O}(N^2)$ and often even require interpolation. They would not be significantly more accurate than the trapezoidal rule anyway, as we will shortly explain.

[^2]: This is not an issue for us, since the values of $a,b$ will usually also be numerical parameters (see below).

[^3]: Remarks concerning the $\omega$-grid, made in the context of the naive approach, of course also apply here.

# Numerical Fourier Transforms

Calculating continuous FT's, which are integrals of type (5) with $b=-a=\infty$, viz.
```math
\begin{align}
    f(t) = \int_{-\infty}^\infty \frac{d\omega}{2 \pi} \; e^{-i \omega t} \hat{f}(\omega), \qquad
    \hat{f}(\omega) = \int_{-\infty}^\infty dt \; e^{i \omega t} f(t),
\end{align}
```
is in general very difficult. If the function falls off reasonably quickly at infinity (We found that a decay as $\mathcal{O}(t^{-2},\omega^{-2})$ is usually fast enough.), we can split the integration interval at some $s_{max}$ and neglect the boundary terms $\sim \int_{s_{max}}^{\infty} \dots$, leaving us with an integral that can be performed using (8). The resulting frequency array has the form $\omega_j = \frac{\pi}{s_{max}} [-N/2, \dots, N/2]_j$, so that $\omega_{max} \sim N/s_{max}$ and $\delta \omega \sim s\_{max}^{-1}$. Both parameters can be controlled by our discretization and cutoff. In general, the convergence in $N, s_{max}$ has to be checked on a case-by-case basis.

If the function is not decaying fast enough, as is the case for the free retarted Greens function $G^R(\omega) \stackrel{\omega \to \infty}{\sim} (\omega + i0^+)^{-1} \equiv G_{asy.}^R(\omega)$ we can still use above formalism, given that the asymptotic behavior is known and that the FT of the asymptotic expression (here the FT of $G_{asy.}^R(\omega)$) can be worked out analytically. If this is the case, we numerically calculate the FT of $(G^R - G^R\_{asy.})(\omega)$ and add the analytically obtained $ \hat{G}^R\_{asy.}(t)$ to the result
```math
\begin{equation}
    G^R(t) \approx \int_{-s_{max}}^{s_{max}} \frac{d\omega}{2 \pi} e^{-i \omega t} 
    \underbrace{\big(G^R - G_{asy.}^R\big)(\omega)}_{\in \mathcal{O}(\omega^{-2})}
    + G\_{asy.}\^R(t).
\end{equation}
```
In a numerical setting $0^+$ in $G_{asy.}^R(\omega)$ is replaced by a finite broadening factor $0^+ \to \eta=0.1$, leading to the analytical result $G_{asy.}^R(t) = -i\Theta(t) e^{-\eta t}$.

In addition to continuous FT's appearing in the real time formalisms, we are also working in imaginary time, where the corresponding Matsubara Fourier Transforms (MFT) are defined as 
```math
\begin{align}
    f(\tau) = \frac{1}{\beta} \sum_{\omega_n} e^{-i\omega_n\tau} \hat{f}(i\omega_n),  \qquad
    \hat{f}(i\omega_n) = \int_0^\beta d\tau \; e^{i\omega_n\tau} f(\tau),
\end{align}
```
with inverse temperature $\beta=\frac{1}{T}$ and Matsubara frequencies $\omega_n = 2\pi \beta^{-1} \left( \mathbb{Z} + \zeta/2 \right)_n$. We have $\zeta = 1$ for fermions and $\zeta = 0$ for bosons. Discretizing imaginary time as $\tau_j = \Delta [0, \dots, N]_j$ with $\Delta = \beta / N$, the transform $f(\tau) \to \hat{f}(i \omega_n)$ can be calculated using (7) as
```math
\begin{equation}
    \hat{f}(i\omega_n) =
    \int_0^\beta d\tau \; e^{i \frac{2\pi}{\beta} n \tau} \big[ e^{\zeta \frac{i\pi}{\beta} \tau}  f(\tau) \big],
\end{equation}
```
where the additional phase factor appears because of the definition of Matsubara frequencies.

The transform $\hat{f}(i \omega_n) \to f(\tau)$ is again more tricky because of the infinite boundaries. If the function decays fast enough we can split the Matsubara sum at $\omega_{max} \sim N_{max}$ and neglect the boundary term $\sim \sum_{N_{max+1}}^\infty \dots$, leaving us with a sum that can be calculated using the DFT, after an appropriate circshift. The resulting $\tau$-array of the DFT has the form $\tilde{\tau}_j = \frac{\beta}{N}[-N/2, \dots, N/2]_j$, but because of the periodicity $f(\tau + \beta) = (-1)^{\zeta} f(\tau)$, this is equivalent to $\tau \in [0, \beta]$. Explicitly, we thus implement
```math
\begin{equation}
    f(\tau_j) 
    \approx \frac{1}{\beta} e^{-\zeta \frac{i\pi}{\beta} \tau_j} \sum_{ |n| \leq N_{max}} 
    e^{-i\frac{2\pi}{\beta} n \tau_j} \hat{f}(i\omega_n).
\end{equation}
```
If the $\hat{f}(i \omega)$ is not decaying fast enough, we again use the trick of subtracting and adding the asymptotic contribution. Such a slow decay is seen for $G(i \omega_n) \stackrel{n \to \infty}{\sim}(i \omega_n)^{-1} \equiv G\_{asy.}(i \omega)$, where the analytically calculated transform of the asymptotic reads $G_{asy.}(\tau) = -1/2$ for $\tau \in [0, \beta]$.

In practice, we often successively apply FT's to the same set of functions in an iterative loop. In this case one can eliminate the need of doing a circshift after each iteration, if the integration interval is symmetric around the origin. In this case, applying the DFT to the ordered $f$-array followed by multiplication of $e^{i\omega_n a}$ [see (7)] is equivalent to just applying the DFT to the *FFT-ordered* array.
