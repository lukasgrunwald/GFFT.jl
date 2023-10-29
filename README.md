# GFFT

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lukasgrunwald.github.io/GFFT.jl/dev/)
[![Build Status](https://github.com/lukasgrunwald/GFFT.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/lukasgrunwald/GFFT.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Numerical evaluation of strongly oscillatory integrals of the form
<!-- $$
\begin{equation}
\hat{f}(y) = \int_a^b \rm{d}x e^{\pm i x \cdot y} f(x).
\end{equation}
$$ -->

![Equation](https://latex.codecogs.com/svg.image?%5Chat%7Bf%7D(y)=%5Cint_a%5Eb%5Crm%7Bd%7Dx%5C;e%5E%7B%5Cpm%20i%20x%5Ccdot%20y%7Df(x).)

using an interpolation based approximation introduced in [Numerical Recipies](http://numerical.recipes/). With an appropriate truncation, this can in particular be applied to Fourier transforms, both regular ones, and Matsubara versions. The focus of this module is on an implementation that yields a fast and accurate approximation to continuous Fourier transforms, using in place mutation and FFT-plans. 

The library mainly exports the function `gfft` (and `gfft!`), which allows the evaluation of FFT using different discretization schemes:
- `:riem`: Riemann sum
- `:trap`: Trapezoidal rule
- `:spl3`: Spline-Interpolation scheme (third order), see Numerical recipies. This approach is in particular exact for polynomials up to 3rd-order. 

The basic usage is illustrated in the example below (for more information see documentation, docstrings, and benchmark/test folders)

```julia
using GFFT

f(t) = exp(-t^2)
N = 2^10 # Number of points (should be 2^x)
r = 1 # Zero padding ratio (interpolation of frequency data)
a = -5 # lower bound of integral
b = 13 # Upper bound of integral

# Time array and associated frequency array (both normal ordering)
t = a .+ collect(0:1:N) * (b - a) / N
w = 2π / (r * (b - a)) * (-r*N/2:1:r*N/2) |> collect
ft = complex(f.(t)) # gfft expexts complex input array

# Calculate Fourier transform f̂(w) directly
fw = gfft(+1, ft; a, b)
```

The fidelity of the approach is illustrated in the file `benchmark/gfft_exact_comparison.jl` for various functions. From these examples, the usage of the module should also become clear (both for normal and Matsubara FFT's). For more details see documentation.