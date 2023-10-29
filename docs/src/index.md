```@meta
CurrentModule = GFFT
```

# GFFT

Documentation for [GFFT](https://github.com/lukasgrunwald/GFFT.jl), that numerically evaluates strongly oscillatory integrals of the form
```math
\begin{equation}
\hat{f}(y) = \int_a^b \rm{d}x e^{\pm i x \cdot y} f(x).
\end{equation}
```

using an interpolation based approximation scheme. With an appropriate truncation, this can in particular be applied to Fourier transforms, both regular ones, and Matsubara versions. The focus of this module is on an implementation that yields a fast and accurate approximation to continuous Fourier transforms, using in place mutation and FFT-plans. 

# Getting Started

The main functionality oft this library is implemented by the [gfft!](@ref gfft) function, that evaluates (1) using an interpolation based approximation scheme, explained in the theory section. Next to the interpolation based scheme we provide other discretization schemes 
- `:riem`: Riemann sum
- `:trap`: Trapezoidal rule
- `:spl3`: Spline-Interpolation scheme (third order), see Numerical recipies. This approach is in particular exact for polynomials up to 3rd-order. 

A minimal working example, in which we numerically calculate the oscillatory integral $\hat{f}(\omega) = \int_{-5}^{13} \text{d}t \; e^{i \omega t} e^{-t^2} \sinh(3t)$, reads

```julia
using GFFT

f(t) = exp(-t^2) * sinh(3 * t)
N = 2^10 # Number of points (should be 2^x)
r = 1 # Zero padding ratio (interpolation of frequency data)
a = -5 # lower bound of integral
b = 13 # Upper bound of integral

# Time array and associated frequency array (both normal ordering)
t = a .+ collect(0:1:N) * (b - a) / N
w = 2π / (r * (b - a)) * (-r*N/2:1:r*N/2) |> collect
ft = complex(f.(t)) # gfft expexts complex input array

# Direct application
fw = gfft(+1, ft; a, b, method = :spl3)
```

Instead of directly calculating the Fourier transform, we can also preallocate the output array and generate FFTW plans, using the in-place interface

```julia
# [...] (Definitions above)
fw2 = Vector{ComplexF64}(undef, N + 1) # container for output
Offt = gfft_data(+1; N=N, r = 1, a=a, b=b, method = :spl3) # gfft_data struct, containig FFT plan etc.
gfft!(fw2, ft; param=Offt)

fw2 == fw # true
```

Further usage and options is explained in the docstrings and illustrated further in the `benchmark` and `test`.

# Index
```@index
```
