module GFFT

export GfftData, gfft!, gfft
export fftshift, fftshift_odd, fftshift_odd!

using Base.Threads: @threads
using LinearAlgebra: mul!
using FFTW
using UnPack

import Dierckx.Spline1D

# ————————————————————————————— Structs and setup functions ———————————————————————————— #

"""
    GfftData{FType, TI<:Integer, TF<:Real}

Container for gfft!-data that makes repeated application fast.

# Fields
- `method::Symbol`: The method to use for the FFT (e.g., :fft or :ifft).
- `sgn::TI`: The sign used in the expression ĥ(y) = ∫ₐᵇ dx exp(sgn⋅ixy)h(x).

- `N::TI`: The length of the FFT-array (the full array is N+1).
- `Nh::TI`: The index of the middle of the array (non-padded).
- `Nr::TI`: The length of the array with padding.
- `Nrh::TI`: The index of the middle of the array with padding.

- `r::TI`: The padding ratio.
- `a::TF`: The lower boundary of the integral ∫ₐ.
- `b::TF`: The upper boundary of the integral ∫ᵇ.
- `dx::TF`: The discretization spacing of data to be transformed, calculated as (b - a) / N.

- `x::Vector{Float64}`: An array of data values, representing either time (t) or frequency (w).
- `ea::Vector{ComplexF64}`: Corrective phase factors for the lower boundary.
- `eb::Vector{ComplexF64}`: Corrective phase factors for the upper boundary.
- `Mfft::FType`: The "Matrix" that applies the FFT or IFFT operation.
- `pad_container::Vector{ComplexF64}`: A container for the padded array.

- `W::Vector{Float64}`: Attenuation factors used for Spline-FFT.
- `A::Matrix{ComplexF64}`: Boundary corrections for Spline-FFT.
"""
struct GfftData{FType,TI<:Integer,TF<:Real}
    method::Symbol # Method to use for the FFT
    sgn::TI # sign in ĥ(y) = ∫ₐᵇ dx exp(sgn⋅ixy)h(x)

    N::TI # Length of FFT-array (full array is N+1)
    Nh::TI # Index of the middel of the array (non-padded)
    Nr::TI # Length of array with padding
    Nrh::TI # Index of the middel of the array with padding

    r::TI # Padding ratio
    a::TF # Lower boundary of integral ∫ₐ
    b::TF # Upper boundary of integral ∫ᵇ
    dx::TF # Discretization-spacing of data to be transformed (b - a) / N

    x::Vector{Float64} # t or w array of the data
    ea::Vector{ComplexF64} # Corrective phase lower boundary
    eb::Vector{ComplexF64} # Corrective phase upper boundary
    Mfft::FType # "Matrix" that applies fft or ifft
    pad_container::Vector{ComplexF64} # Container for padded array

    # Arrays for spline-fft
    W::Vector{Float64} # Attenuation-factor for Spline-FFT
    A::Matrix{ComplexF64} # boundary corrections for Spline-FFT
end

"""
    GfftData(sgn; N, r, a, b, method)

Constructor for GfftData struct, that contains FFT-data. `sgn` describes which transform is caluclated.

Numerically calculate `ĥ(y) = ∫ₐᵇ dx exp(sgn⋅ixy)h(x)` for many values of y.
The resulting argument-array y of a FFT is: \\
`input ∈ [a, b] -> output = 2π/(r * (b-a)) * fftshift(-N*r/2, ..., N*r/2-1)`

# Arguments
- `sgn::Integer:` sgn = ±1 `+1: fft [f(t) -> f̂(ω)] and -1: ifft [f̂(ω) -> f(t)]`
- `N::Integer:` Length of initial-data to perform the FFT on (Array is N+1)
- `r::Integer:` Pading ratio (sinc(x) interpolation of the initial-data)
- `a, b::Float64:` Lower and upper bound of x-array (No fftshift!)
- `method{:spl3 (default), :trap, :riem}:` Method used for the FFT
- `eps::Float64:` Crossover value, at which we switch between the analytical \
 expression and the Taylor expansion
- `return:` GfftData struct
"""
function GfftData(sgn::Integer; N::Integer, r::Integer,
                   a::Real, b::Real, method::Symbol=:spl3, eps::Real=0.11)
    (a >= b) && error("GfftData: Invalid boundaries (a,b)!")
    (sgn != 1 && sgn != -1) && error("GfftData: sgn=$sgn invalid option!")
    (method ∉ [:spl3, :trap, :riem]) && error("GfftData: Invalid method=$(method)!")

    Nr = r * N
    Nh = N ÷ 2
    Nrh = Nr ÷ 2

    x = 2π / (r * (b - a)) * fftshift(-Nr/2:1:Nr/2-1) |> collect
    dx = (b - a) / N

    if method == :spl3
        W, A = generate_fft_weights(x, sgn * dx; eps)
    else # (method == :trap || method == :riem)
        W, A = zeros(Float64, 0), zeros(ComplexF64, 0, 0)
    end

    ea = cis.(x * a * sgn)
    eb = cis.(x * b * sgn)

    pad_container = zeros(ComplexF64, Nr) # Container for padded array

    if sgn == -1
        Mfft = plan_fft(pad_container; flags = FFTW.ESTIMATE, timelimit=Inf) # Plan fft
    else
        Mfft = inv(plan_fft(pad_container; flags = FFTW.ESTIMATE, timelimit=Inf)) # Plan fft
    end
    @. pad_container = 0.0 + 0.0im # Reset everything to 0

    obj = GfftData(method, sgn, N, Nh, Nr, Nrh, r, float(a), float(b), float(dx),
                    x, ea, eb, Mfft, pad_container, W, A)
    return obj
end

"""
    generate_fft_weights(x::Vector{<:AbstractFloat}, delta::Real; eps::Real=5.0e-2)

Generate W and A for the cubic spline-FFT. See numerical-recipies for definition of coeffs
(http://numerical.recipes/book.html).

# Arguments
- `eps::Float64` denotes the crossover value, at which we switch between the analytical expression
    and the Taylor expansion (default experimentally found to optimze error)
- `return:` W, A where A[:, i] is the i-th boundary correction
"""
function generate_fft_weights(x::Vector{<:AbstractFloat}, delta::Real; eps::Real=0.11)

    N = length(x)
    W = Vector{Float64}(undef, N)
    A = Matrix{ComplexF64}(undef, N, 5)
    # a[:, 5] is A(θ) from numerical recipies

    @inbounds @threads for i = 1:N
        th = x[i] * delta

        if (abs(th) < eps) # Use series.

            t = th
            t2 = t * t
            t4 = t2 * t2
            t6 = t4 * t2

            W[i] = 1.0 - (11.0 / 720.0) * t4 + (23.0 / 15120.0) * t6
            A[i, 1] = (-2.0 / 3.0) + t2 / 45.0 + (103.0 / 15120.0) * t4 - (169.0 / 226800.0) * t6 +
                      im * t * (2.0 / 45.0 + (2.0 / 105.0) * t2 - (8.0 / 2835.0) * t4 + (86.0 / 467775.0) * t6)
            A[i, 2] = (7.0 / 24.0) - (7.0 / 180.0) * t2 + (5.0 / 3456.0) * t4 - (7.0 / 259200.0) * t6 +
                      im * t * (7.0 / 72.0 - t2 / 168.0 + (11.0 / 72576.0) * t4 - (13.0 / 5987520.0) * t6)
            A[i, 3] = (-1.0 / 6.0) + t2 / 45.0 - (5.0 / 6048.0) * t4 + t6 / 64800.0 +
                      im * t * (-7.0 / 90.0 + t2 / 210.0 - (11.0 / 90720.0) * t4 + (13.0 / 7484400.0) * t6)
            A[i, 4] = (1.0 / 24.0) - t2 / 180.0 + (5.0 / 24192.0) * t4 - t6 / 259200.0 +
                      im * t * (7.0 / 360.0 - t2 / 840.0 + (11.0 / 362880.0) * t4 - (13.0 / 29937600.0) * t6)
            A[i, 5] = 1 / 3 + 1 / 45 * t2 - 8 / 945 * t4 + 11 / 14175 * t6 - im * imag(A[i, 1])
        else
            # Use trigonometric formulas.
            cth = cos(th)
            sth = sin(th)
            ctth = cth * cth - sth * sth
            stth = 2.0e0 * sth * cth
            th2 = th * th
            th4 = th2 * th2
            tmth2 = 3.0e0 - th2
            spth2 = 6.0e0 + th2
            sth4i = 1.0 / (6.0e0 * th4)
            tth4i = 2.0e0 * sth4i
            W[i] = tth4i * spth2 * (3.0e0 - 4.0e0 * cth + ctth)
            A[i, 1] = sth4i * (-42.0e0 + 5.0e0 * th2 + spth2 * (8.0e0 * cth - ctth)) +
                      im * sth4i * (th * (-12.0e0 + 6.0e0 * th2) + spth2 * stth)
            A[i, 2] = sth4i * (14.0e0 * tmth2 - 7.0e0 * spth2 * cth) +
                      im * sth4i * (30.0e0 * th - 5.0e0 * spth2 * sth)
            A[i, 3] = tth4i * (-4.0e0 * tmth2 + 2.0e0 * spth2 * cth) +
                      im * tth4i * (-12.0e0 * th + 2.0e0 * spth2 * sth)
            A[i, 4] = sth4i * (2.0e0 * tmth2 - spth2 * cth) +
                      im * sth4i * (6.0e0 * th - spth2 * sth)
            A[i, 5] = sth4i * ((-6.0 + 11 * th2) + spth2 * cos(2 * th)) - im * imag(A[i, 1])
        end
    end

    return W, A
end

# ————————————————————————————————— Application of FFT ————————————————————————————————— #
"""
    apply_correction!(sgn, hy, endpts, param, Isort, method)

Apply corrections to hy, for cubic-spline fft (:spl3) or trapezoidal-fft (:trap)correction.
If `Isort=true`, one needs to multiply the data with `ea`.

# Arguments
- `endpts`: Array of the endpoints in input array (len=8 for :spl2)
- `method{:spl3(default), :trap, :riem}:` Method endpoint corrections
- `return`: nothing
"""
function apply_correction!(f::AbstractVector, endpts::AbstractVector, param::GfftData, Isort::Bool,
                   method::Symbol)
    @unpack sgn, a, b, r, dx = param
    @unpack W, A, ea, eb = param

    Δ = sgn == +1 ? r*(b-a) : dx # Using if statement leads to spurious allocations

    if method == :spl3
        if Isort # Multiply data with ea
            @inbounds @threads for i in eachindex(f)
                f[i] = Δ * W[i] * f[i] * ea[i] +
                       (dx * ea[i] *
                        ((A[i, 1]) * endpts[1] + (A[i, 2]) * endpts[2] +
                         (A[i, 3]) * endpts[3] + (A[i, 4]) * endpts[4]) +
                        dx * eb[i] *
                        ((A[i, 5]) * endpts[8] + (A[i, 2]') * endpts[7] +
                         (A[i, 3]') * endpts[6] + (A[i, 4]') * endpts[5]))
            end
        else
            @inbounds @threads for i in eachindex(f)
                f[i] = Δ * W[i] * f[i] +
                       (dx * ea[i] *
                        ((A[i, 1]) * endpts[1] + (A[i, 2]) * endpts[2] +
                         (A[i, 3]) * endpts[3] + (A[i, 4]) * endpts[4]) +
                        dx * eb[i] *
                        ((A[i, 5]) * endpts[8] + (A[i, 2]') * endpts[7] +
                         (A[i, 3]') * endpts[6] + (A[i, 4]') * endpts[5]))
            end
        end
    elseif method == :trap
        if Isort
            @inbounds @threads for i in eachindex(f)
                f[i] = Δ * ea[i] * f[i] - dx / 2 * (ea[i] * endpts[1] - eb[i] * endpts[2])
            end
        else
            @inbounds @threads for i in eachindex(f)
                f[i] = Δ * f[i] - dx / 2 * (ea[i] * endpts[1] - eb[i] * endpts[2])
            end
        end
    elseif method == :riem
        if Isort
            @inbounds @threads for i in eachindex(f)
                f[i] = Δ * ea[i] * f[i]
            end
        else
            @inbounds @threads for i in eachindex(f)
                f[i] = Δ * f[i]
            end
        end
    end

    return nothing
end

"""
    gfft!(hy, hx; param::GfftData, kwargs...)
    gfft!(sgn, hy, hx; a, b, r, kwargs...)

Numerically calculate ``ĥ(y) = ∫ₐᵇ dx exp(sgn⋅ixy)h(x)`` for many values of y.

Resulting y(=t, ω) vectors will have the form
``x ∈ [a, b] -> y = 2π/(r * (b-a)) * (-N*r/2, ..., N*r/2)`` (Assuming `Isort=true`). \\
Note: Isort, Osort decides wether input/output are ordered or in
the "FFT-ordering" `(0, ..., x_{N/2-1}, x_{-N/2}, ..., x_{-1}, x_{N/2})``

# Arguments
In iterative version (provides GfftData) only:
- `sgn::Int:` sgn = ±1 in ``h(y) = ∫ₐᵇ dx exp(sgn⋅ixy)h(x)`` (+1: fft (t->ω) and -1: ifft (ω->t))
- `param::GfftData` Struct generated with GfftData

In standalone-version only:
- `r:` Pading ratio (sinc(x) interpolation of the data)
- `a, b:` Lower and upper bound of w array (No fftshift!)

General:
- `hx:` h(x)-data (x=a+dx*j with j=0,...,N or FFT-ordering) `(len=N+1)`
- `hy:` ĥ(y) container
- `Isort/Osort:` If true, Input/Output in physical ordering. \
- `boundary{:nearest (default), :spl3, :matsubara}:` Decide method for extrapolation. \
    `:nearest` approximates `h(N/2)≈h(N/2-1)` while `:spl3` uses k=3 extrapolation. Matsubara
    uses `:spl3` for the final point (different because of ordering)
- `method{:spl3, :trap, :riem}:` Method to use for calculating FT
 (usually set when generating GfftData).

- `return:` nothing
"""
function gfft!(hy::AbstractVector{ComplexF64}, hx::AbstractVector{ComplexF64};
               param::GfftData,
               Isort::Bool = true, Osort::Bool = true,
               boundary::Symbol = :nearest, method::Symbol = param.method)
    @unpack N, Nh, Nr, Nrh, r = param
    @unpack a, b, x = param
    @unpack sgn, Mfft = param

    # Assertions
    (boundary ∉ [:spl3, :nearest, :matsubara]) && error("gfft!: Invalid endpoint option!")
    (method == :spl3 && length(param.A) < N) && error("gfft!: A, W not generated!")
    (length(hx) != N + 1) && error("gfft!: hx has invalid length!")
    (length(hy) != Nr + 1) && error("gfft!: hy has invalid length!")
    (!Isort && r != 1) && error("gfft!: No padding if not Isort ordering")
    (!Isort && a != -b) && error("gfft!: Non symmetric interval without fftshift")

    if r > 1 # 0-padding the array
        hpad = param.pad_container
        hpad[1:N] .= @view hx[1:N]
    else
        hpad = @view hx[1:N]
    end

    # hy[1:Nr] .= (Mfft * hpad)
    mul!((@view hy[1:Nr]), Mfft, hpad) # Apply FFT

    # Extract array of boundary corrections
    if method == :spl3
        if Isort # Indices are ordered
            hboundary = @views [hx[1:4]; hx[(N - 2):(N + 1)]]
        else
            hboundary = zeros(ComplexF64, 8)
            hboundary[1:4] .= @view hx[(Nh + 1):(Nh + 4)]
            hboundary[5:7] .= @view hx[(Nh - 2):Nh]
            hboundary[8] = hx[N + 1]
        end

    else # Trapz or Riemann
        if Isort # Indices ordered
            hboundary = [hx[1]; hx[N + 1]]
        else
            hboundary = [hx[Nh + 1]; hx[N + 1]]
        end
    end

    apply_correction!((@view hy[1:Nr]), hboundary, param, Isort, method)

    # Calculate endpoint that is left out by fft procedure
    if boundary == :nearest # Approximate by previous value
        h_end = hy[Nrh]
    elseif boundary === :matsubara # f(τ) = ∑_{iω} transform
        # Need to extrapolate to τ = β. param::gfft has b = π * N / β
        # hy is already correctly ordred in imaginary time
        # Since the x-array is generated as [-N/2, ..., N/2-1] and not as [0, ..., N-1]
        # we need to explicitly generate a τ array
        β = π * N / b
        τ_boundary = β / Nr * ((Nr - 5):(Nr - 1))

        h_re = Spline1D(τ_boundary, real.(@view hy[(Nr - 4):Nr]);
                       k = 3, bc = "extrapolate")(β)
        h_im = Spline1D(τ_boundary, imag.(@view hy[(Nr - 4):Nr]);
                        k = 3, bc = "extrapolate")(β)
        h_end = complex(h_re, h_im)

    else # Extrapolate with order-3 spline
        h_re = Spline1D((@view x[(Nrh - 4):Nrh]), real.(@view hy[(Nrh - 4):Nrh]);
                        k = 3, bc = "extrapolate")(-x[Nrh + 1])
        h_im = Spline1D((@view x[(Nrh - 4):Nrh]), imag.(@view hy[(Nrh - 4):Nrh]);
                        k = 3, bc = "extrapolate")(-x[Nrh + 1])
        h_end = complex(h_re, h_im)
    end

    if Osort # Order-Array - Allocates memeory
        hy[1:Nr] .= fftshift(@view hy[1:Nr])
    end
    hy[Nr + 1] = h_end

    return nothing
end

function gfft!(sgn::Integer, hy::AbstractVector{ComplexF64}, hx::AbstractVector{ComplexF64};
               a::Real, b::Real, r::Integer = 1,
               Isort::Bool = true, Osort::Bool = true,
               method = :spl3, boundary::Symbol = :nearest)
    N = length(hx) - 1
    @assert iseven(N)

    param = GfftData(sgn; N, r, a, b, method)
    gfft!(hy, hx; param, Isort, Osort, boundary)

    return nothing
end

"""
    gfft(sgn, hx; a, b, r=1, kwargs...)

Numerically calculate ``ĥ(y) = ∫ₐᵇ dx exp(sgn⋅ixy)h(x)`` for many values of y. Allocates new
array for hy-output. Returns Vector of FFT.

Resulting y(=t, ω) vectors will have the form
``x ∈ [a, b] -> y = 2π/(r * (b-a)) * (-N*r/2, ..., N*r/2)`` (Assuming `Isort=true`). \\
Note: Isort, Osort decides wether input/output are ordered or in
the "FFT-ordering" `(0, ..., x_{N/2-1}, x_{-N/2}, ..., x_{-1}, x_{N/2})``

# Arguments
In iterative version (provides GfftData) only:
- `sgn::Int:` sgn = ±1 in ``h(y) = ∫ₐᵇ dx exp(sgn⋅ixy)h(x)`` (+1: fft (t->ω) and -1: ifft (ω->t))
- `r:` Pading ratio (sinc(x) interpolation of the data)
- `a, b:` Lower and upper bound of w array (No fftshift!)
- `hx:` h(x)-data (x=a+dx*j with j=0,...,N or FFT-ordering) `(len=N+1)`
- `Isort/Osort:` If true, Input/Output in physical ordering. \
- `boundary{:nearest (default), :spl3, :matsubara}:` Decide method for extrapolation. \
    `:nearest` approximates `h(N/2)≈h(N/2-1)` while `:spl3` uses k=3 extrapolation. Matsubara
    uses `:spl3` for the final point (different because of ordering)
- `method{:spl3, :trap, :riem}:` Method to use for calculating FT
 (usually set when generating GfftData).

- `return:` nothing
"""
function gfft(sgn::Integer, hx::AbstractVector{ComplexF64};
              a::Real, b::Real, r::Integer = 1,
              Isort::Bool = true, Osort::Bool = true,
              method::Symbol = :spl3, boundary::Symbol = :nearest)
    N = length(hx) - 1
    @assert iseven(N)
    hy = Vector{ComplexF64}(undef, r * N + 1) # Preallocate array

    gfft!(sgn, hy, hx; a, b, r, Isort, Osort, method, boundary)

    return hy
end

function gfft(sgn, times, hx; r = 1, method = :spl3, boundary = :nearest)
    # gfft! Only implemented for odd inputs!
    flag = isodd(length(hx))
    _hx = flag ? hx : @view hx[1:end-1]
    _times = flag ? times : @view times[1:end-1]

    N = length(_hx) - 1
    hy = Vector{ComplexF64}(undef, r * N + 1)

    a, b = _times[1], _times[end]
    y = 2π / (r * (b - a)) * (-r*N/2:1:r*N/2) |> collect
    gfft!(sgn, hy, complex.(_hx); a, b, r, Isort = true, Osort = true, method, boundary)

    return y, hy
end

# —————————————————————————————————— Helper functions —————————————————————————————————— #
"""
    fftshift_odd(x::AbstractArray)

Copy FFT-shift of `x` (odd-array) which has index-structure `[fftshift(-N/2:1:N/2-1); N/2]`,
i.e. while keeping the last index fixed.
"""
function fftshift_odd(x::AbstractArray)
    N = length(x)-1
    @assert iseven(N)

    return [fftshift(x[1:N]); x[N+1]]
end

"""
    fftshift_odd!(x::AbstractArray)

Inplace FFT-shift of `x` (odd-array) which has index-structure `[fftshift(-N/2:1:N/2-1); N/2]`.

These arrays apear e.g. when using the [`gfft`](@ref) for the iterative solution. This function
still allocates, but modifies the underlying x.
"""
function fftshift_odd!(x::AbstractArray)
    N = length(x)-1
    @assert iseven(N)

    x[1:N] .= fftshift(x[1:N])

    return nothing
end

function __init__()
    # Switch to mkl backend if not already done before!
    # if FFTW.get_provider() != "mkl"
    #     FFTW.set_provider!("mkl")
    # end
end

end
