#=
Testing numerical implementation of the oscillatory integral against known analytical results.
In particular test against polynomials for which we expect the method to be numerically Extract
(here max. error < 1e-9)
=#
using GFFT
using Test

# Fermi functions
function nf(x::T1, β::T2=1.0) where {T1 <: Number, T2 <: Number}
    return 1 / (exp(x * β) + 1)
end

function nb(x::T1, β::T2=1.0) where {T1 <: Number, T2 <: Number}
    return 1 / (exp(x * β) - 1)
end

maximum_noNaN(x) = maximum(map(x->isnan(x) ? -Inf : x, x))
argmax_noNaN(x) = argmax(map(x->isnan(x) ? -Inf : x, x))

# hw obtained as integral over a, b = -5, 13
ht1(t) = t
hw1(w) = 2exp(4im * w) / w^2 * (-9im * w * cos(9w) + (im + 4w) * sin(9w))
ht2(t) = t^2
hw2(w) = exp(-5im * w) / w^3 * (-2im + 5 * (2 + 5im * w) * w + exp(18im * w) *
                                                            (2im + 13 * (2 - 13im * w) * w))
ht3(t) = t^3
hw3(w) = 2exp(4im * w) / w^4 * (27w * (2im + (8 - 43im * w) * w) * cos(9w) +
                               (-6im + w * (-24 + w * (291im + 1036 * w))) * sin(9w))

@testset verbose = false "ImagTime" begin
    N, a = 2^12, 0.6
    β, μ = 100, 0.3
    Nhalf = Int(N / 2)

    # Matsubara freq.
    n = [fftshift(-N/2:1:N/2-1); N / 2]
    w = 2π / β .* (n .+ 1 / 2) # fermionic
    v = 2π / β .* n # bosonic
    k = collect(0:N)    # Imaginary time discretization
    τ = β / N .* k
    ϕ₊ = cis.(π / N * k) # phase for fermionic fft

    Στ = @. -exp(μ * τ) * 1 / (exp(μ * β) + 1)
    Πτ = @. -1 / 2 * (exp(-τ) * nb(-1, β) - exp(τ) * nb(1, β))
    Σw_ref = @. 1 / (im * w + μ)
    Πw_ref = @. 1 / (v^2 + 1.0)

    Σ, Π = (similar(Σw_ref) for j = 1:2)
    bounds = [6 * 1e-9, 0.008, 0.02]

    for (i, meth) in enumerate([:spl3, :trap, :riem])
        @testset "$meth" begin
            pfft = gfft_data(+1; N=N, r=1, a=0, b=β, method=meth)
            gfft!(Σ, (ϕ₊ .* Στ), param=pfft, Isort=true, Osort=false, boundary=:spl3)
            gfft!(Π, complex(Πτ), param=pfft, Isort=true, Osort=false, boundary=:spl3)

            @test maximum(abs.(Σ .- Σw_ref)) < bounds[i]
            @test maximum(abs.(Π .- Πw_ref)) < bounds[i]
        end

    end
end

@testset "Polynomials" begin
    bound = 1e-9 # Maximum difference between the polynomials
    N, r, a, b = 2^10, 1, -5, 13
    dt = (b - a) / N
    println(dt)
    t = a .+ collect(0:1:N) * dt # real t to use
    w = 2π / (r * (b - a)) * (-r*N/2:1:r*N/2-1) |> collect

    hts = [ht1, ht2, ht3]
    hws = [hw1, hw2, hw3]

    for i in 1:3
        ht = hts[i]
        hw = hws[i]
        htj = ht.(t)
        hw_ref = hw.(w)

        hwj = zeros(ComplexF64, r * N + 1)
        Offt = gfft_data(+1; N=N, r=r, a=a, b=b, method=:spl3)
        gfft!(complex(hwj), complex(htj); param=Offt, Isort=true, Osort=true, method=:spl3, boundary=:spl3)

        δ = @. abs(hwj[1:end-1] .- hw_ref)
        idx_max = argmax_noNaN(δ)
        ϵ = eps(abs.(hw_ref[idx_max]))
        println(δ[idx_max], " - ", ϵ)

        @test δ[idx_max] < bound
    end
end

##
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

# Direct application
fw = gfft(+1, ft; a, b, method = :spl3)

# Preallocated version
fw2 = Vector{ComplexF64}(undef, N + 1) # container for output
Offt = gfft_data(+1; N=N, r = 1, a=a, b=b) # gfft_data struct, containig FFT plan etc.
gfft!(fw2, ft; param=Offt)

fw2 == fw # true
