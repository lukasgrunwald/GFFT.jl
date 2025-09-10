using GFFT
using Test

# Test functions
function isapprox_eps(x::Complex{T}, y::Complex{T}; factor::Integer=1) where {T<:Real}
    f1 = isapprox(real.(x), real.(y),
                  atol=factor * eps(maximum([(abs ∘ real)(x), (abs ∘ real)(y)])))
    f2 = isapprox(imag.(x), imag.(y),
                  atol=factor * eps(maximum([(abs ∘ imag)(x), (abs ∘ imag)(y)])))
    return all([f1, f2])
end

function isapprox_eps(x::T, y::T; factor::Integer=1) where {T<:Real}
    return isapprox(x, y, atol=factor * eps(maximum([abs(x), abs(y)])))
end

@testset verbose = false "Isort-equiv" begin
    rel_bound = 1e-8

    N, r, a, b = 2^16, 1, -30.3, 30.3
    htj = rand(ComplexF64, N + 1)

    hp1, hp2, hm1, hm2 = (zeros(ComplexF64, N * r + 1) for i in 1:4)
    rel_err(Δa, a, idx) = Δa[idx] / abs(a[idx]) * 100

    for meth in [:spl3, :trap, :riem]
        @testset "$meth" begin
            pfft = GfftData(+1; N=N, r=r, a=a, b=b, method=meth)
            ipfft = GfftData(-1; N=N, r=r, a=a, b=b, method=meth)

            gfft!(hp1, htj; param=pfft, Isort=true, method=meth, boundary=:nearest)
            gfft!(hm1, htj; param=ipfft, Isort=true, method=meth, boundary=:nearest)

            gfft!(hp2, htj |> fftshift_odd;
                param=pfft, Isort=false, method=meth, boundary=:nearest)
            gfft!(hm2, htj |> fftshift_odd;
                param=ipfft, Isort=false, method=meth, boundary=:nearest)

            Δ₊ = @. abs(hp1 - hp2)
            Δ₋ = @. abs(hm1 - hm2)
            idx₊ = argmax(Δ₊)
            idx₋ = argmax(Δ₋)

            @test rel_err(Δ₊, hp1, idx₊) < rel_bound
            @test rel_err(Δ₋, hm1, idx₋) < rel_bound
        end
    end
end

@testset "Equiv gfft's" begin
    sgn, N, a, b, r = +1, 2^16, 1.3, 20.48, 2
    Isort, Osort = true, true
    x = rand(ComplexF64, N + 1)

    h_repeated, h_inplace = (zeros(ComplexF64, r * N + 1) for i = 1:2)

    for method in [:spl3, :trap, :riem], boundary in [:spl3, :nearest]
        Offt = GfftData(+1; N, r, a, b, method)
        gfft!(h_repeated, x; param=Offt, Isort, Osort, method, boundary)
        gfft!(sgn, h_inplace, x; a, b, r, Isort, Osort, method, boundary)
        h_external = gfft(sgn, x; a, b, r, Isort, Osort, method, boundary)

        @test all(isapprox_eps.(h_repeated, h_inplace))
        @test all(isapprox_eps.(h_external, h_repeated))
    end
end
