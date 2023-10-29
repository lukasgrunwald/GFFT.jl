#=
Experimantally determined crossover value from full formulas to taylor expansion
=#
using GFFT
using MyPlot

function optimize_eps(eps_array, ht, hw; N, r, a, b)
    # N, r = 2^14, 1
    Nexp = log(2, N) |> Int
    dt = (b - a) / N
    t = a .+ collect(0:1:N) * dt # real t to use
    w = 2π / (r * (b - a)) * (-r*N/2:1:r*N/2-1) |> collect
    htj = ht.(t)

    maxeps = Float64[]

    for eps in eps_array
        hwj = zeros(ComplexF64, r * N + 1)
        Offt = gfft_data(+1; N=N, r=r, a=a, b=b, eps)
        gfft!(complex(hwj), complex(htj); param=Offt, Isort=true, Osort=true)

        temp = maximum(x->isnan(x) ? -Inf : x, @views @. abs(hwj[1:end-1] - hw(w)))
        push!(maxeps, temp)
    end
    return maxeps
end

# For Polynomials a, b = -5, 13
ht1(t) = t
hw1(w) = 2exp(4im * w) / w^2 * (-9im * w * cos(9w) + (im + 4w) * sin(9w))

ht2(t) = t^2
hw2(w) = exp(-5im * w) / w^3 * (-2im + 5 * (2 + 5im * w) * w + exp(18im * w) *
                                                            (2im + 13 * (2 - 13im * w) * w))

ht3(t) = t^3
hw3(w) = 2exp(4im * w) / w^4 * (27w * (2im + (8 - 43im * w) * w) * cos(9w) +
                               (-6im + w * (-24 + w * (291im + 1036 * w))) * sin(9w))

ht_sinh(t) = sinh(t)
hw_sinh(w) = exp(-3im * w) * (-cosh(3) - im * w * sinh(3) +
                        exp(-8im * w) * (cosh(5) - im * w * sinh(5))) / (1 + w^2)

ht_exp(t) = exp(-t^2 / 2)
hw_exp(w) = √(2π) * exp(-w^2 / 2)

eps_array = 0.0025:0.0025:0.2 |> collect
maxeps1 = optimize_eps(eps_array, ht1, hw1; N=2^16, r=1, a=-5, b=13)
maxeps2 = optimize_eps(eps_array, ht2, hw2; N=2^16, r=1, a=-5, b=13)
maxeps3 = optimize_eps(eps_array, ht3, hw3; N=2^16, r=1, a=-5, b=13)

maxeps1_18 = optimize_eps(eps_array, ht1, hw1; N=2^18, r=1, a=-5, b=13)
maxeps2_18 = optimize_eps(eps_array, ht2, hw2; N=2^18, r=1, a=-5, b=13)
maxeps3_18 = optimize_eps(eps_array, ht3, hw3; N=2^18, r=1, a=-5, b=13)

maxeps_sh = optimize_eps(eps_array, ht_sinh, hw_sinh; N=2^16, r=1, a=-3, b=5)
maxeps_sh18 = optimize_eps(eps_array, ht_sinh, hw_sinh; N=2^18, r=1, a=-3, b=5)

maxeps_exp = optimize_eps(eps_array, ht_exp, hw_exp; N=2^16, r=1, a=-100, b=100)
maxeps_exp18 = optimize_eps(eps_array, ht_exp, hw_exp; N=2^18, r=1, a=-100, b=100)

##
plt.yscale("log")
b1, = plt.plot(eps_array, maxeps1; label="1")
b2, = plt.plot(eps_array, maxeps2; label="2")
b3, = plt.plot(eps_array, maxeps3; label="3")
b4, = plt.plot(eps_array, maxeps_sh; label="sinh")
b5, = plt.plot(eps_array, maxeps_exp; label="exp")

plt.plot(eps_array, maxeps1_18; linestyle=":", color=b1.get_color())
plt.plot(eps_array, maxeps2_18; linestyle=":", color=b2.get_color())
plt.plot(eps_array, maxeps3_18; linestyle=":", color=b3.get_color())
plt.plot(eps_array, maxeps_sh18; linestyle=":", color=b4.get_color())
plt.plot(eps_array, maxeps_exp18; linestyle=":", color=b5.get_color())

plt.legend()
# plt.savefig(string(@__DIR__)*"/Opt_eps.png")
plt.show()
