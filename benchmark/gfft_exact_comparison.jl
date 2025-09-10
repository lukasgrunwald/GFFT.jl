#=
Comparison of numerical integration with exact results for different methods
=#
using GFFT
using PConv
using MyPlot
set_style(:paper)

function compare_fftmethods(ht, hw; N, r, a, b, showfunction=false, save=false)
    # N, r = 2^14, 1
    Nexp = log(2, N) |> Int
    dt = (b - a) / N
    t = a .+ collect(0:1:N) * dt # real t to use
    w = 2π / (r * (b - a)) * (-r*N/2:1:r*N/2-1) |> collect
    htj = ht.(t)

    hriem, htrap, hsp3 = (zeros(CF64, r * N + 1) for i = 1:3)
    Offt = GfftData(+1; N=N, r=r, a=a, b=b, method=:spl3)
    gfft!(hriem, complex(htj); param=Offt, Isort=true, Osort=true, method=:riem, boundary=:nearest)
    gfft!(htrap, complex(htj); param=Offt, Isort=true, Osort=true, method=:trap, boundary=:nearest)
    gfft!(hsp3, complex(htj); param=Offt, Isort=true, Osort=true, method=:spl3, boundary=:nearest)

    fig, ax = plt.subplots(dpi=200)
    ax.set_yscale("log")
    ax.plot(w, abs.(hriem[1:end-1] .- hw.(w)), marker="x", markersize=3, label="Riemann")
    ax.plot(w, abs.(htrap[1:end-1] .- hw.(w)), marker=".", markersize=3, label="Trapz")
    ax.plot(w, abs.(hsp3[1:end-1] .- hw.(w)), marker=".", markersize=3, label="Spline")
    # plt.plot(w, hw.(w))
    ax.set_ylim(eps() / 4,)
    ax.set_xlabel(L"\omega")
    ax.set_ylabel(L"\vert f(\omega) - f^{\text{ex}}(\omega) \vert")
    plt.legend(title=L"N=2^{%$(Nexp)}")

    if showfunction
        fig, ax = plt.subplots(dpi=200, ncols=2)
        ax[1].set_title("Re")
        ax[1].plot(w, hriem[1:end-1] |> real, marker="x", markersize=3, label="Riemann")
        ax[1].plot(w, htrap[1:end-1] |> real, marker=".", markersize=3, label="Trapz")
        # ax[1].plot(w, hsp3[1:end-1] |> real, marker=".", markersize=3, label="Spline")
        ax[1].plot(w, hw.(w) |> real, label="Ref")

        ax[2].set_title("Im")
        ax[2].plot(w, hriem[1:end-1] |> imag, marker="x", markersize=3, label="Riemann")
        ax[2].plot(w, htrap[1:end-1] |> imag, marker=".", markersize=3, label="Trapz")
        # ax[2].plot(w, hsp3[1:end-1] |> imag, marker=".", markersize=3, label="Spline")
        ax[2].plot(w, hw.(w) |> imag, label="Ref")

        ax[1].legend()
        [(it.set_xlabel(L"\omega"), it.set_ylabel(L"f(\omega)")) for it ∈ ax]
        plt.tight_layout()
        plt.show()
    end
    path = versionized_path("pictures/fft_comparison.png", abspath=@__DIR__)
    save && plt.savefig(path, bbox_inches="tight", format="png", dpi=300)
    plt.show()
end

function compare_imagTime(; save=false)
    N = 2^12
    β, μ = 100, 0.0
    Nhalf = Int(N / 2)

    # Matsubara freq.
    n = [fftshift(-N/2:1:N/2-1); N / 2]
    w = 2π / β .* (n .+ 1 / 2) # fermionic
    v = 2π / β .* n # bosonic
    k = 0:N    # Imaginary time discretization
    τ = β / N .* k

    pfft_spl3 = GfftData(+1; N=N, r=1, a=0, b=β, method=:spl3)
    pfft_trap = GfftData(+1; N=N, r=1, a=0, b=β, method=:trap)
    pfft_riem = GfftData(+1; N=N, r=1, a=0, b=β, method=:riem)

    ϕ₊ = cis.(π / N * k) # phase for fermionic fft
    ϕ₋ = conj.(ϕ₊)

    # tail contributions for matsubara sum
    Gw_tail = @. 1 / (im * w) # dont need the δ, since w ≂̸ 0
    Gτ_tail = -1 / 2
    Dv_tail = @. 1 / (v^2 + 1.0 + 0im)
    Dτ_tail = @. -1 / 2 * (exp(-τ) * nb(-1, β) - 1 / (exp(β - τ) - exp(-τ)))

    Στ = @. -exp(μ * τ) * 1 / (exp(μ * β) + 1)
    Πτ = @. -1 / 2 * (exp(-τ) * nb(-1, β) - exp(τ) * nb(1, β)) |> complex
    Σw_ref = @. 1 / (im * w + μ)
    Πw_ref = @. 1 / (v^2 + 1.0)

    Σspl3, Σtrap, Σriem = (similar(Σw_ref) for j = 1:3)
    Πspl3, Πtrap, Πriem = (similar(Σw_ref) for j = 1:3)

    gfft!(Σspl3, (ϕ₊ .* Στ), param=pfft_spl3, Isort=true, Osort=false, boundary=:spl3)
    gfft!(Σtrap, (ϕ₊ .* Στ), param=pfft_trap, Isort=true, Osort=false, boundary=:spl3)
    gfft!(Σriem, (ϕ₊ .* Στ), param=pfft_riem, Isort=true, Osort=false, boundary=:spl3)

    gfft!(Πspl3, Πτ, param=pfft_spl3, Isort=true, Osort=false, boundary=:spl3)
    gfft!(Πtrap, Πτ, param=pfft_trap, Isort=true, Osort=false, boundary=:spl3)
    gfft!(Πriem, Πτ, param=pfft_riem, Isort=true, Osort=false, boundary=:spl3)

    fig, ax = plt.subplots(ncols=2, dpi=200, figsize=(15CM, 12CM))
    fig.suptitle(L"N=2^{%$(Int(log(2, N)))}, \beta=%$(β), \mu=%$(μ)")

    ax[1].plot(w, abs.(Σspl3 .- Σw_ref); markerx..., label="Spl3")
    ax[1].plot(w, abs.(Σriem .- Σw_ref); markers..., label="riem")
    ax[1].plot(w, abs.(Σtrap .- Σw_ref); markero..., label="trap")

    ax[2].plot(w, abs.(Πspl3 .- Πw_ref); markerx..., label="Spl3")
    ax[2].plot(w, abs.(Πriem .- Πw_ref); markers..., label="riem")
    ax[2].plot(w, abs.(Πtrap .- Πw_ref); markero..., label="trap")

    [(it.set_yscale("log"),
        it.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))) for it ∈ ax]
    ax[1].legend(shadow=true)
    ax[1].set_xlabel(L"\omega_n")
    ax[2].set_xlabel(L"\nu_n")
    ax[1].set_ylabel(L"\big| \Sigma - \Sigma_{\text{ref}}\big| (\omega_n)")
    ax[2].set_ylabel(L"\big| \Pi - \Pi_{\text{ref}}\big| (\nu_n)")

    plt.tight_layout()
    path = versionized_path("pictures/fft_comparison_imagTime.png", abspath=@__DIR__)
    save && plt.savefig(path, bbox_inches="tight", format="png", dpi=300)
    plt.show()
end

compare_imagTime()

##
ht(t) = exp(-t^2 / 2)
hw(w) = √(2π) * exp(-w^2 / 2)
a, b = -100, 100
compare_fftmethods(ht, hw; N=2^16, r=4, a, b, showfunction=false)
##
ht(t) = sinh(t)
hw(w) = exp(-3im * w) * (-cosh(3) - im * w * sinh(3) +
                         exp(-8im * w) * (cosh(5) - im * w * sinh(5))) / (1 + w^2)
a, b = -3, 5 # for the exact solution
compare_fftmethods(ht, hw; N=2^16, r=1, a, b, showfunction=false, save=false)
##
ht(t) = exp(-abs(t)) / 2
hw(w) = 1 / (w^2 + 1)
a, b = -100, 100
compare_fftmethods(ht, hw; N=2^16, r=1, a=a, b=b, showfunction=false)
##
ht(t) = -1im * exp(-0.1 * t) * heaviside(real(t), 1)
hw(w) = 1.0 ./ (w .+ 1im * 0.1)
a, b = 0, 1000
compare_fftmethods(ht, hw; N=2^16, r=2, a=a, b=b, showfunction=true)
