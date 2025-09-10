#=
Manual benchmarking for package
=#
using BenchmarkTools
using GFFT

N, r, a, b = 2^22, 1, -30.1, 30.1
htj = rand(ComplexF64, N + 1)
method, boundary = :spl3, :nearest
Isort, Osort = true, false

h1, h2 = (zeros(ComplexF64, r * N + 1) for _ in 1:2)
obj = GfftData(+1; N, r, a, b, method)
gfft!(h1, htj; param = obj, Isort = Isort, Osort = Osort, method, boundary)

@btime gfft!($h1, $htj; param = $obj, Isort = $Isort, Osort = $Osort, method = $:(method),
             boundary = :($boundary))

# N = 2^16
# 420.566 μs (46 allocations: 1.00 MiB)

# N = 2^22
# 75.321 ms (46 allocations: 64.00 MiB)

##
using Profile
using ProfileView

Profile.clear()
@profile for i=1:100 gfft!(h1, htj; param = obj, Isort = true, Osort = false, boundary = :nearest) end
ProfileView.view()
