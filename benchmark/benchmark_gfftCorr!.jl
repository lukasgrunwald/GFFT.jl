# Fun bug: If one removes the type annotation from Δ, the variable becomes boxed in the
# threaded for loop, tanking the performance!

function g!(f::AbstractVector, flag::Bool, dx::Float64, dy::Float64)
    Δ::Float64 = 0.0 # Not really needed, since if-statement does not introduce new scope
    if flag Δ = dy else Δ = dx end

    @inbounds Threads.@threads for i in eachindex(f)
        f[i] = Δ * f[i]
    end
    return nothing
end

function g_ternary!(f::AbstractVector, flag::Bool, dx::Float64, dy::Float64)
    Δ = flag ? dy : dx

    @inbounds Threads.@threads for i in eachindex(f)
        f[i] = Δ * f[i]
    end
    return nothing
end

using BenchmarkTools
N, flag = 2^16, false
y = rand(ComplexF64, N)
dx, dy = rand(2)

@btime g!($y, $flag, $dx, $dy)
@btime g_ternary!($y, $flag, $dx, $dy)
