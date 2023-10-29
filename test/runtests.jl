using GFFT
using Test

@testset "interface test" begin include("interface_test.jl") end
@testset "convergence test" begin include("convergence_test.jl") end
