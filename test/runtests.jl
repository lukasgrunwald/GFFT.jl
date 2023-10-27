using GFFT
using Test
using Aqua

@testset "GFFT.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GFFT; ambiguities = false,)
    end
    # Write your tests here.
end
