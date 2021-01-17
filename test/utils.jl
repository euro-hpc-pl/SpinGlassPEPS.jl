@testset "state indexing" begin
    σ = (1, -1, -1, 1, 1)
    d = 2^length(σ)
    x = rand(2, 2, d)
    # @show @macroexpand @state x[1, 1, σ]
    @state x[1, 1, σ]# == x[1, 1, 20]
end