import Distributions: Normal
import StatsBase: mean_and_var

const S2PI = sqrt(2 * pi)

"PDF of a normal distribution."
function p(x::Float64, μ::Float64, σ2::Float64)
    exp(-(x - μ)^2 / (2 * σ2)) / (S2PI * sqrt(σ2))
end

function EM!(X::Vector{Float64}, n::Integer, iters::Integer=100)
    N = length(X)

    # Guess initial model parameters
    X = sort(X)
    k = int(ceil(N / n))
    params = [mean_and_var(X[(1 + (i - 1) * k):min(i * k, end)]) for i = 1:n]
    iμ = map(first, params)
    iσ2 = map(last, params)

    # Initialize model parameters
    π::Vector{Float64} = [1 / n for i = 1:n]
    μ::Vector{Float64} = iμ
    σ2::Vector{Float64} = iσ2

    # Cache this
    const X2::Vector{Float64} = X.^2

    for j = 1:iters
        # E step
        r = Array(Float64, N, n)

        for i = 1:N
            R = dot(π, [p(X[i], μ[k], σ2[k]) for k = 1:n])

            r[i, :] = π .* [p(X[i], μ[k], σ2[k]) for k = 1:n] / R
        end

        # M step
        rk = [sum(r[:, k]) for k = 1:n]
        nπ = rk / N
        nμ = [dot(r[:, k], X) for k = 1:n] ./ rk
        nσ2 = [dot(r[:, k], X2) for k = 1:n] ./ rk - μ.^2

        π, μ, σ2 = nπ, nμ, nσ2
    end

    Mixture{Normal}([Normal(μ[i], sqrt(σ2[i])) for i = 1:n], Categorical(π))
end
