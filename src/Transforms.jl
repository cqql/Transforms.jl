module Transforms

import GaussianMixtures: GMM, History
import Distributions: Normal, mean
import FastGaussQuadrature: gausshermite

function resample(a, b, op)
    n = 10000
    as = rand(a, n)
    bs = rand(b, n)

    cs = map(op, as, bs)

    # Much higher number of components leads to everything being NaN
    GMM(40, cs)
end

# Functions for inspection of GMMs
mean(a::GMM) = sum(a.w .* a.μ)

# Monadic operations
-(a::GMM) = GMM(a.weights, -a.μ, a.Σ, a.hist)

# Diadic operations with scalars
function +(a::GMM, b::Real)
    history = vcat(a.hist, History(@sprintf("Add %f", b)))

    GMM(a.weights, a.μ + b, a.Σ, a.hist)
end

+(a::Real, b::GMM) = b + a
-(a::GMM, b::Real) = a + (-b)
-(a::Real, b::GMM) = (-b) + a

function *(a::GMM, b::Real)
    if b == 0
        0
    else
        history = vcat(a.hist, History(@sprintf("Multiply by %f", b)))

        GMM(a.weights, b * a.μ, b^2 * a.Σ, history)
    end
end

*(a::Real, b::GMM) = b * a

# Diadic operations with two GMMs
function +(a::GMM, b::GMM)
    # Reshape a matrix into a 2-dimensional "vector"
    rs = v -> reshape(v, (a.n * b.n, a.d))

    weights = vec(a.w * b.w')
    means = rs([a.μ[i] + b.μ[j] for i = 1:a.n, j = 1:b.n])
    variances = rs([a.Σ[i][1] + b.Σ[j][1] for i = 1:a.n, j = 1:b.n])
    history = vcat(a.hist, b.hist, History("Sum of two GMMs"))

    GMM(weights, means, variances, history, a.nx + b.nx)
end

-(a::GMM, b::GMM) = a + (-b)

function *(a::GMM, b::GMM)
    # Reshape a matrix into a 2-dimensional "vector"
    rs = v -> reshape(v, (a.n * b.n, a.d))

    gmms = [Normal(a.μ[i], a.Σ[i][1]) * Normal(b.μ[j], b.Σ[j][1])
            for i = 1:a.n, j = 1:b.n]

    weights = vcat(rs([a.w[i] * b.w[j] * gmms[i, j].w for i = 1:a.n, j = 1:b.n])...)
    μ = vcat(rs([gmms[i, j].μ for i = 1:a.n, j = 1:b.n])...)
    σ = vcat(rs([gmms[i, j].Σ for i = 1:a.n, j = 1:b.n])...)

    history = vcat(a.hist, b.hist, History("Product of two GMMs"))

    GMM(weights, μ, σ, history, a.nx + b.nx)
end

/(a::GMM, b::GMM) = resample(a, b, /)

# Positions and weights for Gauss-Hermite quadrature
(X, W) = gausshermite(20)
sqrtPi = sqrt(pi)
precomputedWeights = W / sqrtPi

function *(a::Normal, b::Normal)
    rs = v -> reshape(v, (20, 1))

    t = sqrt(2 * b.σ) * X + b.μ
    μ = rs(t * a.μ)
    σ = rs((t .* t) * a.σ)

    GMM(precomputedWeights, μ, σ, [], 0)
end

end
