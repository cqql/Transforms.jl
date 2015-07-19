# Implement basic arithmetic on normal distributions

import Distributions: Normal
import CurveFit: linear_fit

function -(x::Normal)
    Normal(-x.μ, x.σ)
end

function +(x::Normal, y::Real)
    Normal(x.μ + y, x.σ)
end

function *(x::Normal, y::Real)
    Normal(y * x.μ, y * x.σ)
end

function +(x::Normal, y::Normal)
    Normal(x.μ + y.μ, x.σ + y.σ)
end

function *(params::GaussHermiteQuadrature, x::Normal, y::Normal)
    t = sqrt(2) * y.σ
    normals = [Normal((t * ξ + y.μ) * x.μ,
                      abs(t * ξ + y.μ) * x.σ)
               for ξ = params.X]

    Mixture{Normal}(normals, Categorical(params.W / sqrt(pi)))
end

function *(p::GaussLaguerreQuadrature, x::Normal, y::Normal)
    n, W, X, ϵ = p.n, p.W, p.X, p.ϵ
    μ, σ, σ2 = x.μ, x.σ, x.σ^2
    ν, τ, τ2 = y.μ, y.σ, y.σ^2

    # Constants and formulas for parameters for the left sum
    lS = (ϵ + ν)^2
    lT = lS / (2 * τ2)
    lnumerator = exp(-lT)
    lw(w, x) = w * lnumerator / (2 * sqrt(pi * (x + lT)))
    lμ(x) = -(sqrt(2 * τ2 * x + lS) - ν) * μ
    lσ(x) = (sqrt(2 * τ2 * x + lS) - ν) * σ

    lWeights = Float64[lw(W[i], X[i]) for i = 1:n]
    lComponents = Normal[Normal(lμ(x), lσ(x)) for x = X]

    # Constants and formulas for parameters for the right sum
    rS = (ϵ - ν)^2
    rT = rS / (2 * τ2)
    rnumerator = exp(-rT)
    rw(w, x) = w * rnumerator / (2 * sqrt(pi * (x + rT)))
    rμ(x) = (sqrt(2 * τ2 * x + rS) + ν) * μ
    rσ(x) = (sqrt(2 * τ2 * x + rS) + ν) * σ

    rWeights = Float64[rw(W[i], X[i]) for i = 1:n]
    rComponents = Normal[Normal(rμ(x), rσ(x)) for x = X]

    w = [lWeights, rWeights]
    c = [lComponents, rComponents]

    # Normalize the weights to 1
    w = w / sum(w)

    Mixture{Normal}(c, Categorical(w))
end

function /(params::GaussHermiteQuadrature, x::Normal, y::Normal)
    normals = [Normal(x.μ / (sqrt(2) * y.σ * ξ + y.μ),
                      abs(x.σ / (sqrt(2) * y.σ * ξ + y.μ)))
               for ξ = params.X]

    Mixture{Normal}(normals, Categorical(params.W / sqrt(pi)))
end
