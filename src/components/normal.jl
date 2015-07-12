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

function *(params::FlattenedGaussHermiteQuadrature, x::Normal, y::Normal)
    mixture = *(params.ghparams, x, y)
    cs = components(mixture)
    n = length(cs)
    tenpercent = int(0.1 * n)
    range = tenpercent:(n - tenpercent)

    a, b = linear_fit(range, log(map(var, cs[range])))
    f(x) = a + b * x

    function flatten(d::Normal)
        if d.σ < params.ϵ
            Normal(d.μ, 1)
        else
            d
        end
    end

    prior = mixture.prior
    cs = map(flatten, cs)

    Mixture{Normal}(cs, prior)
end

function *(p::GaussLaguerreQuadrature, x::Normal, y::Normal)
    n, W, X, ϵ = p.n, p.W, p.X, p.ϵ
    μ, σ2 = x.μ, x.σ^2
    ν, τ2 = y.μ, y.σ^2

    # Constants and formulas for parameters for the left sum
    lS = (ϵ + ν)^2
    lT = lS / (2 * τ2)
    lnumerator = exp(-lT)
    lw(w, x) = w * lnumerator / (2 * sqrt(pi * (x + lT)))
    lμ(x) = -(sqrt(2 * τ2 * x + lS) - ν) * μ
    lσ(x) = (sqrt(2 * τ2 * x + lS) - ν) * σ2

    lWeights = Float64[lw(W[i], X[i]) for i = 1:n]
    lComponents = Normal[Normal(lμ(x), lσ(x)) for x = X]

    # Constants and formulas for parameters for the left sum
    lS = (ϵ - ν)^2
    lT = lS / (2 * τ2)
    lnumerator = exp(-lT)
    rw(w, x) = w * lnumerator / (2 * sqrt(pi * (x + lT)))
    rμ(x) = -(sqrt(2 * τ2 * x + lS) + ν) * μ
    rσ(x) = (sqrt(2 * τ2 * x + lS) + ν) * σ2

    rWeights = Float64[rw(W[i], X[i]) for i = 1:n]
    rComponents = Normal[Normal(rμ(x), rσ(x)) for x = X]

    w = [lWeights, rWeights]
    c = [lComponents, rComponents]

    # Normalize the weights to 1
    w = w / sum(w)

    Mixture{Normal}(c, Categorical(w))
end

function *(params::ExpVar, x::Normal, y::Normal)
    μ = x.μ * y.μ
    σ2 = x.σ^2 * y.σ^2 + x.μ^2 * y.σ^2 + y.μ^2 * x.σ^2

    Mixture{Normal}([Normal(μ, sqrt(σ2))], Categorical([1.0]))
end

function /(params::GaussHermiteQuadrature, x::Normal, y::Normal)
    normals = [Normal(x.μ / (sqrt(2 * y.σ) * ξ + y.μ),
                      x.σ / (sqrt(2 * y.σ) * ξ + y.μ)^2)
               for ξ = params.X]

    Mixture{Normal}(normals, Categorical(params.W / sqrt(pi)))
end
