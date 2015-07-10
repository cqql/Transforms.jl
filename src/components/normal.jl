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
    Normal(y * x.μ, y^2 * x.σ)
end

function +(x::Normal, y::Normal)
    Normal(x.μ + y.μ, x.σ + y.σ)
end

function *(params::GaussHermiteQuadrature, x::Normal, y::Normal)
    normals = [Normal((sqrt(2 * y.σ) * ξ + y.μ) * x.μ,
                      (sqrt(2 * y.σ) * ξ + y.μ)^2 * x.σ)
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

function /(params::GaussHermiteQuadrature, x::Normal, y::Normal)
    normals = [Normal(x.μ / (sqrt(2 * y.σ) * ξ + y.μ),
                      x.σ / (sqrt(2 * y.σ) * ξ + y.μ)^2)
               for ξ = params.X]

    Mixture{Normal}(normals, Categorical(params.W / sqrt(pi)))
end
