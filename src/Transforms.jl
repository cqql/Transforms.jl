module Transforms

import Distributions: mean
import GaussianMixtures: GMM

# Load some utility definitions
include("utils.jl")

"""
A random variable to do computations with
"""
immutable RandomVariable
    distribution::GMM

    function RandomVariable(distribution::GMM)
        new(distribution)
    end
end

function mean(x::RandomVariable)
    d = x.distribution

    sum(d.w .* d.μ)
end

function -(x::RandomVariable)
    d = x.distribution

    RandomVariable(GMM(d.w, -d.μ, d.Σ))
end

function +(x::RandomVariable, y::Real)
    d = x.distribution

    RandomVariable(GMM(d.w, d.μ + y, d.Σ))
end

+(x::Real, y::RandomVariable) = y + x
-(x::RandomVariable, y::Real) = x + (-y)
-(x::Real, y::RandomVariable) = (-y) + x

function *(x::RandomVariable, y::Real)
    if y == 0
        0
    else
        d = x.distribution

        RandomVariable(GMM(d.w, y * d.μ, y^2 * d.Σ))
    end
end

*(x::Real, y::RandomVariable) = y * x
/(x::RandomVariable, y::Real) = x * (1 / y)

function +(x::RandomVariable, y::RandomVariable)
    dx = x.distribution
    dy = y.distribution

    w = vec(dx.w * dy.w')
    μ = float(vec([dx.μ[i] + dy.μ[j] for i = 1:dx.n, j = 1:dy.n]))
    Σ = float(vec([dx.Σ[i] + dy.Σ[j] for i = 1:dx.n, j = 1:dy.n]))

    RandomVariable(GMM(w, μ, Σ))
end

-(x::RandomVariable, y::RandomVariable) = x + (-y)

end
