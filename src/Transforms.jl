module Transforms

import Distributions: Distribution, Univariate, Continuous, MixtureModel, components, probs, component_type, Categorical, mean

include("integration.jl")
include("integration/gauss-hermite.jl")

include("components/normal.jl")

"The special case of mixture models, that we work with."
typealias Mixture{T<:Distribution} MixtureModel{Univariate, Continuous, T}

"""
A random variable to do computations with
"""
immutable RandomVariable{Component<:Distribution, T<:IntegrationAlgorithm}
    distribution::Mixture{Component}
    alg::T

    function RandomVariable(distribution::Mixture{Component}, alg::T)
        new(distribution, alg)
    end
end

"A convenience constructor, so you do not have to specify the types."
function RandomVariable{
    T<:Distribution,
    S<:IntegrationAlgorithm}(distribution::Mixture{T}, alg::S)
    RandomVariable{T, S}(distribution, alg)
end

"Construct a random variable with a default integration algorithm."
function RandomVariable(distribution::Mixture)
    RandomVariable(distribution, GaussHermiteQuadrature(5))
end

function mean(x::RandomVariable)
    mean(x.distribution)
end

function -(x::RandomVariable)
    d = x.distribution

    RandomVariable(Mixture{component_type(d)}(map(-, components(d)), d.prior),
                   x.alg)
end

function +(x::RandomVariable, y::Real)
    d = x.distribution
    f = X -> X + y

    RandomVariable(Mixture{component_type(d)}(map(f, components(d)), d.prior),
                   x.alg)
end

+(x::Real, y::RandomVariable) = y + x
-(x::RandomVariable, y::Real) = x + (-y)
-(x::Real, y::RandomVariable) = (-y) + x

function *(x::RandomVariable, y::Real)
    if y == 0
        0
    else
        d = x.distribution
        f = X -> X * y

        RandomVariable(Mixture{component_type(d)}(map(f, components(d)),
                                                  d.prior),
                       x.alg)
    end
end

*(x::Real, y::RandomVariable) = y * x
/(x::RandomVariable, y::Real) = x * (1 / y)

function +(x::RandomVariable, y::RandomVariable)
    dx = x.distribution
    dy = y.distribution

    prior = vec(probs(dx) * probs(dy)')
    cs = vec([i + j for i = components(dx), j = components(dy)])

    # The type of the resulting components
    t = typeof(cs[1])

    RandomVariable(Mixture{t}(cs, Categorical(prior)))
end

-(x::RandomVariable, y::RandomVariable) = x + (-y)

function *(x::RandomVariable, y::RandomVariable)
    dx = x.distribution
    dy = y.distribution

    # Weights if the components were not mixtures themselves
    basicPrior = vec(probs(dx) * probs(dy)')

    # Components approximated by mixtures
    mixtures = vec([*(x.alg, i, j) for i = components(dx), j = components(dy)])

    prior = vcat([basicPrior[i] * probs(mixtures[i]) for i = 1:length(mixtures)]...)
    cs = vcat(map(components, mixtures)...)

    # The type of the resulting components
    t = typeof(cs[1])

    RandomVariable(Mixture{t}(cs, Categorical(prior)))
end

end
