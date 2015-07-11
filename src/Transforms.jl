module Transforms

import Distributions: Distribution, Univariate, Continuous, MixtureModel, components, probs, component_type, Categorical, mean
import GaussianMixtures: GMM, em!

include("integration.jl")
include("integration/gauss-hermite.jl")
include("integration/flattened-gauss-hermite.jl")
include("integration/gauss-laguerre.jl")

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

"Approximate an arbitrary distribution with a mixture of Gaussians."
function RandomVariable(distribution::Distribution,
                        samples::Integer, components::Integer)
    # Dimensionality of samples
    d = length(distribution)

    samples = rand(distribution, samples)
    samples = reshape(samples, (length(samples), d))
    gmm = GMM(components, d)
    em!(gmm, samples)

    normals = [Normal(gmm.μ[i], gmm.Σ[i][1]) for i = 1:gmm.n]

    RandomVariable(MixtureModel(normals, gmm.w))
end

function mean(x::RandomVariable)
    mean(x.distribution)
end

"Transform a random variable with a monadic function."
function monadictransform(op::Function, x::RandomVariable)
    d = x.distribution

    RandomVariable(Mixture{component_type(d)}(map(op, components(d)), d.prior),
                   x.alg)
end

-(x::RandomVariable) = monadictransform(-, x)

+(x::RandomVariable, y::Real) = monadictransform(X -> X + y, x)
+(x::Real, y::RandomVariable) = y + x
-(x::RandomVariable, y::Real) = x + (-y)
-(x::Real, y::RandomVariable) = (-y) + x

*(x::RandomVariable, y::Real) = y == 0 ? 0 : monadictransform(X -> X * y, x)
*(x::Real, y::RandomVariable) = y * x
/(x::RandomVariable, y::Real) = x * (1 / y)

"Transform two random variables with a diadic function."
function diadictransform(op::Function, x::RandomVariable, y::RandomVariable)
    dx = x.distribution
    dy = y.distribution

    prior = vec(probs(dx) * probs(dy)')
    cs = vec([op(i, j) for i = components(dx), j = components(dy)])

    # The type of the resulting components
    t = typeof(cs[1])

    # Cast from Vector{Any} to Vector{t}
    cs = t[c for c = cs]

    RandomVariable(Mixture{t}(cs, Categorical(prior)), x.alg)
end

+(x::RandomVariable, y::RandomVariable) = diadictransform(+, x, y)
-(x::RandomVariable, y::RandomVariable) = x + (-y)

"""
Transform two random variables with a diadic function.

Use this instead of #{diadictransform}, if op already returns a mixture
distribution.
"""
function diadicmixturetransform(op::Function,
                                x::RandomVariable, y::RandomVariable)
    dx = x.distribution
    dy = y.distribution

    # Weights if the components were not mixtures themselves
    basicPrior = vec(probs(dx) * probs(dy)')

    # Components approximated by mixtures
    mixtures = vec([op(x.alg, i, j) for i = components(dx), j = components(dy)])

    prior = vcat([basicPrior[i] * probs(mixtures[i]) for i = 1:length(mixtures)]...)
    cs = vcat(map(components, mixtures)...)

    # The type of the resulting components
    t = typeof(cs[1])

    RandomVariable(Mixture{t}(cs, Categorical(prior)), x.alg)
end

*(x::RandomVariable, y::RandomVariable) = diadicmixturetransform(*, x, y)
/(x::RandomVariable, y::RandomVariable) = diadicmixturetransform(/, x, y)

end
