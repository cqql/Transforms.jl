import FastGaussQuadrature: gausshermite
import GaussQuadrature: laguerre

"Exchangable algorithm for integral approximation"
abstract IntegrationAlgorithm

"Parameters for Gauss-Hermite quadrature"
immutable GaussHermiteQuadrature <: IntegrationAlgorithm
    # Number of points
    n::Integer

    # Evaluation points for the integrand
    X::Vector{Float64}

    # Weights for the weighted sum approximation
    W::Vector{Float64}

    function GaussHermiteQuadrature(n::Integer)
        (X, W) = gausshermite(n)

        new(n, X, W)
    end
end

"Parameters for Gauss-Laguerre quadrature"
immutable GaussLaguerreQuadrature <: IntegrationAlgorithm
    # Number of points
    n::Integer

    # Evaluation points for the integrand
    X::Vector{Float64}

    # Weights for the weighted sum approximation
    W::Vector{Float64}

    # How much to ignore around the pole. Has to be >= 0.
    ϵ::Real

    function GaussLaguerreQuadrature(n::Integer, ϵ::Real)
        (X, W) = laguerre(n, 0.0)

        new(n, X, W, ϵ)
    end
end
