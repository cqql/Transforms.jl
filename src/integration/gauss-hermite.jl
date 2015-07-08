import FastGaussQuadrature: gausshermite

"""
Parameters for Gauss-Hermite quadrature
"""
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
