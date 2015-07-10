import FastGaussQuadrature: gausshermite

"""
A variation of Gauss-Hermite quadrature, were we flatten the components with a
variance lower than a given threshold.
"""
immutable FlattenedGaussHermiteQuadrature <: IntegrationAlgorithm
    # Parameters for normal Gauss-Hermite quadrature
    ghparams::GaussHermiteQuadrature

    # Variance threshold below which we flatten the components
    ϵ::Real

    function FlattenedGaussHermiteQuadrature(n::Integer, ϵ::Real)
        new(GaussHermiteQuadrature(n), ϵ)
    end
end
