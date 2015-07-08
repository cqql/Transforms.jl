# Implement basic arithmetic on normal distributions

import Distributions: Normal

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
