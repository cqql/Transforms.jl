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
