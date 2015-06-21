module Transforms

import GaussianMixtures: GMM, History

function resample(a, b, op)
    n = 1000
    as = rand(a, n)
    bs = rand(b, n)

    cs = map(op, as, bs)

    GMM(max(a.n, b.n), cs)
end

# Monadic operations
-(a::GMM) = GMM(a.weights, -a.μ, a.Σ, a.hist)

# Diadic operations with scalars
function +(a::GMM, b::Real)
    history = vcat(a.hist, History(@sprintf("Add %f", b)))

    GMM(a.weights, a.μ + b, a.Σ, a.hist)
end

+(a::Real, b::GMM) = b + a
-(a::GMM, b::Real) = a + (-b)
-(a::Real, b::GMM) = (-b) + a

function *(a::GMM, b::Real)
    if b == 0
        0
    else
        history = vcat(a.hist, History(@sprintf("Multiply by %f", b)))

        GMM(a.weights, b * a.μ, b^2 * a.Σ, history)
    end
end

*(a::Real, b::GMM) = b * a

# Diadic operations with two GMMs
function +(a::GMM, b::GMM)
    # Reshape a matrix into a 2-dimensional "vector"
    rs = v -> reshape(v, (a.n * b.n, a.d))

    weights = vec(a.w * b.w')
    means = rs([a.μ[i] + b.μ[j] for i = 1:a.n, j = 1:b.n])
    variances = rs([a.Σ[i][1] + b.Σ[j][1] for i = 1:a.n, j = 1:b.n])
    history = vcat(a.hist, b.hist, History("Sum of two GMMs"))

    GMM(weights, means, variances, history, a.nx + b.nx)
end

-(a::GMM, b::GMM) = a + (-b)

*(a::GMM, b::GMM) = resample(a, b, *)
/(a::GMM, b::GMM) = resample(a, b, /)

end
