module Transforms

import GaussianMixtures: GMM

function resample(a, b, op)
    n = 1000
    as = rand(a, n)
    bs = rand(b, n)

    cs = map(op, as, bs)

    GMM(max(a.n, b.n), cs)
end

+(a::GMM, b::GMM) = resample(a, b, +)
-(a::GMM, b::GMM) = resample(a, b, -)
*(a::GMM, b::GMM) = resample(a, b, *)
/(a::GMM, b::GMM) = resample(a, b, /)

# import Distributions: Normal, MvNormal, MixtureModel, mean, var, cov, probs, components

# const AnyNormal = Union(Normal, MvNormal)

# for op in (:.+, :.-, :.*, :./)
#     f = symbol(string(op)[2:end])

#     @eval begin
#         $op(a::Real, b::AnyNormal) = $f(a, b)
#         $op(a::Real, b::AnyNormal) = $f(a, b)
#     end
# end

# convert(::Type{MvNormal}, a::Normal) = MvNormal([mean(a)], [var(a)])

# # One dimensional

# -(a::Normal) = Normal(-mean(a), var(a))

# +(a::Normal, b::Real) = Normal(mean(a) + b, var(a))
# +(a::Real, b::Normal) = b + a
# -(a::Normal, b::Real) = a + (-b)
# -(a::Real, b::Normal) = a + (-b)
# *(a::Normal, b::Real) = Normal(mean(a) * b, var(a) * b^2)
# *(a::Real, b::Normal) = b * a
# /(a::Normal, b::Real) = a * (1 / b)

# +(a::Normal, b::Normal) = Normal(mean(a) + mean(b), var(a) + var(b))
# -(a::Normal, b::Normal) = a + (-b)

# # Multi dimensional

# -(a::MvNormal) = MvNormal(-mean(a), var(a))

# +(a::MvNormal, b::Vector{Real}) = MvNormal(mean(a) + b, cov(a))
# +(a::Vector{Real}, b::MvNormal) = b + a
# -(a::MvNormal, b::Vector{Real}) = a + (-b)
# -(a::Vector{Real}, b::MvNormal) = a + (-b)
# *(a::MvNormal, b::Real) = MvNormal(mean(a) * b, var(a) * b^2)
# *(a::Real, b::MvNormal) = b * a
# /(a::MvNormal, b::Real) = a * (1 / b)

# +(a::MvNormal, b::MvNormal) = MvNormal(mean(a) + mean(b), var(a) + var(b))
# -(a::MvNormal, b::MvNormal) = a + (-b)

# # Mixtures

# +(a::Real, b::MixtureModel) = MixtureModel(a .+ components(b), probs(b))
# +(a::MixtureModel, b::Real) = b + a
# *(a::Real, b::MixtureModel) = MixtureModel(a .* components(b), probs(b))
# *(a::MixtureModel, b::Real) = b * a

end
