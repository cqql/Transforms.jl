import GaussianMixtures: DiagCov, FullCov

"""
Instantiate a GMM with just the actual distribution parameters
"""
function GMM{T<:FloatingPoint}(w::Vector{T}, μ::Matrix{T},
                               Σ::Union(DiagCov{T}, FullCov{T}))
    GMM{T, typeof(Σ)}(w, μ, Σ, [], 0)
end

"""
A shorthand for 1-dimensional mixtures
"""
function GMM{T<:FloatingPoint}(w::Vector{T}, μ::Vector{T}, Σ::Vector{T})
    GMM(w, reshape(μ, (length(μ), 1)), reshape(Σ, (length(Σ), 1)))
end
