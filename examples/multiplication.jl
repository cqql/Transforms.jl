using Distributions
using Transforms
using GaussianMixtures

using PyPlot

function mm(a)
    normals = [Normal(a.μ[i], a.Σ[i][1]) for i = 1:a.n]

    MixtureModel(normals, a.w)
end

a = rand(GMM, 2, 1)
b = rand(GMM, 2, 1)

c = a * b
d = Transforms.resample(a, b, *)

pygui(true)

X = [-10:0.1:10]

plot(X, pdf(mm(a), X), label="a")
plot(X, pdf(mm(b), X), label="b")
plot(X, pdf(mm(c), X), label="c")
plot(X, pdf(mm(d), X), label="Sampled")
legend()
