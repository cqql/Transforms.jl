using Distributions
using Transforms
using GaussianMixtures

using PyPlot

function mm(a)
    normals = [Normal(a.μ[i], a.Σ[i][1]) for i = 1:a.n]

    MixtureModel(normals, a.w)
end

a = rand(GMM, 5, 1)
b = rand(GMM, 3, 1)

c = a + b

pygui(true)

X = [-5:0.1:5]

plot(X, pdf(mm(a), X), label="a")
plot(X, pdf(mm(b), X), label="b")
plot(X, pdf(mm(c), X), label="c")
legend()
