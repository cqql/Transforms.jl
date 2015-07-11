import Transforms
import Distributions: MixtureModel, Normal, Exponential, Gamma, pdf

using PyPlot

x1 = Normal(1, 1)
x2 = Gamma(1, 1)
y = Exponential(1)

components = 25
samples = 10000
x1rv = Transforms.RandomVariable(x1, samples, components)
x2rv = Transforms.RandomVariable(x2, samples, components)
yrv = Transforms.RandomVariable(y, samples, components)

X = linspace(-10, 10, 200)

#plot(X, pdf(x1, X), label="x1", lw=8)
plot(X, pdf(x2, X), label="x2", lw=8)
#plot(X, pdf(y, X), label="y", lw=8)
#plot(X, pdf(x1rv.distribution, X), label="x1rv", lw=4)
plot(X, pdf(x2rv.distribution, X), label="x2rv", lw=4)
#plot(X, pdf(yrv.distribution, X), label="yrv", lw=4)

# for i = 1:5
#     n = i * 400

#     x = Transforms.RandomVariable(xmodel, Transforms.GaussHermiteQuadrature(n))
#     y = Transforms.RandomVariable(ymodel, Transforms.GaussHermiteQuadrature(n))

#     plot(X, pdf((x * y).distribution, X), label="$n components", lw=i)
# end

# N = 1000000
# plt.hist(rand(x.distribution, N) .* rand(y.distribution, N),
#          bins=200, range=(-10, 10), normed=true)
legend()
