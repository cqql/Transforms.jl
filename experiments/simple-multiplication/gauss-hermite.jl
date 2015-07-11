import Transforms
import Distributions: MixtureModel, Normal, pdf

using PyPlot

xmodel = MixtureModel([Normal(1., 2.)])
ymodel = MixtureModel([Normal(-2., 1.)])

X = linspace(-10, 10, 200)

for i = 1:2:10
    n = i * 100
    alg = Transforms.GaussHermiteQuadrature(n)

    x = Transforms.RandomVariable(xmodel, alg)
    y = Transforms.RandomVariable(ymodel, alg)

    plot(X, pdf((x * y).distribution, X), label="$n components", lw=4)
end

N = 1000000
plt.hist(rand(x.distribution, N) .* rand(y.distribution, N),
         bins=200, range=(-10, 10), normed=true)
legend()
