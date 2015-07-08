import Transforms
import Distributions: MixtureModel, Normal, pdf

using PyPlot

x = Transforms.RandomVariable(MixtureModel([Normal(1., 2.)]))
y = Transforms.RandomVariable(MixtureModel([Normal(-2., 1.)]))

X = linspace(-10, 10, 200)

plot(X, pdf(x.distribution, X), label="x", lw=2)
plot(X, pdf(y.distribution, X), label="y", lw=2)

for i = 1:10
    n = i * 100

    x = Transforms.RandomVariable(MixtureModel([Normal(1., 2.)]),
                                  Transforms.GaussHermiteQuadrature(n))
    y = Transforms.RandomVariable(MixtureModel([Normal(-2., 1.)]),
                                  Transforms.GaussHermiteQuadrature(n))

    plot(X, pdf((x * y).distribution, X), label="$n components", lw=i)
end

N = 1000000
plt.hist(rand(x.distribution, N) .* rand(y.distribution, N),
         bins=200, range=(-10, 10), normed=true)
legend()
