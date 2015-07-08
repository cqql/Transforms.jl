import Transforms
import Distributions: MixtureModel, Normal, pdf

using PyPlot

x = Transforms.RandomVariable(MixtureModel([Normal(1., 2.),
                                            Normal(-4., 10.)],
                                           [0.5, 0.5]))
y = Transforms.RandomVariable(MixtureModel([Normal(1., 2.),
                                            Normal(-4., 10.),
                                            Normal(2.3, 0.3)],
                                           [0.2, 0.5, 0.3]))

X = linspace(-10, 10, 200)

plot(X, pdf(x.distribution, X), label="x", lw=2)
plot(X, pdf(y.distribution, X), label="y", lw=2)

plot(X, pdf((x + y).distribution, X), label="x + y")

N = 1000000
plt.hist(rand(x.distribution, N) .+ rand(y.distribution, N),
         bins=200, range=(-10, 10), normed=true)

plt.hist(rand((x + y).distribution, N),
         bins=200, range=(-10, 10), normed=true)

legend()
