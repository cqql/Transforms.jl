import Transforms
import Distributions: MixtureModel, Normal, pdf

using PyPlot

xmodel = MixtureModel([Normal(1., 2.), Normal(-4., 10.)], [0.5, 0.5])
ymodel = MixtureModel([Normal(1., 2.), Normal(-4., 10.), Normal(2.3, 0.3)],
                      [0.2, 0.5, 0.3])
x = Transforms.RandomVariable(xmodel)
y = Transforms.RandomVariable(ymodel)

X = linspace(-10, 10, 200)
n = 200

plot(X, pdf(x.distribution, X), label="x", lw=2)
plot(X, pdf(y.distribution, X), label="y", lw=2)

for ϵ = linspace(0.1, 0.4, 9)
    alg = Transforms.GaussLaguerreQuadrature(n, ϵ)

    x = Transforms.RandomVariable(xmodel, alg)
    y = Transforms.RandomVariable(ymodel, alg)

    label = @sprintf("%.2f", ϵ)
    plot(X, pdf((x * y).distribution, X), label="ϵ = $label", lw=4)
end

N = 1000000
plt.hist(rand(x.distribution, N) .* rand(y.distribution, N),
         bins=200, range=(-10, 10), normed=true)
legend()
