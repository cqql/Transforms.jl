import Transforms
import Distributions: MixtureModel, Normal, pdf

using PyPlot

xmodel = MixtureModel([Normal(1., 2.)])
ymodel = MixtureModel([Normal(-2., 1.)])

n = 4
X = linspace(-10, 10, 200)

for ϵ = linspace(0.1, 1.0, 9)
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
