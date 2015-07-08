using Base.Test

using Transforms: RandomVariable
using GaussianMixtures: GMM

x = RandomVariable(GMM([0.5, 0.5], [1., -4.], [2., 10.]))

@test_approx_eq -1.5 mean(x)

@test_approx_eq -mean(x) mean(-x)

@test_approx_eq (mean(x) + 5) mean(x + 5)
@test_approx_eq (mean(x) + 5) mean(5 + x)
@test_approx_eq (mean(x) - 5) mean(x - 5)
@test_approx_eq (5 - mean(x)) mean(5 - x)

@test_approx_eq (10 * mean(x)) mean(10 * x)
@test_approx_eq (10 * mean(x)) mean(x * 10)
@test_approx_eq (mean(x) / 10) mean(x / 10)
