using Base.Test

import Transforms: RandomVariable
import Distributions: MixtureModel, Normal, mean

x = RandomVariable(MixtureModel([Normal(1., 2.),
                                 Normal(-4., 10.)],
                                [0.5, 0.5]))

@test_approx_eq -1.5 mean(x)

@test_approx_eq -mean(x) mean(-x)

@test_approx_eq (mean(x) + 5) mean(x + 5)
@test_approx_eq (mean(x) + 5) mean(5 + x)
@test_approx_eq (mean(x) - 5) mean(x - 5)
@test_approx_eq (5 - mean(x)) mean(5 - x)

@test_approx_eq (10 * mean(x)) mean(10 * x)
@test_approx_eq (10 * mean(x)) mean(x * 10)
@test_approx_eq (mean(x) / 10) mean(x / 10)

y = RandomVariable(MixtureModel([Normal(1., 2.),
                                 Normal(-4., 10.),
                                 Normal(2.3, 0.3)],
                                [0.2, 0.5, 0.3]))

@test_approx_eq (mean(x) + mean(y)) mean(x + y)
@test_approx_eq (mean(x) - mean(y)) mean(x - y)

@test_approx_eq (mean(x) * mean(y)) mean(x * y)
@test_approx_eq (mean(x) / mean(y)) mean(x / y)
