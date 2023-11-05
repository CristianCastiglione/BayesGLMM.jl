
"""
    fit(y, X, Z, family, prior, alg)

Wrapper function for variational Bayesian inference
"""
function fit(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T};
    family::Family = Quantile(),
    prior::Prior = Prior(),
    alg::Algorithm = SVB()
    ) where T <: FP

    return fit(y, X, Z, family, prior, alg)
end

precompile(fit, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, ))

"""
    fit(y, X, Z, family, prior, alg)

Semiparametric variational Bayes inference with natual gradient optimization
"""
function fit(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T},
    family::Family = Gaussian();
    prior::Prior = Prior(),
    alg::Algorithm = SVB()
    ) where T <: FP

    return fit(y, X, Z, family, prior, alg)
end

for f in FAMILIES
    precompile(fit, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, f, ))
end

function fit(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T},
    family::Family,
    prior::Prior,
    alg::Algorithm
    ) where T <: FP

    # Fixed and random effect dimensions
    n_fe_par = size(X, 2)
    n_re_par = size.(Z, 2)

    # Model initialization
    model = BayesMixedModel(y, X, Z, family)

    # Optim initialization
    opt = OptSummary(n_fe_par, n_re_par, alg.maxiter)

    # Parameter estimation
    return fit!(model, prior, alg, opt)
end

for f in FAMILIES
    precompile(fit, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, f, Prior, SVB, ))
end

"""
    fit(y, X, Z, family, prior, alg)

Stochastic variationa inference with natural gradient optimization
"""
function fit(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T},
    family::Family,
    prior::Prior,
    alg::SVI
    ) where T <: FP

    # Fixed and random effect dimensions
    n_fe_par = size(X, 2)
    n_re_par = size.(Z, 2)

    # Model initialization
    model = BayesMixedModel(y, X, Z, family)

    # Optim initialization
    opt = OptSummary(n_fe_par, n_re_par, alg.maxiter)

    # Parameter estimation
    return fit!(model, prior, alg, opt)
end

for f in FAMILIES
    precompile(fit, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, f, Prior, SVI, ))
end


"""
    fit(y, X, Z, family, prior, alg)

Data-augmented mean field variational Bayes inference
"""
function fit(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T},
    family::Family,
    prior::Prior,
    alg::MFVB
    ) where T <: FP

    # Fixed and random effect dimensions
    n_fe_par = size(X, 2)
    n_re_par = size.(Z, 2)

    # Model initialization
    model = BayesMixedModel(y, X, Z, family)

    # Optim initialization
    opt = OptSummary(n_fe_par, n_re_par, alg.maxiter)

    # Parameter estimation
    return fit!(model, prior, alg, opt)
end

for f in [Gaussian, Logit, Probit, Quantile, SVR, SVC]
    precompile(fit, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Gaussian,  Prior, MFVB, ))    
end

"""
    fit(y, X, Z, family, prior, alg)

Markov chain Monte Carlo inference
"""
function fit(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T},
    family::Family,
    prior::Prior,
    alg::MCMC
    ) where T <: FP

    # Fixed and random effect dimensions
    n_fe_par = size(X, 2)
    n_re_par = size.(Z, 2)

    # Model initialization
    model = BayesMixedModel(y, X, Z, family)

    # Optim initialization
    opt = OptSummary(n_fe_par, n_re_par, alg.maxiter)

    # Parameter estimation
    return fit!(model, prior, alg, opt)
end

for f in [Gaussian, Logit, Probit, Quantile, SVR, SVC]
    precompile(fit, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, f, Prior, MCMC, ))
end

