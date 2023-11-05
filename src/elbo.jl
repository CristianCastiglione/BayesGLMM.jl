
"""
    elbo(model, prior, alg)

Wrapper function calculating the evidence lower bound of the model depending
on the current variational approximation, on the prior distributions and on
the algorithm parameters (e.g., the number of integration knots to be used 
for the numerical evaluation of the expected log-likelihood)
"""

function elbo(
    model::BayesMixedModel{T},
    prior::Prior = Prior(),
    alg::Algorithm = SVB()
    )::T where T <: FP

    # q_eta expected values
    fq = model.eta.m
    vq = model.eta.V

    # psi functions
    psi0, _, _ = _psi(y, fq, sqrt.(vq), f, 0.0, alg.nknots)

    return _elbo(
        model.y, model.family, psi0, model.theta,
        model.sigma2_e, model.sigma2_u, prior, alg)
end;

precompile(elbo, (BayesMixedModel{FP}, Prior, SVB, ))

"""
    elbo()

Returns the evidence lower bound of the model under a semiparametric
variational Bayes (SVB) approximation obtained using a closed-form natural
gradient optimization algorithm
"""

function elbo(
    y::Vector{T},
    f::Family,
    eta::LinPred{T},
    theta::RegParam{T},
    sigma2_e::ScaleParam{T},
    sigma2_u::VecScaleParam{T},
    prior::Prior,
    alg::SVB
    )::T where T <: FP

    # q_eta expected values
    fq = eta.m
    vq = eta.V

    # psi functions
    psi0, _, _ = _psi(y, fq, sqrt.(vq), f, 0.0, alg.nknots)

    # elbo calculation
    return _elbo(y, f, psi0, theta, sigma2_e, sigma2_u, prior, alg)
end;

for f in FAMILIES
    precompile(elbo, (
        Vector{FP}, f, 
        LinPred{FP}, RegParam{FP}, ScaleParam{FP}, 
        VecScaleParam{FP}, Prior, SVB, ))
end


"""
    elbo()

Returns the evidence lower bound of the model under a semiparametric
variational Bayes (SVB) approximation obtained using a closed-form natural
gradient optimization algorithm
"""

function _elbo(
    y::Vector{T},
    f::Family,
    psi0::T,
    theta::RegParam{T},
    sigma2_e::ScaleParam{T},
    sigma2_u::VecScaleParam{T},
    prior::Prior,
    alg::SVB
    )::T where T <: FP

    # Fixed and random parameter indices
    idx_fe = theta.idx_fe
    idx_re = theta.idx_re
    idx = [[idx_fe]; idx_re]

    # Input dimensions
    n_obs = length(y)
    n_fe = 1
    n_re = length(idx_re)
    n_fe_par = length(idx_fe)
    n_re_par = length.(idx_re)
    n_tot_par = n_fe_par + sum(n_re_par)

    # Prior parameters
    A_e = prior.A_e
    B_e = prior.B_e
    A_u = prior.A_u
    B_u = prior.B_u
    sigma2_b = prior.sigma2_b
    s2_inv_b = 1.0 / sigma2_b

    # q_sigma_u expected values
    Aq_u = [sigma2_u[k].A for k in 1:n_re]
    Bq_u = [sigma2_u[k].B for k in 1:n_re]
    mq_inv_u = _iginvmean.(Aq_u, Bq_u)
    mq_log_u = _iglogmean.(Aq_u, Bq_u)

    # q_sigma_e expected values
    Aq_e = sigma2_e.A
    Bq_e = sigma2_e.B
    mq_inv_e = _iginvmean(Aq_e, Bq_e)
    mq_log_e = _iglogmean(Aq_e, Bq_e)

    # expected penalty matrix
    Q = get_prior_matrix(s2_inv_b, mq_inv_u, idx_fe, idx_re)

    # q_theta expected values
    mq, Lq  = theta.m, theta.R
    mq_sq_t = _sqmean(mq, Matrix(Lq), Matrix(Q))

    # loss tail order
    alpha = tailorder(f)

    # evidence lower bound
    elbo  = .0
    elbo -= (n_obs * mq_log_e + mq_inv_e * psi0) / alpha

    elbo -= .5 * (n_fe_par * log(sigma2_b) + sum(n_re_par .* mq_log_u))
    elbo -= .5 * (mq_sq_t - n_fe_par - sum(n_re_par)) - logdet(Lq)

    elbo += n_re * A_u * log(B_u) - sum(Aq_u .* log.(Bq_u))
    elbo -= n_re * loggamma(A_u) - sum(loggamma.(Aq_u))
    elbo -= sum((A_u .- Aq_u) .* mq_log_u)
    elbo -= sum((B_u .- Bq_u) .* mq_inv_u)

    elbo += A_e * log(B_e) - Aq_e * log(Bq_e)
    elbo -= loggamma(A_e) - loggamma(Aq_e)
    elbo -= (A_e - Aq_e) * mq_log_e + (B_e - Bq_e) * mq_inv_e

    # output
    return elbo
end;

# The compilation fails for a generic abstract type like Family,
# In order to make the compilation success it is needed to specify a
# specific struct belonging to the abstract type Family, like Quantile

for f in FAMILIES
    precompile(_elbo, (
        Vector{FP}, f, FP,
        RegParam{FP}, ScaleParam{FP},
        VecScaleParam{FP}, Prior, SVB, ))
end


"""
    elbo()

Returns the evidence lower bound of the model under a stochastic
variational inference (SVI) approximation obtained using a minibatch natural
gradient optimization algorithm
"""

function _elbo(
    y::Vector{T},
    f::Family,
    n::Int64,
    psi0::T,
    theta::RegParam{T},
    sigma2_e::ScaleParam{T},
    sigma2_u::VecScaleParam{T},
    prior::Prior,
    alg::SVI
    )::T where T <: FP

    # Fixed and random parameter indices
    idx_fe = theta.idx_fe
    idx_re = theta.idx_re
    idx = [[idx_fe]; idx_re]

    # Input dimensions
    n_obs = n
    n_mb = alg.minibatch
    n_fe = 1
    n_re = length(idx_re)
    n_fe_par = length(idx_fe)
    n_re_par = length.(idx_re)
    n_tot_par = n_fe_par + sum(n_re_par)

    # Prior parameters
    A_e = prior.A_e
    B_e = prior.B_e
    A_u = prior.A_u
    B_u = prior.B_u
    sigma2_b = prior.sigma2_b
    s2_inv_b = 1.0 / sigma2_b

    # q_sigma_u expected values
    Aq_u = [sigma2_u[k].A for k in 1:n_re]
    Bq_u = [sigma2_u[k].B for k in 1:n_re]
    mq_inv_u = _iginvmean.(Aq_u, Bq_u)
    mq_log_u = _iglogmean.(Aq_u, Bq_u)

    # q_sigma_e expected values
    Aq_e = sigma2_e.A
    Bq_e = sigma2_e.B
    mq_inv_e = _iginvmean(Aq_e, Bq_e)
    mq_log_e = _iglogmean(Aq_e, Bq_e)

    # expected penalty matrix
    Q = get_prior_matrix(s2_inv_b, mq_inv_u, idx_fe, idx_re)

    # q_theta expected values
    mq, Lq  = theta.m, theta.R
    mq_sq_t = _sqmean(mq, Matrix(Lq), Matrix(Q))

    # loss tail order
    alpha = tailorder(f)

    # evidence lower bound
    elbo  = .0
    elbo -= (n_obs * mq_log_e + mq_inv_e * n_obs * psi0 / n_mb) / alpha

    elbo -= .5 * (n_fe_par * log(sigma2_b) + sum(n_re_par .* mq_log_u))
    elbo -= .5 * (mq_sq_t - n_fe_par - sum(n_re_par)) - logdet(Lq)

    elbo += n_re * A_u * log(B_u) - sum(Aq_u .* log.(Bq_u))
    elbo -= n_re * loggamma(A_u) - sum(loggamma.(Aq_u))
    elbo -= sum((A_u .- Aq_u) .* mq_log_u)
    elbo -= sum((B_u .- Bq_u) .* mq_inv_u)

    elbo += A_e * log(B_e) - Aq_e * log(Bq_e)
    elbo -= loggamma(A_e) - loggamma(Aq_e)
    elbo -= (A_e - Aq_e) * mq_log_e + (B_e - Bq_e) * mq_inv_e

    # output
    return elbo
end;

# The compilation fails for a generic abstract type like Family,
# In order to make the compilation success it is needed to specify a
# specific struct belonging to the abstract type Family, like Quantile

for f in FAMILIES
    precompile(_elbo, (
        Vector{FP}, f, Int64, FP, 
        RegParam{FP}, ScaleParam{FP}, 
        VecScaleParam{FP}, Prior, SVB, ))
end

