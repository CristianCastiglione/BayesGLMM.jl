
# Inplace estimation of the variational distributions
function fit!(
    model::BayesMixedModel{T},
    prior::Prior,
    alg::MCMC,
    opt::OptSummary
    ) where T <: FP

    # initial time
    inittime = time()

    # Input dimensions
    n_obs = get_nobs(model)
    n_fe = get_fe(model)
    n_re = get_re(model)
    n_fe_par = get_nfe_par(model)
    n_re_par = get_nre_par(model)
    n_tot_par = get_ntot_par(model)

    # Fixed and random parameter indices
    idx = get_param_indices(model.X, model.Z)
    idx_fe, idx_re = idx[1], idx[2:end]

    # Get full design matrix and response vector
    C = get_design_matrix(model.X, model.Z)
    y = model.y

    # Model family and threshold parameter
    f = model.family

    # prior parameters
    A_u = prior.A_u
    B_u = prior.B_u
    A_e = prior.A_e
    B_e = prior.B_e
    s2_inv_b = inv(prior.sigma2_b)
    s2_inv_0 = inv(prior.sigma2_0)

    # prior init
    Q = init_prior_matrix(s2_inv_b, s2_inv_0, idx_fe, idx_re)

    # q_theta init
    theta = _init_theta(y, C, Q, f, alg)

    # squared theta
    sqtheta = _sqtheta(theta, idx)

    # q_eta init
    eta = C * theta

    # q_sigma_u init
    sigma2u = _init_sigma2u(A_u, B_u, n_re_par, sqtheta, alg)
    
    # q_sigma_e init
    sigma2e = _init_sigma2e(A_e, B_e, y, eta, f, alg)
    
    # q_omega init
    omega = _simulate_omega(y, eta, sigma2e, f, alg)

    # loss evaluation
    loss = sum(dloss(y, eta, f, order = 0))

    # Posterior init
    logp = _logp(y, f, loss, idx, theta, sigma2e, sigma2u, prior)

    # Expected penalization matrix init
    Q = get_prior_matrix(s2_inv_b, inv.(sigma2u), idx_fe, idx_re)

    # Storing
    opt.fitinit.theta    = theta
    opt.fitinit.sigma2_u = sigma2u
    opt.fitinit.sigma2_e = sigma2e
    opt.fitinit.elbo     = logp

    opt.fitlog.theta[1,:]    = theta
    opt.fitlog.sigma2_u[1,:] = sigma2u
    opt.fitlog.sigma2_e[1]   = sigma2e
    opt.fitlog.elbo[1]       = logp

    
    # print initial results
    iter = 1
    _mcmc_midprint(iter, logp, alg.verbose, alg.report, false)

    # MCMC recursion
    for iter in 2:alg.maxiter
        
        # q_sigma_e full-conditional simulation
        sigma2e = _simulate_sigma2e(A_e, B_e, y, eta, omega, f, alg)
        
        # q_omega full-conditional simulation
        omega = _simulate_omega(y, eta, sigma2e, f, alg)

        # q_theta full-conditional simulation
        theta = _simulate_theta(y, C, Q, sigma2e, theta, omega, f, alg)

        # squared theta vector
        sqtheta = _sqtheta(theta, idx)

        # q_eta full-conditional simulation
        eta = C * theta

        # q_sigma_u full-conditional simulation
        sigma2u = _simulate_sigma2u(A_u, B_u, n_re_par, sqtheta, alg)

        # prior matrix update
        fill_prior_matrix!(Q, s2_inv_b, inv.(sigma2u), idx_fe, idx_re)

        # loss evaluation
        loss = sum(dloss(y, eta, f, order = 0))

        # elbo update
        logp = _logp(y, f, loss, idx, theta, sigma2e, sigma2u, prior)

        # storing
        opt.fitlog.theta[iter,:]    = theta
        opt.fitlog.sigma2_u[iter,:] = sigma2u
        opt.fitlog.sigma2_e[iter]   = sigma2e
        opt.fitlog.elbo[iter]       = logp

        # print mid-results
        _mcmc_midprint(iter, logp, alg.verbose, alg.report, false)
    end

    if iter == alg.maxiter + 1
        opt.niter   = alg.maxiter
        opt.success = false
    else
        opt.niter   = iter-1
        opt.success = true
    end

    model.fitted = true
    opt.fitted = true
    opt.niter = alg.maxiter

    # print final results
    _mcmc_midprint(alg.maxiter, logp, alg.verbose, alg.report, true)

    # Posterior summaries
    evidence = _mcmc_summary_logp(opt.fitlog.elbo, alg.burn)
    mq, Lq = _mcmc_summary_theta(opt.fitlog.theta, alg.burn)
    fq, vq = _mcmc_summary_eta(opt.fitlog.theta, C, alg.burn)
    Aq_u, Bq_u, mq_u, vq_u, mq_inv_u, mq_log_u = _mcmc_summary_sigma2u(opt.fitlog.sigma2_u, alg.burn)
    Aq_e, Bq_e, mq_e, vq_e, mq_inv_e, mq_log_e = _mcmc_summary_sigma2e(opt.fitlog.sigma2_e, alg.burn)
    
    # Model updating
    fill_lin_pred!(model.eta, fq, vq)
    fill_reg_param!(model.theta, mq, Lq)
    fill_scale_param!(model.sigma2_u, Aq_u, Bq_u, mq_u, vq_u, mq_inv_u, mq_log_u)
    fill_scale_param!(model.sigma2_e, Aq_e, Bq_e, mq_e, vq_e, mq_inv_e, mq_log_e)

    model.elbo   = evidence
    model.psi_0  = .0
    model.psi_1 .= zero(y)
    model.psi_2 .= zero(y)

    # Final time
    endtime = time()

    # Execution time
    opt.exetime = endtime - inittime

    # Output
    return (model = model, prior = prior, alg = alg, opt = opt)
end

precompile(fit!, (BayesMixedModel{FP}, Prior, MCMC, OptSummary, )) 

