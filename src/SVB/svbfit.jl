
# Inplace estimation of the variational distributions
function fit!(
    model::BayesMixedModel{T},
    prior::Prior,
    alg::SVB,
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
    t = alg.threshold / sqrt2π
    nk = alg.nknots
    alpha = tailorder(f)

    # prior parameters
    A_u = prior.A_u
    B_u = prior.B_u
    A_e = prior.A_e
    B_e = prior.B_e
    s2_inv_b = inv(prior.sigma2_b)
    s2_inv_0 = inv(prior.sigma2_0)

    # q_theta init
    Q = init_prior_matrix(s2_inv_b, s2_inv_0, idx_fe, idx_re)
    A = init_hessian_matrix(C, Q, f)
    b = init_gradient_vector(y, C, f)

    mq, Lq, mq_sq_t = _init_theta(b, A, idx)

    # q_eta init
    fq, vq = _update_eta(mq, Matrix(Lq), C, alg)

    # Variational expectations init
    psi_0, psi_1, psi_2 = _psi(y, fq, sqrt.(vq), f, t, nk)

    # q_sigma_u init
    Aq_u, Bq_u, mq_inv_u = _update_sigma2u(A_u, B_u, n_re_par, mq_sq_t, alg)

    # q_sigma_e init
    Aq_e, Bq_e, mq_inv_e = _update_sigma2e(A_e, B_e, n_obs, psi_0, f, alg)

    # parameter init
    fill_lin_pred!(model.eta, fq, vq)
    fill_reg_param!(model.theta, mq, Lq)
    fill_scale_param!(model.sigma2_u, Aq_u, Bq_u)
    fill_scale_param!(model.sigma2_e, Aq_e, Bq_e)

    # ELBO init
    elbo = _elbo(y, f, psi_0, model.theta, model.sigma2_e, model.sigma2_u, prior, alg)

    # Expected penalization matrix init
    Q = get_prior_matrix(s2_inv_b, mq_inv_u, idx_fe, idx_re)
    w = get_prior_weights(Q, mq_inv_e, alpha)

    # Storing
    opt.fitinit.theta    = mq
    opt.fitinit.sigma2_u = inv.(mq_inv_u)
    opt.fitinit.sigma2_e = inv(mq_inv_e)
    opt.fitinit.elbo     = elbo

    opt.fitlog.theta[1,:]    = mq
    opt.fitlog.sigma2_u[1,:] = inv.(mq_inv_u)
    opt.fitlog.sigma2_e[1]   = inv(mq_inv_e)
    opt.fitlog.elbo[1]       = elbo

    # relative change init
    df = 1.0
    dx = 1.0

    f_init = elbo
    f_old  = elbo
    x_old  = [mq; vech(Lq); Bq_u; Bq_e]

    rate  = alg.rate0
    iter  = 2
    check = true

    # print initial results
    _midprint(iter, dx, df, alg.verbose, alg.report, false)

    # message passing recursion
    while check

        # q_theta update

        # ... step-size update
        rate = _update_rate(alg.rate0, alg.decay, iter)

        # ... gradient and hessian update
        fill_hessian_matrix!(A, mq_inv_e * psi_2 / alpha, C, Q)
        fill_gradient_vector!(b, mq_inv_e * psi_1 / alpha, C, mq, Q)

        # ... mean and variance update
        mq, Lq, mq_sq_t = _update_theta(b, A, rate, y, C, w, mq, idx, f, alg)
        
        # q_eta update
        fq, vq = _update_eta(mq, Matrix(Lq), C, alg)

        # expected loss function
        psi_0, psi_1, psi_2 = _psi(y, fq, sqrt.(vq), f, t, nk)

        # q_sigma_u update
        Aq_u, Bq_u, mq_inv_u = _update_sigma2u(A_u, B_u, n_re_par, mq_sq_t, alg)

        # q_sigma_e update
        Aq_e, Bq_e, mq_inv_e = _update_sigma2e(A_e, B_e, n_obs, psi_0, f, alg)

        # prior matrix update
        fill_prior_matrix!(Q, s2_inv_b, mq_inv_u, idx_fe, idx_re)
        fill_prior_weights!(w, Q, mq_inv_e, alpha)

        # parameter update
        fill_lin_pred!(model.eta, fq, vq)
        fill_reg_param!(model.theta, mq, Lq)
        fill_scale_param!(model.sigma2_u, Aq_u, Bq_u)
        fill_scale_param!(model.sigma2_e, Aq_e, Bq_e)

        # elbo update
        elbo = _elbo(y, f, psi_0, model.theta, model.sigma2_e, model.sigma2_u, prior, alg)

        # convergence parameters
        f_new = elbo
        x_new = [mq; vech(Lq); Bq_u; Bq_e]

        df = absmax(f_new - f_init, f_old - f_init)
        dx = absmax(x_new, x_old)

        f_old = f_new
        x_old = x_new

        # convergence check
        check_xtol = dx > alg.xtol
        check_ftol = df > alg.ftol
        check_iter = iter < alg.maxiter
        check = (check_xtol | check_ftol) & check_iter

        # storing
        opt.fitlog.theta[iter,:]    = mq
        opt.fitlog.sigma2_u[iter,:] = inv.(mq_inv_u)
        opt.fitlog.sigma2_e[iter]   = inv(mq_inv_e)
        opt.fitlog.elbo[iter]       = elbo
        opt.fitlog.rate[iter-1]     = rate
        opt.fitlog.df[iter-1]       = df
        opt.fitlog.dx[iter-1]       = dx

        # update iteration counter
        iter += 1

        # print mid-results
        _midprint(iter, dx, df, alg.verbose, alg.report, false)
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

    # print final results
    _midprint(iter, dx, df, alg.verbose, alg.report, true)

    # Optimization history
    opt.fitlog.theta    = opt.fitlog.theta[1:iter-1,:]
    opt.fitlog.sigma2_u = opt.fitlog.sigma2_u[1:iter-1,:]
    opt.fitlog.sigma2_e = opt.fitlog.sigma2_e[1:iter-1]
    opt.fitlog.elbo     = opt.fitlog.elbo[1:iter-1]
    opt.fitlog.rate     = opt.fitlog.rate[1:iter-2]
    opt.fitlog.df       = opt.fitlog.df[1:iter-2]
    opt.fitlog.dx       = opt.fitlog.dx[1:iter-2]

    # Model updating
    fill_lin_pred!(model.eta, fq, vq)
    fill_reg_param!(model.theta, mq, Lq)
    fill_scale_param!(model.sigma2_u, Aq_u, Bq_u)
    fill_scale_param!(model.sigma2_e, Aq_e, Bq_e)

    model.elbo = elbo
    model.psi_0  = psi_0
    model.psi_1 .= psi_1
    model.psi_2 .= psi_2

    # Final time
    endtime = time()

    # Execution time
    opt.exetime = endtime - inittime

    # Output
    return (model = model, prior = prior, alg = alg, opt = opt)
end

precompile(fit!, (BayesMixedModel{FP}, Prior, SVB, OptSummary, )) 

