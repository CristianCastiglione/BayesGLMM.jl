
# Inplace estimation of the variational distributions
function fit!(
    model::BayesMixedModel{T},
    prior::Prior,
    alg::SVI,
    opt::OptSummary
    ) where T <: FP

    # Initial time
    inittime = time()

    # Input dimensions
    n_obs = get_nobs(model)
    n_mb = alg.minibatch
    n_init = alg.initbatch
    n_fe = get_fe(model)
    n_re = get_re(model)
    n_fe_par = get_nfe_par(model)
    n_re_par = get_nre_par(model)
    n_tot_par = get_ntot_par(model)
    
    # SVB initialization
    svb = SVB(
        maxiter = alg.maxiter, verbose = alg.verbose, 
        report = alg.report, ftol = alg.ftol, xtol = alg.ftol,
        nknots = alg.nknots, threshold = alg.threshold,
        search = false, random = false,
        rate0 = alg.rate0, decay = alg.decay
    )

    # Fixed and random parameter indices
    idx = get_param_indices(model.X, model.Z)
    idx_fe, idx_re = idx[1], idx[2:end]

    # Get full design matrix and response vector
    C = get_design_matrix(model.X, model.Z)
    y = model.y

    # Model family and threshold parameter
    f = model.family
    t = alg.threshold / sqrt2π
    alpha = tailorder(f)

    # prior parameters
    A_u = prior.A_u
    B_u = prior.B_u
    A_e = prior.A_e
    B_e = prior.B_e
    s2_inv_b = inv(prior.sigma2_b)
    s2_inv_0 = inv(prior.sigma2_0)

    # indices of 0-1 response (for classification)
    idx_s = nothing # success
    idx_f = nothing # failure
    if (typeof(f) <: ClassFamily) & alg.stratified
        idx_s = findall(y .== 1.)
        idx_f = setdiff(1:n_obs, idx_s)
    end

    # minibatch sampling
    idx_0 = minibatch_indices(n_obs, n_init, idx_s, idx_f, alg.stratified)
    idx_mb = minibatch_indices(n_obs, n_mb, idx_s, idx_f, alg.stratified)
    rate_mb = n_obs / n_mb

    # minibatch response and design matrix
    y_mb = y[idx_mb]
    C_mb = C[idx_mb,:]

    # initial minibatch response and design matrix
    y_init = y[idx_0]
    C_init = C[idx_0,:]

    # q_theta init
    Q = init_prior_matrix(s2_inv_b, s2_inv_0, idx_fe, idx_re)
    A = init_minibatch_hessian(C_init, Q, n_obs, n_init, f)
    b = init_minibatch_gradient(y_init, C_init, n_obs, n_init, f)
    # A = init_minibatch_hessian(C_mb, Q, n_obs, n_mb, f)
    # b = init_minibatch_gradient(y_mb, C_mb, n_obs, n_mb, f)

    lambda_1, lambda_2 = _init_lambda(b, A)
    mq, Lq, mq_sq_t = _init_theta(lambda_1, lambda_2, idx)
    lambda_1_avg, lambda_2_avg = lambda_1, lambda_2

    # q_eta init
    fq_mb, vq_mb = _update_eta(mq, Matrix(Lq), C_mb, alg)

    # Variational expectations init
    psi_0_mb, psi_1_mb, psi_2_mb = _psi(y_mb, fq_mb, sqrt.(vq_mb), f, t)

    # q_sigma_u init
    Aq_u, Bq_u, mq_inv_u = _init_sigma2u(A_u, B_u, n_re_par, mq_sq_t, alg)

    # q_sigma_e init
    Aq_e, Bq_e, mq_inv_e = _init_sigma2e(A_e, B_e, n_obs, psi_0_mb, f, alg)

    # parameter init
    fill_reg_param!(model.theta, mq, Lq)
    fill_scale_param!(model.sigma2_u, Aq_u, Bq_u)
    fill_scale_param!(model.sigma2_e, Aq_e, Bq_e)

    # ELBO init
    elbo = _elbo(y_mb, f, n_obs, psi_0_mb, model.theta, model.sigma2_e, model.sigma2_u, prior, alg)

    # Expected penalization matrix init
    Q = get_prior_matrix(s2_inv_b, mq_inv_u, idx_fe, idx_re)

    # Storing
    opt.fitinit.theta = mq
    opt.fitinit.sigma2_u = inv.(mq_inv_u)
    opt.fitinit.sigma2_e = inv(mq_inv_e)
    opt.fitinit.elbo = elbo

    opt.fitlog.theta[1,:] = mq
    opt.fitlog.sigma2_u[1,:] = inv.(mq_inv_u)
    opt.fitlog.sigma2_e[1] = inv(mq_inv_e)
    opt.fitlog.elbo[1] = elbo

    # relative change init
    df = 1.0
    dx = 1.0

    f_init = elbo
    f_old = elbo
    x_old = [mq; vech(Lq); Bq_u; Bq_e]

    rate = alg.rate0
    check = true

    # print initial results
    _midprint(2, dx, df, alg.verbose, alg.report, false)

    # stochastic message passing recursion
    for iter in 2:alg.maxiter

        # q_theta update
        
        # ... step-size update
        rate = _update_rate(alg.rate0, alg.frate, alg.delay, alg.decay, iter)

        # ... gradient and Hessian update
        fill_minibatch_hessian!(A, mq_inv_e * psi_2_mb / alpha, C_mb, Q, n_obs, n_mb)
        fill_minibatch_gradient!(b, mq_inv_e * psi_1_mb / alpha, C_mb, mq, Q, n_obs, n_mb)

        # ... natural parameter update
        lambda_1, lambda_2 = _update_lambda(lambda_1, lambda_2, b - A * mq, A, rate)

        # ... natural parameter averaging
        if (iter ≥ alg.burn) & alg.averaging
            n_avg = iter - alg.burn
            
            lambda_1_avg = (lambda_1 .+ n_avg .* lambda_1_avg) ./ (n_avg + 1)
            lambda_2_avg = (lambda_2 .+ n_avg .* lambda_2_avg) ./ (n_avg + 1)
        else
            lambda_1_avg = lambda_1
            lambda_2_avg = lambda_2
        end
        
        # ... mean and variance update
        mq, Lq, mq_sq_t = _update_theta(lambda_1, lambda_2, idx)

        # minibatch sampling
        idx_mb = minibatch_indices(n_obs, n_mb, idx_s, idx_f, alg.stratified)
        y_mb = y[idx_mb]
        C_mb = C[idx_mb,:]

        # q_eta update
        fq_mb, vq_mb = _update_eta(mq, Matrix(Lq), C_mb, alg)

        # expected loss function
        psi_0_mb, psi_1_mb, psi_2_mb = _psi(y_mb, fq_mb, sqrt.(vq_mb), f, t)

        # q_sigma_u update
        Aq_u, Bq_u, mq_inv_u = _update_sigma2u(Aq_u, Bq_u, A_u, B_u, n_re_par, mq_sq_t, rate, alg)

        # q_sigma_e update
        Aq_e, Bq_e, mq_inv_e = _update_sigma2e(Aq_e, Bq_e, A_e, B_e, n_obs, psi_0_mb, rate, f, alg)

        # prior matrix update
        fill_prior_matrix!(Q, s2_inv_b, mq_inv_u, idx_fe, idx_re)

        # parameter update
        fill_reg_param!(model.theta, mq, Lq)
        fill_scale_param!(model.sigma2_u, Aq_u, Bq_u)
        fill_scale_param!(model.sigma2_e, Aq_e, Bq_e)

        # elbo update
        elbo = _elbo(y_mb, f, n_obs, psi_0_mb, model.theta, model.sigma2_e, model.sigma2_u, prior, alg)

        # convergence parameters
        f_new = elbo
        x_new = [mq; vech(Lq); Bq_u; Bq_e]

        f_cur = (1 - rate) * f_old + rate * f_new
        x_cur = .9 * x_old + .1 * x_new

        df = absmax(f_cur - f_init, f_old - f_init)
        dx = absmax(x_cur, x_old)

        f_old = f_cur
        x_old = x_cur

        # convergence check
        check_xtol = dx > alg.xtol
        check_ftol = df > alg.ftol
        check_miniter = iter > alg.miniter
        check_maxiter = iter < alg.maxiter
        check = check_miniter ? (check_xtol | check_ftol) & check_maxiter : true
        

        # storing
        opt.fitlog.theta[iter,:] = mq
        opt.fitlog.sigma2_u[iter,:] = inv.(mq_inv_u)
        opt.fitlog.sigma2_e[iter] = inv(mq_inv_e)
        opt.fitlog.elbo[iter] = f_cur
        opt.fitlog.rate[iter-1] = rate
        opt.fitlog.df[iter-1] = df
        opt.fitlog.dx[iter-1] = dx

        # update iteration counter
        iter += 1

        # print mid-results
        _midprint(iter, dx, df, alg.verbose, alg.report, false)
    end

    if check
        opt.niter = alg.maxiter
        opt.success = false
    else
        opt.niter = alg.maxiter
        opt.success = true
    end

    model.fitted = true
    opt.fitted = true

    # print final results
    _midprint(alg.maxiter, dx, df, alg.verbose, alg.report, true)

    # Optimization history
    # opt.fitlog.theta = opt.fitlog.theta[1:iter-1,:]
    # opt.fitlog.sigma2_u = opt.fitlog.sigma2_u[1:iter-1,:]
    # opt.fitlog.sigma2_e = opt.fitlog.sigma2_e[1:iter-1]
    # opt.fitlog.elbo = opt.fitlog.elbo[1:iter-1]
    # opt.fitlog.rate = opt.fitlog.rate[1:iter-2]
    # opt.fitlog.df = opt.fitlog.df[1:iter-2]
    # opt.fitlog.dx = opt.fitlog.dx[1:iter-2]

    # Variational averaging
    if alg.averaging
        lambda_1_avg, lambda_2_avg = lambda_1_avg, Symmetric(lambda_2_avg)
        mq, Lq, mq_sq_t = _update_theta(lambda_1_avg, lambda_2_avg, idx)
    end
    
    # predictive distribution
    fq, vq = _update_eta(mq, Matrix(Lq), C, alg)

    # expected loss function
    psi_0, psi_1, psi_2 = _psi(y, fq, sqrt.(vq), f, t, alg.nknots)

    # evidence lower bound
    elbo = _elbo(y, f, psi_0, model.theta, model.sigma2_e, model.sigma2_u, prior, svb)

    # Model updating
    fill_lin_pred!(model.eta, fq, vq)
    fill_reg_param!(model.theta, mq, Lq)
    fill_scale_param!(model.sigma2_u, Aq_u, Bq_u)
    fill_scale_param!(model.sigma2_e, Aq_e, Bq_e)

    model.elbo   = elbo
    model.psi_0 .= psi_0
    model.psi_1 .= psi_1
    model.psi_2 .= psi_2

    # Final time
    endtime = time()

    # Execution time
    opt.exetime = endtime - inittime

    # Output
    return (model = model, prior = prior, alg = alg, opt = opt)
end

precompile(fit!, (BayesMixedModel{FP}, Prior, SVI, OptSummary, )) 

