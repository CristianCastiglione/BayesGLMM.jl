
# print the intermediete results of the estimation process
function _mcmc_midprint(
    iter::Int64,
    logp::Float64,
    verbose::Bool,
    report::Int64,
    final::Bool)

    if (iter == 2) & verbose & !final
        println()
        println(" Model fitting ")
        println("-------------------------------------------")
    elseif (iter % report == 0) & verbose & !final
        @printf(" - iter: %5d  - logp: %1.5f \n", iter, logp)
    elseif (iter > 2) & verbose & final
        @printf(" - iter: %5d  - logp: %1.5f \n", iter, logp)
        println("-------------------------------------------")
    end
end;

precompile(_midprint, (Int64, FP, Bool, Int64, Bool, ))

# log-likelihood for theta
function _log_likelihood_theta(
    theta::V, sigma2e::T, y::V, C::M, f::Family
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    eta = C * theta
    psi0 = sum(dloss(y, eta, f; order = 0))
    loglik = - inv(sigma2e) * psi0 / tailorder(f)

    return loglik
end

for f in FAMILIES
    precompile(_log_likelihood_theta, (
        Vector{FP}, FP, Vector{FP}, Matrix{FP}, f, ))
end

# log-prior for theta
function _log_prior_theta(theta::Vector{FP}, Q::DiagMatrix{FP})

    logpr = - 0.5 * dot(theta, diag(Q) .* theta)

    return logpr
end

precompile(_log_prior_theta, (Vector{FP}, DiagMatrix{FP}, ))

# full-conditional log-posterior for theta
function _log_posterior_theta(
    theta::V, sigma2e::T, y::V, C::M, Q::D, f::Family
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    loglik = _log_likelihood_theta(theta, sigma2e, y, C, f)
    logpr = _log_prior_theta(theta, Q)
    logpost = loglik + logpr

    return logpost
end

for f in FAMILIES
    precompile(_log_posterior_theta, (
        Vector{FP}, FP, Vector{FP}, Matrix{FP}, DiagMatrix{FP}, f, ))
end

# log-proposal for theta
function _log_proposal_theta(
    theta::V, y::V, C::M, w::V, Q::D
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}
    
    A = Symmetric(C' * (w .* C) .+ Q)
    b = C' * (w .* y)

    L = cholesky(A).L
    m = L' \ (L \ b)

    tAt = dot(theta, A * theta)
    tAm = dot(theta, b)
    mAm = dot(b, m)

    sqform_A = tAt + mAm - 2.0 * tAm
    logdet_A = 2.0 .* logdet(L)
    
    logprop = 0.5 * (logdet_A - sqform_A)

    return logprop
end

precompile(_log_proposal_theta, (
    Vector{FP}, Vector{FP}, Matrix{FP}, Vector{FP}, DiagMatrix{FP}))

# acceptance probability for theta (Metropolis-Hastings step)
function _accprob_theta(
    logp_new::T, logp_old::T, logq_new::T, logq_old::T
    ) where T <: FP
    
    log_ratio_p = logp_new - logp_old
    log_ratio_q = logq_old - logq_new

    prob = minimum([1.0, exp(log_ratio_p + log_ratio_q)])
    
    return prob
end

precompile(_accprob_theta, (FP, FP, FP, FP, ))

# mcmc init of q_theta
function _init_theta(
    y::V, C::M, Q::DiagMatrix{T}, f::Family, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    A = C' * C .+ Q
    b = C' * y

    theta = _lssim(A, b)

    return theta
end

for f in CONJUGATE_FAMILIES
    precompile(_init_theta, (Vector{FP}, Matrix{FP}, DiagMatrix{FP}, f, MCMC, ))
end

# mcmc update of q_theta
function _simulate_theta(
    y::V, C::M, Q::D, sigma2e::T, theta::V, omega::M, f::Gaussian, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    A = inv(sigma2e) .* (C' * C) .+ Q
    b = inv(sigma2e) .* (C' * y)

    theta = _lssim(A, b)
    
    return theta
end

function _simulate_theta(
    y::V, C::M, Q::D, sigma2e::T, theta::V, omega::M, f::Logit, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    w = trim.(omega[:,1], 0.001)

    A = C' * (w .* C) .+ Q
    b = C' * (y .- .5)

    theta = _lssim(A, b)

    return theta
end

function _simulate_theta(
    y::V, C::M, Q::D, sigma2e::T, theta::V, omega::M, f::Probit, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    A = C' * C .+ Q
    b = C' * omega[:,1]

    theta = _lssim(A, b)

    return theta
end

function _simulate_theta(
    y::V, C::M, Q::D, sigma2e::T, theta::V, omega::M, f::Quantile, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    tau = f.tau
    lambda = (1.0 - 2.0 * tau) / (tau * (1.0 - tau))
    delta2 = 2.0 / (tau * (1.0 - tau))

    w = inv(delta2 * sigma2e) .* omega[:,2]
    r = w .* (y .- lambda .* omega[:,1])

    A = C' * (w .* C) .+ Q
    b = C' * r

    theta = _lssim(A, b)

    return theta
end

function _simulate_theta(
    y::V, C::M, Q::D, sigma2e::T, theta::V, omega::M, f::Expectile, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    eta = omega[:,1]
    wts = omega[:,2]

    # proposal
    w = inv(sigma2e) .* abs.(f.tau .- (y .≤ C * theta)) / tailorder(f)
    
    A = C' * (w .* C) .+ Q
    b = C' * (w .* y)

    # metropolis step
    metropolis = false

    if metropolis
        theta_old = theta
        theta_new = _lssim(A, b)

        eta_old = C * theta_old
        eta_new = C * theta_new

        # log-posterior
        logp_old = _log_posterior_theta(theta_old, sigma2e, y, C, Q, f)
        logp_new = _log_posterior_theta(theta_new, sigma2e, y, C, Q, f)
        
        # proposal weigths
        w_old = inv(sigma2e) .* abs.(f.tau .- (y .≤ eta_old)) / tailorder(f)
        w_new = inv(sigma2e) .* abs.(f.tau .- (y .≤ eta_new)) / tailorder(f)

        # log-proposal
        logq_old = _log_proposal_theta(theta_new, y, C, w_old, Q)
        logq_new = _log_proposal_theta(theta_old, y, C, w_new, Q)

        # acceptance-rejection step
        prob = _accprob_theta(logp_new, logp_old, logq_new, logq_old)
        theta = rand(Uniform()) ≤ prob ? theta_new : theta_old
    end

    return theta
end

function _simulate_theta(
    y::V, C::M, Q::D, sigma2e::T, theta::V, omega::M, f::SVR, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    n, p = size(C)

    w_lo, w_up = omega[:,1], omega[:,2]
    r_lo, r_up = y .- f.eps, y .+ f.eps

    w = inv(sigma2e) .* (inv.(w_lo) .+ inv.(w_up))
    r = inv(sigma2e) .* (r_lo ./ w_lo .+ r_up ./ w_up)

    A = C' * (w .* C) .+ Q
    b = C' * r

    theta = _lssim(A, b)

    return theta
end

function _simulate_theta(
    y::V, C::M, Q::D, sigma2e::T, theta::V, omega::M, f::SVC, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}, D <: DiagMatrix{T}}

    n, p = size(C)

    w = omega[:,2]
    r = (w .+ 1) .* y

    A = C' * (w .* C) .+ Q
    b = C' * r

    theta = _lssim(A, b)

    return theta
end

for f in CONJUGATE_FAMILIES
    precompile(_simulate_theta, (
        Vector{FP}, Matrix{FP}, DiagMatrix{FP}, 
        Vector{FP}, FP, Matrix{FP}, f, MCMC, ))
end

# mcmc init of q_sigma2_e
function _init_sigma2e(
    A_e::T, B_e::T, y::V, eta::V, f::RegFamily, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}

    Aq_e = A_e + 0.5 * float(length(y))
    Bq_e = B_e + 0.5 * sum((y .- eta).^2)
    sigma2e = rand(InverseGamma(Aq_e, Bq_e))

    return sigma2e
end

function _init_sigma2e(
    A_e::T, B_e::T, y::V, eta::V, f::IntFamily, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}
    return 1.0
end

function _init_sigma2e(
    A_e::T, B_e::T, y::V, eta::V, f::ClassFamily, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}
    return 1.0
end


for f in CONJUGATE_FAMILIES
    precompile(_init_sigma2e, (FP, FP, Vector{FP}, Vector{FP}, f, MCMC, ))
end

# mcmc update of q_sigma2_e
function _simulate_sigma2e(
    A_e::T, B_e::T, y::V, eta::V, omega::M, f::RegFamily, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    alpha = tailorder(f)
    loss = dloss(y, eta, f; order = 0)
    Aq_e = A_e + float(length(y)) / alpha
    Bq_e = B_e + sum(loss) / alpha
    sigma2e = rand(InverseGamma(Aq_e, Bq_e))

    return sigma2e
end

function _simulate_sigma2e(
    A_e::T, B_e::T, y::V, eta::V, omega::M, f::IntFamily, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}
    return 1.0
end

function _simulate_sigma2e(
    A_e::T, B_e::T, y::V, eta::V, omega::M, f::ClassFamily, alg::MCMC
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}
    return 1.0
end

for f in CONJUGATE_FAMILIES
    precompile(_simulate_sigma2e, (FP, FP, Vector{FP}, Vector{FP}, Matrix{FP}, f, MCMC, ))
end


# mcmc init of q_sigma2_u
function _init_sigma2u(
    A_u::T, B_u::T, n_re_par::Vi, sqtheta::Vf, alg::MCMC
    ) where {T <: FP, Vf <: Vector{T}, Vi <: Vector{Int64}}

    Aq_u = A_u .+ 0.5 .* float.(n_re_par)
    Bq_u = B_u .+ 0.5 .* sqtheta[2:end]
    sigma2u = rand.(InverseGamma.(Aq_u, Bq_u))

    return sigma2u
end

precompile(_init_sigma2u, (FP, FP, Vector{Int64}, Vector{FP}, MCMC, ))

# mcmc update of q_sigma2_u
function _simulate_sigma2u(
    A_u::T, B_u::T, n_re_par::Vi, sqtheta::Vf, alg::MCMC
    ) where {T <: FP, Vf <: Vector{T}, Vi <: Vector{Int64}}

    Aq_u = A_u .+ .5 .* float.(n_re_par)
    Bq_u = B_u .+ .5 .* sqtheta[2:end]
    sigma2u = rand.(InverseGamma.(Aq_u, Bq_u))

    return sigma2u
end

precompile(_simulate_sigma2u, (FP, FP, Vector{Int64}, Vector{FP}, MCMC, ))

# mcmc update q_omega
function _simulate_omega(
    y::V, eta::V, sigma2e::T, f::Gaussian, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}
    return ones(length(y), 2)
end

function _simulate_omega(
    y::V, eta::V, sigma2e::T, f::Logit, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}

    omega = rand.(PolyaGamma.(1, eta))
    invomega = inv.(omega)

    return [omega invomega]
end

function _simulate_omega(
    y::V, eta::V, sigma2e::T, f::Probit, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}

    n  = length(y)
    lo = Vector{T}(undef, n)
    up = Vector{T}(undef, n)

    for (i, yi) in enumerate(y)
        if yi == 1.0
            lo[i], up[i] =  0.0, Inf
        else
            lo[i], up[i] = -Inf, 0.0
        end
    end

    omega = rand.(TruncatedNormal.(eta, 1.0, lo, up))

    return [omega omega]
end

function _simulate_omega(
    y::V, eta::V, sigma2e::T, f::Quantile, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}
    
    tau = f.tau
    lambda = (1.0 - 2.0 * tau) / (tau * (1.0 - tau))
    delta2 = 2.0 / (tau * (1.0 - tau))
    
    Aq_w = sqrt.((lambda^2 + 2.0 * delta2) ./ (y .- eta).^2)
    Bq_w = (lambda^2 + 2.0 * delta2) / (delta2 * sigma2e)

    invomega = rand.(InverseGaussian.(Aq_w, Bq_w))
    omega = inv.(invomega)

    return [omega invomega]
end

function _simulate_omega(
    y::V, eta::V, sigma2e::T, f::Expectile, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}
    
    omega = abs.(f.tau .- (y .≤ eta))

    return [eta omega]
end

function _simulate_omega(
    y::V, eta::V, sigma2e::T, f::SVR, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}

    r = y .- eta

    Aq_lo = inv.(abs.(r .- f.eps))
    Aq_up = inv.(abs.(r .+ f.eps))
    Bq_w  = inv(sigma2e)

    omega_lo = inv.(rand.(InverseGaussian.(Aq_lo, Bq_w)))
    omega_up = inv.(rand.(InverseGaussian.(Aq_up, Bq_w)))

    return [omega_lo omega_up]
end

function _simulate_omega(
    y::V, eta::V, sigma2e::T, f::SVC, alg::MCMC
    ) where {T <: FP, V <: Vector{T}}
    
    Aq_w = inv.(abs.(1.0 .- y .* eta))
    
    invomega = rand.(InverseGaussian.(Aq_w, 1.0))
    omega = inv.(invomega)
    
    return [omega invomega]
end

for f in CONJUGATE_FAMILIES
    precompile(_simulate_omega, (Vector{FP}, Vector{FP}, FP, f, MCMC, ))
end


function _mcmc_summary_theta(theta::Matrix{FP}, burn::Int64)

    mq = mean(theta[burn:end,:], dims = 1)[1,:]
    Vq = cov(theta[burn:end,:])
    Dq = Diagonal(diag(Vq))

    if !isposdef(Vq)
        damping = 0.001
        for i in 1:20
            Vq .= (1 - damping) .* Vq .+ damping .* Dq
            isposdef(Vq) ? break : nothing
        end
    end
    
    Lq = cholesky(Vq).L

    return mq, Lq
end

function _mcmc_summary_eta(theta::Matrix{FP}, C, burn::Int64)

    n, p = size(C)

    mq = Vector{FP}(undef, n)
    vq = Vector{FP}(undef, n)

    if n < 100000
        eta = theta[burn:end,:] * C'
        mq = mean(eta, dims = 1)[1,:]
        vq =  var(eta, dims = 1)[1,:]
    else
        for i in 1:n
            etai = theta[burn:end,:] * view(C, i, :)
            mq[i] = mean(etai)
            vq[i] =  var(etai)
        end
    end

    return mq, vq
end

function _mcmc_summary_sigma2u(sigma2u::Matrix{FP}, burn::Int64)

    mq_u = mean(sigma2u[burn:end,:], dims = 1)[1,:]
    vq_u =  var(sigma2u[burn:end,:], dims = 1)[1,:]
    mq_inv_u = mean(inv.(sigma2u[burn:end,:]), dims = 1)[1,:]
    mq_log_u = mean(log.(sigma2u[burn:end,:]), dims = 1)[1,:]
    Aq_u = (mq_u.^2 .+ 2 .* vq_u) ./ vq_u
    Bq_u = mq_u .* (Aq_u .- 1)

    return Aq_u, Bq_u, mq_u, vq_u, mq_inv_u, mq_log_u
end

function _mcmc_summary_sigma2e(sigma2e::Vector{FP}, burn::Int64)

    mq_e = mean(sigma2e[burn:end])
    vq_e =  var(sigma2e[burn:end])
    mq_inv_e = mean(inv.(sigma2e[burn:end]))
    mq_log_e = mean(log.(sigma2e[burn:end]))
    Aq_e = (mq_e^2 + 2*vq_e) / vq_e
    Bq_e = mq_e * (Aq_e - 1)

    return Aq_e, Bq_e, mq_e, vq_e, mq_inv_e, mq_log_e
end

function _mcmc_summary_logp(logp::Vector{FP}, burn::Int64)
    return mean(logp[burn:end])
end

precompile(_mcmc_summary_theta,   (Matrix{FP}, Int64, ))
precompile(_mcmc_summary_eta,     (Matrix{FP}, Int64, ))
precompile(_mcmc_summary_sigma2u, (Matrix{FP}, Int64, ))
precompile(_mcmc_summary_sigma2e, (Vector{FP}, Int64, ))
precompile(_mcmc_summary_logp,    (Vector{FP}, Int64, ))

function _sqtheta(theta::Vector{T}, Q::Matrix{T})::T where T <: FP
    w = sqrt.(diag(Q))
    sq = norm(w .* theta)^2
    return sq
end;

function _sqtheta(theta::Vector{T}, idx::VecUIntRange)::Vector{T} where T <: FP
    sq = [norm(theta[ik])^2 for (k, ik) in enumerate(idx)]
    return sq
end;

function _sqtheta(theta::Vector{T}, Q::Matrix{T}, idx::VecUIntRange)::Vector{T} where T <: FP
    w = sqrt.(diag(Q))
    sq = [norm(w[ik] .* theta[ik])^2 for (k, ik) in enumerate(idx)]
    return sq
end;

precompile(_sqtheta, (Vector{FP}, Matrix{FP}, ))
precompile(_sqtheta, (Vector{FP}, VecUIntRange, ))
precompile(_sqtheta, (Vector{FP}, Matrix{FP}, VecUIntRange, ))
