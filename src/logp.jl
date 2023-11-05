

function _logp(
    y::Vector{T},
    f::Family,
    loss::T,
    idx::VecUIntRange,
    theta::Vector{T},
    sigma2e::T,
    sigma2u::Vector{T},
    prior::Prior
    )::T where T <: FP

    # Fixed and random parameter indices
    idx_fe = idx[1]
    idx_re = idx[2:end]

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

    # Prior variances
    s2_inv_e = inv(sigma2e)
    s2_log_e = log(sigma2e)

    s2_inv_b = inv(prior.sigma2_b)
    s2_log_b = log(prior.sigma2_b)
    
    s2_inv_u = inv.(sigma2u)
    s2_log_u = log.(sigma2u)

    # squared theta
    sqtheta = _sqtheta(theta, idx)

    # loss tail order
    alpha = tailorder(f)

    # log-posterior density 
    logp  = .0
    logp -= (n_obs * s2_log_e + s2_inv_e * loss) / alpha

    logp -= .5 * (n_fe_par * s2_log_b + sum(n_re_par .* s2_log_u))
    logp -= .5 * sum([s2_inv_b; s2_inv_u] .* sqtheta)

    logp += n_re * A_u * log(B_u) - n_re * loggamma(A_u)
    logp -= A_u * sum(s2_log_u) - B_u * sum(s2_inv_u)

    logp += A_e * log(B_e) - loggamma(A_e)
    logp -= A_e * s2_log_e - B_e * s2_inv_e

    # output
    return logp
end

for f in FAMILIES
    precompile(_logp, (
        Vector{FP}, f, FP, VecUIntRange, 
        Vector{FP}, FP, Vector{FP}, Prior, ))
end
