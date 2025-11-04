
# deterministic rate parameter update
function _update_rate(
    rate0::T, frate::T, delay::T, decay::T, iter::Int64
    )::T where T <: FP

    return rate0 / (delay + decay * rate0 * iter)^frate
end

precompile(_update_rate, (FP, FP, FP, FP, Int64, ))

# variational initialization of q_sigma2_e (continuous regresion models)
function _init_sigma2e(
    A_e::T, B_e::T, n_obs::Int64, psi_0::Vector{T}, f::RegFamily, alg::SVI
    ) where T <: FP

    alpha = tailorder(f)
    obsrate = n_obs / alg.minibatch

    Aq_e = A_e + float(n_obs) / alpha
    Bq_e = B_e + obsrate * sum(psi_0) / alpha

    mq_inv_e = _iginvmean(Aq_e, Bq_e)

    return Aq_e, Bq_e, mq_inv_e
end

# variational initialization of q_sigma2_e (integer regresion models)
function _init_sigma2e(
    A_e::T, B_e::T, n_obs::Int64,
    psi_0::Vector{T}, f::IntFamily, alg::SVI
    ) where T <: FP
    
    return A_e, B_e, 1.
end

# variational initialization of q_sigma2_e (categorical regresion models)
function _init_sigma2e(
    A_e::T, B_e::T, n_obs::Int64,
    psi_0::Vector{T}, f::ClassFamily, alg::SVI
    ) where T <: FP
    
    return A_e, B_e, 1.
end

for f in FAMILIES
    precompile(_init_sigma2e, (FP, FP, Int64, FP, f, SVI, ))
end

# variational update of q_sigma2_e (continuous regresion models)
function _update_sigma2e(
    Aq_e_old::T, Bq_e_old::T, A_e::T, B_e::T, n_obs::Int64,
    psi_0::Vector{T}, rate::T, f::RegFamily, alg::SVI
    ) where T <: FP

    alpha = tailorder(f)
    obsrate = n_obs / alg.minibatch

    Aq_e_new = A_e + float(n_obs) / alpha
    Bq_e_new = B_e + obsrate * sum(psi_0) / alpha
    
    Aq_e = (1 - rate) * Aq_e_old + rate * Aq_e_new
    Bq_e = (1 - rate) * Bq_e_old + rate * Bq_e_new

    mq_inv_e = _iginvmean(Aq_e, Bq_e)

    return Aq_e, Bq_e, mq_inv_e
end

# variational update of q_sigma2_e (integer regresion models)
function _update_sigma2e(
    Aq_e_old::T, Bq_e_old::T, A_e::T, B_e::T,
    n_obs::Int64, psi_0::Vector{T}, rate::T, f::IntFamily, alg::SVI
    ) where T <: FP
    return A_e, B_e, 1.
end

# variational update of q_sigma2_e (categorical regresion models)
function _update_sigma2e(
    Aq_e_old::T, Bq_e_old::T, A_e::T, B_e::T,
    n_obs::Int64, psi_0::Vector{T}, rate::T, f::ClassFamily, alg::SVI
    ) where T <: FP
    return A_e, B_e, 1.
end

for f in FAMILIES
    precompile(_update_sigma2e, (FP, FP, FP, FP, Int64, FP, FP, f, SVI, ))
end

# variational initialization of q_sigma2_u
function _init_sigma2u(
    A_u::T, B_u::T, n_re_par::Vector{Int64}, 
    mq_sq_t::V, alg::SVI
    ) where {T <: FP, V <: Vector{T}}

    Aq_u = A_u .+ .5 .* float.(n_re_par)
    Bq_u = B_u .+ .5 .* mq_sq_t[2:end]
    mq_inv_u = _iginvmean.(Aq_u, Bq_u)

    return Aq_u, Bq_u, mq_inv_u
end

precompile(_update_sigma2u, (FP, FP, Vector{Int64}, Vector{FP}, SVI, ))

# variational update of q_sigma2_u
function _update_sigma2u(
    Aq_u_old::V, Bq_u_old::V, A_u::T, B_u::T,
    n_re_par::Vector{Int64}, mq_sq_t::V, rate::FP, alg::SVI
    ) where {T <: FP, V <: Vector{T}}

    Aq_u_new = A_u .+ .5 .* float.(n_re_par)
    Bq_u_new = B_u .+ .5 .* mq_sq_t[2:end]

    Aq_u = (1 - rate) .* Aq_u_old .+ rate .* Aq_u_new
    Bq_u = (1 - rate) .* Bq_u_old .+ rate .* Bq_u_new

    mq_inv_u = _iginvmean.(Aq_u, Bq_u)

    return Aq_u, Bq_u, mq_inv_u
end

precompile(_update_sigma2u, (
    Vector{FP}, Vector{FP}, FP, FP, 
    Vector{Int64}, Vector{FP}, FP, SVI, ))

# variational update of q_eta
function _update_eta(
    mq::V, Lq::M, C::M, alg::SVI
    ) where {V <: Vector{FP}, M <: Matrix{FP}}

    fq = C * mq
    vq = _dgchol(C, Lq, true, false)

    return fq, vq
end

precompile(_update_eta, (Vector{FP}, Matrix{FP}, Matrix{FP}, SVI, ))

# compute the gradient vector
function init_minibatch_gradient(
    y::Vector{T}, C::Matrix{T},
    n_obs::Int64, n_mb::Int64, f::Family
    )::Vector{T} where T <: FP

    z = copy(y)
    if typeof(f) in [Logit; Probit; CLogLog]
        z = y .- .5
    end

    return (n_obs / n_mb) .* (C' * z) ./ tailorder(f)
end;

function get_minibatch_gradient(
    g::Vector{T}, C::Matrix{T},
    m::Vector{T}, Q::DiagMatrix{T},
    n_obs::Int64, n_mb::Int64
    )::Vector{T} where T <: FP

    return - (n_obs / n_mb) .* (C' * g) - Q * m
end;

function fill_minibatch_gradient!(
    b::Vector{T}, g::Vector{T}, C::Matrix{T},
    m::Vector{T}, Q::DiagMatrix{T},
    n_obs::Int64, n_mb::Int64
    )::Vector{T} where T <: FP

    b .= - (n_obs / n_mb) .* (C' * g) - Q * m
    return b
end;

for f in FAMILIES
    precompile(init_minibatch_gradient,  (Vector{FP}, DiagMatrix{FP}, Int64, Int64, f, ))
end
precompile(get_minibatch_gradient,   (Vector{FP}, Matrix{FP}, Vector{FP}, DiagMatrix{FP}, Int64, Int64, ))
precompile(fill_minibatch_gradient!, (Vector{FP}, Vector{FP}, Matrix{FP}, Vector{FP}, DiagMatrix{FP}, Int64, Int64, ))


# compute the hessian matrix
function init_minibatch_hessian(
    C::Matrix{T}, Q::DiagMatrix{T},
    n_obs::Int64, n_mb::Int64, f::Family
    )::SymMatrix{T} where T <: FP

    CWC = (n_obs / n_mb) .* (C' * C) ./ tailorder(f)

    return Symmetric(- CWC - Q)
end;

function get_minibatch_hessian(
    w::Vector{T}, C::Matrix{T}, Q::DiagMatrix{T},
    n_obs::Int64, n_mb::Int64
    )::SymMatrix{T} where T <: FP

    CWC = (n_obs / n_mb) .* (C' * (w .* C))
    return Symmetric(- CWC - Q)
end;

function fill_minibatch_hessian!(
    A::SymMatrix{T}, w::Vector{T},
    C::Matrix{T}, Q::DiagMatrix{T},
    n_obs::Int64, n_mb::Int64
    )::SymMatrix{T} where T <: FP

    CWC = (n_obs / n_mb) .* (C' * (w .* C))
    A  .= Symmetric(- CWC - Q)
    return A
end;

for f in FAMILIES
    precompile(init_minibatch_hessian,  (Matrix{FP}, DiagMatrix{FP}, Int64, Int64, f, ))
end
precompile(get_minibatch_hessian,   (Vector{FP}, Matrix{FP}, DiagMatrix{FP}, Int64, Int64, ))
precompile(fill_minibatch_hessian!, (SymMatrix{FP}, Vector{FP}, Matrix{FP}, DiagMatrix{FP}, Int64, Int64, ))

# variational initialization of lambda (natural parameters)
function _init_lambda(g::V, H::M) where {V <: Vector{FP}, M <: SymMatrix{FP}}

    lambda_1 = g
    lambda_2 = 0.5 * H

    return lambda_1, Symmetric(lambda_2)
end

# variational update of lambda (natural parameters)
function _update_lambda(
    lambda_1_old::V, lambda_2_old::M, dg::V, H::M, rate::FP
    ) where {V <: Vector{FP}, M <: SymMatrix{FP}}

    lambda_1_new = dg # - H * mq_old
    lambda_2_new = 0.5 * H

    lambda_1 = (1 - rate) .* lambda_1_old .+ rate .* lambda_1_new
    lambda_2 = (1 - rate) .* lambda_2_old .+ rate .* lambda_2_new

    return lambda_1, Symmetric(lambda_2)
end

precompile(_init_lambda, (Vector{FP}, SymMatrix{FP}, ))
precompile(_update_lambda, (
    Vector{FP}, SymMatrix{FP}, Vector{FP}, SymMatrix{FP}, FP, ))

# variational initialization of lambda (natural parameters)
function _init_theta(
    g::V, H::M, idx::VecUIntRange
    ) where {V <: Vector{FP}, M <: SymMatrix{FP}}

    mq, Lq = _lschol(-H, -g, true)
    mq_sq_t = _sqmean(mq, Matrix(Lq), idx)

    return mq, Lq, mq_sq_t
end

# variational update of lambda (natural parameters)
function _update_theta(
    lambda_1::V, lambda_2::M, idx::VecUIntRange
    ) where {V <: Vector{FP}, M <: SymMatrix{FP}}

    mq, Lq = _lschol(- 2.0 * lambda_2, lambda_1, true)
    mq_sq_t = _sqmean(mq, Matrix(Lq), idx)

    return mq, Lq, mq_sq_t
end

precompile(_init_theta, (Vector{FP}, SymMatrix{FP}, VecUIntRange, ))
precompile(_update_theta, (Vector{FP}, SymMatrix{FP}, VecUIntRange, ))