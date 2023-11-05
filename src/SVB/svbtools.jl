
# deterministic rate parameter update
function _update_rate(rate0::T, decay::T, iter::Int64)::T where T <: FP
    return rate0 / (1.0 + decay * float(iter))
end;

precompile(_update_rate, (FP, FP, Int64, ))

# line search algorithm
function _linesearch(
    rate::T,
    y::Vector{T},
    C::Matrix{T},
    w::Vector{T},
    m::Vector{T},
    d::Vector{T},
    R::Matrix{T},
    f::Family,
    alg::Algorithm
    )::T where T <: FP

    f_old = C * m
    s_new = _dgchol(C, Matrix(R), true, true)

    a = alg.lbound
    b = alg.ubound

    psi_old, _, _ = _psi(y, f_old, s_new, f, 0.0)
    elbo_old = - psi_old - .5 * m' * (w .* m)

    for k in alg.ntry
        m_new = m - rate * d
        f_new = C * m_new

        psi_new, _, _ = _psi(y, f_new, s_new, f, 0.0)
        elbo_new = - psi_new - .5 * m_new' * (w .* m_new)

        if elbo_new - elbo_old > .0
            break
        else
            rate *= !alg.random ? 0.5 : rand(Uniform(a, b))
        end
    end

    return rate
end;

for f in FAMILIES
    precompile(_linesearch, (
        FP, Vector{FP}, Matrix{FP}, Vector{FP},
        Vector{FP}, Vector{FP}, Matrix{FP}, f,  SVB, ))
end

# variational update of q_sigma2_e (continuous regresion models)
function _update_sigma2e(
    A_e::T, B_e::T, n_obs::Int64, psi_0::T, f::RegFamily, alg::SVB
    ) where T <: FP

    alpha = tailorder(f)
    Aq_e = A_e + float(n_obs) / alpha
    Bq_e = B_e + psi_0 / alpha
    mq_inv_e = _iginvmean(Aq_e, Bq_e)

    return Aq_e, Bq_e, mq_inv_e
end

# variational update of q_sigma2_e (integer regresion models)
function _update_sigma2e(
    A_e::T, B_e::T, n_obs::Int64, psi_0::T, f::IntFamily, alg::SVB
    ) where T <: FP
    return A_e, B_e, 1.
end

# variational update of q_sigma2_e (categorical regresion models)
function _update_sigma2e(
    A_e::T, B_e::T, n_obs::Int64, psi_0::T, f::ClassFamily, alg::SVB
    ) where T <: FP
    return A_e, B_e, 1.
end

for f in FAMILIES
    precompile(_update_sigma2e, (FP, FP, Int64, FP, f, SVB, ))
end

# variational update of q_sigma2_u
function _update_sigma2u(
    A_u::T, B_u::T, n_re_par::Vector{Int64}, mq_sq_t::V, alg::SVB
    ) where {T <: FP, V <: Vector{T}}

    Aq_u = A_u .+ .5 .* float.(n_re_par)
    Bq_u = B_u .+ .5 .* mq_sq_t[2:end]
    mq_inv_u = _iginvmean.(Aq_u, Bq_u)

    return Aq_u, Bq_u, mq_inv_u
end

precompile(_update_sigma2u, (FP, FP, Vector{Int64}, Vector{FP}, SVB, ))

# variational update of q_eta
function _update_eta(
    mq::V, Lq::M, C::M, alg::SVB
    ) where {V <: Vector{FP}, M <: Matrix{FP}}

    fq = C * mq
    vq = _dgchol(C, Lq, true, false)

    return fq, vq
end

precompile(_update_eta, (Vector{FP}, Matrix{FP}, Matrix{FP}, SVB, ))

# variational initialization of q_theta
function _init_theta(
    g::V, H::M, idx::VecUIntRange
    ) where {V <: Vector{FP}, M <: SymMatrix{FP}}

    mq, Lq = _lschol(-H, -g, true)
    mq_sq_t = _sqmean(mq, Matrix(Lq), idx)

    return mq, Lq, mq_sq_t
end

precompile(_init_theta, (Vector{FP}, SymMatrix{FP}, VecUIntRange, ))

# variational initialization of q_theta
function _update_theta(
    g::V, H::S, rate::FP, y::V, C::M, w::V, mq::V, 
    idx::VecUIntRange, f::Family, alg::SVB
    ) where {V <: Vector{FP}, M <: Matrix{FP}, S <: SymMatrix{FP}}

    # direction and Hessian calculation
    dq, Lq = _lschol(-H, -g, true)

    # line-search step
    if alg.search
        rate = _linesearch(rate, y, C, w, mq, dq, Matrix(Lq), f, alg)
    end

    # mean update
    mq .= mq .- rate .* dq
    mq_sq_t = _sqmean(mq, Matrix(Lq), idx)

    return mq, Lq, mq_sq_t
end

precompile(_update_theta, (
    Vector{FP}, SymMatrix{FP}, FP, 
    Vector{FP}, Matrix{FP}, Vector{FP}, Vector{FP}, 
    VecUIntRange, Family, SVB, ))

for f in FAMILIES
    precompile(_update_theta, (
        Vector{FP}, SymMatrix{FP}, FP, 
        Vector{FP}, Matrix{FP}, Vector{FP}, Vector{FP}, 
        VecUIntRange, f, SVB, ))
end
