

# variational init of q_theta
function _init_theta(
    y::V, C::M, Q::DiagMatrix{T}, f::Family, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    A = init_hessian_matrix(C, Q, f)
    b = init_gradient_vector(y, C, f)

    mq, Lq = _lschol(-A, -b, true)

    return mq, Lq
end

for f in CONJUGATE_FAMILIES
    precompile(_init_theta, (Vector{FP}, Matrix{FP}, DiagMatrix{FP}, f, MFVB, ))
end

# variational update of q_theta
function _update_theta(
    y::V, C::M, Q::DiagMatrix{T}, mq_inv_e::T, mq_w::M, f::Gaussian, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    A = mq_inv_e * (C' * C) .+ Q
    b = mq_inv_e * (C' * y)

    mq, Lq = _lschol(A, b, true)

    return mq, Lq
end

function _update_theta(
    y::V, C::M, Q::DiagMatrix{T}, mq_inv_e::T, mq_w::M, f::Logit, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    A = C' * (view(mq_w, :, 1) .* C) .+ Q
    b = C' * (y .- .5)

    mq, Lq = _lschol(A, b, true)

    return mq, Lq
end

function _update_theta(
    y::V, C::M, Q::DiagMatrix{T}, mq_inv_e::T, mq_w::M, f::Probit, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    A = C' * C .+ Q
    b = C' * (y .* view(mq_w, :, 1) .+ (1.0 .- y) .* view(mq_w, :, 2))

    mq, Lq = _lschol(A, b, true)

    return mq, Lq
end

function _update_theta(
    y::V, C::M, Q::DiagMatrix{T}, mq_inv_e::T, mq_w::M, f::Quantile, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    tau = f.tau
    lambda = (1.0 - 2.0 * tau) / (tau * (1.0 - tau))
    delta2 = 2.0 / (tau * (1.0 - tau))

    A = mq_inv_e * (C' * (view(mq_w, :, 2) .* C)) / delta2 .+ Q
    b = mq_inv_e * (C' * (view(mq_w, :, 2) .* y .- lambda)) / delta2

    mq, Lq = _lschol(A, b, true)

    return mq, Lq
end

function _update_theta(
    y::V, C::M, Q::DiagMatrix{T}, mq_inv_e::T, mq_w::M, f::SVR, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    e = f.eps
    
    w_up = view(mq_w, :, 1)
    w_lo = view(mq_w, :, 2)
    
    w = (w_up + w_lo) ./ (w_lo .* w_up)
    z = ((y .- e) .* w_up + (y .+ e) .* w_lo) ./ (w_lo .* w_up)
    
    A = mq_inv_e * (C' * (w .* C)) .+ Q
    b = mq_inv_e * (C' * z)

    mq, Lq = _lschol(A, b, true)

    return mq, Lq
end

function _update_theta(
    y::V, C::M, Q::DiagMatrix{T}, mq_inv_e::T, mq_w::M, f::SVC, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    A = C' * (view(mq_w, :,2) .* C) .+ Q
    b = C' * ((1.0 .+ view(mq_w, :,2)) .* y)

    mq, Lq = _lschol(A, b, true)

    return mq, Lq
end

for f in CONJUGATE_FAMILIES
    precompile(_update_theta, (
        Vector{FP}, Matrix{FP}, DiagMatrix{FP}, 
        FP, Matrix{FP}, f, MFVB, ))
end

# variational update of q_eta
function _update_eta(
    mq::V, Lq::M, C::M, alg::MFVB
    ) where {V <: Vector{FP}, M <: Matrix{FP}}

    fq = C * mq
    vq = _dgchol(C, Lq, true, false)

    return fq, vq
end

precompile(_update_eta, (Vector{FP}, Matrix{FP}, Matrix{FP}, MFVB, ))

# variational init of q_sigma2_e
function _init_sigma2e(
    A_e::T, B_e::T, y::V, fq::V, vq::V, f::RegFamily, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}

    Aq_e = A_e + 0.5 * float(length(y))
    Bq_e = B_e + 0.5 * sum((y .- fq).^2 .+ vq)
    mq_inv_e = _iginvmean(Aq_e, Bq_e)

    return Aq_e, Bq_e, mq_inv_e
end

function _init_sigma2e(
    A_e::T, B_e::T, y::V, fq::V, vq::V, f::IntFamily, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}
    return A_e, B_e, 1.0
end

function _init_sigma2e(
    A_e::T, B_e::T, y::V, fq::V, vq::V, f::ClassFamily, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}
    return A_e, B_e, 1.0
end

for f in CONJUGATE_FAMILIES
    precompile(_init_sigma2e, (
        FP, FP, Vector{FP}, Vector{FP}, Vector{FP}, f, MFVB, ))
end

# variational update of q_sigma2_e
function _update_sigma2e(
    A_e::T, B_e::T, mq_e::V, mq_sq_e::V, mq_w::M, f::Gaussian, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    Aq_e = A_e + 0.5 * float(length(mq_e))
    Bq_e = B_e + 0.5 * sum(mq_sq_e)
    mq_inv_e = _iginvmean(Aq_e, Bq_e)

    return Aq_e, Bq_e, mq_inv_e
end

function _update_sigma2e(
    A_e::T, B_e::T, mq_e::V, mq_sq_e::V, mq_w::M, f::Logit, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}
    return A_e, B_e, 1.0
end

function _update_sigma2e(
    A_e::T, B_e::T, mq_e::V, mq_sq_e::V, mq_w::M, f::Probit, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}
    return A_e, B_e, 1.0
end

function _update_sigma2e(
    A_e::T, B_e::T, mq_e::V, mq_sq_e::V, mq_w::M, f::Quantile, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    lambda = (1.0 - 2.0 * f.tau) / (f.tau * (1.0 - f.tau))
    delta2 = 2.0 / (f.tau * (1.0 - f.tau))

    mq_sq_v  = lambda^2 * sum(view(mq_w, :, 1))
    mq_sq_v -= 2 * lambda * sum(mq_e)
    mq_sq_v += sum(view(mq_w, :, 2) .* mq_sq_e)

    Aq_e = A_e + 1.5 * float(length(mq_e))
    Bq_e = B_e + sum(view(mq_w, :, 1)) + 0.5 * mq_sq_v / delta2
    mq_inv_e = Aq_e / Bq_e
    
    return Aq_e, Bq_e, mq_inv_e
end

function _update_sigma2e(
    A_e::T, B_e::T, mq_e::V, mq_sq_e::M, mq_w::M, f::SVR, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}

    w_lo = view(mq_w, :, 1)
    w_up = view(mq_w, :, 2)

    mq_sq_lo = view(mq_sq_e, :, 1)
    mq_sq_up = view(mq_sq_e, :, 2)

    mq_sum_loss_lo = 0.5 * sum((w_lo.^2 + mq_sq_lo) ./ w_lo)
    mq_sum_loss_up = 0.5 * sum((w_up.^2 + mq_sq_up) ./ w_up)

    mq_sum_loss = mq_sum_loss_lo + mq_sum_loss_up - f.eps

    Aq_e = A_e + 0.5 * float(size(mq_e, 1))
    Bq_e = B_e + 0.5 * mq_sum_loss
    mq_inv_e = _iginvmean(Aq_e, Bq_e)

    return Aq_e, Bq_e, mq_inv_e
end

function _update_sigma2e(
    A_e::T, B_e::T, mq_e::V, mq_sq_e::V, mq_w::M, f::SVC, alg::MFVB
    ) where {T <: FP, V <: Vector{T}, M <: Matrix{T}}
    return A_e, B_e, 1.0
end

for f in CONJUGATE_FAMILIES
    precompile(_update_sigma2e, (
        FP, FP, Vector{FP}, Vector{FP}, Matrix{FP}, f, MFVB, ))
end

# variational init of q_sigma2_u
function _init_sigma2u(
    A_u::T, B_u::T, n_re_par::Vector{Int64}, mq_sq_t::V, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}

    Aq_u = A_u .+ .5 .* float.(n_re_par)
    Bq_u = B_u .+ .5 .* mq_sq_t[2:end]
    mq_inv_u = _iginvmean.(Aq_u, Bq_u)

    return Aq_u, Bq_u, mq_inv_u
end

precompile(_init_sigma2u, (FP, FP, Vector{Int64}, Vector{FP}, MFVB, ))

# variational update of q_sigma2_u
function _update_sigma2u(
    A_u::T, B_u::T, n_re_par::Vector{Int64}, mq_sq_t::V, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}

    Aq_u = A_u .+ .5 .* float.(n_re_par)
    Bq_u = B_u .+ .5 .* mq_sq_t[2:end]
    mq_inv_u = _iginvmean.(Aq_u, Bq_u)

    return Aq_u, Bq_u, mq_inv_u
end

precompile(_update_sigma2u, (FP, FP, Vector{Int64}, Vector{FP}, MFVB, ))


# variational update q_omega
function _update_omega(
    y::V, fq::V, vq::V, mq_inv_e::T, f::Gaussian, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}
    return ones(length(y), 2)
end

function _update_omega(
    y::V, fq::V, vq::V, mq_inv_e::T, f::Logit, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}

    Bq_w = sqrt.(fq.^2 .+ vq)
    mq_w = .5 .* tanh.(.5 .* Bq_w) ./ Bq_w

    return [mq_w mq_w]
end

function _update_omega(
    y::V, fq::V, vq::V, mq_inv_e::T, f::Probit, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}

    phi1 = pdfn.(-fq)
    phi0 = cdfn.(-fq)

    mq_1 = fq .+ phi1 ./ (1.0 .- phi0)
    mq_0 = fq .- phi1 ./ phi0

    return [mq_1 mq_0]
end

function _update_omega(
    y::V, fq::V, vq::V, mq_inv_e::T, f::Quantile, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}
    
    tau = f.tau
    lambda = (1.0 - 2.0 * tau) / (tau * (1.0 - tau))
    delta2 = 2.0 / (tau * (1.0 - tau))

    Aq_w = mq_inv_e .* (lambda^2 + 2*delta2) ./ delta2
    Bq_w = mq_inv_e .* ((y .- fq).^2 .+ vq)  ./ delta2
    mq_w = sqrt.(Bq_w ./ Aq_w) .+ 1.0 ./ Aq_w
    mq_inv_w = sqrt.(Aq_w ./ Bq_w)
    
    return [mq_w mq_inv_w]
end

function _update_omega(
    y::V, fq::V, vq::V, mq_inv_e::T, f::SVR, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}

    e = f.eps

    mq_sq_lo = (y .- e .- fq).^2 .+ vq
    mq_sq_up = (y .+ e .- fq).^2 .+ vq

    mq_inv_lo = sqrt.(mq_sq_lo)
    mq_inv_up = sqrt.(mq_sq_up)

    return [mq_inv_lo mq_inv_up]
end

function _update_omega(
    y::V, fq::V, vq::V, mq_inv_e::T, f::SVC, alg::MFVB
    ) where {T <: FP, V <: Vector{T}}
    
    Bq_w = (1.0 .- y .* fq).^2 .+ vq
    mq_w = sqrt.(Bq_w) .+ 1
    mq_inv_w = sqrt.(1 ./ Bq_w)

    return [mq_w mq_inv_w]
end

for f in CONJUGATE_FAMILIES
    precompile(_update_omega, (
        Vector{FP}, Vector{FP}, Vector{FP}, FP, f, MFVB, ))
end

# variation update squared errors
function _sqerr(y::V, fq::V, vq::V, f::Gaussian) where {V <: Vector{FP}}
    return (y .- fq).^2 .+ vq 
end

function _sqerr(y::V, fq::V, vq::V, f::Logit) where {V <: Vector{FP}}
    return fq.^2 .+ vq 
end

function _sqerr(y::V, fq::V, vq::V, f::Probit) where {V <: Vector{FP}}
    return ones(length(y))
end

function _sqerr(y::V, fq::V, vq::V, f::Quantile) where {V <: Vector{FP}}
    return (y .- fq).^2 .+ vq 
end

function _sqerr(y::V, fq::V, vq::V, f::SVR) where {V <: Vector{FP}}

    mq_sq_lo = (y .- fq .- f.eps).^2 .+ vq
    mq_sq_up = (y .- fq .+ f.eps).^2 .+ vq

    return [mq_sq_lo mq_sq_up]
end

function _sqerr(y::V, fq::V, vq::V, f::SVC) where {V <: Vector{FP}}
    return (1.0 .- y .* fq).^2 .+ vq 
end

for f in CONJUGATE_FAMILIES
    precompile(_sqerr, (Vector{FP}, Vector{FP}, Vector{FP}, f, ))
end
