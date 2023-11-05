
function _igmean end
function _igvar end
function _igstd end
function _iginvmean end
function _iglogmean end
function _igmode end
function _igquantile end
function _igconfint end
function _sqmean end

### INVERSE GAMMA

function _igmode(sigma::VecScaleParam{T})::Vector{T} where T <: Float64
    return [_igmode(sigmak.A, sigmak.B) for sigmak in sigma]
end;

function _igmode(sigma::ScaleParam{T})::T where T <: Float64
    return _igmode(sigma.A, sigma.B)
end;

function _igmode(A::T, B::T)::T where T <: Float64
    return B / (A + 1.0)
end;

precompile(_igmode, (FP, FP, ))
precompile(_igmode, (ScaleParam{FP}, ))
precompile(_igmode, (VecScaleParam{FP}, ))

function _igmedian(sigma::VecScaleParam{T})::Vector{T} where T <: FP
    return [_igmedian(sigmak.A, sigmak.B) for sigmak in sigma]
end;

function _igmedian(sigma::ScaleParam{T})::T where T <: FP
    return _igmedian(sigma.A, sigma.B)
end;

function _igmedian(A::T, B::T)::T where T <: FP
    return median(InverseGamma(A, B))
end;

precompile(_igmedian, (FP, FP, ))
precompile(_igmedian, (ScaleParam{FP}, ))
precompile(_igmedian, (VecScaleParam{FP}, ))

function _igmean(sigma::VecScaleParam{T})::Vector{T} where T <: FP
    return [_igmean(sigmak.A, sigmak.B) for sigmak in sigma]
end;

function _igmean(sigma::ScaleParam{T})::T where T <: FP
    return _igmean(sigma.A, sigma.B)
end;

function _igmean(A::T, B::T)::T where T <: FP
    return B / (A - 1.0)
end;

precompile(_igmean, (FP, FP, ))
precompile(_igmean, (ScaleParam{FP}, ))
precompile(_igmean, (VecScaleParam{FP}, ))

function _igvar(sigma::VecScaleParam{T})::Vector{T} where T <: FP
    return [_igvar(sigmak.A, sigmak.B) for sigmak in sigma]
end;

function _igvar(sigma::ScaleParam{T})::T where T <: FP
    return _igvar(sigma.A, sigma.B)
end;

function _igvar(A::T, B::T)::T where T <: FP
    return B / ((A - 2.0) * (A - 1.0)^2)
end;

precompile(_igvar, (FP, FP, ))
precompile(_igvar, (ScaleParam{FP}, ))
precompile(_igvar, (VecScaleParam{FP}, ))

function _igstd(sigma::VecScaleParam{T})::Vector{T} where T <: FP
    return sqrt.([_igvar(sigmak.A, sigmak.B) for sigmak in sigma])
end;

function _igstd(sigma::ScaleParam{T})::T where T <: FP
    return sqrt(_igvar(sigma.A, sigma.B))
end;

function _igstd(A::T, B::T)::T where T <: FP
    return sqrt(_igvar(A, B))
end;

precompile(_igstd, (FP, FP, ))
precompile(_igstd, (ScaleParam{FP}, ))
precompile(_igstd, (VecScaleParam{FP}, ))

function _iginvmean(sigma::VecScaleParam{T})::Vector{T} where T <: FP
    return [_iginvmean(sigmak.A, sigmak.B) for sigmak in sigma]
end;

function _iginvmean(sigma::ScaleParam{T})::T where T <: FP
    return _iginvmean(sigma.A, sigma.B)
end;

function _iginvmean(A::T, B::T)::T where T <: FP
    return A / B
end;

precompile(_iginvmean, (FP, FP, ))
precompile(_iginvmean, (ScaleParam{FP}, ))
precompile(_iginvmean, (VecScaleParam{FP}, ))

function _iglogmean(sigma::VecScaleParam{T})::Vector{T} where T <: Float64
    return [_iglogmean(sigmak.A, sigmak.B) for sigmak in sigma]
end;

function _iglogmean(sigma::ScaleParam{T})::T where T <: Float64
    return _iglogmean(sigma.A, sigma.B)
end;

function _iglogmean(A::T, B::T)::T where T <: Float64
    return log(B) - digamma(A)
end;

precompile(_iglogmean, (FP, FP, ))
precompile(_iglogmean, (ScaleParam{FP}, ))
precompile(_iglogmean, (VecScaleParam{FP}, ))

function _igquantile(sigma::VecScaleParam{T}, p::T=0.5)::Vector{T} where T <: FP
    return [_igquantile(sigmak.A, sigmak.B) for sigmak in sigma]
end;

function _igquantile(sigma::ScaleParam{T}, p::T=0.5)::T where T <: FP
    return _igquantile(sigma.A, sigma.B)
end;

function _igquantile(A::T, B::T, p::T=0.5)::T where T <: FP
    return quantile(InverseGamma(A, B), p)
end;

precompile(_igquantile, (FP, FP, FP, ))
precompile(_igquantile, (ScaleParam{FP}, FP, ))
precompile(_igquantile, (VecScaleParam{FP}, FP, ))

function _igconfint(sigma::VecScaleParam{T}, p::T = 0.95) where T <: FP
    return [_igconfint(sigmak.A, sigmak.B, p) for sigmak in sigma]
end;

function _igconfint(sigma::ScaleParam{T}, p::T = 0.95) where T <: FP
    return _igconfint(sigma.A, sigma.B, p)
end;

function _igconfint(A::T, B::T, p::T = 0.95) where T <: FP
    lo = Distributions.quantile(InverseGamma(A, B),   (1-p)/2)
    up = Distributions.quantile(InverseGamma(A, B), 1-(1-p)/2)
    return lo, up
end;

precompile(_igconfint, (FP, FP, ))
precompile(_igconfint, (ScaleParam{FP}, ))
precompile(_igconfint, (VecScaleParam{FP}, ))

### GAUSSIAN

function _sqmean(m::Vector{T}, V::SymMatrix{T})::T where T <: FP
    return norm(m)^2 + tr(V)
end;

# expectation of xt*Q*x where x ~ N(m, V)
function _sqmean(m::Vector{T}, V::SymMatrix{T}, Q::DiagMatrix{T})::T where T <: FP
    return m' * Q * m + tr(Q * V)
end;

# expectation of xt*Q*x where x ~ N(m, V) where V = Rt*R
function _sqmean(m::Vector{T}, R::LowTriMatrix{T}, Q::DiagMatrix{T})::T where T <: FP

    w = sqrt.(diag(Q))
    QR = R .* w'
    mQm = norm(w .* m)^2
    trRQR = sum(map(norm, eachcol(QR)).^2)

    return mQm + trRQR
end;

# expectation of xt*x where x ~ N(m, V) where V = Rt*R
function _sqmean(
    m::Vector{T}, R::LowTriMatrix{T}, idx::VecUnitRange{Int64}
    )::Vector{T} where T <: FP

    sqm = zeros(length(idx))
    for (k, ik) in enumerate(idx)
        m2_k = norm(m[ik])^2
        tr_k  = sum(map(norm, eachcol(R[:,ik])).^2)
        sqm[k] = m2_k + tr_k
    end

    return sqm
end;

# expectation of xt*Q*x where x ~ N(m, V) where V = Rt*R
function _sqmean(
    m::Vector{T}, R::LowTriMatrix{T}, Q::DiagMatrix{T}, idx::VecUnitRange{Int64}
    )::Vector{T} where T <: FP

    w = sqrt.(diag(Q))
    sqm = zeros(length(idx))

    for (k, ik) in enumerate(idx)
        m2_k = norm(w[ik] .* m[ik])^2
        QR_k = R[:,ik] .* w[ik]'
        tr_k = sum(map(norm, eachcol(QR_k)).^2)

        sqm[k] = m2_k + tr_k
    end

    return sqm
end;

precompile(_sqmean, (Vector{FP}, SymMatrix{FP}, ))
precompile(_sqmean, (Vector{FP}, SymMatrix{FP}, DiagMatrix{FP}, ))
precompile(_sqmean, (Vector{FP}, LowTriMatrix{FP}, DiagMatrix{FP}, ))
precompile(_sqmean, (Vector{FP}, LowTriMatrix{FP}, VecUnitRange{Int64}, ))
precompile(_sqmean, (Vector{FP}, LowTriMatrix{FP}, DiagMatrix{FP}, VecUnitRange{Int64}, ))


# ------------------------------------------------------------------------------

# expectation of xt*x where x ~ N(m, V)
# function _sqmean(m::Vector{T}, V::Matrix{T})::T where T <: FP
#     return norm(m)^2 + tr(V)
# end;

# expectation of xt*Q*x where x ~ N(m, V)
# function _sqmean(m::Vector{T}, V::Matrix{T}, Q::Matrix{T})::T where T <: FP
#     return m' * Q * m + tr(Q * V)
# end;

# expectation of xt*Q*x where x ~ N(m, V) where V = Rt*R
function _sqmean(m::Vector{T}, R::Matrix{T}, Q::Matrix{T})::T where T <: FP
    w = sqrt.(diag(Q))
    QR = R .* w'
    mQm = norm(w .* m)^2
    trRQR = sum(map(norm, eachcol(QR)).^2)
    return mQm + trRQR
end;

function _sqmean(m::Vector{T}, R::Matrix{T}, idx::VecUnitRange{Int64})::Vector{T} where T <: FP
    sqm = zeros(length(idx))
    for (k, ik) in enumerate(idx)
        m2_k = norm(m[ik])^2
        tr_k  = sum(map(norm, eachcol(R[:,ik])).^2)
        sqm[k] = m2_k + tr_k
    end
    return sqm
end;

function _sqmean(m::Vector{T}, R::Matrix{T}, Q::Matrix{T}, idx::VecUnitRange{Int64})::Vector{T} where T <: FP
    w = sqrt.(diag(Q))
    sqm = zeros(length(idx))
    for (k, ik) in enumerate(idx)
        m2_k = norm(w[ik] .* m[ik])^2
        QR_k = R[:,ik] .* w[ik]'
        tr_k = sum(map(norm, eachcol(QR_k)).^2)

        sqm[k] = m2_k + tr_k
    end
    return sqm
end;

# precompile(_sqmean, (Vector{FP}, Matrix{FP}, ))
# precompile(_sqmean, (Vector{FP}, Matrix{FP}, Matrix{FP}, ))
precompile(_sqmean, (Vector{FP}, Matrix{FP}, Matrix{FP}, ))
precompile(_sqmean, (Vector{FP}, Matrix{FP}, VecUIntRange, ))
precompile(_sqmean, (Vector{FP}, Matrix{FP}, Matrix{FP}, VecUIntRange, ))
