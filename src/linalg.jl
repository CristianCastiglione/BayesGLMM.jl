
"""
    _lssim(A, b)

Simulate a Gaussian random vector x ∼ N(μ, Σ) with mean μ = A⁻¹ b and variance Σ = A⁻¹
using the Cholesky factorization of A, i.e. A = L*Lt.
"""
function _lssim(A::SymMatrix{FP}, b::Vector{FP})
    
    n = length(b)
    L = cholesky(A).L
    m = L \ b
    z = rand(Normal(), n)
    x = L' \ (m .+ z)

    return x
end;

function _lssim(A::Matrix{FP}, b::Vector{FP})
    return _lssim(Symmetric(A), b)
end;

precompile(_lssim, (SymMatrix{Float64}, Vector{FP}, ))
precompile(_lssim, (Matrix{Float64}, Vector{FP}, ))


"""
    _lschol(A, b)

Calculate the solution of the linear system Ax = b using the LS-CHOL algorithm.
The ouput is the solution x and the lower Cholesky factor of A, i.e. A = L*Lt.
If `invcheck` is true it is returned also the inverse of L.
"""
function _lschol(A::SymMatrix{FP}, b::Vector{FP}, invcheck::Bool = false)

    L = cholesky(A).L
    m = L' \ (L \ b)
    if invcheck
        R = L \ I
        return m, R
    else
        return m, L
    end
end;

function _lschol(A::Matrix{FP}, b::Vector{FP}, invcheck::Bool = false)
    return _lschol(Symmetric(A), b, invcheck)
end;

precompile(_lschol, (SymMatrix{Float64}, Vector{FP}, ))
precompile(_lschol, (Matrix{Float64}, Vector{FP}, ))

"""
    _trchol(L)

Calculate the trace of a matrix A = L*Lt only using the lower Cholesky factor L.
If `invcheck` is true, the function calculate the trace of inv(A) = Rt*R where
R = inv(L) is the inverse of L.
"""
function _trchol(L::LowTriMatrix{FP}, invcheck::Bool = false)::FP

    trace = .0
    if invcheck
        R = L \ I
        trace = sum(map(norm, eachcol(R)).^2)
    else
        trace = sum(map(norm, eachcol(L)).^2)
    end
    return trace
end;

"""
    _trchol(L, Q)

Calculate tr(Q*A) for a matrix A = L*Lt with lower Cholesky factor L
and a diagonal matrix Q.
If `invcheck` is true, calculate tr(Q*inv(A)) with inv(A) = Rt*R and
R = inv(L) begin the inverse of L.
"""
function _trchol(L::LowTriMatrix{FP}, Q::DiagMatrix{FP}, invcheck::Bool = false)::FP

    trace = .0
    if invcheck
        R  = L \ I
        QR = R .* sqrt.(diag(Q))'
        trace = sum(map(norm, eachcol(QR)).^2)
    else
        QL = L .* sqrt.(diag(Q))
        trace = sum(map(norm, eachcol(QL)).^2)
    end
    return trace
end;

precompile(_trchol, (LowTriMatrix{FP}, Bool, ))
precompile(_trchol, (LowTriMatrix{FP}, DiagMatrix{FP}, Bool, ))

"""
    _dgchol(C, L)

Calculate efficiently diag(C*A*Ct) by only using the lower Cholesky factor
of A = L*Lt without build explicitelly the matrix A.
If `invcheck` is true then diag(C*inv(A)*Ct) is calculated using the fact that
A = L*Lt and inv(A) = Rt*R with R = inv(L).
If `sqrtchack` is true then the function returns as an output the square-root of
the diagonal instead of the diagonal itself.
"""
function _dgchol(C::Matrix{FP}, L::LowTriMatrix{FP}, _inv::Bool = false, _sqrt::Bool = false)::Vector{FP}

    n, p = size(C)
    dg = Vector{FP}(undef, n)

    if _inv
        # dg = [norm(L * view(C, i, :)) for i in 1:n]
        dg = vec(sum((C * L').^2, dims = 2))
    else
        # dg = [norm(L' * view(C, i, :)) for i in 1:n]
        dg = vec(sum((C * L).^2, dims = 2))
    end

    # !_sqrt ? dg = dg.^2 : nothing
    _sqrt ? dg = sqrt.(dg) : nothing

    return dg
end;

function _dgchol(C::Matrix{FP}, L::Matrix{FP}, _inv::Bool = false, _sqrt::Bool = false)::Vector{FP}

    n, p = size(C)
    dg = Vector{FP}(undef, n)

    if _inv
        # dg = [norm(L * view(C, i, :)) for i in 1:n]
        dg = vec(sum((C * L').^2, dims = 2))
    else
        # dg = [norm(L' * view(C, i, :)) for i in 1:n]
        dg = vec(sum((C * L).^2, dims = 2))
    end

    # !_sqrt ? dg = dg.^2 : nothing
    _sqrt ? dg = sqrt.(dg) : nothing

    return dg
end;

"""
    _dgouter(C, A)

Calculates efficiently diag(C*A*Ct) by only using the lower.
It is equivalent to DgChol(C, L) when L is the lower Cholesky factor of A = L*Lt.
If `sqrtchack` is true then the function returns as an output the square-root of
the diagonal instead of the diagonal itself.
"""
function _dgouter(C::Matrix{FP}, V::SymMatrix{FP}, _sqrt::Bool = false)::Vector{FP}

    n, p = size(C)
    dg = [view(C, i, :)' * V * view(C, i, :) for i in 1:n]
    _sqrt ? dg = sqrt.(dg) : nothing

    return dg
end;

function _dgouter(C::Matrix{FP}, V::Matrix{FP}, _sqrt::Bool = false)::Vector{FP}

    n, p = size(C)
    dg = [view(C, i, :)' * V * view(C, i, :) for i in 1:n]
    _sqrt ? dg = sqrt.(dg) : nothing

    return dg
end;

precompile(_dgchol, (Matrix{FP}, LowTriMatrix{FP}, Bool, Bool, ))
precompile(_dgouter, (Matrix{FP}, SymMatrix{FP}, Bool, ))

precompile(_dgchol, (Matrix{FP}, Matrix{FP}, Bool, Bool, ))
precompile(_dgouter, (Matrix{FP}, Matrix{FP}, Bool, ))


"""
    vech(M)

Returns as output the half-vectorization of the squared symmetrix matrix M.
"""
function vech(M::AbstractMatrix{FP})::Vector{FP}

    n = size(M, 1)
    l = n*(n+1)÷2
    v = Vector{FP}(undef, l)

    k = 0
    @inbounds begin
        for i in 1:n
            for j in 1:i
                v[k+j] = M[i,j]
            end
            k += i
        end
    end

    return v
end

precompile(vech, (Matrix{FP}, ))
precompile(vech, (SymMatrix{FP}, ))
precompile(vech, (LowTriMatrix{FP}, ))
