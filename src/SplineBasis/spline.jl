
using LinearAlgebra
using Distributions
using OffsetArrays
using BSplines

#-------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

function seq(a::T, b::T, l::Int64)::Vector{T} where T <: Float64
    return collect(range(a, b, l))
end

precompile(seq, (Float64, Float64, Int64, ))

#-------------------------------------------------------------------------------
# Boundaries and knots
# ------------------------------------------------------------------------------

function get_bound_knots(x::Vector{T})::Vector{T} where T <: Float64
    a = minimum(x)
    b = maximum(x)

    return [a; b]
end

function get_inner_knots(x::Vector{T}, nknots::Int64)::Vector{T} where T <: Float64
    p = seq(0.0, 1.0, nknots+2)[2:nknots+1]
    k = quantile(x, p)

    return k
end

function get_unif_knots(x::Vector{T}, nknots::Int64)::Vector{T} where T <: Float64
    a = minimum(x)
    b = maximum(x)
    k = seq(a, b, nknots+2)[2:nknots+1]
    return k
end

precompile(get_bound_knots, (Vector{Float64}, ))
precompile(get_inner_knots, (Vector{Float64}, Int64, ))
precompile(get_unif_knots , (Vector{Float64}, Int64, ))

#-------------------------------------------------------------------------------
# B-spline matrices
# ------------------------------------------------------------------------------

function get_bspline_matrix(
    x::Vector{T},
    k::Vector{T},
    a::T, b::T,
    order::Int64 = 4,
    deriv::Int64 = 0
    )::Matrix{T} where T <: Float64

    d = length(k)
    n = length(x)

    B = zeros(n, d+order)

    basis = BSplineBasis(order, [a; k; b])
    for i in 1:n
        b = bsplines(basis, x[i], Derivative(deriv))
        j = b.offsets[1]
        B[i,j+1:j+order] = parent(b)
    end

    return B
end

function get_bspline_energy(
    k::Vector{T}, a::T, b::T,
    )::Matrix{T} where T <: Float64

    d = length(k)
    L = 3 * (d + 8)

    K = [repeat([a], 4); k; repeat([b], 4)]
    x = .5 .* (repeat(K, inner=3)[2:L-2] .+ repeat(K, inner=3)[3:L-1])
    w = repeat(diff(K), inner=3) .* repeat([1; 4; 1] ./ 6, outer=d+7)
    B = get_bspline_matrix(x, K, a, b, 4, 2)
    P = (B' * Diagonal(w) * B)[5:d+8,5:d+8]

    return P
end

function get_bspline_system(
    x::Vector{T},
    k::Vector{T},
    a::T,
    b::T
    )::Tuple where T <: Float64

    B = get_bspline_matrix(x, k, a, b, 4, 0)
    P = get_bspline_energy(k, a, b)

    return B, P
end

precompile(get_bspline_matrix, (Vector{Float64}, Vector{Float64}, Float64, Float64, Int64, Int64, ))
precompile(get_bspline_energy, (Vector{Float64}, Float64, Float64, ))
precompile(get_bspline_system, (Vector{Float64}, Vector{Float64}, Float64, Float64, ))

#-------------------------------------------------------------------------------
# O-spline matrices
# ------------------------------------------------------------------------------

function get_ospline_matrix(
    x::Vector{T}, k::Vector{T}, a::T, b::T
    )::Matrix{T} where T <: Float64

    n = length(x)
    d = length(k)

    B = get_bspline_matrix(x, k, a, b, 4, 0)
    P = get_bspline_energy(k, a, b)

    D , U  = eigen(P)
    DZ, UZ = D[3:end], U[:,3:end]

    L = UZ ./ sqrt.(DZ)'
    X = [ones(n) x]
    Z = B * L

    return [X Z]
end

function get_ospline_energy(
    k::Vector{T}, a::T, b::T
    )::Matrix{T} where T <: Float64

    d = length([a; k; b])
    R = Diagonal(I, d+2)
    R[1,1] = .0
    R[2,2] = .0

    return R
end

function get_ospline_system(
    x::Vector{T}, k::Vector{T}, a::T, b::T
    )::Tuple where T <: Float64

    n = length(x)
    d = length(k)

    B = get_bspline_matrix(x, k, a, b, 4, 0)
    P = get_bspline_energy(k, a, b)

    D , U  = eigen(P)
    DZ, UZ = D[3:end], U[:,3:end]

    L = UZ * Diagonal(1 ./ sqrt.(DZ))
    X = [ones(n) x]
    Z = B * L

    d = length([a; k; b])
    R = Diagonal(I, d)

    return X, Z, R, L
end

precompile(get_ospline_matrix, (Vector{Float64}, Vector{Float64}, Float64, Float64, ))
precompile(get_ospline_energy, (Vector{Float64}, Float64, Float64, ))
precompile(get_ospline_system, (Vector{Float64}, Vector{Float64}, Float64, Float64, ))

#-------------------------------------------------------------------------------
# End of file
# ------------------------------------------------------------------------------

mutable struct SplineBasis{T <: Float64}
    data::Vector{T}
    nknots::Int64
    bounds::Vector{T}
    order::Int64
    unif::Bool
    intercept::Bool

    function SplineBasis(
        data::Vector{T},
        nknots::Int64 = 10,
        bounds::Vector{T} = [minimum(data); maximum(data)],
        order::Int64 = 4,
        unif::Bool = true,
        intercept::Bool = false
        ) where T <: Float64
        return new{T}(data, nknots, bounds, order, unif, intercept)
    end
end

const VecSplineBasis{T <: Float64} = Vector{SplineBasis{T}}

function get_basis_matrix(s::SplineBasis{T}) where T <: Float64

    x = s.data
    n = s.nknots
    a = s.bounds[1]
    b = s.bounds[2]
    k = unif ? get_unif_knots(x, n) : get_inner_knots(x, n)
    X, Z = get_ospline_matrix(x, k, a, b)

    X = s.intercept ? X : X[:,2:end]

    return X, Z
end

function get_basis_matrix(s::VecSplineBasis{T}) where T <: Float64

    X, Z = [], []
    for (k, sk) in enumerate(s)
        Xk, Zk = get_basis_matrix(sk)
        push!(X, Xk)
        push!(X, Zk)
    end
    X = hcat(X...)

    return X, Z
end
