
# ------------------------------------------------------------------------------
# Initialization, calculation and inplace calculation of
# - the prior/penalization matrix
# - the hessian matrix
# - the gradient vector
# ------------------------------------------------------------------------------


# print the intermediete results of the estimation process
function _midprint(
    iter::Int64,
    dx::Float64,
    df::Float64,
    verbose::Bool,
    report::Int64,
    final::Bool)

    if (iter == 2) & verbose & !final
        println()
        println(" Model fitting ")
        println("-------------------------------------------")
    elseif (iter % report == 0) & verbose & !final
        @printf(" - iter: %3d  - dx: %1.5f  - df: %1.5f \n", iter, dx, df)
    elseif (iter > 2) & verbose & final
        @printf(" - iter: %3d  - dx: %1.5f  - df: %1.5f \n", iter, dx, df)
        println("-------------------------------------------")
    end
end;

precompile(_midprint, (Int64, FP, FP, Bool, Int64, Bool, ))

# compute the prior matrix
function init_prior_matrix(
    s2_inv_b::T, s2_inv_u::T,
    idx_fe::UnitRange{Int64}, 
    idx_re::VecUnitRange{Int64}
    )::DiagMatrix{T} where T <: Float64

    n_fe, n_re = 1, length(idx_re)
    s2_inv_u = repeat([s2_inv_u], n_re)

    return get_prior_matrix(s2_inv_b, s2_inv_u, idx_fe, idx_re)
end;

function get_prior_matrix(
    s2_inv_b::T, s2_inv_u::Vector{T},
    idx_fe::UnitRange{Int64}, 
    idx_re::VecUnitRange{Int64}
    )::DiagMatrix{T} where T <: Float64

    n_fe_par = length(idx_fe)
    n_re_par = length.(idx_re)
    n_tot_par = n_fe_par + sum(n_re_par)

    dg = Vector{T}(undef, n_tot_par)
    dg[idx_fe] .= s2_inv_b
    for (k, idx_k) in enumerate(idx_re)
        dg[idx_k] .= s2_inv_u[k]
    end

    return Diagonal(dg)
end;

function fill_prior_matrix!(
    Q::DiagMatrix{T},
    s2_inv_b::T, s2_inv_u::Vector{T},
    idx_fe::UnitRange{Int64}, 
    idx_re::VecUnitRange{Int64}
    )::DiagMatrix{T} where T <: FP

    Q.diag[idx_fe] .= s2_inv_b
    for (k, idx_k) in enumerate(idx_re)
        Q.diag[idx_k] .= s2_inv_u[k]
    end

    return Q
end;

precompile(init_prior_matrix,  (FP, FP, UnitRange{Int64}, VecUnitRange{Int64}, ))
precompile(get_prior_matrix,   (FP, Vector{FP}, UnitRange{Int64}, VecUnitRange{Int64}, ))
precompile(fill_prior_matrix!, (DiagMatrix{FP}, FP, Vector{FP}, UnitRange{Int64}, VecUnitRange{Int64}, ))

# compute the prior weight vector
function get_prior_weights(
    Q::DiagMatrix{T}, s2_inv_e::T, alpha::T
    )::Vector{T} where T <: FP
    return (alpha / s2_inv_e) .* diag(Q)
end;

function fill_prior_weights!(
    w::Vector{T}, Q::DiagMatrix{T}, s2_inv_e::T, alpha::T
    )::Vector{T} where T <: FP
    w .= (alpha / s2_inv_e) .* diag(Q)
    return w
end;

precompile(get_prior_weights,   (DiagMatrix{FP}, FP, FP, ))
precompile(fill_prior_weights!, (Vector{FP}, DiagMatrix{FP}, FP, FP, ))

# compute the hessian matrix
function init_hessian_matrix(
    C::Matrix{T}, Q::DiagMatrix{T}, f::Family
    )::SymMatrix{T} where T <: FP

    w = typeof(f) <: ClassFamily ? 0.25 : 1.0

    return Symmetric(- w .* (C' * C) .- Q)
end;

function get_hessian_matrix(
    w::Vector{T}, C::Matrix{T}, Q::DiagMatrix{T}
    )::SymMatrix{T} where T <: FP

    return Symmetric(- C' * (w .* C) .- Q)
end;

function fill_hessian_matrix!(
    A::SymMatrix{T}, w::Vector{T},
    C::Matrix{T}, Q::DiagMatrix{T}
    )::SymMatrix{T} where T <: FP

    A .= Symmetric(- C' * (w .* C) .- Q)
    return A
end;

for f in FAMILIES
    precompile(init_hessian_matrix,  (Matrix{FP}, DiagMatrix{FP}, FP, f, ))
end
precompile(get_hessian_matrix,   (Vector{FP}, Matrix{FP}, DiagMatrix{FP}, ))
precompile(fill_hessian_matrix!, (SymMatrix{FP}, Vector{FP}, Matrix{FP}, DiagMatrix{FP}, ))

# compute the gradient vector
function init_gradient_vector(
    y::Vector{T}, C::Matrix{T}, f::Family
    )::Vector{T} where T <: FP

    z = copy(y)
    if typeof(f) in [Logit; Probit; CLogLog]
        z = 2. .* (y .- .5)
    end

    return - C' * z
end;

function get_gradient_vector(
    g::Vector{T}, C::Matrix{T},
    m::Vector{T}, Q::DiagMatrix{T}
    )::Vector{T} where T <: FP

    return - C' * g .- Q * m
end;

function fill_gradient_vector!(
    b::Vector{T}, g::Vector{T}, C::Matrix{T},
    m::Vector{T}, Q::DiagMatrix{T}
    )::Vector{T} where T <: FP

    b .= - C' * g .- Q * m
    return b
end;

for f in FAMILIES
    precompile(init_gradient_vector,  (Vector{FP}, DiagMatrix{FP}, FP, f, ))
end
precompile(get_gradient_vector,   (Vector{FP}, Matrix{FP},Vector{FP}, DiagMatrix{FP}, ))
precompile(fill_gradient_vector!, (Vector{FP}, Vector{FP}, Matrix{FP}, Vector{FP}, DiagMatrix{FP}, ))
