
# Linear predictor
mutable struct LinPred{T <: FP}
    m::Vector{T} # posterior mean vector
    V::Vector{T} # posterior variance vector
end;

precompile(LinPred, (Vector{FP}, Vector{FP}, ))

# Regression parameters
mutable struct RegParam{T <: FP}
    m::Vector{T} # posterior mean vector
    R::LowTriMatrix{T} # posterior Cholescky factor of the variance matrix
    idx_fe::UnitRange{Int64} # fixed effect indices
    idx_re::VecUnitRange{Int64} # random effect indices
end;

precompile(RegParam, (
    Vector{FP}, LowTriMatrix{FP},
    UnitRange{Int64}, VecUnitRange{Int64}, ))

function RegParam(idx_fe::UnitRange{Int64}, idx_re::VecUnitRange{Int64})

    n_fe = length(idx_fe)
    n_re = sum(length.(idx_re))
    n_par = n_fe + n_re

    m = Vector{FP}(undef, n_par)
    R = Matrix{FP}(undef, n_par, n_par)
    R = LowerTriangular(R)

    return RegParam(m, R, idx_fe, idx_re)
end;

precompile(RegParam, (UnitRange{Int64}, VecUnitRange{Int64}, ))

# Scale parameter
mutable struct ScaleParam{T <: FP}
    A::T    # posterior shape parameter
    B::T    # posterior rate parameter
    m::T    # posterior mean
    v::T    # posterior variance
    m_inv::T    # posterior inverse mean
    m_log::T    # posterior log mean

    function ScaleParam(
        A::T = 2.0001, 
        B::T = 1.0001,
        m::T = -1.0,
        v::T = -1.0,
        m_inv::T = -1.0,
        m_log::T = -1.0
        ) where T <: FP

        A > 0.0 ? nothing : error("Invalid parameter: `A` > 0.")
        B > 0.0 ? nothing : error("Invalid parameter: `B` > 0.")
        (m > 0.0) | (m == -1.0) ? nothing : error("Invalid parameter: `m` > 0.")
        (v > 0.0) | (m == -1.0) ? nothing : error("Invalid parameter: `v` > 0.")
        (m_inv > 0.0) | (m_inv == -1.0) ? nothing : error("Invalid parameter: `m_inv` > 0.")
        
        return new{T}(A, B, m, v, m_inv, m_log)
    end
end;

function ScaleParam(A::T, B::T) where T <: FP

    m = _igmean(A, B)
    v = _igvar(A, B)
    m_inv = _iginvmean(A, B)
    m_log = _iglogmean(A, B)

    return ScaleParam(A, B, m, v, m_inv, m_log)
end;

precompile(ScaleParam, (FP, FP, FP, FP, FP, FP, ))
precompile(ScaleParam, (FP, FP, ))

abstract type RegEffect{T <: FP} end

# Fixed effect parameters
mutable struct FixedEffect{T} <: RegEffect{T}
    m::Vector{T}        # posterior mean vector
    V::SymMatrix{T}     # posterior variance-covariance matrix
end;

# Random effect parameters
mutable struct RandomEffect{T} <: RegEffect{T}
    m::Vector{T}        # posterior mean vector
    V::SymMatrix{T}     # posterior variance-covariance matrix
end;

precompile(FixedEffect, (Vector{FP}, SymMatrix{FP}, ))
precompile(RandomEffect, (Vector{FP}, SymMatrix{FP}, ))

const VecScaleParam{T <: FP} = Vector{ScaleParam{T}};
const VecRegEffect{T <: FP} = Vector{RegEffect{T}};

# Fill mean and variance vectors of an existing LinPred object
function fill_lin_pred!(
    eta::LinPred{T}, m::Vector{T}, V::Vector{T}
    )::LinPred{T} where T <: FP

    eta.m .= m
    eta.V .= V

    return eta
end;

precompile(fill_lin_pred!, (LinPred{FP}, Vector{FP}, Vector{FP}, ))

# Fill mean and Cholesky factor of an existing RegParam object
function fill_reg_param!(
    theta::RegParam{T}, m::Vector{T}, R::LowTriMatrix{T}
    )::RegParam{T} where T <: FP

    theta.m .= m
    theta.R .= R

    return theta
end;

precompile(fill_reg_param!, (RegParam{FP}, Vector{FP}, LowTriMatrix{FP}, ))

# Fill shape and rate of an existing ScaleParam object
function fill_scale_param!(
    sigma::ScaleParam{T}, A::T, B::T
    )::ScaleParam{T} where T <: FP

    sigma.A = A
    sigma.B = B
    sigma.m = _igmean(A, B)
    sigma.v = _igvar(A, B)
    sigma.m_inv = _iginvmean(A, B)
    sigma.m_log = _iglogmean(A, B)

    return sigma
end;

function fill_scale_param!(
    sigma::ScaleParam{T}, 
    A::T, B::T, m::T, v::T, m_inv::T, m_log::T
    )::ScaleParam{T} where T <: FP

    sigma.A = A
    sigma.B = B
    sigma.m = m
    sigma.v = v
    sigma.m_inv = m_inv
    sigma.m_log = m_log

    return sigma
end;

precompile(fill_scale_param!, (ScaleParam{FP}, FP, FP, ))
precompile(fill_scale_param!, (ScaleParam{FP}, FP, FP, FP, FP, FP, FP, ))

# Fill shape and rate of an existing vector of ScaleParam object
function fill_scale_param!(
    sigma::VecScaleParam{T}, A::Vector{T}, B::Vector{T}
    )::VecScaleParam{T} where T <: FP

    for k in eachindex(sigma)
        sigma[k].A = A[k]
        sigma[k].B = B[k]
        sigma[k].m = _igmean(A[k], B[k])
        sigma[k].v = _igvar(A[k], B[k])
        sigma[k].m_inv = _iginvmean(A[k], B[k])
        sigma[k].m_log = _iglogmean(A[k], B[k])
    end

    return sigma
end;

function fill_scale_param!(
    sigma::VecScaleParam{T}, 
    A::Vector{T}, B::Vector{T},
    m::Vector{T}, v::Vector{T},
    m_inv::Vector{T}, m_log::Vector{T}
    )::VecScaleParam{T} where T <: FP

    for k in eachindex(sigma)
        sigma[k].A = A[k]
        sigma[k].B = B[k]
        sigma[k].m = m[k]
        sigma[k].v = v[k]
        sigma[k].m_inv = m_inv[k]
        sigma[k].m_log = m_log[k]
    end

    return sigma
end;

precompile(fill_scale_param!, (VecScaleParam{FP}, Vector{FP}, Vector{FP}, ))
precompile(fill_scale_param!, (
    VecScaleParam{FP}, Vector{FP}, Vector{FP}, 
    Vector{FP}, Vector{FP}, Vector{FP}, Vector{FP}, ))

# get the completed design matrix
function get_design_matrix(
    X::Matrix{T}, Z::VecMatrix{T}
    )::Matrix{T} where T <: FP
    return hcat(X, Z...)
end;

# Get the indices relative to fixed and random effect parameters
function get_param_indices(
    X::Matrix{T}, Z::VecMatrix{T}
    )::VecUnitRange{Int64} where T <: FP

    n  = length(Z) + 1
    ii = cumsum([0; size(X, 2); size.(Z, 2)])
    jj = [ii[k]+1:ii[k+1] for k in 1:n]

    return jj
end;

precompile(get_design_matrix, (Matrix{FP}, VecMatrix{FP}, ))
precompile(get_param_indices, (Matrix{FP}, VecMatrix{FP}, ))


# Bayesian mixed model
mutable struct BayesMixedModel{T <: FP} <: RegressionModel
    y::Vector{T} # response data vector
    X::Matrix{T} # fixed effect design matrix
    Z::VecMatrix{T} # random effect design matrices
    family::Family # model family
    eta::LinPred{T} # linear predictor
    theta::RegParam{T} # fixed and random effect parameters
    sigma2_e::ScaleParam{T} # data error scale parameter
    sigma2_u::VecScaleParam{T} # Random effect scale parameters
    elbo::T # Evidence lower bound
    psi_0::T # Variational expected loss
    psi_1::Vector{T} # Variational expected first derivative of the loss
    psi_2::Vector{T} # Variational expected second derivative of the loss
    fitted::Bool # indicates whether the model has been fitted or not

    function BayesMixedModel(
        y::Vector{T},
        X::Matrix{T},
        Z::VecMatrix{T},
        family::Family,
        eta::LinPred{T},
        theta::RegParam{T},
        sigma2_e::ScaleParam{T},
        sigma2_u::VecScaleParam{T},
        elbo::T,
        psi_0::T,
        psi_1::Vector{T},
        psi_2::Vector{T},
        fitted::Bool,
        ) where T <: FP

        # Input dimensionss
        n_obs = length(y)
        nr_X, nc_X = size(X)
        nr_Z = size.(Z, 1)
        nc_Z = size.(Z, 2)

        # Parameter checks
        check_X = n_obs != nr_X
        check_Z = any(n_obs .!= nr_Z)

        # Error messages
        check_X ? error("Incompatible matrix dimensions: `X` must have a number of rows equal to the length of `y`.") : nothing
        check_Z ? error("Incompatible matrix dimensions: the matrices contained in `Z` must have a number of rows equal to the length of `y`.") : nothing

        return new{T}(
            y, X, Z, family,
            eta, theta, sigma2_e, sigma2_u,
            elbo, psi_0, psi_1, psi_2, fitted)
    end
end;

function BayesMixedModel(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T},
    f::Family = Quantile()
    ) where T <: FP

    n_obs = length(y)
    n_fe = 1
    n_re = length(Z)
    n_fe_par = size(X, 2)
    n_re_par = size.(Z, 2)
    n_tot_par = n_fe_par + sum(n_re_par)

    idx = get_param_indices(X, Z)
    idx_fe, idx_re = idx[1], idx[2:end]

    eta = LinPred(zeros(n_obs), zeros(n_obs))
    theta = RegParam(idx_fe, idx_re)
    sigma2_e = ScaleParam(2.0001, 1.0001)
    sigma2_u = [ScaleParam(2.0001, 1.0001) for k in 1:n_re]
    elbo = - 1.0
    psi_0 = - 1.0
    psi_1 = zeros(n_obs)
    psi_2 = zeros(n_obs)
    fitted = false

    return BayesMixedModel(
        y, X, Z, f,
        eta, theta, sigma2_e, sigma2_u,
        elbo, psi_0, psi_1, psi_2, fitted)
end;

function BayesMixedModel(
    y::Vector{T},
    X::Matrix{T},
    Z::VecMatrix{T};
    f::Family = Quantile(0.5)
    ) where T <: FP

    n_obs = length(y)
    n_fe = 1
    n_re = length(Z)
    n_fe_par = size(X, 2)
    n_re_par = size.(Z, 2)
    n_tot_par = n_fe_par + sum(n_re_par)

    idx = get_param_indices(X, Z)
    idx_fe, idx_re = idx[1], idx[2:end]

    eta = LinPred(zeros(n_obs), zeros(n_obs))
    theta = RegParam(idx_fe, idx_re)
    sigma2_e = ScaleParam(2.0001, 1.0001)
    sigma2_u = [ScaleParam(2.0001, 1.0001) for k in 1:n_re]
    elbo = - 1.0
    psi_0 = - 1.0
    psi_1 = zeros(n_obs)
    psi_2 = zeros(n_obs)
    fitted = false

    return BayesMixedModel(
        y, X, Z, f,
        eta, theta, sigma2_e, sigma2_u,
        elbo, psi_0, psi_1, psi_2, fitted)
end;

for f in FAMILIES
    precompile(BayesMixedModel, (
        Vector{FP}, Matrix{FP}, VecMatrix{FP}, f,
        LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP},
        FP, FP, Vector{FP}, Vector{FP}, Bool, ))

    precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, f, ))
end

precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, ))

# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Gaussian,  LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Poisson,   LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Logit,     LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Probit,    LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, CLogLog,   LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Quantile,  LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Expectile, LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, SVR,       LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, SVC,       LinPred{FP}, RegParam{FP}, ScaleParam{FP}, VecScaleParam{FP}, FP, FP, Vector{FP}, Vector{FP}, Bool, ))

# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Gaussian,  ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Poisson,   ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Logit,     ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Probit,    ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, CLogLog,   ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Quantile,  ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, Expectile, ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, SVR,       ))
# precompile(BayesMixedModel, (Vector{FP}, Matrix{FP}, VecMatrix{FP}, SVC,       ))


# Number of observations
function get_nobs(model::BayesMixedModel{T})::Int64 where T <: FP
    return length(model.y)
end

# Number of fixed effects
function get_nfe(model::BayesMixedModel{T})::Int64 where T <: FP
    return 1
end

# Number of random effects
function get_nre(model::BayesMixedModel{T})::Int64 where T <: FP
    return length(model.Z)
end

# Number of fixed effect parameters
function get_nfe_par(model::BayesMixedModel{T})::Int64 where T <: FP
    return size(model.X, 2)
end

# Number of random effect parameters
function get_nre_par(model::BayesMixedModel{T})::Vector{Int64} where T <: FP
    return size.(model.Z, 2)
end

# Total number of regression parameters
function get_ntot_par(model::BayesMixedModel{T})::Int64 where T <: FP
    return size(model.X, 2) + sum(size.(model.Z, 2))
end

precompile(get_nobs,     (BayesMixedModel{FP}, ))
precompile(get_nfe,      (BayesMixedModel{FP}, ))
precompile(get_nre,      (BayesMixedModel{FP}, ))
precompile(get_nfe_par,  (BayesMixedModel{FP}, ))
precompile(get_nre_par,  (BayesMixedModel{FP}, ))
precompile(get_ntot_par, (BayesMixedModel{FP}, ))


function summary(model::BayesMixedModel{T}) where T <: FP

    n_obs = get_nobs(model)
    n_fe = get_nfe(model)
    n_re = get_nre(model)
    n_fe_par = get_nfe_par(model)
    n_re_par = get_nre_par(model)
    n_reg_par = get_ntot_par(model)
    n_tot_par = n_reg_par + n_re + 1

    println()
    println(" Model summary ")
    println("--------------------------------------")
    println(" Family = ", model.family)
    println(" Number of observations = ", n_obs)
    println(" Number of fixed effects = ", n_fe)
    println(" Number of random effects = ", n_re)
    println(" Number of regression parameters : ")
    println("  - total = ", n_reg_par)
    println("  - fixed effects = ", n_fe_par)
    for k in 1:n_re
        println("  - random effects ($k) = ", n_re_par[k])
    end
    println(" Total number of parameters = ", n_tot_par)
    println("-------------------------------------")
end;

precompile(summary, (BayesMixedModel{FP}, ))
