
# get the fixed effects from the regressin parameters
function get_fe(model::BayesMixedModel{T}) where T <: FP
    return get_fe(model.theta)
end;

function get_fe(rp::RegParam{T}) where T <: FP
    idx = rp.idx_fe
    m_fe = rp.m[idx]
    V_fe = Symmetric(rp.R[:,idx]' * rp.R[:,idx])
    return FixedEffect{T}(m_fe, V_fe)
end;

precompile(get_fe, (BayesMixedModel{FP}, ))
precompile(get_fe, (RegParam{FP}, ))

# get the random effects from the regressin parameters
function get_re(model::BayesMixedModel{T}) where T <: FP
    return get_re(model.theta)
end;

function get_re(rp::RegParam{T}) where T <: FP
    re = RandomEffect{T}[]
    for (k, idx) in enumerate(rp.idx_re)
        m_re_k = rp.m[idx]
        V_re_k = Symmetric(rp.R[:,idx]' * rp.R[:,idx])
        push!(re, RandomEffect{T}(m_re_k, V_re_k))
    end
    return re
end;

function get_re(model::BayesMixedModel{T}, k::Int64) where T <: FP
    return get_re(model.theta, k)
end;

function get_re(rp::RegParam{T}, k::Int64) where T <: FP
    idx = rp.idx_re[k]
    m_re = rp.m[idx]
    V_re = Symmetric(rp.R[:,idx]' * rp.R[:,idx])
    return re
end;

precompile(get_re, (BayesMixedModel{FP}, ))
precompile(get_re, (RegParam{FP}, ))
precompile(get_re, (BayesMixedModel{FP}, Int64, ))
precompile(get_re, (RegParam{FP}, Int64, ))

fixedeffects(model::BayesMixedModel{FP}) = get_fe(model)
randomeffects(model::BayesMixedModel{FP}) = get_re(model)
randomeffects(model::BayesMixedModel{FP}, k::Int64) = get_re(model, k)

fixedeffects(rp::RegParam{FP}) = get_fe(rp)
randomeffects(rp::RegParam{FP}) = get_re(rp)
randomeffects(rp::RegParam{FP}, k::Int64) = get_re(rp, k)

precompile(fixedeffects,  (BayesMixedModel{FP}, ))
precompile(randomeffects, (BayesMixedModel{FP}, ))
precompile(randomeffects, (BayesMixedModel{FP}, Int64, ))

precompile(fixedeffects,  (RegParam{FP}, ))
precompile(randomeffects, (RegParam{FP}, ))
precompile(randomeffects, (RegParam{FP}, Int64, ))

dispersion(model::BayesMixedModel{FP}) = model.sigma2_e
randeffvar(model::BayesMixedModel{FP}) = model.sigma2_u
randeffvar(model::BayesMixedModel{FP}, k::Int64 = 1) = model.sigma2_u[k]

precompile(dispersion, (BayesMixedModel{FP}, ))
precompile(randeffvar, (BayesMixedModel{FP}, ))
precompile(randeffvar, (BayesMixedModel{FP}, Int64, ))

# Regression effect: mean vector
function mean(re::RegEffect{T})::Vector{T} where T <: FP
    return re.m
end;

function coef(re::RegEffect{T})::Vector{T} where T <: FP
    return re.m
end;

# Regresssion effect: variance vector
function var(re::RegEffect{T})::Vector{T} where T <: FP
    return diag(re.V)
end;

# Regresssion effect: standard deviation vector
function std(re::RegEffect{T})::Vector{T} where T <: FP
    return sqrt.(var(re))
end;

# Regresssion effect: standard deviation vector
function stderror(re::RegEffect{T})::Vector{T} where T <: FP
    return sqrt.(var(re))
end;

# Regresssion effect: covariance matrix
function cov(re::RegEffect{T})::SymMatrix{T} where T <: FP
    return re.V
end;

# Regresssion effect: correlation matrix
function cor(re::RegEffect{T})::SymMatrix{T} where T <: FP
    D = inv.(sqrt.(diag(re.V)))
    R = Symmetric(D .* re.V .* D')
    return R
end;

# Regression effect: credibility intervals
function confint(re::RegEffect{T}, level::T = 0.95)::Matrix{T} where T <: FP
    m = mean(re)
    s = stderror(re)
    p = 1 - (1 - level) / 2
    z = quantile(Normal(), p)
    lo = m .- z .* s
    up = m .+ z .* s
    return [lo up]
end;

precompile(mean,     (RegEffect{FP}, ))
precompile(coef,     (RegEffect{FP}, ))
precompile(var,      (RegEffect{FP}, ))
precompile(std,      (RegEffect{FP}, ))
precompile(stderror, (RegEffect{FP}, ))
precompile(cov,      (RegEffect{FP}, ))
precompile(cor,      (RegEffect{FP}, ))
precompile(confint,  (RegEffect{FP}, FP, ))


function coef(model::BayesMixedModel{T}, k::Int64 = -1)::Vector{T} where T <: FP
    n = length(model.Z)
    if 0 < k ≤ n
        m = get_re(model)[k].m
    elseif k == 0
        m = get_fe(model).m
    elseif k == -1
        m = model.theta.m
    else
        error("Invalid parameter: k ≥ -1.")
    end
    return m
end;

function var(model::BayesMixedModel{T}, k::Int64 = -1)::Vector{T} where T <: FP
    n = length(model.Z)
    if 0 < k ≤ n
        v = diag(get_re(model)[k].V)
    elseif k == 0
        v = diag(get_fe(model).V)
    elseif k == -1
        v = map(norm, eachcol(model.theta.R)).^2
    else
        error("Invalid parameter: k ≥ -1.")
    end
    return v
end;

function std(model::BayesMixedModel{T}, k::Int64 = -1)::Vector{T} where T <: FP
    return sqrt.(var(model, k))
end;

function stderror(model::BayesMixedModel{T}, k::Int64 = -1)::Vector{T} where T <: FP
    return sqrt.(var(model, k))
end;

function confint(model::BayesMixedModel{T}, k::Int64 = -1, level::T = 0.95)::Matrix{T} where T <: FP
    p = 1 - level
    z = quantile(Normal(), 1-p/2)
    co = coef(model, k)
    se = std(model, k)
    lo = co .- z .* se
    up = co .+ z .* se
    return [lo up]
end;

function cov(model::BayesMixedModel{T}, k::Int64 = -1)::SymMatrix{T} where T <: FP
    n = length(model.Z)
    if 0 < k ≤ n
        V = re = get_re(model)[k].V
    elseif k == 0
        V = fe = get_fe(model).V
    elseif k == -1
        V = Symmetric(model.theta.R' * model.theta.R)
    else
        error("Invalid parameter: k ≥ -1.")
    end
    return V
end;

function cor(model::BayesMixedModel{T}, k::Int64 = -1)::SymMatrix{T} where T <: FP
    V = cov(model, k)
    D = inv.(sqrt.(diag(V)))
    R = Symmetric(D .* V .* D')
    return R
end;

precompile(coef,     (BayesMixedModel{FP}, Int64, ))
precompile(var,      (BayesMixedModel{FP}, Int64, ))
precompile(std,      (BayesMixedModel{FP}, Int64, ))
precompile(stderror, (BayesMixedModel{FP}, Int64, ))
precompile(confint,  (BayesMixedModel{FP}, Int64, ))
precompile(cov,      (BayesMixedModel{FP}, Int64, ))
precompile(cor,      (BayesMixedModel{FP}, Int64, ))


mean(sigma::ScaleParam{FP})::FP = _igmean(sigma)
mean(sigma::VecScaleParam{FP})::Vector{FP} = _igmean(sigma)

precompile(mean, (ScaleParam{FP}, ))
precompile(mean, (VecScaleParam{FP}, ))

mode(sigma::ScaleParam{FP})::FP = _igmode(sigma)
mode(sigma::VecScaleParam{FP})::Vector{FP} = _igmode(sigma)

precompile(mode, (ScaleParam{FP}, ))
precompile(mode, (VecScaleParam{FP}, ))

median(sigma::ScaleParam{FP})::FP = _igmedian(sigma)
median(sigma::VecScaleParam{FP})::Vector{FP} = _igmedian(sigma)

precompile(median, (ScaleParam{FP}, ))
precompile(median, (VecScaleParam{FP}, ))

var(sigma::ScaleParam{FP})::FP = _igvar(sigma)
var(sigma::VecScaleParam{FP})::Vector{FP} = _igvar(sigma)

precompile(var, (ScaleParam{FP}, ))
precompile(var, (VecScaleParam{FP}, ))

std(sigma::ScaleParam{FP})::FP = _igstd(sigma)
std(sigma::VecScaleParam{FP})::Vector{FP} = _igstd(sigma)

precompile(std, (ScaleParam{FP}, ))
precompile(std, (VecScaleParam{FP}, ))

invmean(sigma::ScaleParam{FP})::FP = _iginvmean(sigma)
invmean(sigma::VecScaleParam{FP})::Vector{FP} = _iginvmean(sigma)

precompile(invmean, (ScaleParam{FP}, ))
precompile(invmean, (VecScaleParam{FP}, ))

logmean(sigma::ScaleParam{FP})::FP = _iglogmean(sigma)
logmean(sigma::VecScaleParam{FP})::Vector{FP} = _iglogmean(sigma)

precompile(logmean, (ScaleParam{FP}, ))
precompile(logmean, (VecScaleParam{FP}, ))

quantile(sigma::ScaleParam{FP}, p::FP = 0.5)::FP = _igquantile(sigma, p)
quantile(sigma::VecScaleParam{FP}, p::FP = 0.5)::Vector{FP} = _igquantile(sigma, p)

precompile(quantile, (ScaleParam{FP}, FP, ))
precompile(quantile, (VecScaleParam{FP}, FP, ))

confint(sigma::ScaleParam{FP}, p::FP = 0.95) = _igconfint(sigma, p)
confint(sigma::VecScaleParam{FP}, p::FP = 0.95) = _igconfint(sigma, p)

precompile(confint, (ScaleParam{FP}, FP, ))
precompile(confint, (VecScaleParam{FP}, FP, ))

nobs(model::BayesMixedModel{FP})::Int64 = get_nobs(model)
nfe(model::BayesMixedModel{FP})::Int64 = get_nfe(model)
nre(model::BayesMixedModel{FP})::Int64 = get_nre(model)
nfepar(model::BayesMixedModel{FP})::Int64 = get_nfe_par(model)
nrepar(model::BayesMixedModel{FP})::Vector{Int64} = get_nre_par(model)
nregpar(model::BayesMixedModel{FP})::Int64 = get_ntot_par(model)

precompile(nobs,    (BayesMixedModel{FP}, ))
precompile(nfe,     (BayesMixedModel{FP}, ))
precompile(nre,     (BayesMixedModel{FP}, ))
precompile(nfepar,  (BayesMixedModel{FP}, ))
precompile(nrepar,  (BayesMixedModel{FP}, ))
precompile(nregpar, (BayesMixedModel{FP}, ))


# Partial linear predictor
function linpred(model::BayesMixedModel{T}, k::Int64 = -1) where T <: FP

    n_obs = get_nobs(model)
    m_k = Vector{T}(undef, n_obs)
    V_k = Vector{T}(undef, n_obs)

    if k == 0
        fe = get_fe(model)
        m_k .= model.X * fe.m
        V_k .= _dgouter(model.X, fe.V, false)
    elseif k > 0
        re = get_re(model)[k]
        m_k .= model.Z[k] * re.m
        V_k .= _dgouter(model.Z[k], re.V, false)
    elseif k == -1
        C = get_design_matrix(model.X, model.Z)
        m_k .= C * model.theta.m
        V_k .= _dgchol(C, model.theta.R, true, false)
    else
        error("Invalid parameter: k ≥ -1.")
    end

    return LinPred(m_k, V_k)
end;

# Partial linear predictor
function fitted(model::BayesMixedModel{T}, k::Int64 = -1) where T <: FP

    n_obs = get_nobs(model)
    m_k = Vector{T}(undef, n_obs)
    V_k = Vector{T}(undef, n_obs)

    if k == 0
        fe = get_fe(model)
        m_k .= model.X * fe.m
        V_k .= _dgouter(model.X, fe.V, false)
    elseif k > 0
        re = get_re(model)[k]
        m_k .= model.Z[k] * re.m
        V_k .= _dgouter(model.Z[k], re.V, false)
    elseif k == -1
        C = get_design_matrix(model.X, model.Z)
        m_k .= C * model.theta.m
        V_k .= _dgchol(C, model.theta.R, true, false)
    else
        error("Invalid parameter: k ≥ -1.")
    end

    return LinPred(m_k, V_k)
end;

# Partial residuals
function residuals(model::BayesMixedModel{T}, k::Int64 = -1)::Vector{T} where T <: FP

    # Input dimensions
    n_obs = get_nobs(model)
    n_fe = get_nfe(model)
    n_re = get_nre(model)

    # Partial linear predictor
    eta_fe = linpred(model, 0).m
    eta_re = Matrix{T}(undef, n_obs, n_re)

    for j in 1:n_re
        eta_re[:,j] .= linpred(model, j).m
    end

    # Partial residual vector
    res = Vector{T}(undef, n_obs)

    if k == 0
        res .= model.y .- sum(eta_re, dims = 2)
    elseif k > 0
        idx = setdiff(1:n_re, k)
        res .= model.y .- eta_fe .- sum(eta_re[:,idx], dims = 2)
    elseif k == -1
        res .= model.y .- eta_fe .- sum(eta_re, dims = 2)
    else
        error("Invalid parameter: k ≥ -1.")
    end

    # Output
    return res
end;

# Predict the mean and variance of the linear predictor for new values of X and Z
function predict(
    model::BayesMixedModel{T},
    newX::Matrix{T} = model.X,
    newZ::VecMatrix{T} = model.Z,
    k::Int64 = -1
    ) where T <: FP

    n = 0
    if k > 0
        n = size(newZ[k], 1)
    elseif k == 0
        n = size(newX, 1)
    elseif k == -1
        if any(size(newX, 1) .≠ size.(newZ, 1))
            error("Incompatible dimensions of X and Z.")
        end

        n = size(newX, 1)
    else
        error("Invalid parameter: k ≥ -1.")
    end

    m = Vector{T}(undef, n)
    V = Vector{T}(undef, n)

    if k > 0
        re = get_re(model)
        m .= newZ[k] * re[k].m
        V .= _dgouter(newZ[k], re[k].V, false)
    elseif k == 0
        fe = get_fe(model)
        m .= newX * fe.m
        V .= _dgouter(newX, fe.V, false)
    elseif k == -1
        newC = get_design_matrix(newX, newZ)
        m .= newC * model.theta.m
        V .= _dgchol(newC, model.theta.R, true, false)
    end

    return LinPred(m, V)
end;

# summary coefficient table
function coeftable(
    model::BayesMixedModel{T},
    fixeff::Bool = true,
    randeff::Bool = true,
    vareff::Bool = true,
    level::T = 0.95
    ) where T <: FP

    # Input dimensions
    # ----------------
    n_fe = get_nfe(model)
    n_re = get_nre(model)
    n_fe_par = get_nfe_par(model)
    n_re_par = get_nre_par(model)

    cols = join([" "^9;
        lpad("Mean", 13);
        lpad("Std.Dev.", 13);
        lpad("Lower 95%", 13);
        lpad("Upper 95%", 13)
    ])

    # Fixed effect summary
    # --------------------
    if fixeff
        fe = get_fe(model)
        q = quantile(Normal(), 1-(1-level)/2)

        println("\n Fixed effects \n", "-"^62, "\n", cols, "\n", "-"^62)
        for j in 1:n_fe_par
            mj = fe.m[j]
            sj = sqrt(fe.V[j,j])
            lj = mj - q * sj
            uj = mj + q * sj
            if any(abs.([mj, sj, lj, uj]) .< 1e-04)
                @printf(" β[%.0f] \t %13.2e %12.2e %12.2e %12.2e \n", j, mj, sj, lj, uj)
            else
                @printf(" β[%.0f] \t %13.4f %12.4f %12.4f %12.4f \n", j, mj, sj, lj, uj)
            end
        end
        println("-"^62)
    end

    # Random effect summary
    # ---------------------
    if randeff
        re = get_re(model)
        q = quantile(Normal(), 1-(1-level)/2)

        for k in 1:n_re
            println("\n Random effects: ", k, "\n", "-"^62, "\n", cols, "\n", "-"^62)
            for j in 1:n_re_par[k]
                mj = re[k].m[j]
                sj = sqrt(re[k].V[j,j])
                lj = mj - q * sj
                uj = mj + q * sj
                if any(abs.([mj, sj, lj, uj]) .< 1e-04)
                    @printf(" u[%.0f,%2.0f]  %12.2e %12.2e %12.2e %12.2e \n", k, j, mj, sj, lj, uj)
                else
                    @printf(" u[%.0f,%2.0f]  %12.4f %12.4f %12.4f %12.4f \n", k, j, mj, sj, lj, uj)
                end
            end
            println("-"^62)
        end
    end

    cols = join([" "^9;
        lpad("Mean", 11);
        lpad("Mode", 11);
        lpad("Median", 13);
        lpad("Lower 95%", 13);
        lpad("Upper 95%", 13)
    ])

    # Variance parameter summary
    # --------------------------
    if vareff
        println("\n Variance parameters \n", "-"^71, "\n", cols, "\n", "-"^71)
        Ak = model.sigma2_e.A
        Bk = model.sigma2_e.B
        mnk = _igmean(Ak, Bk)
        mok = _igmode(Ak, Bk)
        mek = _igmedian(Ak, Bk)
        lok = _igquantile(Ak, Bk,   (1-level)/2)
        upk = _igquantile(Ak, Bk, 1-(1-level)/2)

        @printf(" ε       %11.4f %10.4f %12.4f %12.4f %12.4f \n", mnk, mok, mek, lok, upk)
        for k in 1:n_re
            Ak = model.sigma2_u[k].A
            Bk = model.sigma2_u[k].B
            mnk = _igmean(Ak, Bk)
            mok = _igmode(Ak, Bk)
            mek = _igmedian(Ak, Bk)
            lok = _igquantile(Ak, Bk,   (1-level)/2)
            upk = _igquantile(Ak, Bk, 1-(1-level)/2)

            @printf(" u[%.0f,:]  %11.4f %10.4f %12.4f %12.4f %12.4f \n", k, mnk, mok, mek, lok, upk)
        end
        println("-"^71)
    end
end;

function coeftable(
    model::BayesMixedModel{T};
    fixeff::Bool = true,
    randeff::Bool = true,
    vareff::Bool = true,
    level::T = 0.95
    ) where T <: FP

    coeftable(model, fixeff, randeff, vareff, level)
end

precompile(linpred,   (BayesMixedModel{FP}, Int64, ))
precompile(fitted,    (BayesMixedModel{FP}, Int64, ))
precompile(residuals, (BayesMixedModel{FP}, Int64, ))
precompile(predict,   (BayesMixedModel{FP}, Matrix{FP}, VecMatrix{FP}, Int64, ))
precompile(coeftable, (BayesMixedModel{FP}, Bool, Bool, Bool, FP, ))
precompile(coeftable, (BayesMixedModel{FP}, ))

