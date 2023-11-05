
"""
    Family

Abstract type collecting the family of models available in the module.
"""

# Family types
abstract type Family end

"""
    RegFamily

Family of regression models for continuous response data.
It includes Gaussian, Quantile, SVR.

    IntFamily

Family of regression models for integer response data.
It includes Poisson.

    ClassFamily

Family of classification data for categorical data.
It includes Logit, Probit, CLogLog, SVC.
"""

# Cantinuous, integer and categorical regression families
abstract type RegFamily <: Family end
abstract type IntFamily <: Family end
abstract type ClassFamily <: Family end

"""
    Gaussian

Family of Gaussian regression models for continuous response.

    Poisson

Family of Poisson regression models for count response.

    Logit

Family of Bernoulli regression models with Logistic 
link function for binary response.

    Probit

Family of Bernoulli regression models with Probit 
link function for binary response.

    CLogLog

Family of Bernoulli regression models with complementary-
log-log link function for binary response.

    Gamma

Family of Gamma regression models for positive response.

    Quantile

Family of Quantile regression models for continuous response.

    Expectile

Family of Expectile regression models for continuous response.

    SVR

Family of support vector regression models for continuous response.

    SVC

Family of support vector classification models for binary response.
"""

# Gaussian regression family
mutable struct Gaussian <: RegFamily
    Gaussian() = new()
end

# Poisson regression family
mutable struct Poisson <: IntFamily
    Poisson() = new()
end

# Logistic regression family
mutable struct Logit <: ClassFamily
    Logit() = new()
end

# Probit regression family
mutable struct Probit <: ClassFamily
    Probit() = new()
end

# Probit regression family
mutable struct CLogLog <: ClassFamily
    CLogLog() = new()
end

# Logistic regression family
mutable struct Gamma <: RegFamily
    Gamma() = new()
end

# Quantile regression family
mutable struct Quantile <: RegFamily
    tau::Float64

    function Quantile(tau::Float64 = 0.5)
        if 0.0 < tau < 1.0
            nothing
        else
            error("Invalid family parameter: `tau` ∈ (0,1).")
        end

        return new(tau)
    end
end

# Expectile regression family
mutable struct Expectile <: RegFamily
    tau::Float64

    function Expectile(tau::Float64 = 0.5)
        if 0.0 < tau < 1.0
            nothing
        else
            error("Invalid family parameter: `tau` ∈ (0,1).")
        end

        return new(tau)
    end
end

# Huber regression family
mutable struct Huber <: RegFamily
    eps::Float64

    function Huber(eps::Float64 = 0.01)
        if eps > 0.0
            nothing
        else
            error("Invalid family parameter: `eps` > 0.")
        end

        return new(eps)
    end
end

# SVR regression family
mutable struct SVR <: RegFamily
    eps::Float64

    function SVR(eps::Float64 = 0.001)
        if eps > 0.0
            nothing
        else
            error("Invalid family parameter: `eps` > 0.")
        end

        return new(eps)
    end
end

# SVC classification family
mutable struct SVC <: ClassFamily
    SVC() = new()
end

# implemented families
const FAMILIES = [Gaussian, Poisson, Logit, Probit, CLogLog, Quantile, Expectile, Huber, SVR, SVC]
const CONJUGATE_FAMILIES = [Gaussian, Logit, Probit, Quantile, SVR, SVC]

for f in FAMILIES
    precompile(f, ())
end

# function returning the asymptotic polynomial order of psi(y,eta)
function tailorder end

tailorder(::Gaussian)::FP = 2.0
tailorder(::Poisson)::FP = 1.0
tailorder(::Logit)::FP = 1.0
tailorder(::Probit)::FP = 1.0
tailorder(::CLogLog)::FP = 1.0
tailorder(::Gamma)::FP = 1.0
tailorder(::Quantile)::FP = 1.0
tailorder(::Expectile)::FP = 2.0
tailorder(::Huber)::FP = 1.0
tailorder(::SVR)::FP = 1.0
tailorder(::SVC)::FP = 1.0

for f in FAMILIES
    precompile(tailorder, (f, ))
end

# loss function corresponding to the family
function loss end

loss(y::FP, eta::FP, f::Gaussian)::FP = (y - eta)^2
loss(y::FP, eta::FP, f::Poisson)::FP = - y * eta + exp(eta)
loss(y::FP, eta::FP, f::Logit)::FP = - y * eta + log1pexp(eta)
# loss(y::FP, eta::FP, f::Probit)::FP = - y * logcdfn(eta) - (1.0 - y) * logcdfn(-eta)
loss(y::FP, eta::FP, f::Probit)::FP = - logcdfn((2 .* y .- 1) .* eta)
loss(y::FP, eta::FP, f::CLogLog)::FP = - y * log(cloglog(eta)) + (1.0 - y) * exp(eta)
loss(y::FP, eta::FP, f::Gamma)::FP = y * exp(-eta) - log(y) - eta
loss(y::FP, eta::FP, f::Quantile)::FP = (y - eta) * (f.tau - 1.0 * (y < eta))
loss(y::FP, eta::FP, f::Expectile)::FP = (y - eta)^2 * abs(f.tau - 1.0 * (y < eta))
loss(y::FP, eta::FP, f::Huber)::FP = abs(y - eta) ≤ f.eps ? 0.5 * (y - eta)^2 : f.eps * (abs(y - eta) - 0.5 * f.eps)
loss(y::FP, eta::FP, f::SVR)::FP = maximum(vcat(0.0, abs(y - eta - f.eps)))
loss(y::FP, eta::FP, f::SVC)::FP = maximum(vcat(0.0, 1.0 - y * eta))

for f in FAMILIES
    precompile(loss, (FP, FP, f, ))
end

# loss function derivatives
function dloss end

function dloss(
    y::V, eta::V, f::Gaussian; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    
    order != 0 ? nothing : l .= (y .- eta).^2
    order != 1 ? nothing : l .= 2.0 .* (eta .- y)
    order != 2 ? nothing : l .= 2.0

    return l
end

function dloss(
    y::V, eta::V, f::Poisson; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    exp_eta = exp.(eta)
    
    order != 0 ? nothing : l .= exp_eta .- y .* eta
    order != 1 ? nothing : l .= exp_eta .- y
    order != 2 ? nothing : l .= exp_eta

    return l
end

function dloss(
    y::V, eta::V, f::Logit; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    llp = log1pexp.(eta)
    
    order != 0 ? nothing : l .= llp .- y .* eta
    order != 1 ? nothing : l .= exp.(eta .- llp) .- y
    order != 2 ? nothing : l .= exp.(eta .- 2.0 .* llp)

    return l
end

function dloss(
    y::V, eta::V, f::Probit; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    if order == 0
        f1 = logcdfn.(+eta)
        f0 = logcdfn.(-eta)

        l .= - y .* f1 .- (1.0 .- y) .* f0
    elseif order == 1
        g1 = + pdfn.(+eta) ./ cdfn.(+eta)
        g0 = - pdfn.(-eta) ./ cdfn.(-eta)
        
        l .= - y .* g1 .- (1.0 .- y) .* g0
    elseif order == 2
        g1 = + pdfn.(+eta) ./ cdfn.(+eta)
        g0 = - pdfn.(-eta) ./ cdfn.(-eta)
        h1 = - eta .* g1 .- g1.^2
        h0 = - eta .* g0 .- g0.^2
        
        l .= - y .* h1 .- (1.0 .- y) .* h0
    end

    return l
end

function dloss(
    y::V, eta::V, f::CLogLog; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    exp_eta = exp.(eta)
    cee_eta = cexpexp.(eta)

    if order == 0
        f1 = log.(cee_eta)
        f0 = - exp_eta

        l .= - y .* f1 .- (1.0 .- y) .* f0
    elseif order == 1
        g1 = exp.(eta .- exp_eta) ./ cee_eta
        g0 = - exp_eta

        l .= - y .* g1 .- (1.0 .- y) .* g0
    elseif order == 2
        g1 = exp.(eta .- exp_eta) ./ cee_eta
        h1 = (1.0 .- exp_eta) .* g1 .- g1.^2
        h0 = - exp_eta
        
        l .= - y .* h1 .- (1.0 .- y) .* h0
    end

    return l
end

function dloss(
    y::V, eta::V, f::Gamma; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    exp_eta = exp.(-eta)
    
    order != 0 ? nothing : l .= y .* exp_eta .- log.(y) .- eta
    order != 1 ? nothing : l .= - y .* exp_eta .- 1.0
    order != 2 ? nothing : l .= y .* exp_eta

    return l
end

function dloss(
    y::V, eta::V, f::Quantile; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    r = y .- eta
    
    order != 0 ? nothing : l .= 0.5 .* abs.(r) .+ (f.tau - 0.5) .* r
    order != 1 ? nothing : l .= - 0.5 .* sign.(r) .- (f.tau - 0.5)
    order != 2 ? nothing : l .= 1.0 .* (r .== 0.0)

    return l
end

function dloss(
    y::V, eta::V, f::Expectile; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    r = float.(y .- eta)
    s = float.(y .< eta)
    w = float.(f.tau .- s)
    
    order != 0 ? nothing : l .= abs.(w) .* r.^2
    order != 1 ? nothing : l .= - 2.0 .* w .* r
    order != 2 ? nothing : l .= - 2.0 .* w

    return l
end

function dloss(
    y::V, eta::V, f::Huber; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    c = f.eps
    r = y .- eta
    w = abs.(r) .≤ c
    
    order != 0 ? nothing : l .= w .* 0.5 .* r.^2 + (1 .- w) .* c .* (abs.(r) .- 0.5 .* c)
    order != 1 ? nothing : l .= w .* r + (1 .- w) .* c .* sign.(r)
    order != 2 ? nothing : l .= w

    return l
end

function dloss(
    y::V, eta::V, f::SVR; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    r = y .- eta
    r_lo = r .- f.eps
    r_up = r .+ f.eps
    
    order != 0 ? nothing : l .= abs.(r_lo) .+ abs.(r_up) .+ r_lo .- r_up
    order != 1 ? nothing : l .= sign.(x_lo) .+ abs(r_up)
    order != 2 ? nothing : l .= 2.0 .* (r_lo .== 0.0) .+ 2.0 .* (r_up .== 0.0)

    return l
end

function dloss(
    y::V, eta::V, f::SVC; order::Int64 = 0
    )::V where {T <: FP, V <: VecOrMat{T}}

    l = zero(y)
    r = 1.0 .- y .* eta
    
    order != 0 ? nothing : l .= abs.(r) .+ r
    order != 1 ? nothing : l .= sign.(r) .+ 1.0
    order != 2 ? nothing : l .= 2.0 .* (r .== 0)

    return l
end

for f in FAMILIES
    precompile(dloss, (Vector{FP}, Vector{FP}, f, ))
    precompile(dloss, (Matrix{FP}, Matrix{FP}, f, ))
end
