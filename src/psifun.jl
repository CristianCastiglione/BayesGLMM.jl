
## Wrapper functions

function psi(
    model::BayesMixedModel{T}, t::T = 0.0, nknots::Int64 = 19
    ) where T <: FP

    psi_0, psi_1, psi_2 = _psi(
        model.y, model.eta.m, sqrt.(model.eta.V), model.family, t, nknots)
    
    return psi_0, psi_1, psi_2
end

precompile(psi, (BayesMixedModel{FP}, FP, Int64, ))

function psi(
    y::Vector{T}, eta::LinPred{T}, f::Family, t::T = 0.0, nknots::Int64 = 19
    ) where T <: FP

    psi_0, psi_1, psi_2 = _psi(y, eta.m, sqrt.(eta.V), f, t, nknots)
    
    return psi_0, psi_1, psi_2
end

for f in FAMILIES
    precompile(psi, (Vector{FP}, LinPred{FP}, f,  FP, Int64, ))
end

function psi(
    y::Vector{T}, C::Matrix{T}, 
    theta::RegParam{T}, f::Family, t::T = 0.0, nknots::Int64 = 19
    ) where T <: FP

    eta = C * theta.m
    sigma = _dgchol(C, theta.R, true, true)
    psi_0, psi_1, psi_2 = _psi(y, eta, sigma, f, t, nknots)

    return psi_0, psi_1, psi_2
end

for f in FAMILIES
    precompile(psi, (Vector{FP}, Matrix{FP}, RegParam{FP}, f, FP, Int64, ))
end


## Gauss-Hermite integration

function _gausshermite(
    y::V, eta::V, sigma::V, f::Family, nknots::Int64 = 19
    ) where {T <: FP, V <: Vector{T}}

    # k, w = gausshermite(nknots)
    k, w = get_gh_points(nknots)
    
    Y = y * ones(nknots)'
    K = eta .+ sqrt2 .* (sigma * k')
    W = w ./ sqrtπ

    psi_0 = dloss(Y, K, f; order = 0) * W
    psi_1 = dloss(Y, K, f; order = 1) * W
    psi_2 = dloss(Y, K, f; order = 2) * W

    return psi_0, psi_1, psi_2
end

for f in [Logit, Probit, CLogLog]
    precompile(_gausshermite, (Vector{FP}, Vector{FP}, Vector{FP}, f, Int64, ))
end

function _monahanstefanski(
    y::V, eta::V, sigma::V, t::T
    ) where {T <: FP, V <: Vector{T}}

    # Monahan, JF & Stefanski, LA (1989)
    # Normal scale mixture approximations to F*(z) and computation of the logistic-normal integral
    # Handbook of the Logistic Distribution 
    # Balakrishnan, N (ed), Marcel Dekker, New York, 529–540.

    # Nolan, Wand (2017)
    # Accurate logistic variational message passing: algebraic and numerical details
    # Stat, 6, 102--112

    # scale knots
    sk = [
        1.365340806296348;
        1.059523971016916;
        0.830791313765644;
        0.650732166639391;
        0.508135425366489;
        0.396313345166341;
        0.308904252267995;
        0.238212616409306]

    # weigths
    wk = [
        0.003246343272134;
        0.051517477033972;
        0.195077912673858;
        0.315569823632818;
        0.274149576158423;
        0.131076880695470;
        0.027912418727972;
        0.001449567805354]

    # one vector
    uk = ones(8)

    ms = eta * sk'
    ss = sqrt.(1. .+ (sigma * sk').^2)
    bs = ss.^2 ./ (ones(length(y)) * sk')

    phi_1 = cdf.(Normal.(-ms, ss), .0)
    phi_2 = pdf.(Normal.(-ms, ss), .0)

    psi_0 = (phi_1 .* (eta * uk') .+ phi_2 .* bs) * wk .- y .* eta
    psi_1 = (phi_1 * wk) .- y
    psi_2 = (phi_2 * (wk .* sk))

    t > .0 ? psi_2 .= trim.(psi_2, t) : nothing
    
    return psi_0, psi_1, psi_2
end

## psi vectors calculation

function _psi(
    y::V, eta::V, sigma::V, f::Gaussian, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    psi_2 = 2.0 * ones(length(y))
    psi_1 = 2.0 * (eta .- y)
    psi_0 = (y .- eta).^2 .+ sigma.^2

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::Poisson, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    psi_2 = exp.(eta .+ 0.5 .* sigma.^2)
    psi_1 = psi_2 .- y
    psi_0 = psi_2 .- y .* eta

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::Logit, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    # Add an option to select the quadrature method
    # default choice: method = "monahanstefanski"

    # psi_0, psi_1, psi_2 = _gausshermite(y, eta, sigma, f, nknots)
    psi_0, psi_1, psi_2 = _monahanstefanski(y, eta, sigma, t)

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::Probit, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    psi_0, psi_1, psi_2 = _gausshermite(y, eta, sigma, f, nknots)

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::CLogLog, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    psi_0, psi_1, psi_2 = _gausshermite(y, eta, sigma, f, nknots)

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::Gamma, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    psi_2 = y .* exp.(- eta .+ 0.5 .* sigma.^2)
    psi_1 = - (psi_2 .+ 1.0)
    psi_0 = psi_2 .- log.(y) .- eta

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::Quantile, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    z = (y .- eta) ./ sigma

    phi_2 = pdf.(Normal(), z)
    phi_1 = cdf.(Normal(), z)

    psi_2 = trim.(phi_2 ./ sigma, t)
    psi_1 = 1.0 .- f.tau .- phi_1
    psi_0 = sigma .* (phi_2 .- psi_1 .* z)

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::Expectile, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    eps = y .- eta
    tau = 1.0 - f.tau
    # lambda = 2.0 * tau - 1.0
    lambda = 1.0 - 2.0 * tau

    phi_2 = pdf.(Normal.(eta, sigma), y)
    phi_1 = cdf.(Normal.(eta, sigma), y)

    # psi_2  = tau .- lambda .* phi_1
    # psi_1  = - eps .* psi_2 .- lambda .* sigma.^2 .* phi_2
    # psi_0  = sum(- eps .* psi_1 .+ sigma.^2 .* phi_2)
    # psi_0 += lambda * sum(eps .* sigma.^2 .* (1.0 .- sigma.^2) .* phi_2)

    weights = tau .+ (1.0 - 2.0 * tau) .* phi_1

    psi_2 = 2.0 * weights
    psi_1 = 2.0 * (- eps .* weights .+ lambda .* sigma.^2 .* phi_2)
    psi_0 = (eps.^2 .+ sigma.^2) .* weights
    psi_0 -= lambda .* eps .* sigma.^2 .* phi_2

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::Huber, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    c = f.eps
    eps = y .- eta
    eps_up = c .+ eps
    eps_lo = c .- eps

    pdf_up = pdf.(Normal.(eps, sigma), +c)
    pdf_lo = pdf.(Normal.(eps, sigma), -c)
    cdf_up = cdf.(Normal.(eps, sigma), +c)
    cdf_lo = cdf.(Normal.(eps, sigma), -c)

    psi_2 = trim.(cdf_up .- cdf_lo, t)

    psi_1  = eps_lo .* cdf_up .+ sigma.^2 .* pdf_up .- c
    psi_1 += eps_up .* cdf_lo .- sigma.^2 .* pdf_lo
    
    psi_0  = 0.5 .* ((eps.^2 .+ sigma.^2) .* cdf_up .- sigma.^2 .* eps_up .* pdf_up)
    psi_0 -= 0.5 .* ((eps.^2 .+ sigma.^2) .* cdf_lo .+ sigma.^2 .* eps_lo .* pdf_lo)
    psi_0 += c .* (eps .- 0.5 .* c) .* (1 .- cdf_up) .+ c .* sigma.^2 .* pdf_up
    psi_0 -= c .* (eps .+ 0.5 .* c) .* cdf_lo .- c .* sigma.^2 .* pdf_lo

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::SVR, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    z_lo = (- f.eps .- y .+ eta) ./ sigma
    z_up = (+ f.eps .- y .+ eta) ./ sigma

    phi_2 = pdf.(Normal(), z_lo) .+ pdf.(Normal(), z_up)
    phi_1 = cdf.(Normal(), z_lo) .+ cdf.(Normal(), z_up)
    phi_0 = z_lo .* cdf.(Normal(), z_lo) .- z_up .* (1 .- cdf.(Normal(), z_up))

    psi_2 = 2.0 .* trim.(phi_2 ./ sigma, t)
    psi_1 = 2.0 .* (phi_1 .- 1.0)
    psi_0 = 2.0 .* sigma .* (phi_2 + phi_0)

    return psi_0, psi_1, psi_2
end

function _psi(
    y::V, eta::V, sigma::V, f::SVC, t::T = 0.0, nknots = 19
    ) where {T <: FP, V <: Vector{T}}

    z = (1.0 .- y .* eta) ./ sigma

    phi_2 = pdf.(Normal(), z)
    phi_1 = cdf.(Normal(), z)

    psi_2 = trim.(2.0 .* phi_2 ./ sigma, 2.0 * t)
    psi_1 = 2.0 .* (- y .* phi_1)
    psi_0 = sigma .* (phi_2 .+ phi_1 .* z)

    return psi_0, psi_1, psi_2
end

for f in FAMILIES
    precompile(_psi, (Vector{FP}, Vector{FP}, Vector{FP}, f, FP, Int64, ))
end

