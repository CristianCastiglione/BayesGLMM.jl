
## Dependences
include("../src/BayesGLMM.jl");

## External libraries
using Printf
using Random
using Distributions
using LinearAlgebra
using KernelDensity
using Plots
using LaTeXStrings
using SpecialFunctions
using KernelDensity
using DataFrames
using CSV
using StatsPlots

## Using
using .BayesGLMM

## Settings
theme(:default)
default(palette = :tab10, grid = :none, titlefontsize = 10, size = (800, 400));

## Simulation functions

function data_sim(x, setting = "A")
    n = length(x)
    mu = nothing
    sigma = .25

    logit(x) = log.(x ./ (1 .- x));
    expit(x) = 1 ./ (1 .+ exp.(-x));
    dnorm(x, m, s) = pdf.(Normal(m, s), x);
    pnorm(x, m, s) = cdf.(Normal(m, s), x);

    if setting == "A"
        mu = sin.(3*π*x.^2)
    elseif setting == "B"
        b = [-1.02; 0.018; 0.4; 0.08]
        Z = [x x.^2 dnorm(x, 0.38, 0.08) dnorm(x, 0.75, 0.03)]
        mu = Z * b
    elseif setting == "C"
        b = [0.35; 1.9; 1.8]
        Z = [dnorm(x, 0.01, 0.08) dnorm(x, 0.45, 0.23) (1 .- dnorm(x, 0.7, 0.14))]
        mu = Z * b
    elseif setting == "D"
        b = [1.0; 1.02; 0.01; 0.4]
        Z = [sin.(3*π*x.^2) x x.^2 dnorm(x, 0.38, 0.08)]
        mu = Z * b
    elseif setting == "E"
        mu = 1.0 * ((x .≥ 0.2) .& (x .≤ 0.5)) .+ ((x .≥ 0.75) .& (x .≤ 0.85))
    else
        error("Setting ranges from `A` to `E`.")
    end
    return mu, sigma
end

## Data simulation

# Dimensions
n, p, d = 500, 2, 40;

# Time points
x = collect(0:1/(n-1):1);

# Design matrix and spline basis
a = 0.0;
b = 1.0;
k = get_unif_knots(x, d);

X = [ones(n) x];
Z = get_ospline_matrix(x, k, a, b)[:,3:end];

mu, sigma = data_sim(x, "D");
eps = rand(TDist(4), n);
y = mu .+ sigma .* eps;

## Data visualization

begin
    plot(legend=:topleft, size=(600,300), title="Data vs True function")
    scatter!(x, y, label="data", alpha=.5, color=:black)
    plot!(x, mu, color=:1, line=:solid, linewidth=1.0, label="Truth")
    plot!(x, mu .- 1.64 * sigma, color=:1, line=:dash, linewidth=1.0, label="")
    plot!(x, mu .+ 1.64 * sigma, color=:1, line=:dash, linewidth=1.0, label="")
end

## Model estimation
tol = 0.001;

family = BayesGLMM.SVR(tau);
mcmc = BayesGLMM.MCMC(maxiter = 5500, report = 500);
mfvb = BayesGLMM.MFVB(maxiter = 300);
svb  = BayesGLMM.SVB(maxiter = 300, search = true);

@time fit_mcmc = BayesGLMM.fit(y, X, [Z], family; alg = mcmc);
@time fit_mfvb = BayesGLMM.fit(y, X, [Z], family; alg = mfvb);
@time fit_svb  = BayesGLMM.fit(y, X, [Z], family; alg = svb);

## Convergence checks

# MCMC
BayesGLMM.plot_trace_elbo(fit_mcmc.opt, marker = :none)
BayesGLMM.plot_trace_regparam(fit_mcmc.opt, marker = :none)
BayesGLMM.plot_trace_sigma2u(fit_mcmc.opt, marker = :none)
BayesGLMM.plot_trace_sigma2e(fit_mcmc.opt, marker = :none)

# MFVB
BayesGLMM.plot_trace_dx(fit_mfvb.opt)
BayesGLMM.plot_trace_df(fit_mfvb.opt)
BayesGLMM.plot_trace_elbo(fit_mfvb.opt)
BayesGLMM.plot_trace_regparam(fit_mfvb.opt)
BayesGLMM.plot_trace_sigma2u(fit_mfvb.opt)
BayesGLMM.plot_trace_sigma2e(fit_mfvb.opt)

# SVB
BayesGLMM.plot_trace_dx(fit_svb.opt)
BayesGLMM.plot_trace_df(fit_svb.opt)
BayesGLMM.plot_trace_elbo(fit_svb.opt)
BayesGLMM.plot_trace_regparam(fit_svb.opt)
BayesGLMM.plot_trace_sigma2u(fit_svb.opt)
BayesGLMM.plot_trace_sigma2e(fit_svb.opt)

## Posterior summary
BayesGLMM.coeftable(fit_mcmc.model, randeff = false)
BayesGLMM.coeftable(fit_mfvb.model, randeff = false)
BayesGLMM.coeftable(fit_svb.model, randeff = false)

## Posterior predictive distributions
begin
    plot(legend=:topleft, size=(600,300), title="Posterior quantile curves")
    scatter!(x, y, label="data", alpha=.5, color=:black)
    plot!(x, mu, label = "true")
    plot!(x, fit_mcmc.model.eta.m, color=:1, label="MCMC")
    plot!(x, fit_mcmc.model.eta.m .- 2 .* sqrt.(fit_mcmc.model.eta.V), color=:1, linestyle=:dash, label="")
    plot!(x, fit_mcmc.model.eta.m .+ 2 .* sqrt.(fit_mcmc.model.eta.V), color=:1, linestyle=:dash, label="")
    plot!(x, fit_mfvb.model.eta.m, color=:3, linestyle=:solid, label="MFVB")
    plot!(x, fit_mfvb.model.eta.m .- 2 .* sqrt.(fit_mfvb.model.eta.V), color=:3, linestyle=:dash, label="")
    plot!(x, fit_mfvb.model.eta.m .+ 2 .* sqrt.(fit_mfvb.model.eta.V), color=:3, linestyle=:dash, label="")
    plot!(x, fit_svb.model.eta.m, color=:4, linestyle=:solid, label="SVB")
    plot!(x, fit_svb.model.eta.m .- 2 .* sqrt.(fit_svb.model.eta.V), color=:4, linestyle=:dash, label="")
    plot!(x, fit_svb.model.eta.m .+ 2 .* sqrt.(fit_svb.model.eta.V), color=:4, linestyle=:dash, label="")
end

## End of file