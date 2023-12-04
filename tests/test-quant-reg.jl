
## Dependences
include("../src/BayesGLMM.jl");
include("../src/SplineBasis/spline.jl");

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
    sigma = nothing

    logit(x) = log.(x ./ (1 .- x));
    expit(x) = 1 ./ (1 .+ exp.(-x));
    dnorm(x, m, s) = pdf.(Normal(m, s), x);
    pnorm(x, m, s) = cdf.(Normal(m, s), x);

    if setting == "A"
        mu = sin.(3*π*x.^2)
        sigma = exp.(0.5 * (0.1 .+ cos.(4*π*x)))
    elseif setting == "B"
        b = [-1.02; 0.018; 0.4; 0.08]
        Z = [x x.^2 dnorm(x, 0.38, 0.08) dnorm(x, 0.75, 0.03)]
        mu = Z * b

        a = [-0.5; 0.3; -1.0]
        W = [ones(n) x.^2 dnorm(x, 0.2, 0.1)]
        sigma = exp.(0.5 * W * a)
    elseif setting == "C"
        b = [0.35; 1.9; 1.8]
        Z = [dnorm(x, 0.01, 0.08) dnorm(x, 0.45, 0.23) (1 .- dnorm(x, 0.7, 0.14))]
        mu = Z * b

        a = [0.3; 0.4]
        W = [dnorm(x, 0.0, 0.2) dnorm(x, 1.0, 0.1)]
        sigma = exp.(0.5 * W * a)
    elseif setting == "D"
        b = [1.0; 1.02; 0.01; 0.4]
        Z = [sin.(3*π*x.^2) x x.^2 dnorm(x, 0.38, 0.08)]
        mu = Z * b

        a = [-0.4; 0.3; 1.0; -0.5]
        W = [ones(n) x.^2 cos.(4*π*x) dnorm(x, 0.2, 0.1)]
        sigma = exp.(0.5 * W * a)
    elseif setting == "E"
        mu = 1.0 * ((x .≥ 0.2) .& (x .≤ 0.5)) .+ ((x .≥ 0.75) .& (x .≤ 0.85))
        sigma = 0.1
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
eps = rand(Normal(), n);
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
tau = 0.9;

family = BayesGLMM.Quantile(tau);
mcmc = BayesGLMM.MCMC(maxiter = 5500, report = 500);
mfvb = BayesGLMM.MFVB(maxiter = 300);
svb  = BayesGLMM.SVB(maxiter = 300, search = true);

@time qreg_mcmc = BayesGLMM.fit(y, X, [Z], family; alg = mcmc);
@time qreg_mfvb = BayesGLMM.fit(y, X, [Z], family; alg = mfvb);
@time qreg_svb  = BayesGLMM.fit(y, X, [Z], family; alg = svb);

## Convergence checks

# MCMC
BayesGLMM.plot_trace_dx(qreg_mcmc.opt)
BayesGLMM.plot_trace_df(qreg_mcmc.opt)
BayesGLMM.plot_trace_elbo(qreg_mcmc.opt)
BayesGLMM.plot_trace_regparam(qreg_mcmc.opt)
BayesGLMM.plot_trace_sigma2u(qreg_mcmc.opt)
BayesGLMM.plot_trace_sigma2e(qreg_mcmc.opt)

# MFVB
BayesGLMM.plot_trace_dx(qreg_mfvb.opt)
BayesGLMM.plot_trace_df(qreg_mfvb.opt)
BayesGLMM.plot_trace_elbo(qreg_mfvb.opt)
BayesGLMM.plot_trace_regparam(qreg_mfvb.opt)
BayesGLMM.plot_trace_sigma2u(qreg_mfvb.opt)
BayesGLMM.plot_trace_sigma2e(qreg_mfvb.opt)

# SVB
BayesGLMM.plot_trace_dx(qreg_svb.opt)
BayesGLMM.plot_trace_df(qreg_svb.opt)
BayesGLMM.plot_trace_elbo(qreg_svb.opt)
BayesGLMM.plot_trace_regparam(qreg_svb.opt)
BayesGLMM.plot_trace_sigma2u(qreg_svb.opt)
BayesGLMM.plot_trace_sigma2e(qreg_svb.opt)

## Posterior predictive distributions

begin
    plot(legend=:topleft, size=(600,300), title="Posterior quantile curves")
    scatter!(x, y, label="data", alpha=.5, color=:black)
    plot!(x, qreg_mcmc.model.eta.m, color=:1, label="MCMC")
    plot!(x, qreg_mcmc.model.eta.m .- 2 .* sqrt.(qreg_mcmc.model.eta.V), color=:1, linestyle=:dash, label="")
    plot!(x, qreg_mcmc.model.eta.m .+ 2 .* sqrt.(qreg_mcmc.model.eta.V), color=:1, linestyle=:dash, label="")
    plot!(x, qreg_mfvb.model.eta.m, color=:3, linestyle=:solid, label="MFVB")
    plot!(x, qreg_mfvb.model.eta.m .- 2 .* sqrt.(qreg_mfvb.model.eta.V), color=:3, linestyle=:dash, label="")
    plot!(x, qreg_mfvb.model.eta.m .+ 2 .* sqrt.(qreg_mfvb.model.eta.V), color=:3, linestyle=:dash, label="")
    plot!(x, qreg_svb.model.eta.m, color=:4, linestyle=:solid, label="SVB")
    plot!(x, qreg_svb.model.eta.m .- 2 .* sqrt.(qreg_svb.model.eta.V), color=:4, linestyle=:dash, label="")
    plot!(x, qreg_svb.model.eta.m .+ 2 .* sqrt.(qreg_svb.model.eta.V), color=:4, linestyle=:dash, label="")
end

## End of file