
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
logit(x) = log.(x ./ (1 .- x));
expit(x) = 1 ./ (1 .+ exp.(-x));
jitter(x) = Float64.(x .+ rand(Uniform(-0.025,+0.025), length(x)))

function data_sim(x, setting = "A")
    n = length(x)
    eta = nothing

    dnorm(x, m, s) = pdf.(Normal(m, s), x);
    pnorm(x, m, s) = cdf.(Normal(m, s), x);

    if setting == "A"
        eta = sin.(3*π*x.^2)
    elseif setting == "B"
        b = [-1.02; 0.018; 0.4; 0.08]
        Z = [x x.^2 dnorm(x, 0.38, 0.08) dnorm(x, 0.75, 0.03)]
        eta = Z * b
    elseif setting == "C"
        b = [0.35; 1.9; 1.8]
        Z = [dnorm(x, 0.01, 0.08) dnorm(x, 0.45, 0.23) (1 .- dnorm(x, 0.7, 0.14))]
        eta = Z * b
    elseif setting == "D"
        b = [1.0; 1.02; 0.01; 0.4]
        Z = [sin.(3*π*x.^2) x x.^2 dnorm(x, 0.38, 0.08)]
        eta = Z * b
    elseif setting == "E"
        eta = 1.0 * ((x .≥ 0.2) .& (x .≤ 0.5)) .+ ((x .≥ 0.75) .& (x .≤ 0.85))
    else
        error("Setting ranges from `A` to `E`.")
    end
    mu = expit(1.2 * (eta .- mean(eta)) ./ std(eta));
    return mu
end

## Data simulation

# Dimensions
n, p, d = 500, 2, 40;

# Time points
x = collect(0:1/(n-1):1);

# Design matrix and spline basis
a, b = 0.0, 1.0;
k = get_unif_knots(x, d);

X = [ones(n) x];
Z = get_ospline_matrix(x, k, a, b)[:,3:end];

mu = data_sim(x, "D");
y = Float64.(rand.(Binomial.(1, mu)));

## Data visualization
begin
    plot(legend=:topleft, size=(600,300), title="Data vs True function")
    scatter!(x, jitter(y), label="data", alpha=.5, color=:black)
    plot!(x, mu, color=:1, line=:solid, linewidth=1.0, label="Truth")
end

## Model estimation
family = BayesGLMM.Logit();
mcmc = BayesGLMM.MCMC(maxiter = 5500, report = 500);
mfvb = BayesGLMM.MFVB(maxiter = 300);
svb  = BayesGLMM.SVB(maxiter = 300, search = true);

@time logit_mcmc = BayesGLMM.fit(y, X, [Z], family; alg = mcmc);
@time logit_mfvb = BayesGLMM.fit(y, X, [Z], family; alg = mfvb);
@time logit_svb  = BayesGLMM.fit(y, X, [Z], family; alg = svb);

## Convergence checks

# MCMC
BayesGLMM.plot_trace_elbo(logit_mcmc.opt, marker = :none)
BayesGLMM.plot_trace_regparam(logit_mcmc.opt, marker = :none)
BayesGLMM.plot_trace_sigma2u(logit_mcmc.opt, marker = :none)

# MFVB
BayesGLMM.plot_trace_dx(logit_mfvb.opt)
BayesGLMM.plot_trace_df(logit_mfvb.opt)
BayesGLMM.plot_trace_elbo(logit_mfvb.opt)
BayesGLMM.plot_trace_regparam(logit_mfvb.opt)
BayesGLMM.plot_trace_sigma2u(logit_mfvb.opt)

# SVB
BayesGLMM.plot_trace_dx(logit_svb.opt)
BayesGLMM.plot_trace_df(logit_svb.opt)
BayesGLMM.plot_trace_elbo(logit_svb.opt)
BayesGLMM.plot_trace_regparam(logit_svb.opt)
BayesGLMM.plot_trace_sigma2u(logit_svb.opt)

## Posterior summary
BayesGLMM.coeftable(logit_mcmc.model, randeff = false)
BayesGLMM.coeftable(logit_mfvb.model, randeff = false)
BayesGLMM.coeftable(logit_svb.model, randeff = false)

## Posterior predictive distributions
begin
    plot(legend=:topleft, size=(600,300), title="Posterior quantile curves")
    scatter!(x, jitter(y), label="data", alpha=.5, color=:black)
    plot!(x, mu, label = "true")
    plot!(x, expit(logit_mcmc.model.eta.m), color=:1, label="MCMC")
    plot!(x, expit(logit_mcmc.model.eta.m .- 2 .* sqrt.(qreg_mcmc.model.eta.V)), color=:1, linestyle=:dash, label="")
    plot!(x, expit(logit_mcmc.model.eta.m .+ 2 .* sqrt.(qreg_mcmc.model.eta.V)), color=:1, linestyle=:dash, label="")
    plot!(x, expit(logit_mfvb.model.eta.m), color=:3, linestyle=:solid, label="MFVB")
    plot!(x, expit(logit_mfvb.model.eta.m .- 2 .* sqrt.(qreg_mfvb.model.eta.V)), color=:3, linestyle=:dash, label="")
    plot!(x, expit(logit_mfvb.model.eta.m .+ 2 .* sqrt.(qreg_mfvb.model.eta.V)), color=:3, linestyle=:dash, label="")
    plot!(x, expit(logit_svb.model.eta.m), color=:4, linestyle=:solid, label="SVB")
    plot!(x, expit(logit_svb.model.eta.m .- 2 .* sqrt.(qreg_svb.model.eta.V)), color=:4, linestyle=:dash, label="")
    plot!(x, expit(logit_svb.model.eta.m .+ 2 .* sqrt.(qreg_svb.model.eta.V)), color=:4, linestyle=:dash, label="")
end

## End of file