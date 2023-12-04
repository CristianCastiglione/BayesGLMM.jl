# Author: Cristian Castiglione
# Contact: cristian.castiglione@phd.unipd.it
# License: MIT
# Creation: 16/04/2022
# Last change: 04/12/2023

include("PolyaGamma/PolyaGammaDistribution.jl")
# include("SplineBasis/spline.jl")

__precompile__()
module BayesGLMM

# Lbraries
using Plots, Random, Distributions, LinearAlgebra, SparseArrays
# using OffsetArrays, BSplines

using SpecialFunctions: loggamma, digamma, trigamma
using StatsBase: CoefTable, StatisticalModel, RegressionModel
using StatsFuns: log1pexp
using FastGaussQuadrature: gausshermite
using Printf: @printf, @sprintf

import Base: show, summary
import StatsBase: fit, fit!, coef, stderror, confint
import StatsBase: coeftable, residuals, predict, nobs
import Statistics: mean, var, std, cov, cor, median, quantile
import Distributions: mode

using Main.PolyaGammaDistribution: PolyaGamma

# Structs
export
    ### general Bayesian mixed model class
    BayesMixedModel,
    ### family definition
    Family, RegFamily, IntFamily, ClassFamily,
    Gaussian, Poisson, Logit, Probit, 
    CLogLog, Quantile, Expectile, SVR, SVC,
    ### linear predictor, regression and scale parameters
    LinPred, RegParam, ScaleParam, VecScaleParam,
    ### fixed and random effects
    RegEffect, FixedEffect, RandomEffect,
    ### prior distributions
    Prior,
    ### optimization history
    OptInit, OptLog, OptSummary,
    ### availbale fitting algorithm
    Algorithm, SVB, MFVB, SVI, CSVI, MCMC

# Functions and methods
export
    ### indices of fixed and random effects
    get_param_indices,
    ### build the completed design matrix
    get_design_matrix,
    ### calculate and fill the prior penalization matrix
    init_prior_matrix, get_prior_matrix, fill_prior_matrix!,
    ### calculate and fill the hessian matrix
    init_hessian_matrix, get_hessian_matrix, fill_hessian_matrix!,
    ### calculate and fill the gradient vector
    init_gradient_vector, get_gradient_vector, fill_gradient_vector!,
    ### loss, expected log-likelihood and ELBO
    loss, psi, elbo, tailorder,
    ### model fitting
    fit, fit!,
    ### model dimensions
    nobs, nfe, nre, nfepar, nrepar, nregpar,
    ### fixed and random effect manipulation
    fixedeffects, randomeffects, dispersion, randeffvar,
    ### posterior summary statistics
    mean, median, mode, var, cov, cor, std,
    stderror, quantile, invmean, logmean,
    coef, confint, coeftable, summary,
    linpred, fitted, residuals, predict
    
# Posterior summary and plots
export 
    ### posterior plots
    plot_trace_delta, plot_trace_elbo,
    plot_trace_df, plot_trace_dx,
    plot_trace_sigma2e, plot_trace_sigma2u,
    plot_trace_regparam, plot_trace_rate

# global constants
const log2 = log(2)
const logπ = log(π)
const log2π = log(2π)
const sqrt2 = sqrt(2)
const sqrtπ = sqrt(π)
const sqrt2π = sqrt(2π)

# derived type definitions
const FP = Float64
const FPVec = Vector{FP}
const FPMat = Matrix{FP}
const FPVecMat = VecOrMat{FP}
const IntVec = Vector{Int64}

const VecMatrix{T <: FP} = Vector{Matrix{T}}
const SymMatrix{T <: FP} = Symmetric{T, Matrix{T}}
const DiagMatrix{T <: FP} = Diagonal{T, Vector{T}}
const LowTriMatrix{T <: FP} = LowerTriangular{T, Matrix{T}}
const UpTriMatrix{T <: FP} = UpperTriangular{T, Matrix{T}}
const VecUnitRange{T <: Int64} = Vector{UnitRange{T}}
const UIntRange = UnitRange{Int64}
const VecUIntRange = Vector{UnitRange{Int64}}

# const SpMatrix{T <: FP} = SparseMatrixCSC{T, Int64}

# number of knots for Gauss-Hermite quadrature
# const NKNOTS = 11

# Solvers and utilities
include("utilities.jl")
include("gaussquad.jl")
include("linalg.jl")
include("prior.jl")
include("algorithms.jl")
include("optsummary.jl")
include("family.jl")
include("models.jl")
include("summary.jl")
include("plots.jl")
include("psifun.jl")
include("expvals.jl")
include("elbo.jl")
include("logp.jl")
include("fittools.jl")
include("SVB/svbtools.jl")
include("SVB/svbfit.jl")
include("SVI/svitools.jl")
include("SVI/svifit.jl")
include("MFVB/mfvbtools.jl")
include("MFVB/mfvbfit.jl")
include("MCMC/mcmctools.jl")
include("MCMC/mcmcfit.jl")
include("fit.jl")


# WARNING: Gamma family still has to be tested!

end
