
# Initial parameters
mutable struct OptInit
    theta::Vector{FP}
    sigma2_u::Vector{FP}
    sigma2_e::FP
    elbo::FP

    function OptInit(
        theta::Vector{FP} = Vector{FP}(undef, 0),
        sigma2_u::Vector{FP} = Vector{FP}(undef, 0),
        sigma2_e::FP = .0,
        elbo::FP = .0,
        )

        return new(theta, sigma2_u, sigma2_e, elbo)
    end

    function OptInit(;
        theta::Vector{FP} = Vector{FP}(undef, 0),
        sigma2_u::Vector{FP} = Vector{FP}(undef, 0),
        sigma2_e::FP = .0,
        elbo::FP = .0,
        )

        return new(theta, sigma2_u, sigma2_e, elbo)
    end
end

function OptInit(
    n_fe_par::Int64,
    n_re_par::Vector{Int64},
    )

    n_fe = 1
    n_re = length(n_re_par)

    theta = Vector{FP}(undef, n_fe_par + sum(n_re_par))
    sigma2_u = Vector{FP}(undef, n_re)
    sigma2_e = .0
    elbo = .0

    return OptInit(theta, sigma2_u, sigma2_e, elbo)
end

precompile(OptInit, (Vector{FP}, Vector{FP}, FP, FP, ))
precompile(OptInit, (Int64, Vector{Int64}, ))
precompile(OptInit, ())

# Optimization history
mutable struct OptLog
    theta::Matrix{FP} # regression parameters
    sigma2_u::Matrix{FP} # random effect variances
    sigma2_e::Vector{FP} # random error variance
    elbo::Vector{FP} # evidence lower bound
    rate::Vector{FP} # learning rate (if any)
    df::Vector{FP} # absolute relative change of the elbo
    dx::Vector{FP} # absolute relative chenge of the parameters

    function OptLog(
        theta::Matrix{FP} = Matrix{FP}(undef, 0, 0),
        sigma2_u::Matrix{FP} = Matrix{FP}(undef, 0, 0),
        sigma2_e::Vector{FP} = Vector{FP}(undef, 0),
        elbo::Vector{FP} = Vector{FP}(undef, 0),
        rate::Vector{FP} = Vector{FP}(undef, 0),
        df::Vector{FP} = Vector{FP}(undef, 0),
        dx::Vector{FP} = Vector{FP}(undef, 0)
        ) where T <: Float64

        return new(theta, sigma2_u, sigma2_e, elbo, rate, df, dx)
    end

    function OptLog(;
        theta::Matrix{FP} = Matrix{FP}(undef, 0, 0),
        sigma2_u::Matrix{FP} = Matrix{FP}(undef, 0, 0),
        sigma2_e::Vector{FP} = Vector{FP}(undef, 0),
        elbo::Vector{FP} = Vector{FP}(undef, 0),
        rate::Vector{FP} = Vector{FP}(undef, 0),
        df::Vector{FP} = Vector{FP}(undef, 0),
        dx::Vector{FP} = Vector{FP}(undef, 0)
        )

        return new(theta, sigma2_u, sigma2_e, elbo, rate, df, dx)
    end
end

function OptLog(
    n_fe_par::Int64,
    n_re_par::Vector{Int64},
    maxiter::Int64
    )

    n_fe = 1
    n_re = length(n_re_par)
    n_par = n_fe_par + sum(n_re_par)

    theta    = Matrix{FP}(undef, maxiter, n_par)
    sigma2_u = Matrix{FP}(undef, maxiter, n_re)
    sigma2_e = Vector{FP}(undef, maxiter)
    elbo     = Vector{FP}(undef, maxiter)
    rate     = Vector{FP}(undef, maxiter-1)
    df       = Vector{FP}(undef, maxiter-1)
    dx       = Vector{FP}(undef, maxiter-1)

    return OptLog(theta, sigma2_u, sigma2_e, elbo, rate, df, dx)
end

precompile(OptLog, (Matrix{FP}, Matrix{FP}, Vector{FP}, Vector{FP}, Vector{FP}, Vector{FP}, Vector{FP}, ))
precompile(OptLog, (Int64, Vector{Int64}, Int64, ))
precompile(OptLog, ())

# Optimization output
mutable struct OptSummary
    niter::Int64
    exetime::FP
    success::Bool
    fitted::Bool
    fitinit::OptInit
    fitlog::OptLog

    function OptSummary(
        niter::Int64 = -1,
        exetime::FP = 0.0,
        success::Bool = false,
        fitted::Bool = false,
        fitinit::OptInit = OptInit(),
        fitlog::OptLog = OptLog()
        )

        return new(niter, exetime, success, fitted, fitinit, fitlog)
    end

    function OptSummary(;
        niter::Int64 = -1,
        exetime::FP = 0.0,
        success::Bool = false,
        fitted::Bool = false,
        fitinit::OptInit = OptInit(),
        fitlog::OptLog = OptLog()
        )

        return new(niter, exetime, success, fitted, fitinit, fitlog)
    end
end

function OptSummary(
    n_fe_par::Int64,
    n_re_par::Vector{Int64},
    maxiter::Int64
    )

    return OptSummary(
        -1, 0.0, false, false, 
        OptInit(n_fe_par, n_re_par), 
        OptLog(n_fe_par, n_re_par, maxiter))
end

function summary(opt::OptSummary)
    println()
    println(" Optimization summary ")
    println("-----------------------------------------")
    println(" niter = ", opt.niter)
    println(" exetime = ", round(opt.exetime, digits = 3))
    println(" fitted = ", opt.fitted)
    println(" success = ", opt.success)
    @printf(" initial elbo = %.5f \n", opt.fitinit.elbo)
    @printf(" final elbo = %.5f \n", opt.fitlog.elbo[end])
    @printf(" final elbo relative change = %.5f \n", opt.fitlog.df[end])
    @printf(" final param. relative change = %.5f \n", opt.fitlog.dx[end])
    println("-----------------------------------------")
end

precompile(OptSummary, (Int64, FP, Bool, Bool, OptInit, OptLog))
precompile(OptSummary, (Int64, Vector{Int64}, Int64, ))
precompile(OptSummary, ())
precompile(summary, (OptSummary, ))