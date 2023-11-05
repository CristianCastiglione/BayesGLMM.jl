
# SVB -> NCVB (non-conjugate variational Bayes)
# MFVB -> CVB (conjugate variational Bayes)
# SVI -> NCSVI (non-conjugate stochastic variational inference)
# CSVI -> CSVI (conjugate stochastic variational inference)

"""
    Algorithm

Abstract type for the estimation algorithm available in the module.
"""

# Optimization algorithm abstract type
abstract type Algorithm end

"""
    SVB

Mutable object describing a semiparametric variational Bayes (SVB) routine.

- maxiter: maximum number of iterations before to stop the optimization
- verbose: boolean value indicating whether printing some information during the optimization
- report: positive integer setting the frequency with which a report has to be printed
- ftol: tollerance value for checking the convergence of the evidence lower bound
- xtol: tollerance value for checking the convergence of the variational parameters
- nknots: number of integration knots used for Guass-Hermite integration
- threshold: lower bound for the second order derivatives
- search: boolean value indicating whether perform a line search update of the learning rate
- rate0: initial learning rate
- decay: decay value for the learning rate dynamics
- random: boolean value indicating whether perform a random line search update of the learning rate
- ntry: maximum number of iterations for the line search update 
- lbound: lower bound for the random line search proposal
- ubound: upper bound for the random line search proposal
"""

# Semiparametric variational Bayes
mutable struct SVB <: Algorithm
    maxiter::Int64
    verbose::Bool
    report::Int64
    ftol::Float64
    xtol::Float64
    nknots::Int64
    threshold::Float64
    search::Bool
    rate0::Float64
    decay::Float64
    random::Bool
    ntry::Int64
    lbound::Float64
    ubound::Float64

    function SVB(
        maxiter::Int64 = 100,
        verbose::Bool = true,
        report::Int64 = 10,
        ftol::Float64 = 1e-04,
        xtol::Float64 = 1e-04,
        nknots::Int64 = 7,
        threshold::Float64 = 0.1,
        search::Bool = true,
        rate0::Float64 = 1.0,
        decay::Float64 = 0.01,
        random::Bool = false,
        ntry::Int64 = 10,
        lbound::Float64 = 0.3,
        ubound::Float64 = 0.9,
        )

        # Parameter check
        check_maxiter = maxiter > 0
        check_report = 0 < report ≤ maxiter
        check_ftol = ftol > .0
        check_xtol = xtol > .0
        check_nknots = nknots > 2
        check_thr = .0 < threshold < 1.
        check_rate0 = .0 < rate0 ≤ 2.
        check_decay = .0 ≤ decay < 1.
        check_ntry = ntry > 0
        check_lbound = .0 < lbound < ubound
        check_ubound = ubound < 1.

        # Error message
        check_maxiter ? nothing : error("Invalid control parameter: `maxiter` must be a positive integer.")
        check_report ? nothing : error("Invalid control parameter: `report` must be a positive integer lower than `maxiter`.")
        check_ftol ? nothing : error("Invalid control parameter: `ftol` small must be a positive real number.")
        check_xtol ? nothing : error("Invalid control parameter: `xtol` small must be a positive real number.")
        check_nknots ? nothing : error("Invalid control parameter: `nknots` must be a positive integer > 2.")
        check_thr ? nothing : error("Invalid control parameter: `threshold` must be a real number lying in (0,1).")
        check_rate0 ? nothing : error("Invalid control parameter: `rate0` must be a real number lying in (0,2].")
        check_decay ? nothing : error("Invalid control parameter: `decay` must be a real number lying in (0,1).")
        check_ntry ? nothing : error("Invalid control parameter: `ntry` must be a positive integer.")
        check_lbound ? nothing : error("Invalid control parameter: `lbound` must be a real number lying in (0,ubound).")
        check_ubound ? nothing : error("Invalid control parameter: `ubound` must be a real number lying in (lbound,1).")

        return new(
            maxiter, verbose, report, ftol, xtol, nknots, threshold,
            search, rate0, decay, random, ntry, lbound, ubound)
    end

    function SVB(;
        maxiter::Int64 = 100,
        verbose::Bool = true,
        report::Int64 = 10,
        ftol::Float64 = 1e-04,
        xtol::Float64 = 1e-04,
        nknots::Int64 = 7,
        threshold::Float64 = 0.1,
        search::Bool = true,
        rate0::Float64 = 1.0,
        decay::Float64 = 0.01,
        random::Bool = false,
        ntry::Int64 = 10,
        lbound::Float64 = 0.3,
        ubound::Float64 = 0.9,
        )

        return new(
            maxiter, verbose, report, ftol, xtol, nknots, threshold,
            search, rate0, decay, random, ntry, lbound, ubound)
    end
end;

function summary(alg::SVB)
    println()
    println(" SVB control parameters")
    println("-------------------------")
    println(" maxiter = ", alg.maxiter)
    println(" verbose = ", alg.verbose)
    println(" report = ", alg.report)
    println(" ftol = ", alg.ftol)
    println(" xtol = ", alg.xtol)
    println(" nknots = ", alg.nknots)
    println(" threshold = ", alg.threshold)
    println(" search = ", alg.search)
    println(" rate0 = ", alg.rate0)
    println(" decay = ", alg.decay)
    println(" random = " , alg.random)
    println(" ntry = ", alg.ntry)
    println(" lbound = " , alg.lbound)
    println(" ubound = " , alg.ubound)
    println("-------------------------")
end;

precompile(SVB, (Int64, Bool, Int64, FP, FP, Int64, FP, Bool, FP, FP, Bool, Int64, FP, FP, ))
precompile(SVB, ())
precompile(summary, (SVB, ))

"""
    SVI

Mutable object describing a stochastic variational inference (SVI) routine.

- miniter: minimum number of iterations before to stop the optimization
- maxiter: maximum number of iterations before to stop the optimization
- verbose: boolean value indicating whether printing some information during the optimization
- report: positive integer setting the frequency with which a report has to be printed
- ftol: tollerance value for checking the convergence of the evidence lower bound
- xtol: tollerance value for checking the convergence of the variational parameters
- nknots: number of integration knots used for Guass-Hermite integration
- threshold: lower bound for the second order derivatives
- minibatch: minibatch sample size 
- rate0: initial learning rate
- frate: forgetting rate
- delay: delay value for the learning rate dynamics
- decay: decay value for the learning rate dynamics
"""

# Stochastic variational inference
mutable struct SVI <: Algorithm
    miniter::Int64
    maxiter::Int64
    verbose::Bool
    report::Int64
    warmup::Bool
    ftol::Float64
    xtol::Float64
    nknots::Int64
    threshold::Float64
    minibatch::Int64
    initbatch::Int64
    stratified::Bool
    averaging::Bool
    burn::Int64
    rate0::Float64
    frate::Float64
    delay::Float64
    decay::Float64

    function SVI(
        miniter::Int64 = 1000,
        maxiter::Int64 = 5000,
        verbose::Bool = true,
        report::Int64 = 250,
        warmup::Bool = true,
        ftol::Float64 = 1e-05,
        xtol::Float64 = 1e-05,
        nknots::Int64 = 7,
        threshold::Float64 = 0.1,
        minibatch::Int64 = 100,
        initbatch::Int64 = 100,
        stratified::Bool = false,
        averaging::Bool = true,
        burn::Int64 = miniter,
        rate0::Float64 = 0.1,
        frate::Float64 = 0.75,
        delay::Float64 = 1.0,
        decay::Float64 = 1.0,
        )

        # Parameter check
        check_miniter = miniter > 0
        check_maxiter = maxiter > miniter
        check_report = 0 < report ≤ maxiter
        check_ftol = ftol > .0
        check_xtol = xtol > .0
        check_nknots = nknots > 2
        check_thr = .0 < threshold < 1.
        check_mbtch = minibatch > 0
        check_ibtch = initbatch > 0
        check_burn = miniter ≤  burn < maxiter
        check_rate0 = .0 < rate0 ≤ 1.
        check_frate = .5 < frate ≤ 1.
        check_delay = delay > 0
        check_decay = .0 ≤ decay < 1.

        # Error message
        check_miniter ? nothing : error("Invalid control parameter: `miniter` must be a positive integer.")
        check_maxiter ? nothing : error("Invalid control parameter: `maxiter` must be a positive integer grater than `miniter`.")
        check_report ? nothing : error("Invalid control parameter: `report` must be a positive integer lower than `maxiter`.")
        check_ftol ? nothing : error("Invalid control parameter: `ftol` small must be a positive real number.")
        check_xtol ? nothing : error("Invalid control parameter: `xtol` small must be a positive real number.")
        check_nknots ? nothing : error("Invalid control parameter: `nknots` must be a positive integer > 2.")
        check_thr ? nothing : error("Invalid control parameter: `threshold` must be a real number lying in (0,1).")
        check_mbtch ? nothing : error("Invalid control parameter: `minibatch` must be a positive integer number")
        check_ibtch ? nothing : error("Invalid control parameter: `initbatch` must be a positive integer number")
        check_burn ? nothing : error("Invalid control parameter: `burn` must be a positive integer between `miniter` and `maxiter`.")
        check_rate0 ? nothing : error("Invalid control parameter: `rate0` must be a real number lying in (0,2].")
        check_frate ? nothing : error("Invalid control parameter: `frate` must be a real number lying in (0.5,1].")
        check_delay ? nothing : error("Invalid control parameter: `delay` must be a real positive number.")
        check_decay ? nothing : error("Invalid control parameter: `decay` must be a real number lying in (0,1).")

        return new(
            miniter, maxiter, verbose, report, warmup, ftol, xtol, 
            nknots, threshold, minibatch, initbatch, stratified, 
            averaging, burn, rate0, frate, delay, decay)
    end

    function SVI(;
        miniter::Int64 = 1000,
        maxiter::Int64 = 500,
        verbose::Bool = true,
        report::Int64 = 50,
        warmup::Bool = false,
        ftol::Float64 = 1e-04,
        xtol::Float64 = 1e-04,
        nknots::Int64 = 7,
        threshold::Float64 = 0.1,
        minibatch::Int64 = 10,
        initbatch::Int64 = 100,
        stratified::Bool = false,
        averaging::Bool = true,
        burn::Int64 = miniter,
        rate0::Float64 = 0.1,
        frate::Float64 = 0.75,
        delay::Float64 = 1.0,
        decay::Float64 = 1.0,
        )

        return new(
            miniter, maxiter, verbose, report, warmup, ftol, xtol,
            nknots, threshold, minibatch, initbatch, stratified, 
            averaging, burn, rate0, frate, delay, decay)
    end
end;

function summary(alg::SVI)
    println()
    println(" SVI control parameters")
    println("-------------------------")
    println(" miniter = ", alg.miniter)
    println(" maxiter = ", alg.maxiter)
    println(" verbose = ", alg.verbose)
    println(" report = ", alg.report)
    println(" ftol = ", alg.ftol)
    println(" xtol = ", alg.xtol)
    println(" nknots = ", alg.nknots)
    println(" threshold = ", alg.threshold)
    println(" minibatch = ", alg.minibatch)
    println(" initbatch = ", alg.initbatch)
    println(" stratified = ", alg.stratified)
    println(" averaging = ", alg.averaging)
    println(" burn = ", alg.burn)
    println(" rate0 = ", alg.rate0)
    println(" frate = ", alg.frate)
    println(" delay = ", alg.delay)
    println(" decay = ", alg.decay)
    println("-------------------------")
end;

precompile(SVI, (Int64, Int64, Bool, Int64, FP, FP, Int64, Int64, Bool, Bool, Int64, FP, FP, FP, FP, ))
precompile(SVI, ())
precompile(summary, (SVI, ))

"""
    MFVB

Mutable object describing a mean field variational Bayes (MFVB) routine.

- maxiter: maximum number of iterations before to stop the optimization
- verbose: boolean value indicating whether printing some information during the optimization
- report: positive integer setting the frequency with which a report has to be printed
- ftol: tollerance value for checking the convergence of the evidence lower bound
- xtol: tollerance value for checking the convergence of the variational parameters
"""

# Mean field variational Bayes
mutable struct MFVB <: Algorithm
    maxiter::Int64
    verbose::Bool
    report::Int64
    ftol::Float64
    xtol::Float64
    nknots::Int64

    function MFVB(
        maxiter::Int64 = 100,
        verbose::Bool = true,
        report::Int64 = 10,
        ftol::Float64 = 1e-04,
        xtol::Float64 = 1e-04,
        nknots::Int64 = 7
        )

        # Parameter check
        check_maxiter = maxiter > 0
        check_report = 0 < report ≤ maxiter
        check_ftol = ftol > .0
        check_xtol = xtol > .0
        check_nknots = nknots > 2

        # Error message
        check_maxiter ? nothing : error("Invalid control parameter: `maxiter` must be a positive integer.")
        check_report ? nothing : error("Invalid control parameter: `report` must be a positive integer lower than `maxiter`.")
        check_ftol ? nothing : error("Invalid control parameter: `ftol` must be a positive real number.")
        check_xtol ? nothing : error("Invalid control parameter: `xtol` must be a positive real number.")
        check_nknots ? nothing : error("Invalid control parameter: `nknots` must be a positive integer grater than 2.")

        return new(maxiter, verbose, report, ftol, xtol, nknots)
    end

    function MFVB(;
        maxiter::Int64 = 100,
        verbose::Bool = true,
        report::Int64 = 10,
        ftol::Float64 = 1e-04,
        xtol::Float64 = 1e-04,
        nknots::Int64 = 7
        )

        return new(maxiter, verbose, report, ftol, xtol, nknots)
    end
end;

function summary(alg::MFVB)
    println()
    println(" MFVB control parameters")
    println("-------------------------")
    println(" maxiter = ", alg.maxiter)
    println(" verbose = ", alg.verbose)
    println(" report = ", alg.report)
    println(" ftol = ", alg.ftol)
    println(" xtol = ", alg.xtol)
    println(" nknots = ", alg.nknots)
    println("-------------------------")
end;

precompile(MFVB, (Int64, Bool, Int64, FP, FP, Int64, ))
precompile(MFVB, ())
precompile(summary, (MFVB, ))

"""
    CSVI

Mutable object describing a conjugate stochastic variational inference (SVI) routine.

- miniter: minimum number of iterations before to stop the optimization
- maxiter: maximum number of iterations before to stop the optimization
- verbose: boolean value indicating whether printing some information during the optimization
- report: positive integer setting the frequency with which a report has to be printed
- ftol: tollerance value for checking the convergence of the evidence lower bound
- xtol: tollerance value for checking the convergence of the variational parameters
- nknots: number of integration knots used for Guass-Hermite integration
- threshold: lower bound for the second order derivatives
- minibatch: minibatch sample size 
- rate0: initial learning rate
- frate: forgetting rate
- delay: delay value for the learning rate dynamics
- decay: decay value for the learning rate dynamics
"""

# Stochastic variational inference
mutable struct CSVI <: Algorithm
    miniter::Int64
    maxiter::Int64
    verbose::Bool
    report::Int64
    warmup::Bool
    ftol::Float64
    xtol::Float64
    nknots::Int64
    threshold::Float64
    minibatch::Int64
    rate0::Float64
    frate::Float64
    delay::Float64
    decay::Float64

    function CSVI(
        miniter::Int64 = 100,
        maxiter::Int64 = 500,
        verbose::Bool = true,
        report::Int64 = 50,
        warmup::Bool = false,
        ftol::Float64 = 1e-04,
        xtol::Float64 = 1e-04,
        nknots::Int64 = 7,
        threshold::Float64 = 0.1,
        minibatch::Int64 = 10,
        rate0::Float64 = 0.6,
        frate::Float64 = 0.75,
        delay::Float64 = 1.0,
        decay::Float64 = 0.1,
        )

        # Parameter check
        check_miniter = miniter > 0
        check_maxiter = maxiter > miniter
        check_report = 0 < report ≤ maxiter
        check_ftol = ftol > .0
        check_xtol = xtol > .0
        check_nknots = nknots > 2
        check_thr = .0 < threshold < 1.
        check_mbtch = minibatch > 0
        check_rate0 = .0 < rate0 ≤ 1.
        check_frate = .5 < frate ≤ 1.
        check_delay = delay > 0
        check_decay = .0 ≤ decay < 1.

        # Error message
        check_miniter ? nothing : error("Invalid control parameter: `miniter` must be a positive integer.")
        check_maxiter ? nothing : error("Invalid control parameter: `maxiter` must be a positive integer grater than `miniter`.")
        check_report ? nothing : error("Invalid control parameter: `report` must be a positive integer lower than `maxiter`.")
        check_ftol ? nothing : error("Invalid control parameter: `ftol` small must be a positive real number.")
        check_xtol ? nothing : error("Invalid control parameter: `xtol` small must be a positive real number.")
        check_nknots ? nothing : error("Invalid control parameter: `nknots` must be a positive integer > 2.")
        check_thr ? nothing : error("Invalid control parameter: `threshold` must be a real number lying in (0,1).")
        check_mbtch ? nothing : error("Invalid control parameter: `minibatch` must be a positive integer number")
        check_rate0 ? nothing : error("Invalid control parameter: `rate0` must be a real number lying in (0,2].")
        check_frate ? nothing : error("Invalid control parameter: `frate` must be a real number lying in (0.5,1].")
        check_delay ? nothing : error("Invalid control parameter: `delay` must be a real positive number.")
        check_decay ? nothing : error("Invalid control parameter: `decay` must be a real number lying in (0,1).")

        return new(
            miniter, maxiter, verbose, report, warmup, ftol, xtol, 
            nknots, threshold, minibatch, rate0, frate, delay, decay)
    end

    function CSVI(;
        miniter::Int64 = 100,
        maxiter::Int64 = 500,
        verbose::Bool = true,
        report::Int64 = 50,
        warmup::Bool = false,
        ftol::Float64 = 1e-04,
        xtol::Float64 = 1e-04,
        nknots::Int64 = 7,
        threshold::Float64 = 0.1,
        minibatch::Int64 = 10,
        rate0::Float64 = 0.6,
        frate::Float64 = 0.75,
        delay::Float64 = 1.0,
        decay::Float64 = 0.1,
        )

        return new(
            miniter, maxiter, verbose, report, warmup, ftol, xtol,
            nknots, threshold, minibatch, rate0, frate, delay, decay)
    end
end;

function summary(alg::CSVI)
    println()
    println(" SVI control parameters")
    println("-------------------------")
    println(" miniter = ", alg.miniter)
    println(" maxiter = ", alg.maxiter)
    println(" verbose = ", alg.verbose)
    println(" report = ", alg.report)
    println(" ftol = ", alg.ftol)
    println(" xtol = ", alg.xtol)
    println(" nknots = ", alg.nknots)
    println(" threshold = ", alg.threshold)
    println(" minibatch = ", alg.minibatch)
    println(" rate0 = ", alg.rate0)
    println(" frate = ", alg.frate)
    println(" delay = ", alg.delay)
    println(" decay = ", alg.decay)
    println("-------------------------")
end;

precompile(CSVI, (Int64, Int64, Bool, Int64, FP, FP, Int64, FP, FP, FP, FP, FP, ))
precompile(CSVI, ())
precompile(summary, (CSVI, ))

"""
    MCMC

Mutable object describing a Markov chain Monte Carlo (MFVB) routine.

- maxiter: length of the chain
- burn: number of burn-in iterations
- verbose: boolean value indicating whether printing some information during the optimization
- report: positive integer setting the frequency with which a report has to be printed
- exact: boolean value indicating whether the sampling has to be exact or approximated
"""

# Markov chain Monte Carlo
mutable struct MCMC <: Algorithm
    maxiter::Int64
    burn::Int64
    verbose::Bool
    report::Int64
    exact::Bool

    function MCMC(
        maxiter::Int64 = 6000,
        burn::Int64 = 1000,
        verbose::Bool = true,
        report::Int64 = 1000,
        exact::Bool = true
        )

        # Parameter check
        check_maxiter = maxiter > 0
        check_report = 0 < report ≤ maxiter

        # Error message
        check_maxiter ? nothing : error("Invalid control parameter: `maxiter` must be a positive integer.")
        check_report ? nothing : error("Invalid control parameter: `report` must be a positive integer lower than `maxiter`.")
        
        return new(maxiter, verbose, report, exact)
    end

    function MCMC(;
        maxiter::Int64 = 6000,
        burn::Int64 = 1000,
        verbose::Bool = true,
        report::Int64 = 1000,
        exact::Bool = true
        )

        return new(maxiter, burn, verbose, report, exact)
    end
end;

function summary(alg::MCMC)
    println()
    println(" MCMC control parameters")
    println("-------------------------")
    println(" maxiter = ", alg.maxiter)
    println(" burn = ", alg.burn)
    println(" verbose = ", alg.verbose)
    println(" report = ", alg.report)
    println(" exact = ", alg.exact)
    println("-------------------------")
end;

precompile(MCMC, (Int64, Int64, Bool, Int64, Bool, ))
precompile(MCMC, ())
precompile(summary, (MCMC, ))
