
"""
    Prior

Object describing the set of prior distribution for the parmeters 
of the model. It contains the prior hyperparameters A_u, B_u, A_e, B_e,
sigma2_b and the initial random effect variance sigma2_0.
"""

# Prior parameter definition
mutable struct Prior
    A_u::FP            # shape prior hyperparameter for the random effect scale parameters
    B_u::FP            # rate prior hyperparameter for the random effect scale parameters
    A_e::FP            # shape prior hyperparameter for the error scale parameter
    B_e::FP            # rate prior hyperparameter for the erro scale parameter
    sigma2_b::FP       # prior variance of the fixed effect parameters
    sigma2_0::FP       # initial variance of the random effect parameters

    function Prior(
        A_u::FP = 2.0001,
        B_u::FP = 1.0001,
        A_e::FP = 2.0001,
        B_e::FP = 1.0001,
        sigma2_b::FP = 1e+06,
        sigma2_0::FP = 1e+03
        )

        # Parameter check
        check_Au = A_u > 0.
        check_Bu = B_u > 0.
        check_Ae = A_e > 0.
        check_Be = B_e > 0.
        check_sigma2b = sigma2_b > 0.
        check_sigma20 = sigma2_0 > 0.

        # Error message
        check_Au ? nothing : error("Invalid prior parameter: `A_u` must be a positive real number.")
        check_Bu ? nothing : error("Invalid prior parameter: `B_u` must be a positive real number.")
        check_Ae ? nothing : error("Invalid prior parameter: `A_e` must be a positive real number.")
        check_Be ? nothing : error("Invalid prior parameter: `B_e` must be a positive real number.")
        check_sigma2b ? nothing : error("Invalid prior parameter: `sigma2_b` must be a positive real number.")
        check_sigma20 ? nothing : error("Invalid prior parameter: `sigma2_0` must be a positive real number.")

        return new(A_u, B_u, A_e, B_e, sigma2_b, sigma2_0)
    end

    function Prior(;
        A_u::FP = 2.0001,
        B_u::FP = 1.0001,
        A_e::FP = 2.0001,
        B_e::FP = 1.0001,
        sigma2_b::FP = 1e+06,
        sigma2_0::FP = 1e+03
        )

        return new(A_u, B_u, A_e, B_e, sigma2_b, sigma2_0)
    end
end;

"""
    summary(prior)

Method of a Prior object which print the main characteristics 
of the object itself.
"""

function summary(prior::Prior)
    println()
    println(" Prior parameters ")
    println("-----------------------------------------")
    println(" sigma2_u : A_u = ", prior.A_u, ",  B_u = ", prior.B_u)
    println(" sigma2_e : A_e = ", prior.A_e, ",  B_e = ", prior.B_e)
    println(" reg.par. : sigma2_b = ", prior.sigma2_b)
    println(" init.var. : sigma2_0 = ", prior.sigma2_0)
    println("-----------------------------------------")
end;

precompile(Prior, (FP, FP, FP, FP, FP, FP, ))
precompile(Prior, ())
precompile(summary, (Prior, ))