

"""
    absmax(u, v)

Returns the absolute difference between u and v normalized
by the maximum of u and v. A tilting constant is applied to
the denominator to avoid division by 0.
"""
function absmax(u::T, v::T)::T where T <: FP
    return abs(u - v) / maximum(abs.(vcat(u, v)) .+ 1e-04)
end

function absmax(u::V, v::V)::T where {T <: FP, V <: Vector{T}}
    return maximum(abs.(u .- v)) / maximum(abs.(vcat(u, v)) .+ 1e-04)
end

precompile(absmax, (FP, FP, ))
precompile(absmax, (Vector{FP}, Vector{FP}, ))

"""
    trim(x, t)

Truncates the value of x if x < t and returns the value.
"""
function trim(x::FP, t::FP)::FP
    return (x > t) * x + (x < t) * t
end

precompile(trim, (FP, FP, ))

"""
    cdfn(x)

Standard Gaussian cumulative density function.
"""
function cdfn(x::FP)::FP
    return cdf(Normal(), x)
end

precompile(cdfn, (FP, ))

"""
    pdfn(x)

Standard Gaussian probability density function.
"""
function pdfn(x::FP)::FP
    return pdf(Normal(), x)
end

precompile(pdfn, (FP, ))

"""
    logcdf(x)

Standard Gaussian log cumulative density function.
"""
function logcdfn(x::FP)::FP
    return logcdf(Normal(), x)
end

precompile(logcdfn, (FP, ))

"""
    logcdf(x)

Standard Gaussian log probability density function.
"""
function logpdfn(x::FP)::FP
    return logpdf(Normal(), x)
end

precompile(logpdfn, (FP, ))

"""
    logit(x)

Logistic function.
"""
function logit(x::FP)::FP
    return log(x) - log(1.0 - x)
end

precompile(logit, (FP, ))


"""
    expit(x)
    expit2(x)

Inverse of logistic function and modified inverse
logistic function.
"""
function expit(x::FP)::FP
    return exp(x - log1pexp(x))
end

function expit2(x::FP)::FP
    return exp(x - 2 * log1pexp(x))
end

precompile(expit, (FP, ))
precompile(expit2, (FP, ))

"""
    cloglog(x)

Complementary log-log function.
"""
function cloglog(x::FP)::FP
    return log(- log(1 - x))
end 

precompile(cloglog, (FP, ))

"""
    cexpexp(x)

Inverse of the complementary log-log function.
"""
function cexpexp(x::FP)::FP
    return 1 - exp(- exp(x))
end

precompile(cexpexp, (FP, ))

"""
    minibatch_indices(n, m, y, alg)

Returns a vector of indices of length m randomly sampled 
without repetition from the index set {1, ..., n}
"""
function random_minibatch(n::Int64, m::Int64)::Vector{Int64}
    return sample(1:n, m, replace = false)
    # return randperm(n)[1:m]
end

function stratified_minibatch(idx_1::Vector{Int64}, idx_0::Vector{Int64}, m::Int64)::Vector{Int64}
    n1 = length(idx_1)
    n0 = length(idx_0)
    n = n1 + n0 

    m1 = maximum([1, floor(Int64, n1 * m / n)])
    m0 = m - m1

    # idx_mb_1 = idx_1[randperm(n1)[1:m1]]
    # idx_mb_0 = idx_0[randperm(n0)[1:m0]]

    idx_mb_1 = idx_1[sample(1:n1, m1, replace = false)]
    idx_mb_0 = idx_0[sample(1:n0, m0, replace = false)]

    return [idx_mb_1; idx_mb_0]
end

function minibatch_indices(
    n::Int64, m::Int64, 
    idx_1::Union{Nothing, Vector{Int64}}, 
    idx_0::Union{Nothing, Vector{Int64}}, 
    stratified::Bool)::Vector{Int64}

    idx = Vector{Int64}(undef, m)
    check = !(isnothing(idx_1) | isnothing(idx_0))

    if check & stratified
        idx .= stratified_minibatch(idx_1, idx_0, m)
    else
        idx .= random_minibatch(n, m)
    end

    return idx
end

precompile(random_minibatch, (Int64, Int64, ))
precompile(stratified_minibatch, (Vector{Int64}, Vector{Int64}, Int64, ))
precompile(minibatch_indices, (Int64, Int64, Nothing, Nothing, Bool, ))
precompile(minibatch_indices, (Int64, Int64, Vector{Int64}, Vector{Int64}, Bool, ))

# """
#     minibatch_list(n, m)

# Returns a list where the k-th component is a vector of indices
# corresponding to the k-th minibatch. The indices are shuffled
# and sampled without repetition from the index set {1, ..., n}.
# """
# function minibatch_list(n::Int64, m::Int64)::Vector{Vector{Int64}}
#     idx = shuffle(collect(1:n_obs))
#     idx = [idx[k:min(k+m-1, n)] for i in 1:m:n]
#     return idx
# end

# function select_minibatch(k::Int64, n::Int64)::Int64
#     m = mod(k, n)
#     idx = m == 0 ? n : m
#     return idx
# end