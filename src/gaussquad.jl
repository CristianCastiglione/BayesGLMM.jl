
# Monahan-Stefanski knots and weights
msk8 = [
    1.365340806296348; 1.059523971016916; 0.830791313765644; 0.650732166639391; 
    0.508135425366489; 0.396313345166341; 0.308904252267995; 0.238212616409306]
msw8 = [
    0.003246343272134; 0.051517477033972; 0.195077912673858; 0.315569823632818; 
    0.274149576158423; 0.131076880695470; 0.027912418727972; 0.001449567805354]

# Gauss-Hermite knots and weights
ghk3, ghw3 = gausshermite(3)
ghk5, ghw5 = gausshermite(5)
ghk7, ghw7 = gausshermite(7)
ghk9, ghw9 = gausshermite(9)
ghk11, ghw11 = gausshermite(11)
ghk13, ghw13 = gausshermite(13)
ghk15, ghw15 = gausshermite(15)
ghk17, ghw17 = gausshermite(17)
ghk19, ghw19 = gausshermite(19)
ghk21, ghw21 = gausshermite(21)

function get_gh_points(n::Int64 = 3)
    k = Vector{FP}(undef, n)
    w = Vector{FP}(undef, n)
    if n == 3
        k[:], w[:] = ghk3, ghw3
    elseif n == 5
        k[:], w[:] = ghk5, ghw5
    elseif n == 7
        k[:], w[:] = ghk7, ghw7
    elseif n == 9
        k[:], w[:] = ghk9, ghw9
    elseif n == 11
        k[:], w[:] = ghk11, ghw11
    elseif n == 13
        k[:], w[:] = ghk13, ghw13
    elseif n == 15
        k[:], w[:] = ghk15, ghw15
    elseif n == 17
        k[:], w[:] = ghk17, ghw17
    elseif n == 19
        k[:], w[:] = ghk19, ghw19
    elseif n == 21
        k[:], w[:] = ghk21, ghw21
    else
        k, w = gausshermite(n)
    end
    return k, w
end
