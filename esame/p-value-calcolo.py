import scipy.stats
 
def p_value(chisq, ndof):
    p = scipy.stats.chi2.cdf(chisq, ndof)
# If the probability is > 50% , take the complement. 
    if p > 0.5:
        p = 1.0 - p
    return p

print(p_value(10.0, 10))
