#!/usr/bin/env python3

import collections
import random
import scipy.optimize as opt
import random as rng
from math import comb as binom
from math import log2, log, ceil, inf
#import numpy


def HHH(c):
    """
    binary entropy
    :param c: float in [0,1]
    """
    if c == 0. or c == 1.:
        return 0.
    if c < 0. or c > 1.:
        return -1000

    return -(c * log2(c) + (1 - c) * log2(1 - c))


def H(c):
    """
    binary entropy
    :param c: float in [0,1]
    """
    if c == 0. or c == 1.:
        return 0.

    if c < 0. or c > 1.:
        return -1000

    return -(c * log2(c) + (1 - c) * log2(1 - c))


def binomH(n, k):
    """
    binary entropy
    :param n: int
    :param k: int
    """
    if k > n:
        return 0
    if k == n:
        return 0
    return n * HHH(k/n)


def binomHH(n, k):
    """
    same as `binomH` without checks
    """
    return n * HHH(k/n)


def time7diss(x):
    """
    magic function for the 7-dissection
    """
    return -2*x/3+2/3


def check_constraints(constraints, solution):
    """
    checks whether constrains are fulfilled or not
    """
    return [(constraint['type'], constraint['fun'](solution))
            for constraint in constraints]


def wrap(f, g):
    """
    helper function injecting variables names into the optimization process.
    """
    def inner(x):
        return f(g(*x))
    return inner


def round_to_str(t):
    """
    Rounds the value 't' to a string with 4 digit precision
    (adding trailing zeroes to emphasize precision).
    """
    s = str(round(t, 4))
    # must be 6 digits
    return (s + "0" * (5 + s.find(".") - len(s)))


def round_upwards_to_str(t):
    """
    Rounds the value 't' *upwards* to a string with 4 digit precision
    (adding trailing zeroes to emphasize precision).
    """
    s = str(ceil(t*10000)/10000)
    # must be 6 digits
    return (s + "0" * (5 + s.find(".") - len(s)))


def xlx(x):
    """
    SOURCE: https://github.com/xbonnetain/optimization-subset-sum
    """
    if x <= 0:
        return 0
    return x*log2(x)


def p_good(a0, b0, a1, b1):
    """
    SOURCE: https://github.com/xbonnetain/optimization-subset-sum
    """
    return -2*xlx(a0/2) - 2*xlx(b0/2) - xlx(a1-a0/2) - xlx(b1-b0/2) \
           - xlx(1-a1-b1-a0/2-b0/2) - 2*g(a1, b1)


def g(a, b):
    """
    SOURCE: https://github.com/xbonnetain/optimization-subset-sum
    """
    return -xlx(a) - xlx(b) - xlx(1-a-b)


def f(a, b, c):
    """
    SOURCE: https://github.com/xbonnetain/optimization-subset-sum
    """
    if a <= 0:
        return g(b, c)
    if b <= 0:
        return g(a, c)
    if c <= 0:
        return g(a, b)
    if a+b+c >= 1:
        return min(g(b, c), g(a, c), g(a, b))
    try:
        return -a*log(a, 2) - b*log(b, 2) - c*log(c, 2)\
                - (1-a-b-c)*log(1-a-b-c, 2)
    except:
        return 0.


def p_good_2(b0, a0, c0, b1, a1, c1):
    """
    SOURCE: https://github.com/xbonnetain/optimization-subset-sum
    """
    def proba(x):
        return 2*xlx(a0/2) + 2*xlx(x+a1-a0/2-b0/2) + xlx(1-c0-2*a1-2*x)\
                + 2*xlx(b0/2-x) + 2*xlx(x) + 2*xlx(x+c0/2-b1/2+a1/2-a0/4-b0/4)\
                + xlx(b1-a1+a0/2+b0/2-2*x)

    bounds = [(max(a0/2+b0/2-a1, 0, b1/2-a1/2+a0/4+b0/4-c0/2),
               min(1/2.-c0/2-a1, b0/2, b1/2-a1/2+a0/4+b0/4))]
    if bounds[0][0] > bounds[0][1]:
        return p_good(b0, a0, b1, a1) - 1
    return - opt.fminbound(proba, bounds[0][0], bounds[0][1], xtol=1e-15,
                           full_output=1)[1] - 2*f(a1, b1, c1)


def p_good_2_aux(b0, a0, c0, b1, a1, c1):
    """
    SOURCE: https://github.com/xbonnetain/optimization-subset-sum
    """
    return -(2*xlx(a1-c1) + xlx(1-2*c1-2*b1) + 2*xlx(c1) + 2*xlx(b0/2-c1))\
            - 2*f(a1, b1, c1)


def H1(value):
    """
    inverse of the bin entropy function. Inverse only on [0,1] -> [0, 1/2]
    """
    if value == 1.0:
        return 0.5

    # approximate inverse binary entropy function
    steps = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
             0.00000001, 0.0000000001, 0.0000000000001, 0.000000000000001]
    r = 0.000000000000000000000000000000001

    for step in steps:
        i = r
        while (i + step < 1.0) and (H(i) < value):
            i += step

        r = i - step

    return r


def Hi(value):
    """
    helper wrapper
    """
    return H1(value)

###############################################################################
##################################BCJ##########################################
###############################################################################
# Beware: level at the bottom is level 0!
# Variables used:
# c_i: total bit constraint at level i
# l_i: list sizes after filtering
# p_i: probability of filtering at level i
# alpha_i: number of negative ones at level i
# gamma_i: number of twos at level i


number_ones_level_0 = lambda x: 1/2.
number_ones_level_1 = lambda x: 1/4.    + x.alpha1
number_ones_level_2 = lambda x: 1/8.    + x.alpha2
number_ones_level_3 = lambda x: 1/16.   + x.alpha3
number_ones_level_4 = lambda x: 1/32.   + x.alpha4
number_ones_level_5 = lambda x: 1/64.   + x.alpha5
number_ones_level_6 = lambda x: 1/128.  + x.alpha6
number_ones_level_7 = lambda x: 1/256.  + x.alpha7
number_ones_level_8 = lambda x: 1/512.  + x.alpha8
number_ones_level_9 = lambda x: 1/1024. + x.alpha9

number_neg_ones_level_0 = lambda x: 0.
number_neg_ones_level_1 = lambda x: x.alpha1
number_neg_ones_level_2 = lambda x: x.alpha2
number_neg_ones_level_3 = lambda x: x.alpha3
number_neg_ones_level_4 = lambda x: x.alpha4
number_neg_ones_level_5 = lambda x: x.alpha5
number_neg_ones_level_6 = lambda x: x.alpha6
number_neg_ones_level_7 = lambda x: x.alpha7
number_neg_ones_level_8 = lambda x: x.alpha8
number_neg_ones_level_9 = lambda x: x.alpha9

number_twos_level_0 = lambda x: 0
number_twos_level_1 = lambda x: x.gamma1
number_twos_level_2 = lambda x: x.gamma2
number_twos_level_3 = lambda x: x.gamma3
number_twos_level_4 = lambda x: x.gamma4
number_twos_level_5 = lambda x: x.gamma5
number_twos_level_6 = lambda x: x.gamma6
number_twos_level_7 = lambda x: x.gamma7
number_twos_level_8 = lambda x: x.gamma8
number_twos_level_9 = lambda x: x.gamma9

domain_level_1 = lambda x: g(number_ones_level_1(x), number_neg_ones_level_1(x))
domain_level_2 = lambda x: g(number_ones_level_2(x), number_neg_ones_level_2(x))
domain_level_3 = lambda x: g(number_ones_level_3(x), number_neg_ones_level_3(x))
domain_level_4 = lambda x: g(number_ones_level_4(x), number_neg_ones_level_4(x))
domain_level_5 = lambda x: g(number_ones_level_5(x), number_neg_ones_level_5(x))
domain_level_6 = lambda x: g(number_ones_level_6(x), number_neg_ones_level_6(x))
domain_level_7 = lambda x: g(number_ones_level_7(x), number_neg_ones_level_7(x))
domain_level_8 = lambda x: g(number_ones_level_8(x), number_neg_ones_level_8(x))
domain_level_9 = lambda x: g(number_ones_level_9(x), number_neg_ones_level_9(x))

bbss_domain_level_1 = lambda x: f(number_ones_level_1(x)-2*number_twos_level_1(x), number_neg_ones_level_1(x), number_twos_level_1(x))
bbss_domain_level_2 = lambda x: f(number_ones_level_2(x)-2*number_twos_level_2(x), number_neg_ones_level_2(x), number_twos_level_2(x))
bbss_domain_level_3 = lambda x: f(number_ones_level_3(x)-2*number_twos_level_3(x), number_neg_ones_level_3(x), number_twos_level_3(x))
bbss_domain_level_4 = lambda x: f(number_ones_level_4(x)-2*number_twos_level_4(x), number_neg_ones_level_4(x), number_twos_level_4(x))
bbss_domain_level_5 = lambda x: f(number_ones_level_5(x)-2*number_twos_level_5(x), number_neg_ones_level_5(x), number_twos_level_5(x))
bbss_domain_level_6 = lambda x: f(number_ones_level_6(x)-2*number_twos_level_6(x), number_neg_ones_level_6(x), number_twos_level_6(x))
bbss_domain_level_7 = lambda x: f(number_ones_level_7(x)-2*number_twos_level_7(x), number_neg_ones_level_7(x), number_twos_level_7(x))
bbss_domain_level_8 = lambda x: f(number_ones_level_8(x)-2*number_twos_level_8(x), number_neg_ones_level_8(x), number_twos_level_8(x))
bbss_domain_level_9 = lambda x: f(number_ones_level_9(x)-2*number_twos_level_9(x), number_neg_ones_level_9(x), number_twos_level_9(x))

filtering_0 = lambda x: p_good(number_ones_level_0(x), number_neg_ones_level_0(x), number_ones_level_1(x), number_neg_ones_level_1(x))
filtering_1 = lambda x: p_good(number_ones_level_1(x), number_neg_ones_level_1(x), number_ones_level_2(x), number_neg_ones_level_2(x))
filtering_2 = lambda x: p_good(number_ones_level_2(x), number_neg_ones_level_2(x), number_ones_level_3(x), number_neg_ones_level_3(x))
filtering_3 = lambda x: p_good(number_ones_level_3(x), number_neg_ones_level_3(x), number_ones_level_4(x), number_neg_ones_level_4(x))
filtering_4 = lambda x: p_good(number_ones_level_4(x), number_neg_ones_level_4(x), number_ones_level_5(x), number_neg_ones_level_5(x))
filtering_5 = lambda x: p_good(number_ones_level_5(x), number_neg_ones_level_5(x), number_ones_level_6(x), number_neg_ones_level_6(x))
filtering_6 = lambda x: p_good(number_ones_level_6(x), number_neg_ones_level_6(x), number_ones_level_7(x), number_neg_ones_level_7(x))
filtering_7 = lambda x: p_good(number_ones_level_7(x), number_neg_ones_level_7(x), number_ones_level_8(x), number_neg_ones_level_8(x))
filtering_8 = lambda x: p_good(number_ones_level_8(x), number_neg_ones_level_8(x), number_ones_level_9(x), number_neg_ones_level_9(x))

bbss_filtering_0 = lambda x: p_good_2_aux(number_ones_level_0(x)-2*number_twos_level_0(x), number_neg_ones_level_0(x), number_twos_level_0(x), \
                                          number_ones_level_1(x)-2*number_twos_level_1(x), number_neg_ones_level_1(x), number_twos_level_1(x))
bbss_filtering_1 = lambda x: p_good_2(number_ones_level_1(x)-2*number_twos_level_1(x), number_neg_ones_level_1(x), number_twos_level_1(x), \
                                      number_ones_level_2(x)-2*number_twos_level_2(x), number_neg_ones_level_2(x), number_twos_level_2(x))
bbss_filtering_2 = lambda x: p_good_2(number_ones_level_2(x)-2*number_twos_level_2(x), number_neg_ones_level_2(x), number_twos_level_2(x), \
                                      number_ones_level_3(x)-2*number_twos_level_3(x), number_neg_ones_level_3(x), number_twos_level_3(x))
bbss_filtering_3 = lambda x: p_good_2(number_ones_level_3(x)-2*number_twos_level_3(x), number_neg_ones_level_3(x), number_twos_level_3(x), \
                                      number_ones_level_4(x)-2*number_twos_level_4(x), number_neg_ones_level_4(x), number_twos_level_4(x))
bbss_filtering_4 = lambda x: p_good_2(number_ones_level_4(x)-2*number_twos_level_4(x), number_neg_ones_level_4(x), number_twos_level_4(x), \
                                      number_ones_level_5(x)-2*number_twos_level_5(x), number_neg_ones_level_5(x), number_twos_level_5(x))
bbss_filtering_5 = lambda x: p_good_2(number_ones_level_5(x)-2*number_twos_level_5(x), number_neg_ones_level_5(x), number_twos_level_5(x), \
                                      number_ones_level_6(x)-2*number_twos_level_6(x), number_neg_ones_level_6(x), number_twos_level_6(x))
bbss_filtering_6 = lambda x: p_good_2(number_ones_level_6(x)-2*number_twos_level_6(x), number_neg_ones_level_6(x), number_twos_level_6(x), \
                                      number_ones_level_7(x)-2*number_twos_level_7(x), number_neg_ones_level_7(x), number_twos_level_7(x))
bbss_filtering_7 = lambda x: p_good_2(number_ones_level_7(x)-2*number_twos_level_7(x), number_neg_ones_level_7(x), number_twos_level_7(x), \
                                      number_ones_level_8(x)-2*number_twos_level_8(x), number_neg_ones_level_8(x), number_twos_level_8(x))
bbss_filtering_8 = lambda x: p_good_2(number_ones_level_8(x)-2*number_twos_level_8(x), number_neg_ones_level_8(x), number_twos_level_8(x), \
                                      number_ones_level_9(x)-2*number_twos_level_9(x), number_neg_ones_level_9(x), number_twos_level_9(x))

it1 = lambda x: x.c1 - x.c2
it2 = lambda x: x.c2 - x.c3
it3 = lambda x: x.c3 - x.c4
it4 = lambda x: x.c4 - x.c5
it5 = lambda x: x.c5 - x.c6
it6 = lambda x: x.c6 - x.c7
it7 = lambda x: x.c7 - x.c8
it8 = lambda x: x.c8 - x.c9

def optimize_bcj_2(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 2-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_2 = collections.namedtuple('bcj_2', 'p0 l1 c1 alpha1')
    
    def bcj_2(f): 
        return wrap(f, set_bcj_2)

    def bcj_2_memory_MITM(x):
        x = set_bcj_2(*x)
        return max(domain_level_1(x)/2., x.l1)
    
    def bcj_2_time_MITM_SS(x):
        x = set_bcj_2(*x)
        return max(domain_level_1(x)/2., x.l1, -x.p0)

    def bcj_2_memory_SS(x):
        x = set_bcj_2(*x)
        return max(domain_level_1(x)/4., x.l1)

    def bcj_2_memory_dissection_tradeoff(x):
        x = set_bcj_2(*x)
        min_memory = x.l1
        min_memory_fac = min_memory / domain_level_1(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_1(x))

    def bcj_2_time_dissection_tradeoff(x):
        x = set_bcj_2(*x)
        min_memory = x.l1
        min_memory_fac = min_memory / domain_level_1(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_1(x)*timefac, x.l1, -x.p0)
    
    def bcj_2_time_repetition_subtrees(x):
        x = set_bcj_2(*x)
        min_memory = x.l1
        memfac = min_memory / domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max(domain_level_1(x)*timefac, x.l1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)

    def bcj_2_memory_repetition_subtrees(x):
        x = set_bcj_2(*x)
        min_memory = x.l1
        memfac = min_memory / domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_1(x) * memfac, x.l1)

    bcj_2_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_2(lambda x : filtering_0(x) - x.p0)},
        # sizes of the lists
        {'type' : 'ineq', 'fun' : bcj_2(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        # memory bound
        {'type' : 'ineq', 'fun' : bcj_2(lambda x: membound - bcj_memory(x))},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_2(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_2_time_MITM_SS
        bcj_memory = bcj_2_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_2_time_MITM_SS
        bcj_memory = bcj_2_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_2_time_dissection_tradeoff
        bcj_memory = bcj_2_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_2_constraints.copy()
    if repetition_subtrees:
        time = bcj_2_time_repetition_subtrees
        bcj_memory = bcj_2_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_2(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(1)] + [random.uniform(0,0.3) for _ in range(2)] + [random.uniform(0,0.3) for _ in range(1)]
    bounds = [(-1,0)]*1 + [(0,1)]*3
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_2(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bcj_3(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 3-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_3 = collections.namedtuple('bcj_3', 'p0 p1 l1 l2 c1 c2 alpha1 alpha2')
    
    def bcj_3(f): 
        return wrap(f, set_bcj_3)

    def bcj_3_memory_MITM(x):
        x = set_bcj_3(*x)
        return max(domain_level_2(x)/2., x.l2, x.l1)
    
    def bcj_3_time_MITM_SS(x):
        x = set_bcj_3(*x)
        return max(domain_level_2(x)/2., x.l2, x.l1 - x.p1, -x.p0)

    def bcj_3_memory_SS(x):
        x = set_bcj_3(*x)
        return max(domain_level_2(x)/4., x.l2, x.l1)

    def bcj_3_memory_dissection_tradeoff(x):
        x = set_bcj_3(*x)
        min_memory = max(x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_2(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_2(x))

    def bcj_3_time_dissection_tradeoff(x):
        x = set_bcj_3(*x)
        min_memory = max(x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_2(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_2(x)*timefac, x.l2, x.l1 - x.p1, -x.p0)
    
    def bcj_3_time_repetition_subtrees(x):
        x = set_bcj_3(*x)
        min_memory = max(x.l2, x.l1)
        memfac = min_memory / domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max(
                    max(domain_level_2(x)*timefac, x.l2)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0), 
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_3_memory_repetition_subtrees(x):
        x = set_bcj_3(*x)
        min_memory = max(x.l2, x.l1)
        memfac = min_memory / domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_2(x) * memfac, x.l2, x.l1)

    bcj_3_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_3(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_3(lambda x : filtering_1(x) - x.p1)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_3(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_3(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_3(lambda x : domain_level_2(x) - x.c2 - x.l2)},       
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_3(lambda x : x.alpha2 - x.alpha1/2)},
        # memory bound
        {'type' : 'ineq', 'fun' : bcj_3(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_3(lambda x : x.c1 - x.c2)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_3(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_3(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_3_time_MITM_SS
        bcj_memory = bcj_3_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_3_time_MITM_SS
        bcj_memory = bcj_3_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_3_time_dissection_tradeoff
        bcj_memory = bcj_3_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_3_constraints.copy()
    if repetition_subtrees:
        time = bcj_3_time_repetition_subtrees
        bcj_memory = bcj_3_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_3(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(2)] + [random.uniform(0,0.3) for _ in range(4)] + [random.uniform(0,0.3) for _ in range(2)]
    bounds = [(-1,0)]*2 + [(0,1)]*6
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_3(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bcj_4(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 4-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_4 = collections.namedtuple('bcj_4', 'p0 p1 p2 l1 l2 l3 c1 c2 c3 alpha1 alpha2 alpha3')
    
    def bcj_4(f): 
        return wrap(f, set_bcj_4)

    def bcj_4_memory_MITM(x):
        x = set_bcj_4(*x)
        return max(domain_level_3(x)/2., x.l3, x.l2, x.l1)
    
    def bcj_4_time_MITM_SS(x):
        x = set_bcj_4(*x)
        return max(domain_level_3(x)/2., x.l3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_4_memory_SS(x):
        x = set_bcj_4(*x)
        return max(domain_level_3(x)/4., x.l3, x.l2, x.l1)

    def bcj_4_memory_dissection_tradeoff(x):
        x = set_bcj_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_3(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_3(x))

    def bcj_4_time_dissection_tradeoff(x):
        x = set_bcj_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_3(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_3(x)*timefac, x.l3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)
    
    def bcj_4_time_repetition_subtrees(x):
        x = set_bcj_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max(
                    max(domain_level_3(x)*timefac, x.l3)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0), 
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_4_memory_repetition_subtrees(x):
        x = set_bcj_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_3(x) * memfac, x.l3, x.l2, x.l1)

    bcj_4_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_4(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_4(lambda x : filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bcj_4(lambda x : filtering_2(x) - x.p2)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_4(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bcj_4(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : domain_level_3(x) - x.c3 - x.l3)},        
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : x.alpha3 - x.alpha2/2)},
        # memory bound
        {'type' : 'ineq', 'fun' : bcj_4(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : x.c2 - x.c3)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
        {'type' : 'ineq', 'fun' : bcj_4(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_4_time_MITM_SS
        bcj_memory = bcj_4_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_4_time_MITM_SS
        bcj_memory = bcj_4_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_4_time_dissection_tradeoff
        bcj_memory = bcj_4_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_4_constraints.copy()
    if repetition_subtrees:
        time = bcj_4_time_repetition_subtrees
        bcj_memory = bcj_4_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_4(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(3)] + [random.uniform(0,0.3) for _ in range(6)] + [random.uniform(0,0.3) for _ in range(3)]
    bounds = [(-1,0)]*3 + [(0,1)]*9
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_4(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bcj_5(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 5-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_5 = collections.namedtuple('bcj_5', 'p0 p1 p2 p3 l1 l2 l3 l4 c1 c2 c3 c4 alpha1 alpha2 alpha3 alpha4')
    
    def bcj_5(f): 
        return wrap(f, set_bcj_5)

    def bcj_5_memory_MITM(x):
        x = set_bcj_5(*x)
        return max(domain_level_4(x)/2., x.l4, x.l3, x.l2, x.l1)
    
    def bcj_5_time_MITM_SS(x):
        x = set_bcj_5(*x)
        return max(domain_level_4(x)/2., x.l4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_5_memory_SS(x):
        x = set_bcj_5(*x)
        return max(domain_level_4(x)/4., x.l4, x.l3, x.l2, x.l1)

    def bcj_5_memory_dissection_tradeoff(x):
        x = set_bcj_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_4(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_4(x))

    def bcj_5_time_dissection_tradeoff(x):
        x = set_bcj_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_4(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_4(x)*timefac, x.l4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_5_time_repetition_subtrees(x):
        x = set_bcj_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_4(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(domain_level_4(x)*timefac, x.l4)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0), 
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_5_memory_repetition_subtrees(x):
        x = set_bcj_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_4(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_4(x) * memfac, x.l4, x.l3, x.l2, x.l1)
    
    bcj_5_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_5(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_5(lambda x : filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bcj_5(lambda x : filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bcj_5(lambda x : filtering_3(x) - x.p3)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_5(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bcj_5(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bcj_5(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : domain_level_4(x) - x.c4 - x.l4)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : x.alpha4 - x.alpha3/2)},
        # memory bound
        {'type' : 'ineq', 'fun' : bcj_5(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : x.c3 - x.c4)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x))},
        {'type' : 'ineq', 'fun' : bcj_5(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_5_time_MITM_SS
        bcj_memory = bcj_5_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_5_time_MITM_SS
        bcj_memory = bcj_5_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_5_time_dissection_tradeoff
        bcj_memory = bcj_5_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0

    mycons = bcj_5_constraints.copy()
    if repetition_subtrees:
        time = bcj_5_time_repetition_subtrees
        bcj_memory = bcj_5_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_5(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(4)] + [random.uniform(0,0.3) for _ in range(8)] + [random.uniform(0,0.3) for _ in range(4)]
    bounds = [(-1,0)]*4 + [(0,1)]*12
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_5(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))
    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0
    

def optimize_bcj_6(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 6-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_6 = collections.namedtuple('bcj_6', 'p0 p1 p2 p3 p4 l1 l2 l3 l4 l5 c1 c2 c3 c4 c5 alpha1 alpha2 alpha3 alpha4 alpha5')
    
    def bcj_6(f): 
        return wrap(f, set_bcj_6)

    def bcj_6_memory_MITM(x):
        x = set_bcj_6(*x)
        return max(domain_level_5(x)/2., x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bcj_6_time_MITM_SS(x):
        x = set_bcj_6(*x)
        return max(domain_level_5(x)/2., x.l5, x.l4 -x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_6_memory_SS(x):
        x = set_bcj_6(*x)
        return max(domain_level_5(x)/4., x.l5, x.l4, x.l3, x.l2, x.l1)

    def bcj_6_memory_dissection_tradeoff(x):
        x = set_bcj_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_5(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_5(x))

    def bcj_6_time_dissection_tradeoff(x):
        x = set_bcj_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_5(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_5(x)*timefac, x.l5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)
    
    def bcj_6_time_repetition_subtrees(x):
        x = set_bcj_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_5(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(domain_level_5(x)*timefac, x.l5)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0), 
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_6_memory_repetition_subtrees(x):
        x = set_bcj_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_5(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_5(x) * memfac, x.l5, x.l4, x.l3, x.l2, x.l1)

    bcj_6_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_6(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_6(lambda x : filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bcj_6(lambda x : filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bcj_6(lambda x : filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bcj_6(lambda x : filtering_4(x) - x.p4)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_6(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bcj_6(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bcj_6(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bcj_6(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : domain_level_5(x) - x.c5 - x.l5)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.alpha5 - x.alpha4/2)},
        # memory bound
        {'type' : 'ineq', 'fun' : bcj_6(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : x.c4 - x.c5)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x))},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x))},
        {'type' : 'ineq', 'fun' : bcj_6(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_6_time_MITM_SS
        bcj_memory = bcj_6_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_6_time_MITM_SS
        bcj_memory = bcj_6_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_6_time_dissection_tradeoff
        bcj_memory = bcj_6_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_6_constraints.copy()
    if repetition_subtrees:
        time = bcj_6_time_repetition_subtrees
        bcj_memory = bcj_6_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_6(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(5)] + [random.uniform(0,0.3) for _ in range(10)] + [random.uniform(0,0.3) for _ in range(5)]
    bounds = [(-1,0)]*5 + [(0,1)]*15
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_6(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0
    

def optimize_bcj_7(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 7-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_7 = collections.namedtuple('bcj_7', 'p0 p1 p2 p3 p4 p5 l1 l2 l3 l4 l5 l6 c1 c2 c3 c4 c5 c6 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6')
    
    def bcj_7(f): 
        return wrap(f, set_bcj_7)

    def bcj_7_memory_MITM(x):
        x = set_bcj_7(*x)
        return max(domain_level_6(x)/2., x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bcj_7_time_MITM_SS(x):
        x = set_bcj_7(*x)
        return max(domain_level_6(x)/2., x.l6, x.l5 - x.p5, x.l4 -x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_7_memory_SS(x):
        x = set_bcj_7(*x)
        return max(domain_level_6(x)/4., x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bcj_7_memory_dissection_tradeoff(x):
        x = set_bcj_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_6(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_6(x))

    def bcj_7_time_dissection_tradeoff(x):
        x = set_bcj_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_6(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_6(x)*timefac, x.l6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)
    
    def bcj_7_time_repetition_subtrees(x):
        x = set_bcj_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_6(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(domain_level_6(x)*timefac, x.l6)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0), 
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_7_memory_repetition_subtrees(x):
        x = set_bcj_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_6(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_6(x) * memfac, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    bcj_7_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_7(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : filtering_5(x) - x.p5)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_7(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bcj_7(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : domain_level_6(x) - x.c6 - x.l6)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.alpha6 - x.alpha5/2)},
        # memory bound
        {'type' : 'ineq', 'fun' : bcj_7(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : x.c5 - x.c6)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x))},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x))},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x))},
        {'type' : 'ineq', 'fun' : bcj_7(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_7_time_MITM_SS
        bcj_memory = bcj_7_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_7_time_MITM_SS
        bcj_memory = bcj_7_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_7_time_dissection_tradeoff
        bcj_memory = bcj_7_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_7_constraints.copy()
    if repetition_subtrees:
        time = bcj_7_time_repetition_subtrees
        bcj_memory = bcj_7_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_7(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(6)] + [random.uniform(0,0.3) for _ in range(12)] + [random.uniform(0,0.3) for _ in range(6)]
    bounds = [(-1,0)]*6 + [(0,1)]*18
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_7(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bcj_8(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original BCJ algorithm for subset-sum, using {0,-1,1}-representations with a 8-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_8 = collections.namedtuple('bcj_8', 'p0 p1 p2 p3 p4 p5 p6 l1 l2 l3 l4 l5 l6 l7 c1 c2 c3 c4 c5 c6 c7 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6 alpha7')
    
    def bcj_8(f): 
        return wrap(f, set_bcj_8)

    def bcj_8_memory_MITM(x):
        x = set_bcj_8(*x)
        return max(domain_level_7(x)/2., x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bcj_8_time_MITM_SS(x):
        x = set_bcj_8(*x)
        return max(domain_level_7(x)/2., x.l7, x.l6 - x.p6, x.l5 - x.p5, x.l4 -x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_8_memory_SS(x):
        x = set_bcj_8(*x)
        return max(domain_level_7(x)/4., x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bcj_8_memory_dissection_tradeoff(x):
        x = set_bcj_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_7(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_7(x))

    def bcj_8_time_dissection_tradeoff(x):
        x = set_bcj_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_7(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_7(x)*timefac, x.l7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_8_time_repetition_subtrees(x):
        x = set_bcj_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_7(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(domain_level_7(x)*timefac, x.l7)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x), 0), 
                    max(x.l7, x.l6 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0),
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_8_memory_repetition_subtrees(x):
        x = set_bcj_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_7(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_7(x) * memfac, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    bcj_8_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_8(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : filtering_5(x) - x.p5)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : filtering_6(x) - x.p6)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_8(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'eq',   'fun' : bcj_8(lambda x : 2*x.l7 - (x.c6 - x.c7) + x.p6 - x.l6)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : domain_level_6(x) - x.c6 - x.l6)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : domain_level_7(x) - x.c7 - x.l7)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.alpha6 - x.alpha5/2)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.alpha7 - x.alpha6/2)},
        # memory bound
        {'type' : 'ineq', 'fun': bcj_8(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.c5 - x.c6)},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : x.c6 - x.c7)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x))},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x))},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x))},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x))},
        {'type' : 'ineq', 'fun' : bcj_8(lambda x : 1 - number_ones_level_7(x) - number_neg_ones_level_7(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_8_time_MITM_SS
        bcj_memory = bcj_8_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_8_time_MITM_SS
        bcj_memory = bcj_8_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_8_time_dissection_tradeoff
        bcj_memory = bcj_8_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_8_constraints.copy()
    if repetition_subtrees:
        time = bcj_8_time_repetition_subtrees
        bcj_memory = bcj_8_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_8(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(7)] + [random.uniform(0,0.3) for _ in range(14)] + [random.uniform(0,0.3) for _ in range(7)]
    bounds = [(-1,0)]*7 + [(0,1)]*21
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_8(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bcj_9(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 9-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_9 = collections.namedtuple('bcj_9', 'p0 p1 p2 p3 p4 p5 p6 p7 l1 l2 l3 l4 l5 l6 l7 l8 c1 c2 c3 c4 c5 c6 c7 c8 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6 alpha7 alpha8')
    
    def bcj_9(f): 
        return wrap(f, set_bcj_9)

    def bcj_9_memory_MITM(x):
        x = set_bcj_9(*x)
        return max(domain_level_8(x)/2., x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bcj_9_time_MITM_SS(x):
        x = set_bcj_9(*x)
        return max(domain_level_8(x)/2., x.l8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 -x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_9_memory_SS(x):
        x = set_bcj_9(*x)
        return max(domain_level_8(x)/4., x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bcj_9_memory_dissection_tradeoff(x):
        x = set_bcj_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_8(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_8(x))

    def bcj_9_time_dissection_tradeoff(x):
        x = set_bcj_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_8(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_8(x)*timefac, x.l8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_9_time_repetition_subtrees(x):
        x = set_bcj_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_8(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(domain_level_8(x)*timefac, x.l8)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x) + 127*it7(x), 0), 
                    max(x.l8, x.l7 - x.p7)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x), 0),
                    max(x.l7, x.l6 - x.p6)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0),
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_9_memory_repetition_subtrees(x):
        x = set_bcj_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_8(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_8(x) * memfac, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    bcj_9_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_5(x) - x.p5)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_6(x) - x.p6)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : filtering_7(x) - x.p7)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l7 - (x.c6 - x.c7) + x.p6 - x.l6)},
        {'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l8 - (x.c7 - x.c8) + x.p7 - x.l7)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_6(x) - x.c6 - x.l6)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_7(x) - x.c7 - x.l7)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : domain_level_8(x) - x.c8 - x.l8)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.alpha6 - x.alpha5/2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.alpha7 - x.alpha6/2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.alpha8 - x.alpha7/2)},
        # memory bound
        {'type' : 'ineq', 'fun': bcj_9(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.c5 - x.c6)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.c6 - x.c7)},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : x.c7 - x.c8)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x))},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x))},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x))},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x))},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_7(x) - number_neg_ones_level_7(x))},
        {'type' : 'ineq', 'fun' : bcj_9(lambda x : 1 - number_ones_level_8(x) - number_neg_ones_level_8(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_9_time_MITM_SS
        bcj_memory = bcj_9_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_9_time_MITM_SS
        bcj_memory = bcj_9_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_9_time_dissection_tradeoff
        bcj_memory = bcj_9_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_9_constraints.copy()
    if repetition_subtrees:
        time = bcj_9_time_repetition_subtrees
        bcj_memory = bcj_9_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_9(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(8)] + [random.uniform(0,0.3) for _ in range(16)] + [random.uniform(0,0.3) for _ in range(8)]
    bounds = [(-1,0)]*8 + [(0,1)]*24
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_9(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bcj_10(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BCJ-representations with a 10-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    """
    set_bcj_10 = collections.namedtuple('bcj_10', 'p0 p1 p2 p3 p4 p5 p6 p7 p8 l1 l2 l3 l4 l5 l6 l7 l8 l9 c1 c2 c3 c4 c5 c6 c7 c8 c9 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6 alpha7 alpha8 alpha9')
    
    def bcj_10(f): 
        return wrap(f, set_bcj_10)

    def bcj_10_memory_MITM(x):
        x = set_bcj_10(*x)
        return max(domain_level_9(x)/2., x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bcj_10_time_MITM_SS(x):
        x = set_bcj_10(*x)
        return max(domain_level_9(x)/2., x.l9, x.l8 - x.p8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 -x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_10_memory_SS(x):
        x = set_bcj_10(*x)
        return max(domain_level_9(x)/4., x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bcj_10_memory_dissection_tradeoff(x):
        x = set_bcj_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_9(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*domain_level_9(x))

    def bcj_10_time_dissection_tradeoff(x):
        x = set_bcj_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / domain_level_9(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(domain_level_9(x)*timefac, x.l9, x.l8 - x.p8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bcj_10_time_repetition_subtrees(x):
        x = set_bcj_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_9(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(domain_level_9(x)*timefac, x.l9)      - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x) + 127*it7(x) + 255*it8(x), 0), 
                    max(x.l9, x.l8 - x.p8)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x) + 127*it7(x), 0),
                    max(x.l8, x.l7 - x.p7)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x), 0),
                    max(x.l7, x.l6 - x.p6)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0),
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bcj_10_memory_repetition_subtrees(x):
        x = set_bcj_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / domain_level_9(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(domain_level_9(x) * memfac, x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    bcj_10_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_5(x) - x.p5)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_6(x) - x.p6)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_7(x) - x.p7)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : filtering_8(x) - x.p8)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l7 - (x.c6 - x.c7) + x.p6 - x.l6)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l8 - (x.c7 - x.c8) + x.p7 - x.l7)},
        {'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l9 - (x.c8 - x.c9) + x.p8 - x.l8)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_6(x) - x.c6 - x.l6)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_7(x) - x.c7 - x.l7)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_8(x) - x.c8 - x.l8)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : domain_level_9(x) - x.c9 - x.l9)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha6 - x.alpha5/2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha7 - x.alpha6/2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha8 - x.alpha7/2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.alpha9 - x.alpha8/2)},
        # memory bound
        {'type' : 'ineq', 'fun': bcj_10(lambda x: membound - bcj_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c5 - x.c6)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c6 - x.c7)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c7 - x.c8)},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : x.c8 - x.c9)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_7(x) - number_neg_ones_level_7(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_8(x) - number_neg_ones_level_8(x))},
        {'type' : 'ineq', 'fun' : bcj_10(lambda x : 1 - number_ones_level_9(x) - number_neg_ones_level_9(x))},
    ]

    if base_list_alg == 'MITM':
        time = bcj_10_time_MITM_SS
        bcj_memory = bcj_10_memory_MITM
    elif base_list_alg == 'SS':
        time = bcj_10_time_MITM_SS
        bcj_memory = bcj_10_memory_SS
    elif base_list_alg == 'Dissection':
        time = bcj_10_time_dissection_tradeoff
        bcj_memory = bcj_10_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    mycons = bcj_10_constraints.copy()
    if repetition_subtrees:
        time = bcj_10_time_repetition_subtrees
        bcj_memory = bcj_10_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bcj_10(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [-random.uniform(0,1) for _ in range(9)] + [random.uniform(0,0.3) for _ in range(18)] + [random.uniform(0,0.3) for _ in range(9)]
    bounds = [(-1,0)]*9 + [(0,1)]*27
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bcj_10(*result.x)

    if verb:
        print("memory ", bcj_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0
    

###############################################################################
#################################BBSS##########################################
###############################################################################
# Beware: level at the bottom is level 0!
# Variables used:
# c_i: total bit constraint at level i
# l_i: list sizes after filtering
# p_i: probability of filtering at level i
# alpha_i: number of negative ones at level i
# gamma_i: number of twos at level i
    
def optimize_bbss_2(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 2-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_2 = collections.namedtuple('bbss_3', 'p0 l1 c1 alpha1 gamma1')
    
    def bbss_2(f): 
        return wrap(f, set_bbss_2)

    def bbss_2_memory_MITM(x):
        x = set_bbss_2(*x)
        return max(bbss_domain_level_1(x)/2., x.l1)
    
    def bbss_2_time_MITM_SS(x):
        x = set_bbss_2(*x)
        return max(bbss_domain_level_1(x)/2., x.l1, -x.p0)

    def bbss_2_memory_SS(x):
        x = set_bbss_2(*x)
        return max(bbss_domain_level_1(x)/4., x.l1)

    def bbss_2_memory_dissection_tradeoff(x):
        x = set_bbss_2(*x)
        min_memory = x.l1
        min_memory_fac = min_memory / bbss_domain_level_1(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_1(x))

    def bbss_2_time_dissection_tradeoff(x):
        x = set_bbss_2(*x)
        min_memory = x.l1
        min_memory_fac = min_memory / bbss_domain_level_1(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_1(x)*timefac, x.l1, -x.p0)
    
    def bbss_2_time_repetition_subtrees(x):
        x = set_bbss_2(*x)
        min_memory = x.l1
        memfac = min_memory / bbss_domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max(bbss_domain_level_1(x)*timefac, x.l1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)

    def bbss_2_memory_repetition_subtrees(x):
        x = set_bbss_2(*x)
        min_memory = x.l1
        memfac = min_memory / bbss_domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_1(x) * memfac, x.l1)

    bbss_2_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_2(lambda x : bbss_filtering_0(x) - x.p0)},
        # sizes of the lists
        {'type' : 'ineq', 'fun' : bbss_2(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_2(lambda x : x.alpha1 - 2*x.gamma1)},
        # memory bound
        {'type' : 'ineq', 'fun' : bbss_2(lambda x: membound - bbss_memory(x))},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_2(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
    ]

    mycons = bbss_2_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_2(lambda x : -x.gamma1)})

    if base_list_alg == 'MITM':
        time = bbss_2_time_MITM_SS
        bbss_memory = bbss_2_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_2_time_MITM_SS
        bbss_memory = bbss_2_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_2_time_dissection_tradeoff
        bbss_memory = bbss_2_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_2_time_repetition_subtrees
        bbss_memory = bbss_2_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_2(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})
    
    start = [-random.uniform(0,1) for _ in range(1)] + [random.uniform(0,0.3) for _ in range(2)] + [random.uniform(0,0.3)]*1 + [0.000]*1
    bounds = [(-1,0)]*1 + [(0,1)]*4
    if use_bcj:
        bounds = [(-1,0)]*1 + [(0,1)]*3+ [(0,0)]*1

    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_2(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0    
   

def optimize_bbss_3(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 3-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_3 = collections.namedtuple('bbss_3', 'p0 p1 l1 l2 c1 c2 alpha1 alpha2 gamma1 gamma2')
    
    def bbss_3(f): 
        return wrap(f, set_bbss_3)

    def bbss_3_memory_MITM(x):
        x = set_bbss_3(*x)
        return max(bbss_domain_level_2(x)/2., x.l2, x.l1)
    
    def bbss_3_time_MITM_SS(x):
        x = set_bbss_3(*x)
        return max(bbss_domain_level_2(x)/2., x.l2, x.l1 - x.p1, -x.p0)

    def bbss_3_memory_SS(x):
        x = set_bbss_3(*x)
        return max(bbss_domain_level_2(x)/4., x.l2, x.l1)

    def bbss_3_memory_dissection_tradeoff(x):
        x = set_bbss_3(*x)
        min_memory = max(x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_2(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_2(x))

    def bbss_3_time_dissection_tradeoff(x):
        x = set_bbss_3(*x)
        min_memory = max(x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_2(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_2(x)*timefac, x.l2, x.l1 - x.p1, -x.p0)
    
    def bbss_3_time_repetition_subtrees(x):
        x = set_bbss_3(*x)
        min_memory = max(x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max(
                    max(bbss_domain_level_2(x)*timefac, x.l2) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0), 
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_3_memory_repetition_subtrees(x):
        x = set_bbss_3(*x)
        min_memory = max(x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_2(x) * memfac, x.l2, x.l1)

    bbss_3_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_3(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_3(lambda x : bbss_filtering_1(x) - x.p1)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_3(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : x.alpha2 - 2*x.gamma2)},
        # memory bound
        {'type' : 'ineq', 'fun': bbss_3(lambda x: membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : x.c1 - x.c2)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_3(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
    ]

    mycons = bbss_3_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_3(lambda x : -x.gamma1)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_3(lambda x : -x.gamma2)})

    if base_list_alg == 'MITM':
        time = bbss_3_time_MITM_SS
        bbss_memory = bbss_3_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_3_time_MITM_SS
        bbss_memory = bbss_3_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_3_time_dissection_tradeoff
        bbss_memory = bbss_3_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_3_time_repetition_subtrees
        bbss_memory = bbss_3_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_3(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})
    
    start = [-random.uniform(0,1) for _ in range(2)] + [random.uniform(0,0.3) for _ in range(4)] + [random.uniform(0,0.3)]*2 + [0.000]*2
    bounds = [(-1,0)]*2 + [(0,1)]*8
    if use_bcj:
        bounds = [(-1,0)]*2 + [(0,1)]*6+ [(0,0)]*2

    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_3(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0    
    
    
def optimize_bbss_4(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 4-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_4 = collections.namedtuple('bbss_4', 'p0 p1 p2 l1 l2 l3 c1 c2 c3 alpha1 alpha2 alpha3 gamma1 gamma2 gamma3')
    
    def bbss_4(f): 
        return wrap(f, set_bbss_4)

    def bbss_4_memory_MITM(x):
        x = set_bbss_4(*x)
        return max(bbss_domain_level_3(x)/2., x.l3, x.l2, x.l1)
    
    def bbss_4_time_MITM_SS(x):
        x = set_bbss_4(*x)
        return max(bbss_domain_level_3(x)/2., x.l3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_4_memory_SS(x):
        x = set_bbss_4(*x)
        return max(bbss_domain_level_3(x)/4., x.l3, x.l2, x.l1)

    def bbss_4_memory_dissection_tradeoff(x):
        x = set_bbss_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_3(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_3(x))

    def bbss_4_time_dissection_tradeoff(x):
        x = set_bbss_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_3(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_3(x)*timefac, x.l3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)
    
    def bbss_4_time_repetition_subtrees(x):
        x = set_bbss_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max(
                    max(bbss_domain_level_3(x)*timefac, x.l3) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0), 
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_4_memory_repetition_subtrees(x):
        x = set_bbss_4(*x)
        min_memory = max(x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_3(x) * memfac, x.l3, x.l2, x.l1)

    bbss_4_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_4(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_4(lambda x : bbss_filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bbss_4(lambda x : bbss_filtering_2(x) - x.p2)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_4(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bbss_4(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : bbss_domain_level_3(x) - x.c3 - x.l3)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : x.alpha2 - 2*x.gamma2)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : x.alpha3 - 2*x.gamma3)},
        # memory bound
        {'type' : 'ineq', 'fun': bbss_4(lambda x: membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : x.c2 - x.c3)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
        {'type' : 'ineq', 'fun' : bbss_4(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x) + number_twos_level_3(x))},
    ]

    mycons = bbss_4_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_4(lambda x : -x.gamma1)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_4(lambda x : -x.gamma2)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_4(lambda x : -x.gamma3)})

    if base_list_alg == 'MITM':
        time = bbss_4_time_MITM_SS
        bbss_memory = bbss_4_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_4_time_MITM_SS
        bbss_memory = bbss_4_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_4_time_dissection_tradeoff
        bbss_memory = bbss_4_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_4_time_repetition_subtrees
        bbss_memory = bbss_4_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_4(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})
    
    start = [-random.uniform(0,1) for _ in range(3)] + [random.uniform(0,0.3) for _ in range(6)] + [random.uniform(0,0.3)]*3 + [0.000]*3
    bounds = [(-1,0)]*3 + [(0,1)]*12
    if use_bcj:
        bounds = [(-1,0)]*3 + [(0,1)]*9 + [(0,0)]*3

    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_4(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bbss_5(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 5-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees shoud be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations shoud be used
    """
    set_bbss_5 = collections.namedtuple('bbss_5', 'p0 p1 p2 p3 l1 l2 l3 l4 c1 c2 c3 c4 alpha1 alpha2 alpha3 alpha4 gamma1 gamma2 gamma3 gamma4')
    
    def bbss_5(f): 
        return wrap(f, set_bbss_5)

    def bbss_5_memory_MITM(x):
        x = set_bbss_5(*x)
        return max(bbss_domain_level_4(x)/2., x.l4, x.l3, x.l2, x.l1)
    
    def bbss_5_time_MITM_SS(x):
        x = set_bbss_5(*x)
        return max(bbss_domain_level_4(x)/2., x.l4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_5_memory_SS(x):
        x = set_bbss_5(*x)
        return max(bbss_domain_level_4(x)/4., x.l4, x.l3, x.l2, x.l1)

    def bbss_5_memory_dissection_tradeoff(x):
        x = set_bbss_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_4(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_4(x))

    def bbss_5_time_dissection_tradeoff(x):
        x = set_bbss_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_4(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_4(x)*timefac, x.l4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)
    
    def bbss_5_time_repetition_subtrees(x):
        x = set_bbss_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_4(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(bbss_domain_level_4(x)*timefac, x.l4) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0), 
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_5_memory_repetition_subtrees(x):
        x = set_bbss_5(*x)
        min_memory = max(x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_4(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_4(x) * memfac, x.l4, x.l3, x.l2, x.l1)

    bbss_5_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_5(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_5(lambda x : bbss_filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bbss_5(lambda x : bbss_filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bbss_5(lambda x : bbss_filtering_3(x) - x.p3)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_5(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bbss_5(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bbss_5(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : bbss_domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : bbss_domain_level_4(x) - x.c4 - x.l4)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.alpha2 - 2*x.gamma2)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.alpha3 - 2*x.gamma3)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.alpha4 - 2*x.gamma4)},
        # memory bound
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : x.c3 - x.c4)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x) + number_twos_level_3(x))},
        {'type' : 'ineq', 'fun' : bbss_5(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x) + number_twos_level_4(x))},
    ]
    mycons = bbss_5_constraints.copy()
    if use_bcj:
        mycons.append({'type' : 'ineq', 'fun' : bbss_5(lambda x : -x.gamma1)})
        mycons.append({'type' : 'ineq', 'fun' : bbss_5(lambda x : -x.gamma2)})
        mycons.append({'type' : 'ineq', 'fun' : bbss_5(lambda x : -x.gamma3)})
        mycons.append({'type' : 'ineq', 'fun' : bbss_5(lambda x : -x.gamma4)})

    if base_list_alg == 'MITM':
        time = bbss_5_time_MITM_SS
        bbss_memory = bbss_5_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_5_time_MITM_SS
        bbss_memory = bbss_5_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_5_time_dissection_tradeoff
        bbss_memory = bbss_5_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_5_time_repetition_subtrees
        bbss_memory = bbss_5_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_5(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})
    
    start = [-0.2]*4 + [random.uniform(0.18,0.22) for _ in range(8)] + [0.03]*4 + [0.000]*4
    bounds = [(-1,0)]*4 + [(0,1)]*16
    if use_bcj:
        bounds = [(-1,0)]*4 + [(0,1)]*12 + [(0,0)]*4

    result = opt.minimize(time, start, bounds=bounds, tol=1e-10,constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_5(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bbss_6(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 6-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_6 = collections.namedtuple('bbss_6', 'p0 p1 p2 p3 p4 l1 l2 l3 l4 l5 c1 c2 c3 c4 c5 alpha1 alpha2 alpha3 alpha4 alpha5 gamma1 gamma2 gamma3 gamma4 gamma5')
    
    def bbss_6(f): 
        return wrap(f, set_bbss_6)

    def bbss_6_memory_MITM(x):
        x = set_bbss_6(*x)
        return max(bbss_domain_level_5(x)/2., x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bbss_6_time_MITM_SS(x):
        x = set_bbss_6(*x)
        return max(bbss_domain_level_5(x)/2., x.l5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_6_memory_SS(x):
        x = set_bbss_6(*x)
        return max(bbss_domain_level_5(x)/4., x.l5, x.l4, x.l3, x.l2, x.l1)

    def bbss_6_memory_dissection_tradeoff(x):
        x = set_bbss_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_5(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_5(x))

    def bbss_6_time_dissection_tradeoff(x):
        x = set_bbss_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_5(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_5(x)*timefac, x.l5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_6_time_repetition_subtrees(x):
        x = set_bbss_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_5(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(bbss_domain_level_5(x)*timefac, x.l5) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0), 
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_6_memory_repetition_subtrees(x):
        x = set_bbss_6(*x)
        min_memory = max(x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_5(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_5(x) * memfac, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    bbss_6_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_6(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_6(lambda x : bbss_filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bbss_6(lambda x : bbss_filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bbss_6(lambda x : bbss_filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bbss_6(lambda x : bbss_filtering_4(x) - x.p4)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_6(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bbss_6(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bbss_6(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bbss_6(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : bbss_domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : bbss_domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : bbss_domain_level_5(x) - x.c5 - x.l5)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha2 - 2*x.gamma2)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha3 - 2*x.gamma3)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha4 - 2*x.gamma4)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.alpha5 - 2*x.gamma5)},
        # memory bound
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : x.c4 - x.c5)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x) + number_twos_level_3(x))},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x) + number_twos_level_4(x))},
        {'type' : 'ineq', 'fun' : bbss_6(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x) + number_twos_level_5(x))},
    ]

    mycons = bbss_6_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_6(lambda x : -x.gamma1)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_6(lambda x : -x.gamma2)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_6(lambda x : -x.gamma3)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_6(lambda x : -x.gamma4)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_6(lambda x : -x.gamma5)})

    if base_list_alg == 'MITM':
        time = bbss_6_time_MITM_SS
        bbss_memory = bbss_6_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_6_time_MITM_SS
        bbss_memory = bbss_6_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_6_time_dissection_tradeoff
        bbss_memory = bbss_6_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_6_time_repetition_subtrees
        bbss_memory = bbss_6_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_6(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [(-0.2)]*5 + [random.uniform(0.18,0.22) for _ in range(10)] + [(0.03)]*5 + [(0.000)]*5
    bounds = [(-1,0)]*5 + [(0,1)]*20
    if use_bcj:
        bounds = [(-1,0)]*5 + [(0,1)]*15 + [(0,0)]*5

    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_6(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bbss_7(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 7-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_7 = collections.namedtuple('bbss_7', 'p0 p1 p2 p3 p4 p5 l1 l2 l3 l4 l5 l6 c1 c2 c3 c4 c5 c6 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6 gamma1 gamma2 gamma3 gamma4 gamma5 gamma6')
    
    def bbss_7(f): 
        return wrap(f, set_bbss_7)

    def bbss_7_memory_MITM(x):
        x = set_bbss_7(*x)
        return max(bbss_domain_level_6(x)/2., x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bbss_7_time_MITM_SS(x):
        x = set_bbss_7(*x)
        return max(bbss_domain_level_6(x)/2., x.l6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_7_memory_SS(x):
        x = set_bbss_7(*x)
        return max(bbss_domain_level_6(x)/4., x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bbss_7_memory_dissection_tradeoff(x):
        x = set_bbss_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_6(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_6(x))

    def bbss_7_time_dissection_tradeoff(x):
        x = set_bbss_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_6(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_6(x)*timefac, x.l6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_7_time_repetition_subtrees(x):
        x = set_bbss_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_6(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(bbss_domain_level_6(x)*timefac, x.l6) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0), 
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_7_memory_repetition_subtrees(x):
        x = set_bbss_7(*x)
        min_memory = max(x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_6(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_6(x) * memfac, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    bbss_7_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_7(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : bbss_filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : bbss_filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : bbss_filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : bbss_filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : bbss_filtering_5(x) - x.p5)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_7(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bbss_7(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : bbss_domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : bbss_domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : bbss_domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : bbss_domain_level_6(x) - x.c6 - x.l6)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha6 - x.alpha5/2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha2 - 2*x.gamma2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha3 - 2*x.gamma3)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha4 - 2*x.gamma4)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha5 - 2*x.gamma5)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.alpha6 - 2*x.gamma6)},
        # memory bound
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : x.c5 - x.c6)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x) + number_twos_level_3(x))},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x) + number_twos_level_4(x))},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x) + number_twos_level_5(x))},
        {'type' : 'ineq', 'fun' : bbss_7(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x) + number_twos_level_6(x))},
    ]

    mycons = bbss_7_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_7(lambda x : -x.gamma1)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_7(lambda x : -x.gamma2)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_7(lambda x : -x.gamma3)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_7(lambda x : -x.gamma4)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_7(lambda x : -x.gamma5)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_7(lambda x : -x.gamma6)})

    if base_list_alg == 'MITM':
        time = bbss_7_time_MITM_SS
        bbss_memory = bbss_7_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_7_time_MITM_SS
        bbss_memory = bbss_7_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_7_time_dissection_tradeoff
        bbss_memory = bbss_7_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_7_time_repetition_subtrees
        bbss_memory = bbss_7_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_7(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})

    start = [(-0.2)]*6 + [random.uniform(0.18,0.22) for _ in range(12)] + [(0.03)]*6 + [(0.000)]*6
    bounds = [(-1,0)]*6 + [(0,1)]*24
    if use_bcj:
        bounds = [(-1,0)]*6 + [(0,1)]*18 + [(0,0)]*6

    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_7(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bbss_8(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 8-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_8 = collections.namedtuple('bbss_8', 'p0 p1 p2 p3 p4 p5 p6 l1 l2 l3 l4 l5 l6 l7 c1 c2 c3 c4 c5 c6 c7 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6 alpha7 gamma1 gamma2 gamma3 gamma4 gamma5 gamma6 gamma7')
    
    def bbss_8(f): 
        return wrap(f, set_bbss_8)

    def bbss_8_memory_MITM(x):
        x = set_bbss_8(*x)
        return max(bbss_domain_level_7(x)/2., x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bbss_8_time_MITM_SS(x):
        x = set_bbss_8(*x)
        return max(bbss_domain_level_7(x)/2., x.l7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_8_memory_SS(x):
        x = set_bbss_8(*x)
        return max(bbss_domain_level_7(x)/4., x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bbss_8_memory_dissection_tradeoff(x):
        x = set_bbss_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_7(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_7(x))

    def bbss_8_time_dissection_tradeoff(x):
        x = set_bbss_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_7(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_7(x)*timefac, x.l7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_8_time_repetition_subtrees(x):
        x = set_bbss_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_7(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(bbss_domain_level_7(x)*timefac, x.l7) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x), 0), 
                    max(x.l7, x.l6 - x.p6)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0),
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_8_memory_repetition_subtrees(x):
        x = set_bbss_8(*x)
        min_memory = max(x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_7(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_7(x) * memfac, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    bbss_8_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_8(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : bbss_filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : bbss_filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : bbss_filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : bbss_filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : bbss_filtering_5(x) - x.p5)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : bbss_filtering_6(x) - x.p6)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_8(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'eq',   'fun' : bbss_8(lambda x : 2*x.l7 - (x.c6 - x.c7) + x.p6 - x.l6)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : bbss_domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : bbss_domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : bbss_domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : bbss_domain_level_6(x) - x.c6 - x.l6)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : bbss_domain_level_7(x) - x.c7 - x.l7)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha6 - x.alpha5/2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha7 - x.alpha6/2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha2 - 2*x.gamma2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha3 - 2*x.gamma3)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha4 - 2*x.gamma4)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha5 - 2*x.gamma5)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha6 - 2*x.gamma6)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.alpha7 - 2*x.gamma7)},
        # memory bound
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.c5 - x.c6)},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : x.c6 - x.c7)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x) + number_twos_level_3(x))},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x) + number_twos_level_4(x))},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x) + number_twos_level_5(x))},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x) + number_twos_level_6(x))},
        {'type' : 'ineq', 'fun' : bbss_8(lambda x : 1 - number_ones_level_7(x) - number_neg_ones_level_7(x) + number_twos_level_7(x))},
    ]

    mycons = bbss_8_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_8(lambda x : -x.gamma1)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_8(lambda x : -x.gamma2)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_8(lambda x : -x.gamma3)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_8(lambda x : -x.gamma4)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_8(lambda x : -x.gamma5)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_8(lambda x : -x.gamma6)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_8(lambda x : -x.gamma7)})

    if base_list_alg == 'MITM':
        time = bbss_8_time_MITM_SS
        bbss_memory = bbss_8_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_8_time_MITM_SS
        bbss_memory = bbss_8_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_8_time_dissection_tradeoff
        bbss_memory = bbss_8_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_8_time_repetition_subtrees
        bbss_memory = bbss_8_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_8(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})
        
    start = [(-0.2)]*7 + [random.uniform(0.18,0.22) for _ in range(14)] + [(0.03)]*7 + [(0.000)]*7
    bounds = [(-1,0)]*7 + [(0,1)]*28
    if use_bcj:
        bounds = [(-1,0)]*7 + [(0,1)]*21 + [(0,0)]*7
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_8(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0
    

def optimize_bbss_9(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 9-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_9 = collections.namedtuple('bbss_9', 'p0 p1 p2 p3 p4 p5 p6 p7 l1 l2 l3 l4 l5 l6 l7 l8 c1 c2 c3 c4 c5 c6 c7 c8 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6 alpha7 alpha8 gamma1 gamma2 gamma3 gamma4 gamma5 gamma6 gamma7 gamma8')
    
    def bbss_9(f): 
        return wrap(f, set_bbss_9)

    def bbss_9_memory_MITM(x):
        x = set_bbss_9(*x)
        return max(bbss_domain_level_8(x)/2., x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bbss_9_time_MITM_SS(x):
        x = set_bbss_9(*x)
        return max(bbss_domain_level_8(x)/2., x.l8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_9_memory_SS(x):
        x = set_bbss_9(*x)
        return max(bbss_domain_level_8(x)/4., x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bbss_9_memory_dissection_tradeoff(x):
        x = set_bbss_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_8(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_8(x))

    def bbss_9_time_dissection_tradeoff(x):
        x = set_bbss_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_8(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_8(x)*timefac, x.l8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_9_time_repetition_subtrees(x):
        x = set_bbss_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_8(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(bbss_domain_level_8(x)*timefac, x.l8) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x) + 127*it7(x), 0), 
                    max(x.l8, x.l7 - x.p7)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x), 0),
                    max(x.l7, x.l6 - x.p6)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0),
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_9_memory_repetition_subtrees(x):
        x = set_bbss_9(*x)
        min_memory = max(x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_8(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_8(x) * memfac, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    bbss_9_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_5(x) - x.p5)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_6(x) - x.p6)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : bbss_filtering_7(x) - x.p7)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l7 - (x.c6 - x.c7) + x.p6 - x.l6)},
        {'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l8 - (x.c7 - x.c8) + x.p7 - x.l7)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_6(x) - x.c6 - x.l6)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_7(x) - x.c7 - x.l7)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : bbss_domain_level_8(x) - x.c8 - x.l8)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha6 - x.alpha5/2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha7 - x.alpha6/2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha8 - x.alpha7/2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha2 - 2*x.gamma2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha3 - 2*x.gamma3)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha4 - 2*x.gamma4)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha5 - 2*x.gamma5)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha6 - 2*x.gamma6)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha7 - 2*x.gamma7)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.alpha8 - 2*x.gamma8)},
        # memory bound
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.c5 - x.c6)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.c6 - x.c7)},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : x.c7 - x.c8)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x) + number_twos_level_3(x))},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x) + number_twos_level_4(x))},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x) + number_twos_level_5(x))},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x) + number_twos_level_6(x))},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_7(x) - number_neg_ones_level_7(x) + number_twos_level_7(x))},
        {'type' : 'ineq', 'fun' : bbss_9(lambda x : 1 - number_ones_level_8(x) - number_neg_ones_level_8(x) + number_twos_level_8(x))},
    ]

    mycons = bbss_9_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma1)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma2)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma3)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma4)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma5)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma6)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma7)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_9(lambda x : -x.gamma8)})

    if base_list_alg == 'MITM':
        time = bbss_9_time_MITM_SS
        bbss_memory = bbss_9_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_9_time_MITM_SS
        bbss_memory = bbss_9_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_9_time_dissection_tradeoff
        bbss_memory = bbss_9_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_9_time_repetition_subtrees
        bbss_memory = bbss_9_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_9(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})
        
    start = [(-0.2)]*8 + [random.uniform(0.18,0.22) for _ in range(16)] + [(0.03)]*8 + [(0.000)]*8
    bounds = [(-1,0)]*8 + [(0,1)]*32
    if use_bcj:
        bounds = [(-1,0)]*8 + [(0,1)]*24 + [(0,0)]*8
    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_9(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bbss_10(base_list_alg='MITM', repetition_subtrees=False, verb=True, membound=1., iters=10000, use_bcj=False):
    """
    Optimization target: original HGJ algorithm for subset-sum, using BBSS-representations with a 10-level merging tree.

    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: set the maximal memory the optimization should use, value between [0,1]
    :param iters: number of iterations scipy is using.
    :param use_bcj: set whether {0,-1,1} representations should be used
    """
    set_bbss_10 = collections.namedtuple('bbss_10', 'p0 p1 p2 p3 p4 p5 p6 p7 p8 l1 l2 l3 l4 l5 l6 l7 l8 l9 c1 c2 c3 c4 c5 c6 c7 c8 c9 alpha1 alpha2 alpha3 alpha4 alpha5 alpha6 alpha7 alpha8 alpha9 gamma1 gamma2 gamma3 gamma4 gamma5 gamma6 gamma7 gamma8 gamma9')
    
    def bbss_10(f): 
        return wrap(f, set_bbss_10)

    def bbss_10_memory_MITM(x):
        x = set_bbss_10(*x)
        return max(bbss_domain_level_9(x)/2., x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    def bbss_10_time_MITM_SS(x):
        x = set_bbss_10(*x)
        return max(bbss_domain_level_9(x)/2., x.l9, x.l8 - x.p8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_10_memory_SS(x):
        x = set_bbss_10(*x)
        return max(bbss_domain_level_9(x)/4., x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)

    def bbss_10_memory_dissection_tradeoff(x):
        x = set_bbss_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_9(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        return max(min_memory, min_memory_fac*bbss_domain_level_9(x))

    def bbss_10_time_dissection_tradeoff(x):
        x = set_bbss_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        min_memory_fac = min_memory / bbss_domain_level_9(x)
        min_memory_fac = max(1/7, min(min_memory_fac, 1/4))
        timefac = time7diss(min_memory_fac)
        return max(bbss_domain_level_9(x)*timefac, x.l9, x.l8 - x.p8, x.l7 - x.p7, x.l6 - x.p6, x.l5 - x.p5, x.l4 - x.p4, x.l3 - x.p3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)

    def bbss_10_time_repetition_subtrees(x):
        x = set_bbss_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_9(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        return max( 
                    max(bbss_domain_level_9(x)*timefac, x.l9) - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x) + 127*it7(x) + 255*it8(x), 0),
                    max(x.l9, x.l8 - x.p8)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x) + 127*it7(x), 0), 
                    max(x.l8, x.l7 - x.p7)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x) + 63*it6(x), 0),
                    max(x.l7, x.l6 - x.p6)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x) + 31*it5(x), 0),
                    max(x.l6, x.l5 - x.p5)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x) + 15*it4(x), 0),
                    max(x.l5, x.l4 - x.p4)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x) + 7*it3(x), 0),
                    max(x.l4, x.l3 - x.p3)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x) + 3*it2(x), 0),
                    max(x.l3, x.l2 - x.p2)                    - min(2*x.l1 - (1-x.c1) + x.p0 + it1(x), 0),
                    max(x.l2, x.l1 - x.p1, 2*x.l1 - (1-x.c1)) - min(2*x.l1 - (1-x.c1) + x.p0, 0)
                )

    def bbss_10_memory_repetition_subtrees(x):
        x = set_bbss_10(*x)
        min_memory = max(x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
        memfac = min_memory / bbss_domain_level_9(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(bbss_domain_level_9(x) * memfac, x.l9, x.l8, x.l7, x.l6, x.l5, x.l4, x.l3, x.l2, x.l1)
    
    bbss_10_constraints = [
        # filtering terms
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_0(x) - x.p0)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_1(x) - x.p1)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_2(x) - x.p2)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_3(x) - x.p3)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_4(x) - x.p4)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_5(x) - x.p5)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_6(x) - x.p6)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_7(x) - x.p7)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : bbss_filtering_8(x) - x.p8)},
        # sizes of the lists
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l5 - (x.c4 - x.c5) + x.p4 - x.l4)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l6 - (x.c5 - x.c6) + x.p5 - x.l5)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l7 - (x.c6 - x.c7) + x.p6 - x.l6)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l8 - (x.c7 - x.c8) + x.p7 - x.l7)},
        {'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l9 - (x.c8 - x.c9) + x.p8 - x.l8)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_1(x) - x.c1 - x.l1)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_2(x) - x.c2 - x.l2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_3(x) - x.c3 - x.l3)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_4(x) - x.c4 - x.l4)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_5(x) - x.c5 - x.l5)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_6(x) - x.c6 - x.l6)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_7(x) - x.c7 - x.l7)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_8(x) - x.c8 - x.l8)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : bbss_domain_level_9(x) - x.c9 - x.l9)},
        # coherence of the -1
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha2 - x.alpha1/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha3 - x.alpha2/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha4 - x.alpha3/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha5 - x.alpha4/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha6 - x.alpha5/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha7 - x.alpha6/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha8 - x.alpha7/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha9 - x.alpha8/2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha1 - 2*x.gamma1)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha2 - 2*x.gamma2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha3 - 2*x.gamma3)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha4 - 2*x.gamma4)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha5 - 2*x.gamma5)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha6 - 2*x.gamma6)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha7 - 2*x.gamma7)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha8 - 2*x.gamma8)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.alpha9 - 2*x.gamma9)},
        # memory bound
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : membound - bbss_memory(x))},
        # bit constraints
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c1 - x.c2)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c2 - x.c3)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c3 - x.c4)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c4 - x.c5)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c5 - x.c6)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c6 - x.c7)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c7 - x.c8)},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : x.c8 - x.c9)},
        # domain correctness
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_1(x) - number_neg_ones_level_1(x) + number_twos_level_1(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_2(x) - number_neg_ones_level_2(x) + number_twos_level_2(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_3(x) - number_neg_ones_level_3(x) + number_twos_level_3(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_4(x) - number_neg_ones_level_4(x) + number_twos_level_4(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_5(x) - number_neg_ones_level_5(x) + number_twos_level_5(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_6(x) - number_neg_ones_level_6(x) + number_twos_level_6(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_7(x) - number_neg_ones_level_7(x) + number_twos_level_7(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_8(x) - number_neg_ones_level_8(x) + number_twos_level_8(x))},
        {'type' : 'ineq', 'fun' : bbss_10(lambda x : 1 - number_ones_level_9(x) - number_neg_ones_level_9(x) + number_twos_level_9(x))},
    ]

    mycons = bbss_10_constraints.copy()
    if use_bcj:
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma1)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma2)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma3)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma4)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma5)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma6)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma7)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma8)})
        mycons.append({ 'type' : 'ineq', 'fun' : bbss_10(lambda x : -x.gamma9)})

    if base_list_alg == 'MITM':
        time = bbss_10_time_MITM_SS
        bbss_memory = bbss_10_memory_MITM
    elif base_list_alg == 'SS':
        time = bbss_10_time_MITM_SS
        bbss_memory = bbss_10_memory_SS
    elif base_list_alg == 'Dissection':
        time = bbss_10_time_dissection_tradeoff
        bbss_memory = bbss_10_memory_dissection_tradeoff
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    if repetition_subtrees:
        time = bbss_10_time_repetition_subtrees
        bbss_memory = bbss_10_memory_repetition_subtrees
    else:
        mycons.append({'type' : 'eq',   'fun' : bbss_10(lambda x : 2*x.l1 - (1-x.c1) + x.p0)})
        
    start = [(-0.2)]*9 + [random.uniform(0.18,0.22) for _ in range(18)] + [(0.03)]*9 + [(0.000)]*9
    bounds = [(-1,0)]*9 + [(0,1)]*36
    if use_bcj:
        bounds = [(-1,0)]*9 + [(0,1)]*27 + [(0,0)]*9

    result = opt.minimize(time, start, bounds=bounds, tol=1e-10, constraints=mycons, options={'maxiter': iters})
    astuple = set_bbss_10(*result.x)

    if verb:
        print("memory ", bbss_memory(result.x))
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_upwards_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0
    

###############################################################################
###################################MMT#########################################
###############################################################################

def optimize_mmt_2(k=0.488, w=Hi(1-0.488)/2, base_list_alg='MITM', repetition_subtrees=False, verb=False, membound=1.0, iters=10000):
    """
    Optimization target: MMT algorithm for syndrome decoding, using 2-level merging tree.
    
    :param k: code rate
    :param w: error weight
    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS), 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: optimize under memory constraint: in [0, 1]
    :param iters: number of iterations scipy is using.
    """
    mmt_domain_level_1 = lambda x: binomHH(k + x.l, x.p/2)
    perms = lambda x: binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)

    def mmt_2(f):
        return wrap(f, set_mmt_2)

    def mmt_2_time_MITM_SS(x):
        x = set_mmt_2(*x)
        if repetition_subtrees:
            return perms(x) + x.c1 - x.r1 + max(mmt_domain_level_1(x)/2., x.L1, x.L0)
        else:
            return perms(x) + max(mmt_domain_level_1(x)/2., x.L1, x.L0)

    def mmt_2_memory_MITM(x):
        x = set_mmt_2(*x)
        return max(mmt_domain_level_1(x)/2., x.L1, x.L0)

    def mmt_2_memory_SS(x):
        x = set_mmt_2(*x)
        return max(mmt_domain_level_1(x)/4., x.L1, x.L0)

    def mmt_2_memory_dissection(x):
        x = set_mmt_2(*x)
        m = max(x.L1, x.L0)
        memfac = m / mmt_domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(m, memfac * mmt_domain_level_1(x))

    def mmt_2_time_dissection(x):
        x = set_mmt_2(*x)
        m = max(x.L0, x.L1)
        memfac = m / mmt_domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        if repetition_subtrees:
            return perms(x) + x.c1 - x.r1 + max(m, timefac * mmt_domain_level_1(x))
        else:
            return perms(x) + max(m, timefac * mmt_domain_level_1(x))

    if repetition_subtrees:
        set_mmt_2 = collections.namedtuple('MMT_2R', 'l p L0 L1 r1 c1')
        constraints_mmt = [
            # list
            {'type': 'eq',   'fun': mmt_2(lambda x: x.L1 - (mmt_domain_level_1(x) - x.c1))},
            {'type': 'eq',   'fun': mmt_2(lambda x: x.L0 - (2*x.L1 - (x.l - x.c1)))},
            # reps
            {'type': 'eq',   'fun': mmt_2(lambda x: x.r1 - binomHH(x.p, x.p/2.))},
            # memory
            {'type': 'ineq', 'fun': mmt_2(lambda x: membound-mmt_memory(x))},
            # correctness
            {'type': 'ineq', 'fun': mmt_2(lambda x: x.l - x.c1)},
            {'type': 'ineq', 'fun': mmt_2(lambda x: x.c1 - x.r1)},
        ]
        start = [(rng.uniform(0.001, 0.2))]*2 +[(0.1)]*2 + [(rng.uniform(0.001, 0.22))]*2
        bounds = [(0., 1.0)]*6
    else:
        set_mmt_2 = collections.namedtuple('MMT_2', 'l p L0 L1 r1')
        constraints_mmt = [
            # list
            {'type': 'eq',   'fun': mmt_2(lambda x: x.L1 - (mmt_domain_level_1(x) - x.r1))},
            {'type': 'eq',   'fun': mmt_2(lambda x: x.L0 - (2*x.L1 - (x.l - x.r1)))},
            # reps
            {'type': 'eq',   'fun': mmt_2(lambda x: x.r1 - binomHH(x.p, x.p/2.))},
            # memory
            {'type': 'ineq', 'fun': mmt_2(lambda x: membound-mmt_memory(x))},
        ]
        start = [(rng.uniform(0.001, 0.2))]*2 +[(0.1)]*2 + [(rng.uniform(0.001, 0.22))]*1
        bounds = [(0., 1.0)]*5

    if base_list_alg == "MITM":
        time = mmt_2_time_MITM_SS
        mmt_memory = mmt_2_memory_MITM
    elif base_list_alg == "SS":
        time = mmt_2_time_MITM_SS
        mmt_memory = mmt_2_memory_SS
    elif base_list_alg == "Dissection":
        time = mmt_2_time_dissection
        mmt_memory = mmt_2_memory_dissection
    else:
        print("ERROR: NO SUCH BASE LIST ALGORITHM")
        return inf, 0
    
    result = opt.minimize(time, start, bounds=bounds, tol=1e-7, constraints=constraints_mmt, options={'maxiter': iters})
    astuple = set_mmt_2(*result.x)
    if verb:
        print("Validity: ", result.success)
        print("Time: ", time(astuple))
        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(constraints_mmt, result.x))

    if not result.success:
        return inf, 0

    t = check_constraints(constraints_mmt, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_mmt_3(k=0.488, w=Hi(1-0.488)/2, base_list_alg='MITM', repetition_subtrees=False, verb=False, membound=1.0, iters=10000):
    """
    Optimization target: MMT algorithm for syndrome decoding, using 3-level merging tree.
    
    :param k: code rate
    :param w: error weight
    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS), 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: optimize under memory constraint: in [0, 1]
    :param iters: number of iterations scipy is using.
    """
    mmt_domain_level_1 = lambda x: binomHH(k + x.l, x.p/2)
    mmt_domain_level_2 = lambda x: binomHH(k + x.l, x.p/4)
    perms = lambda x: binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)

    def mmt_3(f):
        return wrap(f, set_mmt_3)

    def mmt_3_time_MITM_SS(x):
        x = set_mmt_3(*x)
        if repetition_subtrees:
            T3 = max(mmt_domain_level_2(x)/2., x.L2)
            T2 = max(x.L2, x.L1)
            T1 = max(x.L1, x.L0)
            return perms(x) + max(
                T3 + max(3*x.c2 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(2*x.c2 + x.c1 - 2*x.r2 - x.r1, 0)
            )
        else:
            return perms(x) + max(mmt_domain_level_2(x)/2., x.L2, x.L1, x.L0)

    def mmt_3_memory_MITM(x):
        x = set_mmt_3(*x)
        return max(mmt_domain_level_2(x)/2., x.L2, x.L1, x.L0)

    def mmt_3_memory_SS(x):
        x = set_mmt_3(*x)
        return max(mmt_domain_level_2(x)/4., x.L2, x.L1, x.L0)

    def mmt_3_memory_dissection(x):
        x = set_mmt_3(*x)
        m = max(x.L2, x.L1, x.L0)
        memfac = m / mmt_domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(m, memfac * mmt_domain_level_2(x))

    def mmt_3_time_dissection(x):
        x = set_mmt_3(*x)
        m = max(x.L2, x.L1, x.L0)
        memfac = m / mmt_domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        if repetition_subtrees:
            T3 = max(mmt_domain_level_2(x)*timefac, x.L2)
            T2 = max(x.L2, x.L1)
            T1 = max(x.L1, x.L0)
            return perms(x) + max(
                T3 + max(3*x.c2 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(2*x.c2 + x.c1 - 2*x.r2 - x.r1, 0)
            )
        else:
            return perms(x) + max(m, timefac * mmt_domain_level_2(x))

    if repetition_subtrees:
        set_mmt_3 = collections.namedtuple('MMT_3R', 'l p L0 L1 L2 r1 r2 c1 c2')
        constraints_mmt = [
            # list
            {'type': 'eq',   'fun': mmt_3(lambda x: x.L2 - (mmt_domain_level_2(x) - x.c2))},
            {'type': 'eq',   'fun': mmt_3(lambda x: x.L1 - (2*x.L2 - (x.c1 - x.c2)))},
            {'type': 'eq',   'fun': mmt_3(lambda x: x.L0 - (2*x.L1 - (x.l - x.c1)))},
            # reps
            {'type': 'eq',   'fun': mmt_3(lambda x: x.r1 - binomHH(x.p, x.p/2.))},
            {'type': 'eq',   'fun': mmt_3(lambda x: x.r2 - binomHH(x.p/2, x.p/4.))},
            # memory
            {'type': 'ineq', 'fun': mmt_3(lambda x: membound-mmt_memory(x))},
            # correctness
            {'type': 'ineq', 'fun': mmt_3(lambda x: x.l - x.c1)},
            {'type': 'ineq', 'fun': mmt_3(lambda x: x.c1 - x.r1)},
            {'type': 'ineq', 'fun': mmt_3(lambda x: x.c2 - x.r2)},
            {'type': 'ineq', 'fun': mmt_3(lambda x: x.c1 - x.c2)},
        ]
        start = [(rng.uniform(0.001, 0.2))]*2 +[(0.1)]*3 + [(rng.uniform(0.001, 0.22))]*4
        bounds = [(0., 0.3)]*9

    else:
        set_mmt_3 = collections.namedtuple('MMT_3', 'l p L0 L1 L2 r1 r2')
        constraints_mmt = [
            # list
            {'type': 'eq',   'fun': mmt_3(lambda x: x.L2 - (mmt_domain_level_2(x) - x.r2))},
            {'type': 'eq',   'fun': mmt_3(lambda x: x.L1 - (mmt_domain_level_1(x) - x.r1))},
            {'type': 'eq',   'fun': mmt_3(lambda x: x.L0 - (2*x.L1 - (x.l - x.r1)))},
            # reps
            {'type': 'eq',   'fun': mmt_3(lambda x: x.r1 - binomHH(x.p, x.p/2.))},
            {'type': 'eq',   'fun': mmt_3(lambda x: x.r2 - binomHH(x.p/2., x.p/4.))},
            # memory
            {'type': 'ineq', 'fun': mmt_3(lambda x: membound-mmt_memory(x))},
        ]
        start = [(rng.uniform(0.001, 0.2))]*2 +[(0.1)] * 3 + [(rng.uniform(0.001, 0.22))]*2
        bounds = [(0., 0.3)]*7

    if base_list_alg == "MITM":
        time = mmt_3_time_MITM_SS
        mmt_memory = mmt_3_memory_MITM
    elif base_list_alg == "SS":
        time = mmt_3_time_MITM_SS
        mmt_memory = mmt_3_memory_SS
    elif base_list_alg == "Dissection":
        time = mmt_3_time_dissection
        mmt_memory = mmt_3_memory_dissection
    else:
        print("No such base_list_alg")
        return inf, 0
    
    result = opt.minimize(time, start, bounds=bounds, tol=1e-7, constraints=constraints_mmt, options={'maxiter': iters})
    astuple = set_mmt_3(*result.x)

    if verb:
        print("Validity: ", result.success)
        print("Time: ", time(astuple))
        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(constraints_mmt, result.x))

    if not result.success:
        return inf, 0

    t = check_constraints(constraints_mmt, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_mmt_4(k=0.448, w=Hi(1-0.448)/2, base_list_alg='MITM', repetition_subtrees=False, verb=False, membound=1.0, iters=10000):
    """
    Optimization target: MMT algorithm for syndrome decoding, using 4-level merging tree.
    
    :param k: code rate
    :param w: error weight
    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS), 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: optimize under memory constraint: in [0, 1]
    :param iters: number of iterations scipy is using.
    """
    mmt_domain_level_1 = lambda x: binomHH(k + x.l, x.p/2)
    mmt_domain_level_2 = lambda x: binomHH(k + x.l, x.p/4)
    mmt_domain_level_3 = lambda x: binomHH(k + x.l, x.p/8)
    perms = lambda x: binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)

    def mmt_4(f):
        return wrap(f, set_mmt_4)

    def mmt_4_time_MITM_SS(x):
        x = set_mmt_4(*x)
        if repetition_subtrees:
            T4 = max(mmt_domain_level_3(x)/2., x.L3)
            T3 = max(x.L3, x.L2)
            T2 = max(x.L2, x.L1)
            T1 = max(x.L1, x.L0)
            return perms(x) + max(
                T4 + max(7*x.c3 - 4*x.r3 - 2*x.r2 - x.r1, 0),
                T3 + max(4*x.c3 + 3*x.c2 - 4*x.r3 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(4*x.c3 + 2*x.c2 + x.c1 - 4*x.r3 - 2*x.r2 - x.r1, 0)
            )
        else:
            return perms(x) + max(x.L0, x.L1, x.L2, x.L3, mmt_domain_level_3(x)/2.)

    def mmt_4_memory_MITM(x):
        x = set_mmt_4(*x)
        return max(x.L0, x.L1, x.L2, x.L3, mmt_domain_level_3(x)/2.)

    def mmt_4_memory_SS(x):
        x = set_mmt_4(*x)
        return max(x.L0, x.L1, x.L2, x.L3, mmt_domain_level_3(x)/4.)

    def mmt_4_memory_dissection(x):
        x = set_mmt_4(*x)
        m = max(x.L0, x.L1, x.L2, x.L3)
        memfac = m / mmt_domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(mmt_domain_level_3(x) * memfac, m)

    def mmt_4_time_dissection(x):
        x = set_mmt_4(*x)
        m = max(x.L0, x.L1, x.L2, x.L3)
        memfac = m / mmt_domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        if repetition_subtrees:
            T4 = max(mmt_domain_level_3(x)*timefac, x.L3)
            T3 = max(x.L3, x.L2)
            T2 = max(x.L2, x.L1)
            T1 = max(x.L1, x.L0)
            return perms(x) + max(
                T4 + max(7*x.c3 - 4*x.r3 - 2*x.r2 - x.r1, 0),
                T3 + max(4*x.c3 + 3*x.c2 - 4*x.r3 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(4*x.c3 + 2*x.c2 + x.c1 - 4*x.r3 - 2*x.r2 - x.r1, 0)
            )
        else:
            return perms(x) + max(m, timefac * mmt_domain_level_3(x))

    if repetition_subtrees:
        set_mmt_4 = collections.namedtuple('MMT_4R', 'l p L0 L1 L2 L3 r1 r2 r3 c1 c2 c3')
        constraints_mmt = [
            # list
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L3 - (mmt_domain_level_3(x) - x.c3))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L2 - (2*x.L3 - (x.c2 - x.c3)))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L1 - (2*x.L2 - (x.c1 - x.c2)))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L0 - (2*x.L1 - (x.l - x.c1)))},
            # reps
            {'type': 'eq',   'fun': mmt_4(lambda x: x.r1 - binomHH(x.p, x.p/2.))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.r2 - binomHH(x.p/2., x.p/4.))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.r3 - binomHH(x.p/4, x.p/8.))},
            # memory
            {'type': 'ineq', 'fun': mmt_4(lambda x: membound-mmt_memory(x))},
            # correctness
            {'type': 'ineq', 'fun': mmt_4(lambda x: x.l - x.c1)},
            {'type': 'ineq', 'fun': mmt_4(lambda x: x.c1 - x.r1)},
            {'type': 'ineq', 'fun': mmt_4(lambda x: x.c2 - x.r2)},
            {'type': 'ineq', 'fun': mmt_4(lambda x: x.c3 - x.r3)},
        ]
        start = [(rng.uniform(0.001, 0.2))]*2 +[(0.1)]*4 + [(rng.uniform(0.001, 0.22))]*6
        bounds = [(0., 1.0)]*12

    else:
        set_mmt_4 = collections.namedtuple('MMT_4', 'l p L0 L1 L2 L3 r1 r2 r3')
        constraints_mmt = [
            # list
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L3 - (mmt_domain_level_3(x) - x.r3))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L2 - (mmt_domain_level_2(x) - x.r2))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L1 - (mmt_domain_level_1(x) - x.r1))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.L0 - (2*x.L1 - (x.l - x.r1)))},
            # reps
            {'type': 'eq',   'fun': mmt_4(lambda x: x.r1 - binomHH(x.p, x.p/2.))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.r2 - binomHH(x.p/2., x.p/4.))},
            {'type': 'eq',   'fun': mmt_4(lambda x: x.r3 - binomHH(x.p/4., x.p/8.))},
            # memory
            {'type': 'ineq', 'fun': mmt_4(lambda x: membound-mmt_memory(x))},
        ]
        start = [(rng.uniform(0.001, 0.2))]*2 + [(0.1)]*4 + [(rng.uniform(0.001, 0.22))]*3
        bounds = [(0., 1.0)]*9

    if base_list_alg == "MITM":
        time = mmt_4_time_MITM_SS
        mmt_memory = mmt_4_memory_MITM
    elif base_list_alg == "SS":
        time = mmt_4_time_MITM_SS
        mmt_memory = mmt_4_memory_SS
    elif base_list_alg == "Dissection":
        time = mmt_4_time_dissection
        mmt_memory = mmt_4_memory_dissection
    else:
        print("No such base_list_alg")
        return inf, 0
    
    result = opt.minimize(time, start, bounds=bounds, tol=1e-7, constraints=constraints_mmt, options={'maxiter': iters})
    astuple = set_mmt_4(*result.x)

    if verb:
        print("Validity: ", result.success)
        print("Time: ", time(astuple))
        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(constraints_mmt, result.x))

    if not result.success:
        return inf, 0

    t = check_constraints(constraints_mmt, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


###############################################################################
##################################BJMM#########################################
###############################################################################

bjmm_membound = 1.

def optimize_bjmm_2(k=0.448, w=Hi(1-0.448)/2, base_list_alg="MITM", repetition_subtrees=False, verb=False, membound=1., iters=10000):
    """
    Optimization target: BJMM algorithm for syndrome decoding, using 2-level merging tree.
    
    :param k: code rate
    :param w: error weight
    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS), 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: optimize under memory constraint: in [0, 1]
    :param iters: number of iterations scipy is using.
    """
    global bjmm_membound
    bjmm_membound = membound
    bjmm_domain_level_1 = lambda x: binomHH(k+x.l, x.p1)

    def bjmm(f):
        return wrap(f, set_bjmm_2)

    def bjmm_reps(p, p2, l):
        if p == 0. or p2 == 0. or l == 0.:
            return 0
        if l < p2 or p < p2/2. or l - p2 < p - p2/2.:
            return 0.
        if repetition_subtrees:
            return binomHH(p2, p2/2.) + binomHH(l-p2, p-p2/2.)
        else:
            return binomH(p2, p2/2.) + binomH(l-p2, p-p2/2.)

    def bjmm_2_time_MITM_SS(x):
        x = set_bjmm_2(*x)
        perms = binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)
        T2 = max(1/2 * bjmm_domain_level_1(x), x.L1)
        if repetition_subtrees:
            T1 = max(x.L1, 2*x.L1 - (x.l - x.c1))
            return perms + max(
                max(T2, T1) + max(2*x.c2 + x.c1 - 2*x.r2 - x.r1, 0)
            )
        else:
            T1 = max(x.L1, 2*x.L1 - (x.l - x.r1))
            return perms + max(T1, T2)

    def bjmm_2_memory_MITM(x):
        x = set_bjmm_2(*x)
        return max(x.L1, bjmm_domain_level_1(x)/2.)
    
    def bjmm_2_memory_SS(x):
        x = set_bjmm_2(*x)
        return max(x.L1, bjmm_domain_level_1(x)/4.)
    
    def bjmm_2_time_dissection(x):
        x = set_bjmm_2(*x)
        perms = binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)
        m = max(2*x.L1 - (x.l - x.r1), x.L1)
        memfac = m / bjmm_domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        if repetition_subtrees:
            T2 = max(timefac * bjmm_domain_level_1(x), x.L1)
            T1 = max(x.L1, 2*x.L1 - (x.l - x.c1))
            return perms + max(T2, T1) + max(x.c1 - x.r1, 0)
        else:
            return perms + max(m, timefac * bjmm_domain_level_1(x))
    
    def bjmm_2_memory_dissection(x):
        x = set_bjmm_2(*x)
        m = max(2*x.L1 - (x.l - x.r1), x.L1)
        memfac = m / bjmm_domain_level_1(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(m, memfac * bjmm_domain_level_1(x))

    if repetition_subtrees:
        set_bjmm_2 = collections.namedtuple('bjmm_2', 'l p p1 L1 r1 c1')
        constraints_bjmm_2 = [
            # weights
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p1) - x.p)},
            # representations and constrains
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c1 - x.r1)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.l - x.c1)},
            # reps
            {'type': 'eq',   'fun': bjmm(lambda x: x.r1 - bjmm_reps(x.p1, x.p,  k + x.l))},
            # list
            {'type': 'eq',   'fun': bjmm(lambda x: x.L1 - (bjmm_domain_level_1(x) - x.c1))},
            # memory
            {'type': 'ineq', 'fun': bjmm(lambda x: bjmm_membound-bjmm_memory(x))},
            # domain correctness
            {'type': 'ineq', 'fun': bjmm(lambda x: 1. - k - x.l)},
            {'type': 'ineq', 'fun': bjmm(lambda x: k + x.l - x.p)},
            {'type': 'ineq', 'fun': bjmm(lambda x: 1-k-x.l - (w - x.p))},
        ]
        start = [(rng.uniform(0.05, 0.09))]+[(rng.uniform(0.01, 0.02))] + [(rng.uniform(0.001, 0.015))]*1 + [(0.031)]*1 + [(rng.uniform(0.001, 0.2))]*2
        bounds = [(0.0, 1.0)]*6
    else:
        set_bjmm_2 = collections.namedtuple('bjmm_3', 'l p p1 L1 r1')
        constraints_bjmm_2 = [
            # weights
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p1) - x.p)},
            # representations and constrains
            {'type': 'ineq',   'fun': bjmm(lambda x: x.l - x.r1)},
            # reps
            {'type': 'eq',   'fun': bjmm(lambda x: x.r1 - bjmm_reps(x.p1, x.p,  k + x.l))},
            # list
            {'type': 'eq',   'fun': bjmm(lambda x: x.L1 - (bjmm_domain_level_1(x) - x.r1))},
            # memory
            {'type': 'ineq', 'fun': bjmm(lambda x: bjmm_membound-bjmm_memory(x))},
            # domain correctness
            {'type': 'ineq', 'fun': bjmm(lambda x: 1. - k - x.l)},
            {'type': 'ineq', 'fun': bjmm(lambda x: k + x.l - x.p)},
            {'type': 'ineq', 'fun': bjmm(lambda x: 1-k-x.l - (w - x.p))},
        ]

        start = [(rng.uniform(0.05, 0.09))]+[(rng.uniform(0.01, 0.02))] + [(rng.uniform(0.001, 0.015))]*1 + [(0.031)]*1 + [(rng.uniform(0.001, 0.2))]*1
        bounds = [(0.0, 1.0)]*5

    if base_list_alg == "MITM":
        time = bjmm_2_time_MITM_SS
        bjmm_memory = bjmm_2_memory_MITM
    elif base_list_alg == "SS":
        time = bjmm_2_time_MITM_SS
        bjmm_memory = bjmm_2_memory_SS 
    elif base_list_alg == "Dissection":
        time = bjmm_2_time_dissection
        bjmm_memory = bjmm_2_memory_dissection
    else:
        print("No such base_list_alg")
        return inf, 0

    mycons = constraints_bjmm_2
    result = opt.minimize(time, start, bounds=bounds, tol=1e-7, constraints=mycons, options={'maxiter': iters})
    astuple = set_bjmm_2(*result.x)

    if verb:
        print("Validity: ", result.success)
        print("Time: ", time(astuple))

        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    if not result.success:
        return inf, 0

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bjmm_3(k=0.448, w=Hi(1-0.448)/2, base_list_alg="MITM", repetition_subtrees=False, verb=False, membound=1., iters=10000):
    """
    Optimization target: BJMM algorithm for syndrome decoding, using 3-level merging tree.
    
    :param k: code rate
    :param w: error weight
    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS), 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: optimize under memory constraint: in [0, 1]
    :param iters: number of iterations scipy is using.
    """
    global bjmm_membound
    bjmm_membound = membound
    bjmm_domain_level_2 = lambda x: binomHH(k+x.l, x.p2)

    def bjmm(f):
        return wrap(f, set_bjmm_3)

    def bjmm_reps(p, p2, l):
        if p == 0. or p2 == 0. or l == 0.:
            return 0
        if l < p2 or p < p2/2. or l - p2 < p - p2/2.:
            return 0.
        if repetition_subtrees:
            return binomHH(p2, p2/2.) + binomHH(l-p2, p-p2/2.)
        else:
            return binomH(p2, p2/2.) + binomH(l-p2, p-p2/2.)

    def bjmm_3_time_MITM_SS(x):
        x = set_bjmm_3(*x)
        perms = binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)
        T3 = max(1/2. * bjmm_domain_level_2(x), x.L2)
        T2 = max(x.L2, x.L1)
        if repetition_subtrees:
            T1 = max(x.L1, 2*x.L1 - (x.l - x.c1))
            return perms + max(
                T3 + max(3*x.c2 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(2*x.c2 + x.c1 - 2*x.r2 - x.r1, 0)
            )
        else:
            T1 = max(x.L1, 2*x.L1 - (x.l - x.r1))
            return perms + max(T1, T2, T3)

    def bjmm_3_memory_MITM(x):
        x = set_bjmm_3(*x)
        return max(x.L1, x.L2, bjmm_domain_level_2(x)/2.)
    
    def bjmm_3_memory_SS(x):
        x = set_bjmm_3(*x)
        return max(x.L1, x.L2, bjmm_domain_level_2(x)/4.)
    
    def bjmm_3_time_dissection(x):
        x = set_bjmm_3(*x)
        perms = binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)
        m = max(2*x.L1 - (x.l - x.r1), x.L1, x.L2)
        memfac = m / bjmm_domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        if repetition_subtrees:
            T3 = max(timefac * bjmm_domain_level_2(x), x.L2)
            T2 = max(x.L2, x.L1)
            T1 = max(x.L1, 2*x.L1 - (x.l - x.c1))
            return perms + max(
                T3 + max(3*x.c2 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(2*x.c2 + x.c1 - 2*x.r2 - x.r1, 0)
            )
        else:
            return perms + max(m, timefac * bjmm_domain_level_2(x))
    
    def bjmm_3_memory_dissection(x):
        x = set_bjmm_3(*x)
        m = max(2*x.L1 - (x.l - x.r1), x.L1, x.L2)
        memfac = m / bjmm_domain_level_2(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(m, memfac * bjmm_domain_level_2(x))

    if repetition_subtrees:
        set_bjmm_3 = collections.namedtuple('bjmm_3', 'l p p1 p2 L1 L2 r1 r2 c1 c2')
        constraints_bjmm_3 = [
            # weights
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p2) - x.p1)},
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p1) - x.p)},
            # representations and constrains
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c1 - x.c2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c1 - x.r1)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c2 - x.r2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.l - x.c1)},
            # reps
            {'type': 'eq',   'fun': bjmm(lambda x: x.r2 - bjmm_reps(x.p2, x.p1, k + x.l))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.r1 - bjmm_reps(x.p1, x.p,  k + x.l))},
            # list
            {'type': 'eq',   'fun': bjmm(lambda x: x.L2 - (bjmm_domain_level_2(x) - x.c2))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.L1 - (2*x.L2 - (x.c1 - x.c2)))},
            # memory
            {'type': 'ineq', 'fun': bjmm(lambda x: bjmm_membound-bjmm_memory(x))},
            # domain correctness
            {'type': 'ineq', 'fun': bjmm(lambda x: 1. - k - x.l)},
            {'type': 'ineq', 'fun': bjmm(lambda x: k + x.l - x.p)},
            {'type': 'ineq', 'fun': bjmm(lambda x: 1-k-x.l - (w - x.p))},
        ]
        start = [(rng.uniform(0.05, 0.09))]+[(rng.uniform(0.01, 0.02))] + [(rng.uniform(0.001, 0.015))]*2 + [(0.031)]*2 + [(rng.uniform(0.001, 0.2))]*4
        bounds = [(0.0, 1.0)]*10
    else:
        set_bjmm_3 = collections.namedtuple('bjmm_3', 'l p p1 p2 L1 L2 r1 r2')
        constraints_bjmm_3 = [
            # weights
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p2) - x.p1)},
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p1) - x.p)},
            # representations and constrains
            {'type': 'ineq',   'fun': bjmm(lambda x: x.r1 - x.r2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.l - x.r1)},
            # reps
            {'type': 'eq',   'fun': bjmm(lambda x: x.r2 - bjmm_reps(x.p2, x.p1, k + x.l))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.r1 - bjmm_reps(x.p1, x.p,  k + x.l))},
            # list
            {'type': 'eq',   'fun': bjmm(lambda x: x.L2 - (bjmm_domain_level_2(x) - x.r2))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.L1 - (2*x.L2 - (x.r1 - x.r2)))},
            # memory
            {'type': 'ineq', 'fun': bjmm(lambda x: bjmm_membound-bjmm_memory(x))},
            # domain correctness
            {'type': 'ineq', 'fun': bjmm(lambda x: 1. - k - x.l)},
            {'type': 'ineq', 'fun': bjmm(lambda x: k + x.l - x.p)},
            {'type': 'ineq', 'fun': bjmm(lambda x: 1-k-x.l - (w - x.p))},
        ]

        start = [(rng.uniform(0.05, 0.09))]+[(rng.uniform(0.01, 0.02))] + [(rng.uniform(0.001, 0.015))]*2 + [(0.031)]*2 + [(rng.uniform(0.001, 0.2))]*2
        bounds = [(0.0, 1.0)]*8

    if base_list_alg == "MITM":
        time = bjmm_3_time_MITM_SS
        bjmm_memory = bjmm_3_memory_MITM
    elif base_list_alg == "SS":
        time = bjmm_3_time_MITM_SS
        bjmm_memory = bjmm_3_memory_SS 
    elif base_list_alg == "Dissection":
        time = bjmm_3_time_dissection
        bjmm_memory = bjmm_3_memory_dissection
    else:
        print("No such base_list_alg")
        return inf, 0

    mycons = constraints_bjmm_3
    result = opt.minimize(time, start, bounds=bounds, tol=1e-7, constraints=mycons, options={'maxiter': iters})
    astuple = set_bjmm_3(*result.x)

    if verb:
        print("Validity: ", result.success)
        print("Time: ", time(astuple))

        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    if not result.success:
        return inf, 0

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


def optimize_bjmm_4(k=0.448, w=Hi(1-0.448)/2, base_list_alg="MITM", repetition_subtrees=False, verb=False, membound=1., iters=10000):
    """
    Optimization target: BJMM algorithm for syndrome decoding, using 4-level merging tree.
    
    :param k: code rate
    :param w: error weight
    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS), 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param verb: verbose output
    :param membound: optimize under memory constraint: in [0, 1]
    :param iters: number of iterations scipy is using.
    """
    global bjmm_membound
    bjmm_membound = membound
    bjmm_domain_level_3 = lambda x: binomHH(k+x.l, x.p3)

    def bjmm(f):
        return wrap(f, set_bjmm_4)

    def bjmm_reps(p, p2, l):
        if p == 0. or p2 == 0. or l == 0.:
            return 0
        if l < p2 or p < p2/2. or l - p2 < p - p2/2.:
            return 0.
        if repetition_subtrees:
            return binomHH(p2, p2/2.) + binomHH(l-p2, p-p2/2.)
        else:
            return binomH(p2, p2/2.) + binomH(l-p2, p-p2/2.)
        
    def bjmm_4_time_MITM_SS(x):
        x = set_bjmm_4(*x)
        perms = binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)
        T4 = max(bjmm_domain_level_3(x)/2., x.L3)
        T3 = max(x.L3, x.L2)
        T2 = max(x.L2, x.L1)
        if repetition_subtrees:
            T1 = max(x.L1, 2*x.L1 - (x.l - x.c1))
            return perms + max(
                T4          + max(7*x.c3                 - 4*x.r3 - 2*x.r2 - x.r1, 0),
                T3          + max(4*x.c3 + 3*x.c2        - 4*x.r3 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(4*x.c3 + 2*x.c2 + x.c1 - 4*x.r3 - 2*x.r2 - x.r1, 0)
            )
        else:
            T1 = max(x.L1, 2*x.L1 - (x.l - x.r1))
            return perms + max(T1, T2, T3, T4)
        
    def bjmm_4_memory_MITM(x):
        x = set_bjmm_4(*x)
        return max(x.L1, x.L2, x.L3, bjmm_domain_level_3(x)/2.)

    def bjmm_4_memory_SS(x):
        x = set_bjmm_4(*x)
        return max(x.L1, x.L2, x.L3, bjmm_domain_level_3(x)/4.)
    
    def bjmm_4_time_dissection(x):
        x = set_bjmm_4(*x)
        perms = binomHH(1., w) - binomHH(k + x.l, x.p) - binomHH(1. - k - x.l, w - x.p)
        m = max(2*x.L1 - (x.l - x.r1), x.L1, x.L2, x.L3)
        memfac = m / bjmm_domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        timefac = time7diss(memfac)
        if repetition_subtrees:
            T4 = max(bjmm_domain_level_3(x)*timefac, x.L3)
            T3 = max(x.L3, x.L2)
            T2 = max(x.L2, x.L1)
            T1 = max(x.L1, 2*x.L1 - (x.l - x.c1))
            return perms + max(
                T4          + max(7*x.c3                 - 4*x.r3 - 2*x.r2 - x.r1, 0),
                T3          + max(4*x.c3 + 3*x.c2        - 4*x.r3 - 2*x.r2 - x.r1, 0),
                max(T2, T1) + max(4*x.c3 + 2*x.c2 + x.c1 - 4*x.r3 - 2*x.r2 - x.r1, 0)
            )
        else:
            return perms + max(m, timefac * bjmm_domain_level_3(x))
    
    def bjmm_4_memory_dissection(x):
        x = set_bjmm_4(*x)
        m = max(2*x.L1 - (x.l - x.r1), x.L1, x.L2, x.L3)
        memfac = m / bjmm_domain_level_3(x)
        memfac = max(1/7, min(memfac, 1/4))
        return max(m, memfac * bjmm_domain_level_3(x))

    if repetition_subtrees:
        set_bjmm_4 = collections.namedtuple('bjmm_4', 'l p p1 p2 p3 L1 L2 L3 L4 r1 r2 r3 c1 c2 c3')
        constraints_bjmm_4 = [
            # weights
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p3) - x.p2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p2) - x.p1)},
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p1) - x.p)},
            # representations and constrains
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c1 - x.r1)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c2 - x.r2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c3 - x.r3)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c2 - x.c3)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.c1 - x.c2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.l - x.c1)},
            # reps
            {'type': 'eq',   'fun': bjmm(lambda x: x.r3 - bjmm_reps(x.p3, x.p2, k + x.l))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.r2 - bjmm_reps(x.p2, x.p1, k + x.l))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.r1 - bjmm_reps(x.p1, x.p,  k + x.l))},
            # list
            {'type': 'eq',   'fun': bjmm(lambda x: x.L3 - (bjmm_domain_level_3(x) - x.c3))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.L2 - (2*x.L3 - (x.c2 - x.c3)))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.L1 - (2*x.L2 - (x.c1 - x.c2)))},
            # memory
            {'type': 'ineq', 'fun': bjmm(lambda x: bjmm_membound-memory_bjmm(x))},
            # domain correctness
            {'type': 'ineq', 'fun': bjmm(lambda x: 1. - k - x.l)},
            {'type': 'ineq', 'fun': bjmm(lambda x: k + x.l - x.p)},
            {'type': 'ineq', 'fun': bjmm(lambda x: 1-k-x.l - (w - x.p))},
        ]

        start = [(rng.uniform(0.05, 0.09))]+[(rng.uniform(0.01, 0.02))] + [(rng.uniform(0.001, 0.015))]*3 + [(0.031)]*4 + [(rng.uniform(0.001, 0.2))]*6
        bounds = [(0.0, 1.0)]*15

    else:
        set_bjmm_4 = collections.namedtuple('bjmm_4', 'l p p1 p2 p3 L1 L2 L3 L4 r1 r2 r3')
        constraints_bjmm_4 = [
            # weights
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p3) - x.p2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p2) - x.p1)},
            {'type': 'ineq',   'fun': bjmm(lambda x: (2. * x.p1) - x.p)},
            # representations and constrains
            {'type': 'ineq',   'fun': bjmm(lambda x: x.r2 - x.r3)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.r1 - x.r2)},
            {'type': 'ineq',   'fun': bjmm(lambda x: x.l - x.r1)},
            # reps
            {'type': 'eq',   'fun': bjmm(lambda x: x.r3 - bjmm_reps(x.p3, x.p2, k + x.l))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.r2 - bjmm_reps(x.p2, x.p1, k + x.l))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.r1 - bjmm_reps(x.p1, x.p,  k + x.l))},
            # list
            {'type': 'eq',   'fun': bjmm(lambda x: x.L3 - (bjmm_domain_level_3(x) - x.r3))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.L2 - (2*x.L3 - (x.r2 - x.r3)))},
            {'type': 'eq',   'fun': bjmm(lambda x: x.L1 - (2*x.L2 - (x.r1 - x.r2)))},
            # memory
            {'type': 'ineq', 'fun': bjmm(lambda x: bjmm_membound-memory_bjmm(x))},
            # domain correctness
            {'type': 'ineq', 'fun': bjmm(lambda x: 1. - k - x.l)},
            {'type': 'ineq', 'fun': bjmm(lambda x: k + x.l - x.p)},
            {'type': 'ineq', 'fun': bjmm(lambda x: 1-k-x.l - (w - x.p))},
        ]

        start = [(rng.uniform(0.05, 0.09))]+[(rng.uniform(0.01, 0.02))] + [(rng.uniform(0.001, 0.015))]*3 + [(0.031)]*4 + [(rng.uniform(0.001, 0.2))]*3
        bounds = [(0.0, 1.0)]*12


    if base_list_alg == "MITM":
        time = bjmm_4_time_MITM_SS
        memory_bjmm = bjmm_4_memory_MITM
    elif base_list_alg == "SS":
        time = bjmm_4_time_MITM_SS
        memory_bjmm = bjmm_4_memory_SS
    elif base_list_alg == "Dissection":
        time = bjmm_4_time_dissection
        memory_bjmm = bjmm_4_memory_dissection
    else:
        print("no such base list alg")
        return inf, 0
    
    mycons = constraints_bjmm_4
    result = opt.minimize(time, start, bounds=bounds, tol=1e-7, constraints=mycons, options={'maxiter': iters})
    astuple = set_bjmm_4(*result.x)

    if verb:
        print("Validity: ", result.success)
        print("Time: ", time(astuple))

        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]))
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    if not result.success:
        return inf, 0

    t = check_constraints(mycons, result.x)

    if all(-10**(-7) <= i[1] <= 10**(-7) for i in t if i[0] == "eq") \
       and all(-10**(-7) <= i[1] for i in t if i[0] == "ineq"):
        return time(astuple), astuple
    else:
        return inf, 0


################################################################################
##############################OPTIMIZATION SCRIPTS##############################
################################################################################
    
def optimize_subset_sum(depth=4, max_bound=0.29, min_bound=0.09, step_size=0.01, number_of_retries=500, representations="bcj", base_list_alg="MITM", repetition_subtrees=False, iters=10000):
    """
    Optimization target: HGJ algorithm with BCJ / BBSS representations

    :param depth: integer in [3,10] for BCJ, [4, 9] for BBSS
    :param max_bound: set maximum memory bound, value between [0,1]
    :param min_bound: set minimum memory bound, value between [0,1]
    :param step_size: set step size for memory bound, value between [0,1]
    :param number_of_retries: integer between [0, inf]
    :param base_list_alg: Meet-in-the-Middle (MITM), Schroeppel-Shamir (SS) or 7-Dissection (Dissection)
    :param repetition_subtrees: set whether subtrees should be reused
    :param iters: set how many iterations per try, integer between [0, inf]
    """
    algo = globals()["optimize_" + representations + "_" + str(depth)]
    membound = max_bound
    char = base_list_alg[0]
    if repetition_subtrees==True:
        char = "R"

    while membound >= min_bound:
        found_solution = False
        mini = inf
        min_parameters = 0
        c = 0
        while c < number_of_retries:
            try:
                t, parameters = algo(base_list_alg=base_list_alg, repetition_subtrees=repetition_subtrees, verb=False, membound=membound, iters=iters)
                t = float(t)
            except ValueError:
                print("error")
                continue

            if t != inf:
                found_solution = True
                print(f"({depth}{char}: {membound}, {c}, {t}, {parameters})")
                if mini > t:
                    mini = t
                    min_parameters = parameters
            
            c += 1
        print(f"[{membound}, {mini}, {min_parameters}]\n\n")
        membound -= step_size
        if found_solution == False:
            return
    return


def optimize_decoding(k=0.45, w=Hi(1-0.45)/2, depth=2, max_bound=0.06, min_bound=0.00, step_size=0.001, number_of_retries=100, decoding_algorithm="mmt", base_list_alg="MITM", repetition_subtrees=False, iters=10000):
    algo = globals()["optimize_" + decoding_algorithm + "_" + str(depth)]
    membound = min_bound
    char = base_list_alg[0]
    if repetition_subtrees:
        char = "R"

    #graph_output = []
    while membound <= max_bound:
        mini = inf
        min_parameters = 0
        c = 0
        while c < number_of_retries:
            try:
                t, parameters = algo(k=k, w=w, base_list_alg=base_list_alg, repetition_subtrees=repetition_subtrees, verb=False, membound=membound, iters=iters)
                t = float(t)
            except ValueError:
                print("Error")
                continue

            if t != inf:
                print(f"({depth}{char}: {membound}, {c}, {t}, {parameters})")
                if mini > t:
                    mini = t
                    min_parameters = parameters
            
            c += 1
        print(f"[{membound}, {mini}, {min_parameters}]\n\n")
        #graph_output.append([membound, mini])
        membound += step_size

    #for item in graph_output:
    #    print(f"({item[0]}, {item[1]})")
    return

if __name__ == "__main__":
    optimize_decoding(number_of_retries=5, depth=3, base_list_alg="MITM", repetition_subtrees=False, decoding_algorithm="bjmm", min_bound=0.0299, max_bound=0.03, k=0.448, w = Hi(1-0.448)/2)
    #optimize_subset_sum(depth = 6, base_list_alg="Dissection", representations="bbss", repetition_subtrees=False, max_bound=0.15, min_bound=0.145)