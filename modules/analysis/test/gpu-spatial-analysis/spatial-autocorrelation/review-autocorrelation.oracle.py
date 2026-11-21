# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/review-autocorrelation.oracle.py

Independent Fraction definitions of the reviewer counterexamples. No implementation imports.
Complete-graph identity: sum_ij (xi-xj)^2 = 2*n*sum_i (xi-mean)^2 under every permutation.
"""
from fractions import Fraction as F
import json
import math
import struct

values = list(map(F, [3,2,4,2,4,3,3,3]))
neighbors = [[1,2],[3],[4],[1],[2],[0],[5],[6]]
assert all(value == sum(values[other] for other in neighbors[row])/len(neighbors[row])
           for row,value in enumerate(values))
print('R1', {'residualCount': 8, 'residuals': [0]*8, 'status': [2]*8})
constant = [F(1,10)]*8
assert sum((value-sum(constant)/8)**2 for value in constant) == 0
print('R2', 'either constant marginal has variance 0, correlation undefined')
for shift in [0,1]:
    distance = F(1,2**23)
    values = [F(shift)]*4 + [F(shift)+distance]*4
    mean = sum(values)/8
    variance = sum((value-mean)**2 for value in values)/8
    observed = [(value-values[(row+1)%8])**2/variance for row,value in enumerate(values)]
    assert observed == [0,0,0,4,0,0,0,4]
print('R3', list(map(float, observed)))
values = [F(1)]+[F(0)]*8
mean = sum(values)/9
squares = sum((value-mean)**2 for value in values)
assert len(set(values[1:])) == 1
assert 8*(values[0]-mean)*(values[1]-mean)/squares == -F(8,9)
print('R4', {'localI': '-8/9', 'conditionalVariance': 0})
gamma=[]; geary=[]
for values in [[0,1,2,4], [0,4,1,2]]:
    differences = [(F(value)-values[(row+1)%4])**2 for row,value in enumerate(values)]
    gamma.append((differences[0]/4+sum(differences[1:]))/F(13,4))
    mean = sum(map(F, values))/4
    geary.append(3*sum(differences)/(8*sum((value-mean)**2 for value in values)))
assert gamma == [F(85,13), F(72,13)] and geary == [F(33,35),F(9,7)]
print('R5', list(map(str,gamma)), list(map(str,geary)))
assert sum(F(4)-4 for _ in range(10)) == 0
print('R6', {'p': '1/11', 'class': 2})
assert 256*129 > 32768
print('R7', {'simulationNodes': 256*129, 'maximumNodes': 32768})
for rows in [8,9]:
    values=list(map(F, range(rows))); mean=sum(values)/rows
    squared=sum((value-mean)**2 for value in values)
    pair_sum=sum((first-second)**2 for first in values for second in values)
    assert pair_sum == 2*rows*squared
    assert (rows-1)*pair_sum/(2*rows*(rows-1)*squared) == 1
print('R8', {'C': 1, 'variance': 0, 'z': None, 'p': None})
lower = F(25*3,10)-F(1,2); upper=F(25*4,10)-F(1,2)
assert [lower,upper] == [7,F(19,2)]
assert [row for row in range(25) if lower <= row < upper] == [7,8,9]
print('R9', {'boundaries': [float(lower),float(upper)], 'members': [7,8,9]})

# Review 2: independent exact arithmetic, not the CPU twins or WGSL expressions.
w = F(struct.unpack('f', struct.pack('f', .1))[0])
values = list(map(F, [2,3,3,1,1,2,2,2]))
neighbors = [[1,2,3,4],[2],[1],[4],[3],[0],[5],[6]]
weights = [[F(1),w,F(1),w]]+[[F(1)]]*7
assert all(sum(weight*(values[other]-values[row]) for weight,other in zip(weights[row],links)) == 0
           for row,links in enumerate(neighbors))
print('review2 R1', {'residualCount': 8, 'exactResiduals': [0]*8, 'status': [2]*8})

MASK = 2**32-1
def philox(counter):
    words = list(counter); key = [0,0]
    for _ in range(10):
        first, second = 0xd2511f53*words[0], 0xcd9e8d57*words[2]
        words = [((second>>32)^words[1]^key[0])&MASK, second&MASK,
                 ((first>>32)^words[3]^key[1])&MASK, first&MASK]
        key = [(key[0]+0x9e3779b9)&MASK, (key[1]+0xbb67ae85)&MASK]
    return words

def donor(replicate):
    # Random123 Philox4x32-10, Lemire rejection, row 1, purpose local, seed [0,0].
    for block in range(100):
        for word in philox([block,1,replicate,0x10ca1]):
            product = word*7
            if (product&MASK) >= (2**32-7)%7:
                return [0,2,3,4,5,6,7][product>>32]
    raise RuntimeError('rejection budget')

d = F(1,2**26)
values = [F(1),F(0),d,F(0),d,F(0),d,F(0)]
mean = sum(values)/8
variance = sum((value-mean)**2 for value in values)/8
observed = d*d/variance
upper = sum((values[1]-values[donor(replicate)])**2 >= d*d for replicate in range(1,20))
assert upper == 9
assert F(min(upper,19-upper)+1,20) == F(1,2)
print('review2 R3a', {'observed': float(observed), 'upper': upper, 'p': .5})

values = list(map(F, range(6))); mean = sum(values)/6
assert 5*(values[0]-mean)*sum((value-mean)/5 for value in values[1:])/sum((value-mean)**2 for value in values) == -F(5,14)
print('review2 R4', {'localI0': '-5/14', 'all conditional lag variances': 0})
print('review2 R5a', {'constant squared Gamma': 0, 'Geary scale': 0})
assert 2*16385 > 32768
print('review2 R7', {'observed plus simulation nodes': 32770})

epsilon = F(1,2**23)
# Only the perturbed edge depends on labels: C-1 = epsilon/S0 * (8*(a-b)^2/120 - 1).
deviations = [F(8*(first-second)**2,120)-1 for first in range(9) for second in range(9) if first != second]
assert len(deviations) == 72 and sum(deviations) == 0
variance = (epsilon/(72+epsilon))**2 * sum(value*value for value in deviations)/72
difference = epsilon/(72+epsilon)*(F(1,15)-1)
z = float(difference)/math.sqrt(float(variance))
assert abs(float(variance)/3.0702463789201337e-18-1) < 1e-14
print('review2 R8', {'uniform scaled variance': 0, 'perturbed variance': float(variance), 'z': z, 'p': math.erfc(abs(z)/math.sqrt(2))})
# On the centered Gaussian sphere of dimension n-1, (n-1)*projection^2 has
# mean 1 and variance 2*(n-2)/(n+1), independently of the selected edge.
normal_variance = (epsilon/(72+epsilon))**2 * F(2*(9-2),9+1)
normal_z = float(difference)/math.sqrt(float(normal_variance))
print('review2 R8 normality', {'variance': float(normal_variance), 'z': normal_z, 'p': math.erfc(abs(normal_z)/math.sqrt(2))})
lower,upper = F(21*9,14)-F(1,2), F(21*10,14)-F(1,2)
assert [lower,upper] == [13,F(29,2)]
assert [row for row in range(21) if lower <= row < upper] == [13,14]
print('review2 R9', {'bounds': [float(lower),float(upper)], 'members': [13,14]})
values = [F(1)]+[F(0)]*7  # masked outlier 99 is absent from this population.
mean = sum(values)/8
assert 7*(values[0]-mean)*(values[1]-mean)/sum((value-mean)**2 for value in values) == -F(7,8)
print('review2 R10', {'masked outlier localI0': '-7/8', 'conditional lag variance': 0})
