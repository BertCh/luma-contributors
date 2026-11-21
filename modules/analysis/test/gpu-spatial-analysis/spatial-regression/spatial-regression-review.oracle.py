# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent dense review probes. Run:
/usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/spatial-regression-review.oracle.py
Only fixture INPUTS are read; no implementation or expected outputs are used.
"""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

fixture = json.loads(Path(__file__).with_name('spatial-two-stage-least-squares-audit-fixtures.json').read_text())[0]
count = 12
matrix = np.zeros((count, count))
for row in range(count):
    for slot in range(fixture['weights']['offsets'][row], fixture['weights']['offsets'][row+1]):
        matrix[row, fixture['weights']['neighbors'][slot]] = fixture['weights']['weights'][slot]
design = np.column_stack([np.ones(count), np.array(fixture['predictors']).reshape(count, 2)])
response = np.array(fixture['response'])
residuals = response - design @ np.linalg.lstsq(design, response, rcond=None)[0]
ratio = np.trace(matrix.T @ matrix) / count
moments = [(matrix.T @ matrix - ratio*np.eye(count))/(1+ratio**2), (matrix+matrix.T)/2]
bound = .99 / np.linalg.norm(matrix, ord=np.inf)
def criterion(parameter):
    filtered = residuals - parameter * matrix @ residuals
    return sum((filtered @ moment @ filtered/count)**2 for moment in moments)
grid = np.linspace(-bound, bound, 8193)
position = np.argmin([criterion(value) for value in grid])
parameter = minimize_scalar(criterion, bounds=(grid[max(0, position-1)], grid[min(len(grid)-1, position+1)]), method='bounded', options={'xatol':1e-15}).x
filtered_design = design - parameter * matrix @ design
coefficients = np.linalg.lstsq(filtered_design, response-parameter*matrix@response, rcond=None)[0]
residuals = response-design@coefficients
filtered = residuals-parameter*matrix@residuals
variance = np.mean(filtered**2)
fourth = np.mean(filtered**4)
psi = np.array([[(2*variance**2*np.trace(first@second)+(fourth-3*variance**2)*np.diag(first)@np.diag(second))/count for second in moments] for first in moments])
with np.errstate(over='ignore', invalid='ignore'):
    for scale in [1, 2**18]:
        covariance = psi*scale**4
        rounded = covariance.astype(np.float32)
        raw_determinant = np.float32(np.float32(rounded[0,0]*rounded[1,1])-np.float32(rounded[0,1]*rounded[1,0]))
        print('R8', scale, 'psi', covariance.tolist(), 'det64', np.linalg.det(covariance), 'raw float32 det', raw_determinant)

single = np.float32(.1)
total = np.float32(0)
for _ in range(8): total = np.float32(total+single)
mean = np.float32(total/np.float32(8))
deviation = np.float32(single-mean)
squares = np.float32(8)*np.float32(deviation*deviation)
corrected = squares-np.float32(np.float32(8)*deviation)**2/np.float32(8)
print('R5 input/mean/old diagonal/corrected diagonal', single, mean, squares, corrected)

for offset in [0, 4096]:
    predictor = np.tile([1,1,-1,-1], 4)
    weights = np.roll(np.eye(16), 1, axis=1)
    residual = weights@predictor
    response = offset+predictor+residual
    design = np.column_stack([np.ones(16), predictor])
    lag = weights@(response-residual)
    annihilator = np.eye(16)-design@np.linalg.pinv(design)
    norm = (annihilator@lag)@(annihilator@lag)
    old = np.float32(lag@lag)-np.float32(lag@(np.eye(16)-annihilator)@lag)
    print('R2', offset, 'norm64', norm, 'old float32 norm', old)

predictors = np.array([[1,11],[8,11],[1,16],[18,7],[2,3]], dtype=float)
response = np.array([3,10,0,7,14], dtype=float)
weights = np.roll(np.eye(5), 1, axis=1)
instruments = np.column_stack([np.ones(5), predictors, weights@predictors])
design = np.column_stack([np.ones(5), predictors, weights@response])
projector = instruments@np.linalg.pinv(instruments)
coefficients = np.linalg.solve(design.T@projector@design, design.T@projector@response)
print('R4 instrument rank, first-stage df, coefficients', np.linalg.matrix_rank(instruments), 5-instruments.shape[1], coefficients.tolist())
for count in [8,12]:
    print('R7', count, 'Moran I', -1/(count-1), 'variance', 0, 'z and p undefined')

def exact_rank(rows):
    """Rational row elimination: exact rank, independent of QR residual rounding."""
    basis = {}
    for row in rows:
        values = list(map(Fraction, row))
        for column, pivot in sorted(basis.items()):
            multiple = values[column]
            values = [value - multiple*entry for value, entry in zip(values, pivot)]
        nonzero = next((column for column, value in enumerate(values) if value), None)
        if nonzero is not None:
            divisor = values[nonzero]
            basis[nonzero] = [value/divisor for value in values]
    return len(basis)

base_predictors = np.array(fixture['predictors'], dtype=np.float32).reshape(12, 2)
for count, changed_row, step in [(12, 0, Fraction(1, 2**25)), (300, 6, Fraction(1, 2**23))]:
    exact_design = [[Fraction(1), *map(lambda value: Fraction(float(value)), base_predictors[row % 12])] for row in range(count)]
    exact_response = [row[0] + Fraction(3, 4)*row[1] - Fraction(1, 2)*row[2] for row in exact_design]
    for departure in [False, True]:
        response = exact_response.copy()
        if departure: response[changed_row] += step
        rank = exact_rank(exact_design)
        augmented_rank = exact_rank([row + [value] for row, value in zip(exact_design, response)])
        print('R3', count, 'departure', departure, 'rank X / [X,y]', rank, augmented_rank,
              'witness residual', response[changed_row]-exact_response[changed_row])
        assert (rank, augmented_rank) == (3, 4 if departure else 3)
        # A zero true residual makes every spatial-error moment zero at every lambda.
        if not departure: assert all(value == row[0]+Fraction(3,4)*row[1]-Fraction(1,2)*row[2] for row,value in zip(exact_design,response))
assert 6 not in {index*299//255 for index in range(256)}

# Review 2: an exact rational slope cannot be represented by one f32 coefficient.
response = np.array([1,3,-1,-3,2,-2,7,-7,5,-5,6,-6], dtype=np.float32)
predictor = 7*response
for departure in [False, True]:
    changed = response.copy()
    if departure: changed[0] = np.nextafter(changed[0], np.float32(np.inf))
    print('review2 R3 ranks', exact_rank([[1,Fraction(float(x))] for x in predictor]), exact_rank([[1,Fraction(float(x)),Fraction(float(y))] for x,y in zip(predictor,changed)]))
slope = np.float32(1/7)
print('review2 R3 adjacent slope products at x=21', float(np.float32(np.nextafter(slope, np.float32(-np.inf))*np.float32(21))), float(np.float32(slope*np.float32(21))))

for count, value, changed in [(512,1.9793816,True),(2**23,1.973755,False)]:
    values = np.full(count, value, dtype=np.float32)
    if changed: values[-1] = np.nextafter(values[-1], np.float32(np.inf))
    tile = max(64, (count+4095)//4096)
    def tiled_sum(values):
        partials = np.cumsum(values.reshape(-1,tile),axis=1,dtype=np.float32)[:,-1]
        while len(partials)>1: partials = np.float32(partials[::2]+partials[1::2])
        return partials[0]
    mean = tiled_sum(values)/np.float32(count)
    deviations = np.float32(values-mean)
    total = tiled_sum(deviations)
    norm = tiled_sum(np.float32(deviations*deviations))
    corrected = np.float32(norm - np.float32(total*np.float32(total/np.float32(count))))
    exact = np.sum((values.astype(float)-values.astype(float).mean())**2)
    print('review2 R5', count, 'old corrected/raw', float(corrected/norm), 'true norm', exact)

# Exact M for [1,x], x alternating 0/1: subtract within-parity means.
M = [[Fraction(int(i==j))-Fraction(int(i%2==j%2),4) for j in range(8)] for i in range(8)]
def multiply(first, second):
    return [[sum(a*b for a,b in zip(row,column)) for column in zip(*second)] for row in first]
for increment in [Fraction(0),Fraction(1,2**20),Fraction(1,32)]:
    B = [[increment if (i,j) in [(0,1),(1,0)] else Fraction(0) for j in range(8)] for i in range(8)]
    projected = multiply(multiply(M,B),M)
    trace = sum(projected[i][i] for i in range(8))
    centered = [[projected[i][j]-trace*M[i][j]/6 for j in range(8)] for i in range(8)]
    scale = Fraction(8)/(56+2*increment)
    variance = 2*scale**2*sum(value**2 for row in centered for value in row)/(6*8)
    residual = [1,1,-1,-1]*2
    numerator = scale*(sum(residual[i]*B[i][j]*residual[j] for i in range(8) for j in range(8))/8-trace/6)
    z = float(numerator)/np.sqrt(float(variance)) if variance else np.nan
    from scipy.special import erfc
    print('review2 R7',float(increment),'variance',float(variance),'z',z,'p',erfc(abs(z)/np.sqrt(2)))

import contextlib, io, runpy
with contextlib.redirect_stdout(io.StringIO()):
    reference = runpy.run_path(str(Path(__file__).with_name('spatial-error-gm-audit.oracle.py')))['fixture']('homoskedastic','exact',False)
response = np.array(reference['response'])*2**30
residual = np.array(reference['expected']['residuals'])*2**30
fitted = response-residual
covariance = np.dot(response-response.mean(), fitted-fitted.mean())
response_squares = np.sum((response-response.mean())**2)
fitted_squares = np.sum((fitted-fitted.mean())**2)
with np.errstate(over='ignore',invalid='ignore'):
    old = np.float32(np.float32(covariance)*np.float32(covariance))/np.float32(np.float32(response_squares)*np.float32(fitted_squares))
print('review2 R12 numerator/denominator',covariance**2,response_squares*fitted_squares,'old f32',old,'oracle',(covariance/np.sqrt(response_squares)/np.sqrt(fitted_squares))**2)

# v3-1: model the allowed fast-math regrouping of the two denominator roots. This is a
# float32 arithmetic probe, not a claim to have inspected Metal's generated machine code.
for exponent in [18,29,30]:
    scale_squared = 2.0**(2*(exponent-30))
    response_variance = np.float32(response_squares*scale_squared)
    fitted_variance = np.float32(fitted_squares*scale_squared)
    cross = np.float32(covariance*scale_squared)
    with np.errstate(over='ignore'):
        regrouped = np.float32(cross/np.sqrt(np.float32(response_variance*fitted_variance)))
    first = np.float32(cross/np.sqrt(response_variance))
    staged = np.float32(first/np.sqrt(fitted_variance))
    print('v3-1 R12 scale',exponent,'regrouped f32',np.float32(regrouped*regrouped),'staged f32',np.float32(staged*staged),'oracle',0.7094877554580391)
    assert abs(float(np.float32(staged*staged))-0.7094877554580391)<1e-7
    if exponent == 30: assert regrouped == 0

# Constant fitted lag probe: row-order f32 summation of 64 equal constants loses the low part.
constant = np.float32(4096+1/256)
projected = np.float32(np.cumsum(np.full(64,constant,dtype=np.float32))[-1]/np.float32(64))
print('review2 R2 fitted/projection',constant,projected,'old J',64+64*float(constant-projected)**2/(1/256)**2,'true J',64)
