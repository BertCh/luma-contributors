# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-dataframe/composite-indicators/inequality-output-oracles.oracle.py
import json
import numpy as np
from decimal import Decimal, localcontext

# Independent high-precision definition, avoiding a float64 intermediate rounded near 1.
with localcontext() as context:
    context.prec = 70
    power = -Decimal(2) ** -23
    relative = [Decimal('0.5'), Decimal('0.5'), Decimal('1.5'), Decimal('1.5')]
    power_mean = sum((power * value.ln()).exp() for value in relative) / 4
    print('adjacent epsilon Atkinson:', 1 - (power_mean.ln() / power).exp())
    power = 1 - Decimal.from_float(float(np.float32(0.999)))
    mean = Decimal(2048) / 1025
    power_mean = (512 * (power * (1/mean).ln()).exp() + 512 * (power * (3/mean).ln()).exp()) / 1025
    print('one zero, 512 ones and 512 threes:', 1 - (power_mean.ln() / power).exp())
    weights = [2**110, 2**126, 2**126]
    incomes = [Decimal(0), Decimal(1) / 256, Decimal(1)]
    mean = sum(value * weight for value, weight in zip(incomes, weights)) / sum(weights)
    power_mean = sum(weight * (value / mean)**power for value, weight in zip(incomes, weights)) / sum(weights)
    print('large centered weights (R1):', 1 - (power_mean.ln() / power).exp())
    low, high = Decimal(1) / 1024, Decimal(1)
    print('large equal weights at epsilon one:', 1 - (low * high).sqrt() / ((low + high) / 2))
    incomes = [Decimal.from_float(float(np.float32(value))) for value in [1e-30, 1e30]]
    mean = sum(incomes)/2
    power = Decimal(2) ** -8
    power_mean = sum((power*(value/mean).ln()).exp() for value in incomes)/2
    print('extreme positive ratio:', 1-(power_mean.ln()/power).exp())

def indices(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    relative = values / mean
    cumulative = np.r_[0, np.cumsum(np.sort(values)) / np.sum(values)]
    return dict(gini=np.abs(values[:, None] - values).sum() / (2 * len(values)**2 * mean),
                theilT=np.mean(relative * np.log(relative)), theilL=-np.mean(np.log(relative)),
                atkinson=1-np.exp(np.mean(np.log(values))) / mean,
                hoover=np.abs(values-mean).sum()/(2*values.sum()),
                palma=(1-np.interp(.75, np.linspace(0,1,len(values)+1), cumulative))/np.interp(.5,np.linspace(0,1,len(values)+1),cumulative),
                mean=mean, counts=len(values), lorenzKnots=np.interp(np.linspace(0,1,5), np.linspace(0,1,len(values)+1),cumulative).tolist())

zones = [np.array([1,1,3,3]), np.array([1,2,6])]
pooled = np.concatenate(zones)
results = [indices(zone) for zone in zones]
income = np.array([zone.sum() for zone in zones]) / pooled.sum()
population = np.array([len(zone) for zone in zones]) / len(pooled)
summary = [indices(pooled)['theilT'], np.sum(income*np.log(income/population)),
           sum(share*result['theilT'] for share,result in zip(income,results)), indices(pooled)['gini'],
           len(pooled), pooled.mean(), len(pooled), pooled.sum()]
print(json.dumps(dict(zones=results, globalSummary=summary), indent=2, default=float))
