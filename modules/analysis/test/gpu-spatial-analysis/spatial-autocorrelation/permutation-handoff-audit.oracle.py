# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/permutation-handoff-audit.oracle.py

Independent dense definitions for bivariate Moran, local/global Lee L and local
join counts. Local join counts use binary edges, BJC=x_i(1-y_i)sum(1-x_j)y_j,
CLC=x_i y_i sum x_j y_j; paired donor rows stay paired. Lee permutes both
variables jointly, bivariate Moran fixes x and permutes y. LOSH uses exact
Fraction residuals and independent arbitrary-integer Random123 draws; ranks
depend on sums of raw residual powers, not on a shared normalization helper.
"""
import importlib.util
import itertools
import json
from fractions import Fraction
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('global_fixture',Path(__file__).parent.parent/'global-spatial-statistics/global-mathematical-audit.oracle.py')
fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
active=np.arange(7); weights=np.zeros((7,7))
for row in active:
    for neighbor,weight in fixture.EDGES[row]:
        if neighbor<7 and neighbor!=row and weight>0: weights[row,neighbor]=weight
x=np.array(fixture.VALUES)[active]; y=np.array(fixture.SECOND)[active]
centered=x-x.mean(); second=y-y.mean(); denominator=np.sqrt((centered@centered)*(second@second)); binary=(weights>0).astype(float)
local_bivariate=6*centered*(weights@second)/denominator
local_lee=7*(weights@centered)*(weights@second)/denominator
global_lee=local_lee.sum()/np.sum(weights.sum(axis=1)**2)
black=(x!=0).astype(float); other=np.array([1,1,0,1,1,0,1],dtype=float)
for name,values in [('localBivariateMoran',local_bivariate),('localLeeL',local_lee),('localJoinCount',black*(binary@black)),('BJC',black*(1-other)*(binary@((1-black)*other))),('CLC',black*other*(binary@(black*other)))]:
    print(name,list(map(float,values))+[None,None,None])
print('global Lee',float(global_lee))

def words(row,replicate):
    key=[43,0]; counter=[0,row,replicate,0x10ca1]; mask=2**32-1
    for _ in range(10):
        first=counter[0]*0xd2511f53; second=counter[2]*0xcd9e8d57
        counter=[((second>>32)^counter[1]^key[0])&mask,second&mask,((first>>32)^counter[3]^key[1])&mask,first&mask]
        key=[(key[0]+0x9e3779b9)&mask,(key[1]+0xbb67ae85)&mask]
    return counter

scene=[0,1,4,10,22]; neighbors=[[1,4],[0,2],[1,3],[2,4],[0,3]]
for power in [1,2]:
    residuals=[abs(Fraction(scene[row])-sum(Fraction(scene[neighbor]) for neighbor in neighbors[row])/2)**power for row in range(5)]
    average=sum(residuals)/5; normalized=[value/average for value in residuals]
    observed=[sum(normalized[neighbor] for neighbor in neighborhood)/2 for neighborhood in neighbors]
    for tail in ['greater','lesser','two-sided']:
        exceedances=[]; probabilities=[]
        for row in range(5):
            actual=sum(residuals[neighbor] for neighbor in neighbors[row]); lower=0; upper=0
            for replicate in range(19):
                pool=[other for other in range(5) if other!=row]; random=iter(words(row,replicate)); chosen=[]
                for draw in range(2):
                    bound=4-draw; threshold=(2**32-bound)%bound
                    while True:
                        product=next(random)*bound
                        if (product & (2**32-1))>=threshold: break
                    selection=draw+(product>>32); pool[draw],pool[selection]=pool[selection],pool[draw]; chosen.append(pool[draw])
                value=sum(residuals[neighbor] for neighbor in chosen)
                lower+=value<=actual; upper+=value>=actual
            count=upper if tail=='greater' else lower if tail=='lesser' else min(lower,upper)
            probability=min(Fraction(1),Fraction((2 if tail=='two-sided' else 1)*(count+1),20))
            exceedances.append(count); probabilities.append(float(probability))
        print('LOSH',power,tail,json.dumps({'statistic':list(map(float,observed)),'normalized':list(map(float,normalized)), 'exceedances':exceedances,'p':probabilities,'significant':[int(value<=.6) for value in probabilities]}))
