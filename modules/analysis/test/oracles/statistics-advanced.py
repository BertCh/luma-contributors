# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct spreg 1.8.3 Panel_RE_Lag/Panel_FE_Lag and mgwr 2.2.1 full MGWR inference."""
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import warnings
import numpy as np
import spreg
from libpysal.weights import W
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR, GWR
from types import SimpleNamespace
warnings.filterwarnings('ignore')
assert importlib.metadata.version('spreg') == '1.8.3'
assert importlib.metadata.version('mgwr') == '2.2.1'
def flat(value): return np.asarray(value).reshape(-1).tolist()
def rounded(value): return np.asarray(value,dtype=np.float32).astype(float)
result={'versions':{'spreg':'1.8.3','mgwr':'2.2.1','numpy':np.__version__},'panels':[]}
for seed in range(40):
    entities=8;periods=6
    weights=W({row:sorted([(row-1)%entities,(row+1)%entities]) for row in range(entities)},silence_warnings=True);weights.transform='r'
    matrix=weights.full()[0];random=np.random.default_rng(seed)
    predictors=rounded(random.normal(size=(entities*periods,2)))
    effects=random.normal(size=entities)*(.7 if seed%2==0 else 0)
    response=rounded(np.concatenate([np.linalg.solve(np.eye(entities)-.3*matrix,1+predictors[time*entities:(time+1)*entities]@np.array([.7,-.4])+effects+random.normal(size=entities)*.4) for time in range(periods)])[:,None])
    with contextlib.redirect_stdout(io.StringIO()):
        random_model=spreg.Panel_RE_Lag(response,predictors,weights,epsilon=1e-9)
        unit_model=spreg.Panel_FE_Lag(response,predictors,weights,epsilon=1e-9)
        statistic,p_value=spreg.panel_Hausman(unit_model,random_model)
    if not any((item['hausman'][0]>=0)==(statistic>=0) for item in result['panels']):
        sparse=weights.sparse.tocsr();sparse.sort_indices()
        result['panels'].append({'seed':seed,'entities':entities,'periods':periods,'predictors':flat(predictors),'response':flat(response),'weights':{'offsets':flat(sparse.indptr),'neighbors':flat(sparse.indices),'weights':flat(sparse.data)},'random':{'coefficients':flat(random_model.betas),'covariance':flat(random_model.varb),'residuals':flat(random_model.u),'fitted':flat(random_model.predy),'variance':float(random_model.sig2),'logLikelihood':float(random_model.logll)},'unit':{'coefficients':flat(unit_model.betas),'covariance':flat(unit_model.varb),'residuals':flat(unit_model.u),'fitted':flat(unit_model.predy),'variance':float(unit_model.sig2),'logLikelihood':float(unit_model.logll)},'hausman':[float(statistic),random_model.k,float(p_value)]})
    if len(result['panels'])==2:break
index=np.arange(24);coordinates=np.column_stack([index%6,index//6]).astype(float)
predictors=rounded(np.column_stack([np.sin(index*.7)+.13,np.cos(index*.37)+.07]));response=rounded((1+.4*predictors[:,0]-.2*predictors[:,1]+np.sin(index*1.8)*.2)[:,None]);bandwidths=[2,1.5,3]
selector=Sel_BW(coordinates,response,predictors,multi=True,fixed=True,kernel='gaussian',n_jobs=1)
with contextlib.redirect_stdout(io.StringIO()):
    selected=selector.search(init_multi=2,multi_bw_min=bandwidths,multi_bw_max=bandwidths,tol_multi=1e-9,max_iter_multi=128,bws_same_times=128,verbose=False)
    model=MGWR(coordinates,response,predictors,selector,fixed=True,kernel='gaussian',hat_matrix=True,n_jobs=1).fit()
smoother=np.sum(model.R,axis=2)
result['multiscale']={'positions':flat(coordinates),'predictors':flat(predictors),'response':flat(response),'bandwidths':flat(selected),'coefficients':flat(model.params),'standardErrors':flat(model.bse),'effectiveParameters':flat(model.ENP_j),'summary':[float(model.tr_S),float(np.sum(smoother*smoother)),float(model.sigma2),float(model.n-model.tr_S)],'sweeps':len(selector.bw[1])}
# Explicit finite-grid LOO partial-residual convention, using actual mgwr univariate
# local solves. This is a composed candidate policy, not an assertion of Sel_BW AICc parity.
design=np.column_stack([np.ones(len(response)),predictors]);parameters=np.zeros_like(design)
candidates=np.array([[1.5,2.5,4],[1,2,3],[1.5,3,4]])
history=[]
for sweep in range(128):
    previous=parameters.copy();chosen=[]
    for column in range(3):
        partial=response[:,0]-np.sum(design*parameters,axis=1)+design[:,column]*parameters[:,column]
        fits=[GWR(coordinates,partial[:,None],design[:,column,None],bw=bandwidth,fixed=True,kernel='gaussian',constant=False,hat_matrix=True,n_jobs=1).fit() for bandwidth in candidates[column]]
        scores=[np.sum((fit.resid_response/(1-fit.influ))**2) for fit in fits]
        winner=int(np.argmin(scores));chosen.append(candidates[column,winner]);parameters[:,column]=fits[winner].params[:,0]
    history.append(chosen)
    change=np.max(np.abs(design*(parameters-previous))/np.maximum(1,np.abs(design*previous)))
    if change<=1e-6:break
selector=SimpleNamespace(bw=(np.array(history[-1]),np.array(history)),bw_init=2,params=parameters)
with contextlib.redirect_stdout(io.StringIO()):
    model=MGWR(coordinates,response,predictors,selector,fixed=True,kernel='gaussian',hat_matrix=True,n_jobs=1).fit()
result['candidateMultiscale']={'candidates':flat(candidates),'bandwidths':history[-1],'coefficients':flat(parameters),'standardErrors':flat(model.bse),'effectiveParameters':flat(model.ENP_j),'sweeps':len(history)}
path=Path(__file__).parent/'fixtures/statistics-advanced.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('Executed direct panel random/unit ML, Hausman, and full MGWR smoother inference')
