# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct pinned PySAL-family references, distinct from the independent NumPy oracle.

PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python statistics-upstream.py
Pins: numpy2.2.6 scipy1.15.3 spreg1.8.3 mgwr2.2.1 esda2.7.0
      giddy2.3.6 libpysal4.13.0 spint1.0.7.
Every listed model is actually executed. Inputs are rounded to f32 before upstream calls.
"""
import contextlib
import importlib.metadata
import io
import json
import warnings
from pathlib import Path
import numpy as np
import scipy
import spreg
import mgwr
import esda
import giddy
import libpysal
import spint
import geopandas as gpd
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon
from mgwr.gwr import GWR
from esda.geary_local_mv import Geary_Local_MV
from giddy.markov import FullRank_Markov, GeoRank_Markov
from libpysal.weights import W, WSP, KNN, fuzzy_contiguity, attach_islands
from spreg.robust import robust_vm

warnings.filterwarnings('ignore')
PINS = {'numpy':'2.2.6','scipy':'1.15.3','spreg':'1.8.3','mgwr':'2.2.1','esda':'2.7.0','giddy':'2.3.6','libpysal':'4.13.0','spint':'1.0.7','scikit-learn':'1.7.2'}
for name,version in PINS.items():
    assert importlib.metadata.version(name) == version
HERE=Path(__file__).parent
independent=json.loads((HERE/'fixtures/statistics-coverage.json').read_text())
def array(values,shape=None):
    result=np.asarray(values,dtype=np.float32).astype(float)
    return result.reshape(shape) if shape else result

def csr(weights):
    matrix=weights.sparse.tocsr(); matrix.sort_indices()
    return {'offsets':matrix.indptr.tolist(),'neighbors':matrix.indices.tolist(),'weights':matrix.data.tolist()}

def from_csr(values):
    size=len(values['offsets'])-1
    return WSP(csr_matrix((array(values['weights']),values['neighbors'],values['offsets']),shape=(size,size))).to_W(silence_warnings=True)

def line_weights(rows):
    weights=W({row:sorted({(row-1)%rows,(row+1)%rows}) for row in range(rows)},silence_warnings=True)
    weights.transform='r';return weights

def flat(values):return np.asarray(values).reshape(-1).tolist()

result={'versions':PINS,'references':{}}
# ST01: actual TSLS coefficients and spreg's robust_vm, including caller-supplied HAC kernel.
x=array(independent['predictors'],(-1,1)); y=array(independent['response'],(-1,1))
design=array(independent['design'],(-1,2)); instruments=array(independent['instruments'],(-1,2))
model=spreg.TSLS(y,x,design[:,1:2],q=instruments[:,1:2],nonspat_diag=False)
result['instrumentalVariables']={'predictors':flat(x),'response':flat(y),'design':flat(design),'instruments':flat(instruments),'residuals':flat(model.u),'coefficients':flat(model.betas),'kernel':independent['kernel'], 'covariance':{'HC0':flat(robust_vm(model,sig2n_k=False)), 'HC1':flat(robust_vm(model,sig2n_k=True)), 'HAC':flat(robust_vm(model,gwk=from_csr(independent['kernel'])))}}
# ST02: partially varying intercepts, shared slope, pooled OLS covariance and Chow stability.
data=independent['regimes']; x=array(data['predictors'],(-1,1)); y=array(data['response'],(-1,1))
model=spreg.OLS_Regimes(y,x,data['regimes'],constant_regi='many',cols2regi=[False],regime_err_sep=False,nonspat_diag=False)
result['regimes']={'predictors':flat(x),'response':flat(y),'regimes':data['regimes'],'coefficients':flat(model.betas),'covariance':flat(model.vm),'stability':flat(model.chow.regi),'joint':flat(model.chow.joint)}
# ST03: exact determinant balanced FE error maximum likelihood.
data=independent['panelError']; weights=from_csr(data['weights']); x=array(data['predictors'],(-1,1));y=array(data['response'],(-1,1))
with contextlib.redirect_stdout(io.StringIO()):model=spreg.Panel_FE_Error(y,x,weights,epsilon=1e-10)
result['panelError']={**data,'coefficient':float(model.betas[0,0]),'parameter':float(model.lam),'logLikelihood':float(model.logll),'variance':float(model.sig2)}
result['panelError']['diagnostics']=flat([spreg.panel_LMerror(y,x,weights),spreg.panel_rLMlag(y,x,weights),spreg.panel_rLMerror(y,x,weights)])
# ST04: direct nonspatial and spatial-error joint SUR.
data=independent['sur'];x=array(data['predictors'],(2,40));y=array(data['responses'],(2,40));weights=line_weights(40)
bigX={equation:np.column_stack([np.ones(40),x[equation]]) for equation in range(2)}
bigy={equation:y[equation,:,None] for equation in range(2)}
with contextlib.redirect_stdout(io.StringIO()):
    plain=spreg.SUR(bigy,bigX,iter=False,nonspat_diag=False)
    spatial=spreg.SURerrorGM(bigy,bigX,weights,nonspat_diag=False)
result['sur']={'predictors':flat(x),'responses':flat(y),'weights':csr(weights),'ordinary':{'coefficients':flat([plain.bSUR[equation] for equation in range(2)]),'covariance':flat(plain.varb),'residualCovariance':flat(plain.sig)},'spatial':{'coefficients':flat([spatial.bSUR[equation] for equation in range(2)]),'covariance':flat(spatial.varb),'residualCovariance':flat(spatial.sig),'parameters':flat(spatial.lamsur),'residuals':flat(spatial.bigE.T)}}
# ST05: direct probit, average partial effects and all three spatial diagnostics.
data=independent['probit'];x=array(data['predictors'],(-1,2));y=array(data['response'],(-1,1));weights=line_weights(40)
with contextlib.redirect_stdout(io.StringIO()):model=spreg.Probit(y,x,w=weights,spat_diag=True,maxiter=200)
result['probit']={'predictors':flat(x),'response':flat(y),'weights':csr(weights),'coefficients':flat(model.betas),'fitted':flat(model.predy),'effects':flat(model.slopes),'logLikelihood':float(model.logl),'diagnostics':flat([model.Pinkse_error,model.KP_error,model.PS_error])}
# ST06: mgwr local_collinearity including its squared-weight uncentered design convention.
rows=24;index=np.arange(rows);coordinates=np.column_stack([index%6,index//6]).astype(float)
x=array(np.column_stack([np.sin(index*.7)+1.5,index/24+np.cos(index*.3)]));y=array((2+x@np.array([.4,-.2])+np.sin(index*1.8)*.3)[:,None])
model=GWR(coordinates,y,x,bw=3,fixed=True,kernel='gaussian',n_jobs=1).fit()
correlations,vif,condition,decomposition=model.local_collinearity()
weights=W({row:list(range(rows)) for row in range(rows)}, {row:model.model._build_wi(row,3).tolist() for row in range(rows)},silence_warnings=True)
result['collinearity']={'predictors':flat(x),'weights':csr(weights),'varianceInflationFactors':flat(vif),'designConditionNumbers':flat(condition),'varianceDecomposition':flat(decomposition)}
# ST06: actual mgwr GLM standard errors and influence (scale one).
from spglm.family import Poisson, Binomial
result['generalizedGWR']={}
for name,family,response in [('poisson',Poisson(),(1+(index*7)%9)[:,None]),('logistic',Binomial(),((index*7)%11<5).astype(float)[:,None])]:
    model=GWR(coordinates,array(response),x,bw=3,fixed=True,kernel='gaussian',family=family,n_jobs=1).fit(tol=1e-10,max_iter=128)
    result['generalizedGWR'][name]={'positions':flat(coordinates),'predictors':flat(x),'response':flat(response),'coefficients':flat(model.params),'standardErrors':flat(model.bse),'influence':flat(model.influ)}
# Exact LOO fits through mgwr's local GLM solver with self spatial weight removed.
class LeaveOneOutGWR(GWR):
    def _build_wi(self,index,bandwidth):
        weights=super()._build_wi(index,bandwidth);weights[index]=0;return weights
training=index%4!=0
scores=[]
response=array(result['generalizedGWR']['poisson']['response'],(-1,1))
for bandwidth in [1.5,3,6]:
    fitted=LeaveOneOutGWR(coordinates[training],response[training],x[training],bw=bandwidth,fixed=True,kernel='gaussian',family=Poisson(),n_jobs=1).fit(tol=1e-10,max_iter=128).predy
    observed=response[training]
    scores.append(float(np.mean(2*(observed*np.log(observed/fitted)-observed+fitted))))
result['generalizedGWR']['selection']={'scores':scores,'bandwidth':[1.5,3,6][np.argmin(scores)]}
# ST07: actual esda multivariate statistic. RNG streams differ; existing Philox replay checks
# the conditional distribution independently. esda's undoubled minimum-tail rule is explicit.
weights=line_weights(12);variables=array(np.column_stack([np.sin(np.arange(12)*.7),np.cos(np.arange(12)*.3)]))
model=Geary_Local_MV(connectivity=weights,permutations=0).fit(variables.T)
result['multivariate']={'values':flat(variables),'weights':csr(weights),'statistic':flat(model.localG)}
# ST08: upstream tie ordering and both transition conventions.
panel=array([[2,3,4],[2,5,1],[4,1,4],[1,2,3],[5,4,2],[3,6,6]])
full=FullRank_Markov(panel,summary=False);geo=GeoRank_Markov(panel,summary=False)
result['rank']={'values':flat(panel.T),'periodCount':3,'ranks':flat(full.ranks.T-1),'fullCounts':flat(full.transitions),'geographicCounts':flat(geo.transitions)}
# ST09: direct buffer/intersects and original-island reciprocal attachments, away from degeneracy.
polygons=[Polygon([(0,0),(1,0),(1,1),(0,1)]),Polygon([(1.05,0),(2.05,0),(2.05,1),(1.05,1)]),Polygon([(4,0),(5,0),(5,1),(4,1)]),Polygon([(4.3,.3),(4.5,.3),(4.5,.5),(4.3,.5)])]
weights=fuzzy_contiguity(gpd.GeoDataFrame(geometry=polygons),buffering=True,buffer=.05,silence_warnings=True)
result['fuzzy']={'positions':flat([list(polygon.exterior.coords)[:-1] for polygon in polygons]),'tolerance':.1,'weights':csr(weights)}
positions=array([[0,0],[1,0],[4,0],[4.3,1]])
initial=W({0:[1],1:[0],2:[],3:[]},silence_warnings=True)
attached=attach_islands(initial,KNN.from_array(positions,k=1),silence_warnings=True)
result['islands']={'positions':flat(positions),'initial':csr(initial),'attached':csr(attached)}
# ST11: fitted GLM coefficients/predictions, not evaluation of an arbitrary coefficient grid.
origins=np.repeat(np.arange(4),5);destinations=np.tile(np.arange(5),4)
cost=array(1+(origins-destinations)**2*.2+np.sin(np.arange(20)*.7)**2)
flows=np.array([20,15,10,8,4,12,22,15,7,6,5,13,25,15,10,4,8,16,24,18], dtype=np.int64)[:,None]
empty=np.ones((20,1));cost_column=cost[:,None]
models={'unconstrained':spint.Gravity(flows,empty,empty,cost_column,'exp'),'origin':spint.Production(flows,origins[:,None],empty,cost_column,'exp'),'destination':spint.Attraction(flows,destinations[:,None],empty,cost_column,'exp'),'doubly':spint.Doubly(flows,origins[:,None],destinations[:,None],cost_column,'exp')}
result['interaction']={'origins':flat(origins),'destinations':flat(destinations),'flows':flat(flows),'cost':flat(cost),'models':{name:{'coefficient':float(model.params[-1]),'predictions':flat(model.yhat),'deviance':float(model.deviance)} for name,model in models.items()}}
result['references']={'instrumentalVariables':'spreg.TSLS + spreg.robust.robust_vm','regimes':'spreg.OLS_Regimes(regime_err_sep=False).chow','panelError':'spreg.Panel_FE_Error','sur':'spreg.SUR + spreg.SURerrorGM','probit':'spreg.Probit(spat_diag=True)','collinearity':'mgwr.GWRResults.local_collinearity','multivariate':'esda.Geary_Local_MV','rank':'giddy.FullRank_Markov + GeoRank_Markov','fuzzy':'libpysal.fuzzy_contiguity(buffer=.05)','islands':'libpysal.attach_islands','interaction':'spint.Gravity/Production/Attraction/Doubly GLM'}
(HERE/'fixtures/statistics-upstream.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('Wrote direct upstream statistics references')
