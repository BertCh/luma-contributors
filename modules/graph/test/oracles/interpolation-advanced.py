# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent pinned GSTools/PyKrige/SciPy/CGAL and NumPy reference generation.
PYTHONPATH=/tmp/luma-coverage-oracles/lib/python3.12/site-packages \
 /tmp/luma-interpolation-oracles312/bin/python modules/graph/test/oracles/interpolation-advanced.py
Compile interpolation-cgal.cpp against external CGAL6.0.1/Boost1.87 first.
"""
import json, subprocess
from pathlib import Path
import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay
import gstools as gs
import pykrige
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
rng = np.random.default_rng(19147)
positions = np.array([[0,0],[2,0],[2,2],[0,2],[.6,.7],[1.4,1.2]], dtype=float)
values = np.array([1,2,1.3,3.2,2.4,.8])
queries = np.array([[.8,1.1],[1.1,.4],[1.7,1.6]])
model = gs.Exponential(dim=2, var=1.7, len_scale=.8, nugget=0)
covariance = {'model':'exponential','sill':1.7,'range':2.4,'nugget':0}
model_parameters = {'psill':1.7,'range':2.4,'nugget':0}
kriging = {}
for mode in ['simple','ordinary','universal','external-drift']:
    options = {'unbiased':mode != 'simple','mean':1.2 if mode == 'simple' else None, 'cond_err':0, 'pseudo_inv':False}
    if mode == 'universal': options['drift_functions'] = 'linear'
    if mode == 'external-drift': options['ext_drift'] = positions[:,0]**2
    krige = gs.krige.Krige(model, positions.T, values, **options)
    prediction, variance = krige(queries.T, ext_drift=queries[:,0]**2 if mode == 'external-drift' else None, store=False)
    kriging[mode] = {'gstools':np.column_stack([prediction, variance]).tolist()}
    if mode == 'simple': continue
    if mode == 'ordinary': other = OrdinaryKriging(*positions.T, values, variogram_model='exponential',variogram_parameters=model_parameters)
    else: other = UniversalKriging(*positions.T, values, variogram_model='exponential',variogram_parameters=model_parameters, drift_terms=['regional_linear'] if mode == 'universal' else ['specified'], specified_drift=[positions[:,0]**2] if mode=='external-drift' else None)
    prediction, variance = other.execute('points', *queries.T, **({'specified_drift_arrays':[queries[:,0]**2]} if mode=='external-drift' else {}))
    kriging[mode]['pykrige'] = np.column_stack([prediction, variance]).tolist()
# Weighted block with custom 8-column drift, arbitrary supplied design matrices, 32 samples.
wide_positions = rng.uniform(-2, 2, (32,2)); wide_values = rng.normal(size=32)
supports = rng.uniform(-.5,.5,(3,4,2)); support_weights = np.array([.1,.2,.3,.4])
sample_design = np.column_stack([np.ones(32), rng.normal(size=(32,7))]); query_design = np.column_stack([np.ones(3), rng.normal(scale=.2,size=(3,7))])
def covariance_matrix(first, second, structure=covariance):
    separation=np.linalg.norm(first[:,None,:]-second[None,:,:],axis=2)/structure['range']
    if structure['model']=='exponential': correlation=np.exp(-3*separation)
    elif structure['model']=='gaussian': correlation=np.exp(-3*separation**2)
    else: correlation=np.where(separation<1, 1-1.5*separation+.5*separation**3,0)
    return structure['sill']*correlation + (separation==0)*structure.get('nugget',0)
def constrained_prediction(matrix, cross, prior, design, query_design, response):
    size=len(matrix); dimensions=design.shape[1]
    system=np.block([[matrix, design],[design.T,np.zeros((dimensions,dimensions))]])
    solution=np.linalg.solve(system,np.r_[cross,query_design])
    return [float(solution[:size]@response),float(prior-solution@np.r_[cross,query_design])], solution[:size].tolist()
wide_expected=[]
for support, drift in zip(supports,query_design):
    cross=covariance_matrix(wide_positions,support)@support_weights
    prior=support_weights@covariance_matrix(support,support)@support_weights
    wide_expected.append(constrained_prediction(covariance_matrix(wide_positions,wide_positions)+np.eye(32)*.07,cross,prior,sample_design,drift,wide_values)[0])
# Three-variable, two-structure PSD LMC, independent unknown means with masks.
multivariate_positions=np.repeat(rng.uniform(-2,2,(10,2)),3,axis=0)
variable_indices=np.tile(np.arange(3),10); multivariate_values=rng.normal(size=30)+variable_indices*2
structures=[{'covariance':covariance,'loadings':[1,0,0,.4,.8,0,-.1,.3,.7]}, {'covariance':{'model':'gaussian','sill':.4,'range':1.3},'loadings':[.5,.1,0,0,.6,.1,.2,0,.4]}]
active=np.arange(30)!=4
lmc_design=np.eye(3)[variable_indices[active]]; lmc_support=np.array([[-.2,.3],[.4,.1],[.7,.5]]); lmc_weights=np.array([.2,.3,.5])
lmc_matrix=np.zeros((active.sum(),active.sum())); cross=np.zeros(active.sum()); prior=0
for structure in structures:
    loading=np.array(structure['loadings']).reshape(3,3); coefficients=loading@loading.T
    lmc_matrix += covariance_matrix(multivariate_positions[active],multivariate_positions[active],structure['covariance'])*coefficients[variable_indices[active,None],variable_indices[None,active]]
    cross += (covariance_matrix(multivariate_positions[active],lmc_support,structure['covariance'])@lmc_weights)*coefficients[variable_indices[active],2]
    prior += (lmc_weights@covariance_matrix(lmc_support,lmc_support,structure['covariance'])@lmc_weights)*coefficients[2,2]
lmc_expected,lmc_coefficients=constrained_prediction(lmc_matrix+np.eye(active.sum())*.05,cross,prior,lmc_design,np.array([0,0,1]),multivariate_values[active])
# Direct SciPy8 kernels, degree2 and heteroscedastic smoothing, restricted indexed neighborhoods.
radial_positions=rng.uniform(-2,2,(60,2)); radial_values=np.sin(radial_positions[:,0])+radial_positions[:,1]**2; radial_queries=rng.uniform(-1,1,(4,2))
smoothing=np.linspace(.01,.04,60); neighborhoods=np.argsort(np.linalg.norm(radial_positions[None,:,:]-radial_queries[:,None,:],axis=2),axis=1)[:,:32]
radial={}
for kernel in ['thin_plate_spline','cubic','gaussian','linear','quintic','multiquadric','inverse_multiquadric','inverse_quadratic']:
    radial[kernel.replace('_','-')]=[float(RBFInterpolator(radial_positions[indices],radial_values[indices],kernel=kernel,degree=2,epsilon=.9,smoothing=smoothing[indices])(query[None,:])[0]) for query,indices in zip(radial_queries,neighborhoods)]
# CGAL directly computes full-data Sibson coordinates, SciPy supplies only index topology and seed.
natural_positions=np.r_[np.array([[-2,-2],[2,-2],[2,2],[-2,2]]),rng.uniform(-1.9,1.9,(124,2))]
natural_values=np.sin(natural_positions[:,0])+natural_positions[:,1]**2
natural_queries=np.r_[rng.uniform(-1.7,1.7,(6,2)),natural_positions[[7]],[[0,-2],[3,0]]]
triangulation=Delaunay(natural_positions); packed_triangles=np.column_stack([triangulation.simplices,triangulation.neighbors[:,[2,0,1]]]).astype(np.int64); packed_triangles[packed_triangles<0]=0xffffffff
seeds=triangulation.find_simplex(natural_queries).astype(np.int64); seeds[seeds<0]=0xffffffff
input_text=f'{len(natural_positions)} {len(natural_queries)}\n'+ '\n'.join(' '.join(map(str,row)) for row in np.column_stack([natural_positions,natural_values]))+'\n'+'\n'.join(' '.join(map(str,row)) for row in natural_queries)+'\n'
natural_expected=json.loads(subprocess.run(['/tmp/luma-interpolation-cgal'],input=input_text,text=True,capture_output=True,check=True).stdout)
# GSTools large joint conditioning and a separate independent conditioned-field ensemble.
simulation_positions=rng.uniform(-2,2,(40,2)); simulation_values=rng.normal(size=40); simulation_queries=rng.uniform(-1.8,1.8,(64,2))
krige=gs.krige.Krige(model,simulation_positions.T,simulation_values,mean=1.2,unbiased=False,cond_err=.07,pseudo_inv=False)
simulation_mean,simulation_variance=krige(simulation_queries.T,store=False)
small_krige=gs.krige.Krige(model,positions.T,values,mean=1.2,unbiased=False,cond_err=.07,pseudo_inv=False)
conditioned=gs.CondSRF(small_krige, seed=77, mode_no=2000)
ensemble=np.array([conditioned(queries[:2].T,seed=3000+replicate,store=False) for replicate in range(1024)])
# GSTools direct exact variogram-fit evidence; unlike skgstat defaults all three parameters fitted.
lags=np.linspace(.1,4,40); exact_semivariances=.2+1.5*(1-np.exp(-3*lags/2.2))
fit_model=gs.Exponential(dim=2); fit_model.fit_variogram(lags,exact_semivariances,nugget=True)
reference={'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'gstools':gs.__version__,'pykrige':pykrige.__version__,'CGAL':'6.0.1','Boost':'1.87.0'}, 'positions':positions.tolist(),'values':values.tolist(),'queries':queries.tolist(),'covariance':covariance,'kriging':kriging,
'wide':{'positions':wide_positions.tolist(),'values':wide_values.tolist(),'supports':supports.tolist(),'supportWeights':support_weights.tolist(),'sampleDesign':sample_design.tolist(),'queryDesign':query_design.tolist(),'expected':wide_expected},
'lmc':{'positions':multivariate_positions.tolist(),'observations':np.column_stack([multivariate_values,np.ones(30)*.05,variable_indices,active]).tolist(),'structures':structures,'supports':lmc_support.tolist(),'supportWeights':lmc_weights.tolist(),'expected':lmc_expected,'coefficients':lmc_coefficients},
'radial':{'positions':radial_positions.tolist(),'values':radial_values.tolist(),'queries':radial_queries.tolist(),'smoothing':smoothing.tolist(),'neighborhoods':neighborhoods.tolist(),'expected':radial},
'natural':{'positions':natural_positions.tolist(),'values':natural_values.tolist(),'queries':natural_queries.tolist(),'triangles':packed_triangles.tolist(),'seeds':seeds.tolist(),'expected':natural_expected},
'simulation':{'positions':simulation_positions.tolist(),'values':simulation_values.tolist(),'queries':simulation_queries.tolist(),'mean':simulation_mean.tolist(),'variance':simulation_variance.tolist(),'ensembleMean':ensemble.mean(0).tolist(),'ensembleCovariance':np.cov(ensemble.T).tolist()},
'fit':{'lags':lags.tolist(),'semivariances':exact_semivariances.tolist(),'parameters':[fit_model.nugget,fit_model.var,fit_model.len_scale*3]}}
output=Path(__file__).parents[1]/'gpu-spatial-analysis/spatial-interpolation/interpolation-advanced-reference.json'
output.write_text(json.dumps(reference,indent=2)+'\n')
print(output)
