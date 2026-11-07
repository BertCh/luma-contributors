# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy 2.2.6 dense IV sandwich references; no runtime dependency.

Run with NumPy==2.2.6. Direct linear algebra reference, not a claimed upstream spreg call.
Semantic target: spreg 4dbc809242d9fe8f5fca0257c7dc9c957fc1ce36 white/HAC covariance.
"""
import json
from pathlib import Path
import numpy as np

assert np.__version__ == '2.2.6'
rows = 12
indices = np.arange(rows, dtype=float)
predictor = np.array([1, 3, 2, 5, 4, 8, 6, 9, 7, 11, 10, 13.])
weights = np.zeros((rows, rows))
for row in range(rows):
    weights[row, (row + 1) % rows] = .7
    weights[row, (row + 3) % rows] = .3
response = np.linalg.solve(np.eye(rows) - .3 * weights,
                          2 + 1.2 * predictor + np.sin(indices * 1.7) * (1 + indices / 8))
design = np.column_stack([np.ones(rows), predictor, weights @ response])
instruments = np.column_stack([np.ones(rows), predictor, weights @ predictor])
projection = instruments @ np.linalg.solve(instruments.T @ instruments, instruments.T)
bread = np.linalg.inv(design.T @ projection @ design)
coefficients = bread @ design.T @ projection @ response
residuals = response - design @ coefficients
influence = bread @ design.T @ projection
scores = influence * residuals
kernel = np.eye(rows)
for row in range(rows):
    kernel[row, (row + 1) % rows] = .2
    kernel[row, (row - 1) % rows] = .2
covariances = {'HC0': scores @ scores.T,
               'HC1': scores @ scores.T * rows / (rows - design.shape[1]),
               'HAC': scores @ kernel @ scores.T}
def csr(matrix):
    offsets = [0]
    neighbors = []
    values = []
    for row in matrix:
        selected = np.flatnonzero(row)
        neighbors.extend(selected.tolist())
        values.extend(row[selected].tolist())
        offsets.append(len(neighbors))
    return {'offsets': offsets, 'neighbors': neighbors, 'weights': values}
result = {'source': 'NumPy 2.2.6 independent dense projection and sandwich',
          'predictors': predictor.tolist(), 'response': response.tolist(),
          'design': design[:, 1:].flatten().tolist(),
          'instruments': instruments[:, 1:].flatten().tolist(),
          'coefficients': coefficients.tolist(), 'residuals': residuals.tolist(),
          'weights': csr(weights), 'kernel': csr(kernel),
          'covariances': {name: value.flatten().tolist() for name, value in covariances.items()}}
Path(__file__).with_name('fixtures').joinpath('statistics-coverage.json').write_text(json.dumps(result, indent=2) + '\n')

# Independent likelihood and block-GLS references for the expanded bounded estimators.
import scipy
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtr
assert scipy.__version__ == '1.15.3'
observations = np.arange(40, dtype=float)
probit_predictors = np.column_stack([np.sin(observations*.7), np.cos(observations*.37)])
probit_design = np.column_stack([np.ones(40), probit_predictors])
probit_response = ((observations*17 % 41) / 41 < ndtr(.3 + probit_predictors @ np.array([.6,-.4]))).astype(float)
def probit_likelihood(coefficients):
    probabilities = ndtr(probit_design @ coefficients)
    return -(probit_response*np.log(probabilities)+(1-probit_response)*np.log1p(-probabilities)).sum()
fit = minimize(probit_likelihood,np.zeros(3),method='BFGS',tol=1e-10)
linear = probit_design @ fit.x
probabilities = ndtr(linear)
density = np.exp(-linear*linear/2)/np.sqrt(2*np.pi)
information = probit_design.T @ (probit_design*(density*density/(probabilities*(1-probabilities)))[:,None])
result['probit'] = {'predictors':probit_predictors.flatten().tolist(),'response':probit_response.tolist(),'coefficients':fit.x.tolist(),'standardErrors':np.sqrt(np.diag(np.linalg.inv(information))).tolist(),'fitted':probabilities.tolist(),'effects':(density.mean()*fit.x[1:]).tolist(),'logLikelihood':-fit.fun}

sur_predictors = np.stack([np.sin(observations*.21)+observations/20,np.cos(observations*.43)+observations/18])
noise = np.sin(observations*2.1)
sur_response = np.stack([1+2*sur_predictors[0]+noise, -2+.5*sur_predictors[1]+.7*noise+.3*np.cos(observations*1.7)])
sur_designs = [np.column_stack([np.ones(40),values]) for values in sur_predictors]
initial = [np.linalg.lstsq(design,response,rcond=None)[0] for design,response in zip(sur_designs,sur_response)]
residuals = np.column_stack([response-design@coefficients for design,response,coefficients in zip(sur_designs,sur_response,initial)])
sigma = residuals.T @ residuals / 40
block_design = np.zeros((80,4)); block_design[:40,:2]=sur_designs[0];block_design[40:,2:]=sur_designs[1]
precision = np.kron(np.linalg.inv(sigma),np.eye(40))
normal = block_design.T @ precision @ block_design
coefficients = np.linalg.solve(normal,block_design.T @ precision @ sur_response.flatten())
result['sur']={'predictors':sur_predictors.flatten().tolist(),'responses':sur_response.flatten().tolist(),'coefficients':coefficients.tolist(),'covariance':np.linalg.inv(normal).flatten().tolist(),'residualCovariance':sigma.flatten().tolist(),'residuals':(sur_response.flatten()-block_design@coefficients).tolist()}

regimes = np.arange(40)%2
regime_predictor = np.sin(observations*.3)+observations/20
regime_response = 1+2*regimes+(1+.4*regimes)*regime_predictor+np.sin(observations*1.9)
restricted_design = np.column_stack([regimes==0,regimes==1,regime_predictor])
full_design = np.column_stack([regimes==0,regimes==1,regime_predictor*(regimes==0),regime_predictor*(regimes==1)])
restricted_coefficient = np.linalg.lstsq(restricted_design,regime_response,rcond=None)[0]
full_coefficient = np.linalg.lstsq(full_design,regime_response,rcond=None)[0]
restricted_squares = np.sum((regime_response-restricted_design@restricted_coefficient)**2)
full_squares = np.sum((regime_response-full_design@full_coefficient)**2)
result['regimes']={'predictors':regime_predictor.tolist(),'response':regime_response.tolist(),'regimes':regimes.tolist(),'coefficients':restricted_coefficient.tolist(),'covariance':(np.linalg.inv(restricted_design.T@restricted_design)*restricted_squares/37).flatten().tolist(),'chow':[(restricted_squares-full_squares)/(full_squares/36),1,36]}

entities=6;periods=8
panel_weights=np.zeros((entities,entities))
for row in range(entities):panel_weights[row,(row+1)%entities]=.6;panel_weights[row,(row+2)%entities]=.4
panel_indices=np.arange(entities*periods,dtype=float)
panel_predictor=(np.sin(panel_indices*.6)+panel_indices/20).reshape(periods,entities)
panel_innovation=(np.sin(panel_indices*1.3)+.5*np.cos(panel_indices*.9)).reshape(periods,entities)
panel_error=np.linalg.solve(np.eye(entities)-.25*panel_weights,panel_innovation.T).T
panel_response=np.arange(entities)[None,:]*2+1.3*panel_predictor+panel_error
within_predictor=panel_predictor-panel_predictor.mean(axis=0)
within_response=panel_response-panel_response.mean(axis=0)
def panel_objective(parameter):
    transform=np.eye(entities)-parameter*panel_weights
    filtered_predictor=(transform@within_predictor.T).T.flatten()
    filtered_response=(transform@within_response.T).T.flatten()
    coefficient=filtered_predictor@filtered_response/(filtered_predictor@filtered_predictor)
    squares=np.sum((filtered_response-coefficient*filtered_predictor)**2)
    return entities*periods/2*np.log(squares/(entities*periods))-periods*np.linalg.slogdet(transform)[1]
fit=minimize_scalar(panel_objective,bounds=(-.9,.9),method='bounded',options={'xatol':1e-12})
transform=np.eye(entities)-fit.x*panel_weights
filtered_predictor=(transform@within_predictor.T).T.flatten();filtered_response=(transform@within_response.T).T.flatten()
coefficient=filtered_predictor@filtered_response/(filtered_predictor@filtered_predictor)
sigma=np.sum((filtered_response-coefficient*filtered_predictor)**2)/(entities*periods)
result['panelError']={'weights':csr(panel_weights),'periods':periods,'predictors':panel_predictor.flatten().tolist(),'response':panel_response.flatten().tolist(),'coefficient':coefficient,'parameter':fit.x,'variance':sigma,'logLikelihood':-fit.fun-entities*periods/2*(np.log(2*np.pi)+1)}

flows=np.array([[12.,4,2],[5,9,3],[1,6,10]])
cost=np.array([[0.,1,3],[2,0,1],[4,2,0]])
coefficients=[-1.,-.5,0.,.5]
interaction={}
for constraint in ['unconstrained','origin','destination','doubly']:
    candidates=[]
    for coefficient in coefficients:
        predictions=np.exp(coefficient*cost)
        if constraint=='unconstrained':predictions*=flows.sum()/predictions.sum()
        else:
            for iteration in range(10000):
                if constraint!='destination':predictions*= (flows.sum(axis=1)/predictions.sum(axis=1))[:,None]
                if constraint!='origin':predictions*= (flows.sum(axis=0)/predictions.sum(axis=0))[None,:]
                error=0
                if constraint!='destination':error=max(error,np.max(np.abs(flows.sum(axis=1)-predictions.sum(axis=1))/flows.sum(axis=1)))
                if constraint!='origin':error=max(error,np.max(np.abs(flows.sum(axis=0)-predictions.sum(axis=0))/flows.sum(axis=0)))
                if error<1e-12:break
        deviance=2*np.sum(flows*np.log(flows/predictions)-flows+predictions)
        candidates.append((deviance,coefficient,predictions))
    deviance,coefficient,predictions=min(candidates,key=lambda candidate:candidate[0])
    interaction[constraint]={'coefficient':coefficient,'deviance':deviance,'predictions':predictions.flatten().tolist()}
result['interaction']={'flows':flows.flatten().tolist(),'cost':cost.flatten().tolist(),'coefficients':coefficients,'models':interaction}
Path(__file__).with_name('fixtures').joinpath('statistics-coverage.json').write_text(json.dumps(result, indent=2) + '\n')
