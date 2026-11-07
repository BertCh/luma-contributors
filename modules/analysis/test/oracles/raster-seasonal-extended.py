# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent selected-model reference: NumPy 2.2.6 QR/SVD least squares, not full BFAST inference.
Dummy seasonal regression, conditional BIC level breaks, fixed-history residual monitoring.
Run with NumPy 2.2.6. Production WGSL and its cyclic solver are never imported.
"""
import json
from pathlib import Path
import numpy as np
assert np.__version__ == '2.2.6'
rng=np.random.default_rng(43089)
cases=[]
for name, seasonality, jump, policy, monitoring in [
    ('dummy-break','dummy',3.,'omit',None),
    ('dummy-negative','dummy',-2.,'omit',None),
    ('dummy-interpolation','dummy',3.,'interpolate',None),
    ('harmonic-monitor','harmonic',0.,'omit',dict(historyLength=32,threshold=4.,minimumConsecutive=2)),
    ('dummy-monitor','dummy',0.,'interpolate',dict(historyLength=32,threshold=4.,minimumConsecutive=2)),
]:
    slices=48; period=4; minimum=12 if monitoring else 16
    time=np.arange(slices)
    seasonal=np.array([0,1.2,-.7,.5])[time%period] if seasonality=='dummy' else .8*np.cos(2*np.pi*time/period)-.4*np.sin(2*np.pi*time/period)
    values=(2+.06*time+seasonal+(time>=24)*jump+rng.normal(0,.025,slices)).astype(np.float32).astype(float)
    validity=np.ones(slices,dtype=bool)
    validity[[5,6,19]]=False
    if monitoring:
        values[40:]+=2
        validity[38]=False
    history=monitoring['historyLength'] if monitoring else slices
    training=validity[:history].copy(); response=values[:history].copy()
    if policy=='interpolate':
        indices=np.flatnonzero(training)
        selected=np.arange(indices[0],indices[-1]+1)
        response[selected]=np.interp(selected,indices,response[indices]); training[selected]=True
    seasonal_columns=np.eye(period)[time%period,1:] if seasonality=='dummy' else np.column_stack([np.cos(2*np.pi*time/period),np.sin(2*np.pi*time/period)])
    design=np.column_stack([np.ones(slices),time/slices,seasonal_columns])
    matrix=design[:history][training]; target=response[training]
    coefficients=np.linalg.lstsq(matrix,target,rcond=None)[0]
    residual=np.sum((matrix@coefficients-target)**2)
    energy=np.dot(target,target); floor=max(1e-20,energy*1e-12); count=len(target); baseline_columns=matrix.shape[1]
    baseline_score=count*np.log(max(residual,floor)/count)+baseline_columns*np.log(count)
    chosen=np.r_[coefficients,0.]; selected=0xffffffff; gain=0.; selected_columns=baseline_columns
    for boundary in range(minimum,history-minimum+1):
        if training[:boundary].sum()<minimum or training[boundary:].sum()<minimum: continue
        candidate=np.column_stack([matrix,(time[:history]>=boundary)[training]])
        fit=np.linalg.lstsq(candidate,target,rcond=None)[0]
        error=np.sum((candidate@fit-target)**2)
        improvement=baseline_score-(count*np.log(max(error,floor)/count)+(baseline_columns+2)*np.log(count))
        if improvement>gain: chosen=fit; selected=boundary; gain=improvement; residual=error; selected_columns=baseline_columns+1
    prediction=design@chosen[:-1]+(time>=selected)*chosen[-1]
    scale=np.sqrt(max(residual,floor)/max(1,count-selected_columns))
    alarm=-1; maximum_score=0.; run=0; previous_sign=0
    if monitoring:
        for index in range(history,slices):
            if not validity[index]: run=0; previous_sign=0; continue
            score=(values[index]-prediction[index])/scale; maximum_score=max(maximum_score,abs(score))
            if abs(score)>=monitoring['threshold']:
                direction=np.sign(score); run=run+1 if direction==previous_sign else 1; previous_sign=direction
                if alarm<0 and run>=monitoring['minimumConsecutive']: alarm=index+1-monitoring['minimumConsecutive']
            else: run=0; previous_sign=0
    cases.append(dict(name=name,seasonality=seasonality,sliceCount=slices,period=period,minimumSegmentLength=minimum,missingPolicy=policy,monitoring=monitoring,values=values.tolist(),validity=validity.astype(int).tolist(),coefficients=chosen.tolist(),breakIndex=int(selected),improvement=float(gain),prediction=prediction.tolist(),support=int(count),scale=float(scale),alarm=alarm,maximumScore=float(maximum_score)))
path=Path(__file__).parent/'fixtures/raster-seasonal-extended.json'
path.write_text(json.dumps(dict(versions={'numpy':np.__version__},cases=cases),indent=2)+'\n')
print(f'Wrote {len(cases)} direct NumPy regression/BIC/monitor cases')
