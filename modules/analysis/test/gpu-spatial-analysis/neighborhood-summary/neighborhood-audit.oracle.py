# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/neighborhood-summary/neighborhood-audit.oracle.py
"""Independent NumPy population weighted moments, unweighted median, categorical
bincount frequencies. Numeric missingness filters numeric members before weights
are normalized; categorical normalization uses its own membership. The focal
weight is never normalized. Mode ties prefer the smallest category; entropy is
Shannon natural-log entropy. Zero CSR weights, self and invalid IDs are excluded.
"""
import json
import numpy as np

values = np.array([10,2,6,np.nan,8,4,-2,99], dtype=float)
categories = np.array([5,5,3,3,7,7,3,9])
mask = np.array([1,1,1,1,1,1,1,0], dtype=bool)
rows = [[(0,9),(1,1),(2,3),(3,4),(7,2),(99,1)],[(2,1),(4,1)],[],[(1,0),(4,2),(5,2)],[(1,1),(2,1)],[(6,1)],[(3,1)],[(0,1)]]
for focal, standardized in [(False,False),(True,True)]:
    table, modes, entropy = [], [], []
    for index, entries in enumerate(rows):
        if not mask[index]:
            table.extend([0,*([float('nan')]*7)])
            modes.append(0xffffffff)
            entropy.append(float('nan'))
            continue
        members = [(row,weight) for row,weight in entries if row < len(values) and row != index and weight > 0 and mask[row]]
        numeric = [(row,weight) for row,weight in members if np.isfinite(values[row])]
        def normalize(selected, numeric):
            ids = np.array([row for row,_ in selected], dtype=int)
            mass = np.array([weight for _,weight in selected], dtype=float)
            if standardized and mass.sum() > 0:
                mass /= mass.sum()
            if focal and (not numeric or np.isfinite(values[index])):
                ids = np.r_[index, ids]
                mass = np.r_[2, mass]
            return ids, mass
        ids, mass = normalize(numeric, True)
        data = values[ids]
        if len(data):
            mean = np.average(data, weights=mass)
            table.extend([len(data), mass.sum(), np.dot(data,mass),mean,np.min(data),np.max(data),np.sqrt(np.average((data-mean)**2, weights=mass)),np.median(data)])
        else:
            table.extend([0,0,0,*([float('nan')]*5)])
        ids, mass = normalize(members, False)
        if len(ids):
            distinct, inverse = np.unique(categories[ids], return_inverse=True)
            frequency = np.bincount(inverse, weights=mass)
            modes.append(int(distinct[np.argmax(frequency)]))
            probability = frequency/frequency.sum()
            entropy.append(float(-np.sum(probability*np.log(probability))))
        else:
            modes.append(0xffffffff)
            entropy.append(float('nan'))
    print('focal', focal, 'standardized', standardized, json.dumps(dict(table=table,modes=modes,entropy=entropy,overflow=[0])))
