# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent numerical fixtures. Run with numpy==2.2.6; no production imports."""
import json
from pathlib import Path
import numpy as np

assert np.__version__ == '2.2.6'
positions = np.array([[1024 + horizontal, -2048 + vertical, 64 + .3 * horizontal - .2 * vertical]
                      for vertical in [-1, 0, 1] for horizontal in [-1, 0, 1]], dtype=np.float32).astype(float)
eigenvalues = []
normals = []
covariances = []
for index in range(len(positions)):
    neighbors = np.delete(positions, index, axis=0)
    centered = neighbors - neighbors.mean(axis=0)
    covariance = centered.T @ centered / len(neighbors)
    covariances.append(covariance[np.triu_indices(3)].tolist())
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    if normal[2] < 0:
        normal *= -1
    normals.append(normal.tolist())
    eigenvalues.append(values[::-1].tolist())
source = np.array([[0,0,0],[2,0,0],[0,3,0],[0,0,4],[2,2,2]], dtype=float)
angle = .06
rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
translation = np.array([.05,-.03,.02])
target = source @ rotation.T + translation
first = source - source.mean(axis=0)
second = target - target.mean(axis=0)
left, _, right = np.linalg.svd(first.T @ second)
solved = right.T @ left.T
transform = np.eye(4)
transform[:3,:3] = solved
transform[:3,3] = target.mean(axis=0) - solved @ source.mean(axis=0)
fixture = dict(generator='NumPy 2.2.6 eigh and SVD Kabsch', positions=positions.ravel().tolist(),
               eigenvalues=eigenvalues, normals=normals, covariances=covariances, source=source.ravel().tolist(),
               target=target.ravel().tolist(), transform=transform.T.ravel().tolist())
Path(__file__).with_name('pointcloud.json').write_text(json.dumps(fixture, indent=2) + '\n')
