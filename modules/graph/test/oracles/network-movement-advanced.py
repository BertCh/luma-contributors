# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Direct ecosystem references plus independent full-covariance/expanded-state fixtures.

Pinned small packages: movingpandas==0.22.4 stonesoup==1.8 tslearn==0.6.4
similaritymeasures==1.3.0 access==1.1.9; numerical base numpy==2.2.6 networkx==3.5.
For memory/disk constrained runs reuse the pinned numerical/geospatial environment via PYTHONPATH.
No production TypeScript/shader code is imported. Run this file to regenerate the committed JSON.
"""
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from shapely.geometry import Point, Polygon
from tslearn.metrics import dtw_path_from_metric, lcss
from similaritymeasures import frechet_dist
from access.raam import raam
from pyproj import Transformer
from spopt.locate import PDispersion, LSCPB, PMedian
import pulp

versions = {name: importlib.metadata.version(name) for name in ['numpy', 'networkx', 'movingpandas', 'stonesoup', 'tslearn', 'similaritymeasures', 'access', 'shapely', 'geopandas', 'pandas', 'pyproj', 'spopt', 'pulp']}
for name, expected in {'numpy': '2.2.6', 'networkx': '3.5', 'movingpandas': '0.22.4', 'stonesoup': '1.8', 'tslearn': '0.6.4', 'similaritymeasures': '1.3.0', 'access': '1.1.9', 'spopt': '0.7.0', 'pulp': '3.3.0'}.items():
    assert versions[name] == expected, (name, versions[name])

positions = [[0, 1], [1, 1.2], [3, 1.5], [3.5, 1.8]]
times = [0, 0.5, 2, 3]
frame = gpd.GeoDataFrame({'geometry': [Point(*point) for point in positions]}, index=pd.to_datetime(times, unit='s'), crs='EPSG:3857')
trajectory = mpd.Trajectory(frame, 1)
smoothed = mpd.KalmanSmootherCV(trajectory).smooth(process_noise_std=np.sqrt(0.2), measurement_noise_std=np.sqrt(0.5))
kalman_movingpandas = {'times': times, 'positions': positions, 'parameters': [0.2, 0.5, 0, 10], 'smoothedPositions': [[point.x, point.y] for point in smoothed.df.geometry]}

# Direct MovingPandas linear interpolation/clip with a hole and Shapely boundary semantics.
polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], [[(1, 1), (3, 1), (3, 3), (1, 3)]])
clip_frame = gpd.GeoDataFrame({'geometry': [Point(-1, 2), Point(5, 2)]}, index=pd.to_datetime([0, 6], unit='s'), crs='EPSG:3857')
clipped = mpd.Trajectory(clip_frame, 2).clip(polygon)
clip_intervals = [[(piece.get_start_time() - clip_frame.index[0].to_pydatetime()).total_seconds(), (piece.get_end_time() - clip_frame.index[0].to_pydatetime()).total_seconds()] for piece in clipped]

# Full correlated-axis model independent NumPy matrix operations, column-major covariance export.
parameters = [0.3, 0.08, 0.2, 0.5, 0.15, 0.8, 1.5, -0.2, 2.0, 10]
noise = np.array([[parameters[0], parameters[1]], [parameters[1], parameters[2]]])
measurement = np.array([[parameters[3], parameters[4]], [parameters[4], parameters[5]]])
velocity = np.array([[parameters[6], parameters[7]], [parameters[7], parameters[8]]])
observation = np.column_stack((np.eye(2), np.zeros((2, 2))))
state = np.array([*positions[0], 0., 0.])
covariance = np.block([[measurement, np.zeros((2, 2))], [np.zeros((2, 2)), velocity]])
states, covariances = [], []
for row, point in enumerate(positions):
    if row:
        delta = times[row] - times[row - 1]
        transition = np.block([[np.eye(2), np.eye(2) * delta], [np.zeros((2, 2)), np.eye(2)]])
        process = np.block([[noise * delta**3 / 3, noise * delta**2 / 2], [noise * delta**2 / 2, noise * delta]])
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        gain = np.linalg.solve(observation @ covariance @ observation.T + measurement, observation @ covariance).T
        state += gain @ (point - observation @ state)
        correction = np.eye(4) - gain @ observation
        covariance = correction @ covariance @ correction.T + gain @ measurement @ gain.T
    states.append(state.copy()); covariances.append(covariance.copy())
smoothed_states = [state.copy() for state in states]
smoothed_covariances = [covariance.copy() for covariance in covariances]
for row in reversed(range(len(times) - 1)):
    delta = times[row + 1] - times[row]
    transition = np.block([[np.eye(2), np.eye(2) * delta], [np.zeros((2, 2)), np.eye(2)]])
    process = np.block([[noise * delta**3 / 3, noise * delta**2 / 2], [noise * delta**2 / 2, noise * delta]])
    predicted = transition @ covariances[row] @ transition.T + process
    gain = np.linalg.solve(predicted, transition @ covariances[row]).T
    smoothed_states[row] += gain @ (smoothed_states[row + 1] - transition @ states[row])
    smoothed_covariances[row] += gain @ (smoothed_covariances[row + 1] - predicted) @ gain.T
full_covariance = {'times': times, 'positions': positions, 'parameters': parameters, 'filteredState': np.array(states).ravel().tolist(), 'filteredCovariance': np.concatenate([value.ravel(order='F') for value in covariances]).tolist(), 'smoothedState': np.array(smoothed_states).ravel().tolist(), 'smoothedCovariance': np.concatenate([value.ravel(order='F') for value in smoothed_covariances]).tolist()}

tracks = [[[0., 0.], [1., 0.], [2., 1.]], [[0., 0.5], [1., 0.25], [2., 1.]], [[0., 0.], [0.5, 0.], [1., 0.], [2., 1.]], [[10., 0.], [11., 0.], [12., 1.]]]
similarity = []
for first in range(len(tracks)):
    for second in range(len(tracks)):
        a, b = np.array(tracks[first]), np.array(tracks[second])
        _, distance = dtw_path_from_metric(a, b, metric='euclidean')
        similarity.append({'pair': [first, second], 'dtw': float(distance), 'frechet': float(frechet_dist(a, b)), 'lcss': float(lcss(a, b, eps=0.5))})

raam_demand = pd.DataFrame({'demand': [10, 6]}, index=[0, 1])
raam_supply = pd.DataFrame({'supply': [8, 8]}, index=[0, 1])
raam_costs = pd.DataFrame({'origin': [0, 0, 1, 1], 'dest': [0, 1, 0, 1], 'cost': [0., 2., 2., 0.]})
raam_result = raam(raam_demand, raam_supply, raam_costs, tau=10, rho=1, max_cycles=150)

# Virtual split-edge graph equivalent to proportional costs in pgRouting withPoints, no database dependency.
split = nx.DiGraph()
split.add_weighted_edges_from([(0, 1, 1.), (1, 2, 3.), (2, 3, 2.), (3, 4, 2.), (1, 4, 10.)])
point_routing = {'edgeCosts': [4., 4., 10.], 'turns': [[0, 1], [0, 2]], 'queries': [[0, 1]], 'fractions': [[0.25, 0.5]], 'cost': nx.shortest_path_length(split, 1, 3, weight='weight')}

# Independent time-expanded directed graph, one state per physical vertex/integer time.
time_graph = nx.DiGraph()
for vertex in range(3):
    for time in range(16):
        if time < 15: time_graph.add_edge((vertex, time), (vertex, time + 1), weight=1)
        for source, destination, duration in [(0, 1, 2 + time), (1, 2, 1)]:
            if vertex == source and time + duration <= 15: time_graph.add_edge((source, time), (destination, time + duration), weight=duration)
time_routing = {'departure': 2, 'arrival': min(time for time in range(16) if nx.has_path(time_graph, (0, 2), (2, time)))}

transformer = Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
origin = [-73.9857, 40.7484]
longitude_latitude = [origin, [-73.9856, 40.7485], [-73.9858, 40.7483]]
anchor = np.array(transformer.transform(*origin, 0))
longitude, latitude = np.radians(origin)
rotation = np.array([[-np.sin(longitude), np.cos(longitude), 0], [-np.sin(latitude)*np.cos(longitude), -np.sin(latitude)*np.sin(longitude), np.cos(latitude)]])
projection = {'origin': origin, 'longitudeLatitude': longitude_latitude, 'positions': [(rotation @ (np.array(transformer.transform(*point, 0)) - anchor)).tolist() for point in longitude_latitude]}

# Direct spopt/CBC bounded integer programs; backup uses unit weights to match LSCPB's count objective.
solver = pulp.PULP_CBC_CMD(msg=False)
dispersion_costs = [[0, 1, 4], [1, 0, 3], [4, 3, 0]]
backup_costs = [[0, 1, 5], [5, 0, 1], [1, 5, 0]]
median_costs = [[1, 2], [1, 100]]
dispersion = PDispersion.from_cost_matrix(np.array(dispersion_costs), 2).solve(solver)
backup = LSCPB.from_cost_matrix(np.array(backup_costs), 1, solver).solve(solver)
median = PMedian.from_cost_matrix(np.array(median_costs), np.array([1, 1]), 2, facility_capacities=np.array([1, 1])).solve(solver)
facility = [
    {'operation': 'p-dispersion', 'facilityCount': 3, 'selectCount': 2, 'costs': dispersion_costs, 'demand': [1, 1, 1], 'capacities': [1, 1, 1], 'threshold': 1, 'objective': dispersion.problem.objective.value()},
    {'operation': 'backup-coverage', 'facilityCount': 3, 'selectCount': None, 'costs': backup_costs, 'demand': [1, 1, 1], 'capacities': [1, 1, 1], 'threshold': 1, 'objective': backup.problem.objective.value()},
    {'operation': 'capacitated-p-median', 'facilityCount': 2, 'selectCount': 2, 'costs': median_costs, 'demand': [1, 1], 'capacities': [1, 1], 'threshold': 1, 'objective': median.problem.objective.value()}
]

fixture = {'facility': facility, 'provenance': {'versions': versions, 'dtwContract': 'tslearn dtw_path_from_metric(metric=euclidean): sum Euclidean costs, not tslearn default sqrt(sum squared costs)', 'kalman': 'direct MovingPandas KalmanSmootherCV; initial velocity variance zero', 'temporal': 'direct MovingPandas clip; no MobilityDB server installed', 'routing': 'independent NetworkX virtual split/time-expanded graphs; no pgRouting/Valhalla server installed', 'raam': 'direct access.raam.raam at equilibrium; production recomputes final costs rather than upstream stale last-cycle costs'}, 'movingpandasKalman': kalman_movingpandas, 'fullCovariance': full_covariance, 'clipIntervals': clip_intervals, 'tracks': tracks, 'similarity': similarity, 'raam': {'costs': [0, 2, 2, 0], 'demand': [10, 6], 'supply': [8, 8], 'scores': raam_result.tolist()}, 'pointRouting': point_routing, 'timeRouting': time_routing, 'projection': projection}
path = Path(__file__).with_name('fixtures') / 'network-movement-advanced.json'
path.write_text(json.dumps(fixture, indent=2) + '\n')
print(path)
