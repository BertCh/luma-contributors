// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Parameters of the flagship demos, kept free of luma.gl imports so pages can render the controls
// in prerendered HTML. Each demo module re-exports its list as `parameters`.

import type {GpuSceneParameter} from '../types';

export const KERNEL_DENSITY_PARAMETERS: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'Bandwidth', min: 1.5, max: 12, step: 0.25, default: 6, unit: '%'},
  {
    id: 'kernel',
    label: 'Kernel',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['gaussian', 'quartic', 'epanechnikov']
  }
];

export const LOCAL_MORANS_I_PARAMETERS: GpuSceneParameter[] = [
  {id: 'radius', label: 'Neighbor radius', min: 2, max: 9, step: 0.25, default: 5.5, unit: '%'},
  {id: 'significance', label: 'Significance', min: 0.001, max: 0.3, step: 0.001, default: 0.05},
  {id: 'noise', label: 'Simulated noise', min: 0, max: 200, step: 5, default: 0, unit: 'mm'}
];

export const DBSCAN_PARAMETERS: GpuSceneParameter[] = [
  {id: 'epsilon', label: 'Epsilon', min: 0.8, max: 5, step: 0.1, default: 2.6, unit: '%'},
  {
    id: 'minimumSamples',
    renamedFrom: 'minimumPoints',
    label: 'Minimum samples',
    min: 2,
    max: 20,
    step: 1,
    default: 6
  }
];

export const K_MEANS_PARAMETERS: GpuSceneParameter[] = [
  {id: 'iterations', label: 'Iterations', min: 1, max: 12, step: 1, default: 12},
  {id: 'seed', label: 'Seed', min: 1, max: 24, step: 1, default: 1}
];

export const IDW_PARAMETERS: GpuSceneParameter[] = [
  {id: 'power', label: 'Power', min: 0.5, max: 6, step: 0.1, default: 2},
  {id: 'searchRadius', label: 'Search radius', min: 10, max: 100, step: 5, default: 60, unit: '%'}
];

export const DELAUNAY_VORONOI_PARAMETERS: GpuSceneParameter[] = [
  {id: 'sites', label: 'Data: sites', min: 16, max: 96, step: 16, default: 96},
  {id: 'wander', label: 'Simulated site drift', min: 0, max: 8, step: 0.25, default: 0, unit: '%'}
];

export const TERRAIN_HYDROLOGY_PARAMETERS: GpuSceneParameter[] = [
  {id: 'azimuthDegrees', label: 'Sun azimuth', min: 0, max: 359, step: 1, default: 315, unit: '°'},
  {id: 'altitudeDegrees', label: 'Sun altitude', min: 5, max: 85, step: 1, default: 38, unit: '°'},
  {id: 'zFactor', label: 'Slope exaggeration', min: 0.5, max: 6, step: 0.1, default: 1.5},
  {id: 'streamThreshold', label: 'Stream threshold', min: 10, max: 200, step: 5, default: 80}
];

export const NETWORK_ISOCHRONES_PARAMETERS: GpuSceneParameter[] = [
  {id: 'costLimit', label: 'Travel budget', min: 0.05, max: 1.25, step: 0.05, default: 0.7},
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 2, step: 0.1, default: 0.9}
];

export function validateParameterRenames(
  parameters: readonly GpuSceneParameter[],
  raw?: Readonly<Record<string, number>>
): void {
  if (!raw) return;
  for (const {id, renamedFrom} of parameters) {
    if (renamedFrom !== undefined && Object.hasOwn(raw, renamedFrom)) {
      throw new Error(`use ${id} instead of ${renamedFrom}`);
    }
  }
}

/** The initial value of every parameter, overridden by any `initial` values the page passes. */
export function readParameterValues(
  parameters: readonly GpuSceneParameter[],
  initial?: Record<string, number>
): Record<string, number> {
  validateParameterRenames(parameters, initial);
  const values: Record<string, number> = {};
  for (const parameter of parameters) {
    values[parameter.id] = initial?.[parameter.id] ?? parameter.default;
  }
  return values;
}

export const SHORTEST_PATH_PARAMETERS: GpuSceneParameter[] = [
  {id: 'congestion', label: 'Data: congestion', min: 0, max: 3, step: 0.1, default: 1.5},
  {id: 'originPosition', label: 'Origin', min: 5, max: 95, step: 1, default: 12, unit: '%'}
];

export const HOT_SPOTS_PARAMETERS: GpuSceneParameter[] = [
  {id: 'radius', label: 'Neighbor radius', min: 2, max: 9, step: 0.25, default: 5, unit: '%'},
  {
    id: 'day',
    label: 'Observation day',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    options: ['Sep 26', 'Sep 27']
  }
];
