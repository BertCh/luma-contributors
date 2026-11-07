// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  RasterOperatorEntry,
  RasterOperatorTable
} from '../../../gpu-raster/computation-pack/types';

function makeProcessingEntry(
  filename: string,
  reduction: RasterOperatorEntry['descriptor']['reduction'],
  reason: string
): RasterOperatorEntry {
  return {
    family: 'terrain-processing',
    descriptor: {scope: 'global', reduction},
    variables: {elevation: 'elevation', values: 'measure', status: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'terrain cell', indexSpace: '2d'}],
    neighbourhood: 'path',
    guards: [],
    choices: [],
    formula: {formula: 'none', reason},
    sources: [
      `modules/graph/src/gpu-terrain/terrain-processing/${filename}.ts`,
      ...(filename === 'gpu-terrain-tiled-flow'
        ? ['modules/graph/src/gpu-terrain/terrain-processing/terrain-tiled-resident.ts']
        : [])
    ]
  };
}

/** Explicit semantics for bounded terrain roadmap contributors. */
export const TERRAIN_PROCESSING_OPERATORS: RasterOperatorTable = {
  GPUTerrainBreach: makeProcessingEntry(
    'gpu-terrain-breach',
    'argmin',
    'Minimize summed cut depth on a bounded step-layered D8 graph; merge cuts by minimum elevation and recheck strict descent after explicit optional burning.'
  ),
  GPUTerrainBranchingFlow: makeProcessingEntry(
    'gpu-terrain-branching-flow',
    'solve',
    'Finalize deterministic donor/receiver recurrences for transport, expected/minimum/maximum distance and fractional terminal membership; preserve invalid and unresolved rows.'
  ),
  GPUTerrainFlowFractions: makeProcessingEntry(
    'gpu-terrain-flow-fractions',
    'weighted-mean',
    'Materialize the existing D-infinity, Freeman MFD or Quinn MFD routing fractions and ground edge lengths for downstream consumers.'
  ),
  GPUTerrainDepressionHierarchy: makeProcessingEntry(
    'gpu-terrain-depression-hierarchy',
    'scan-sort',
    'Activate finite raster cells in elevation/ID order to construct a sublevel merge tree; sum descendant-owned stage area and volume exactly once per cell.'
  ),
  GPUTerrainCatchmentSummary: makeProcessingEntry(
    'gpu-terrain-catchment-summary',
    'sum',
    'Sort basin membership into contiguous ranges for area/elevation and cumulative hypsometry reductions; pointer-jump downstream profiles while preserving exact walk budgets and source-basin stream length.'
  ),
  GPUTerrainDEMCoregistration: makeProcessingEntry(
    'gpu-terrain-dem-coregistration',
    'select',
    'Fit the stable-ground median vertical difference, estimate residual NMAD, and propagate explicitly modeled exponential measurement covariance plus common bias error.'
  ),
  GPUTerrainTiledFlow: makeProcessingEntry(
    'gpu-terrain-tiled-flow',
    'sum',
    'Exchange owned tile routing into a resident global DAG, solve atomic donor dependencies and pointer-jump stream-chain ancestry; halos inject no runoff, fractional mass is conserved, and reach IDs cross seams and restart at branches.'
  )
};
