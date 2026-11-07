// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

export const TERRAIN_EXPANSION_OPERATORS: RasterOperatorTable = {
  GPUTerrainFlowDistance: {
    family: 'expansion',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {
      flowDirections: 'category',
      streams: 'category',
      values: 'measure',
      validity: 'category',
      terminalCells: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'path',
    guards: [],
    choices: [],
    formula: {
      id: 'gpu-terrain.GPUTerrainFlowDistance.rule',
      derivation: 'curated',
      tex: String.raw`D(c)=\sum_{(u,v)\in path(c)}\|A(v-u)\|_2`,
      ascii:
        'Follow each D8 receiver and sum affine transition lengths until a stream or terminal cell; cycles and truncated paths stay unresolved.',
      symbols: []
    },
    sources: ['modules/graph/src/gpu-terrain/hydrology/gpu-terrain-flow-distance.ts']
  },
  GPUTerrainStreamNetwork: {
    family: 'expansion',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {
      flowDirections: 'category',
      streams: 'category',
      sourceCells: 'category',
      targetCells: 'category',
      reachLengths: 'measure',
      vertexOffsets: 'measure',
      reachCoordinates: 'measure',
      reachCount: 'measure',
      vertexCount: 'measure',
      overflow: 'category',
      unresolvedCount: 'measure'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'path',
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Trace valid stream reaches between headwaters, confluences and outlets, retaining original raster identities and affine cell-center vertices.'
    },
    sources: ['modules/graph/src/gpu-terrain/hydrology/gpu-terrain-stream-network.ts']
  }
};
