// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainRuggedness({terrainRuggednessAlgorithm, edgeMode, output}).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'terrainRuggednessAlgorithm',
    label: 'terrainRuggednessAlgorithm',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['riley', 'wilson']
  },
  {
    id: 'output',
    label: 'output',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['terrainRuggednessIndex', 'roughness', 'topographicPositionIndex']
  },
  {
    id: 'edgeMode',
    label: 'edgeMode',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['nodata', 'extrapolate']
  }
];
