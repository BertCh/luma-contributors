// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainDepressionHierarchyOperationProps} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy-operation';

export function makeHierarchyInput() {
  return {
    elevation: Float32Array.of(0, 3, 1, 5, 0, 3, 2),
    outlets: Uint32Array.of(1, 0, 0, 0, 0, 0, 1),
    stages: Float32Array.of(0, 2, 3, 5),
    validity: new Uint32Array(7).fill(1)
  };
}

export function makeHierarchyProps(nodeCapacity = 8): TerrainDepressionHierarchyOperationProps {
  return {
    id: 'hierarchy-test',
    grid: {width: 7, height: 1, affine: [2, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
    nodeCapacity,
    stageCount: 4,
    elevation: 'elevation',
    outlets: 'outlets',
    stages: 'stages',
    validity: 'validity',
    output: {
      nodes: 'nodes',
      levels: 'levels',
      cellNodes: 'cellNodes',
      curves: 'curves',
      status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
    }
  };
}
