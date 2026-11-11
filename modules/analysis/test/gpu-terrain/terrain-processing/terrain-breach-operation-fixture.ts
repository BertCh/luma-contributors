// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainBreachOperationProps} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach-operation';

export function makeBreachInput() {
  return {
    elevation: Float32Array.of(0, 4, 1, 5, 0, 4, 2, 4, 0),
    pitCells: Uint32Array.of(2, 6),
    burnDepth: Float32Array.of(0, 0.25, 0, 0, 0, 0, 0, 0.5, 0),
    validity: new Uint32Array(9).fill(1)
  };
}

export function makeBreachProps(): TerrainBreachOperationProps {
  return {
    id: 'breach-test',
    grid: {width: 9, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
    maximumSteps: 3,
    maximumCutDepth: 5,
    elevationDecrement: 0.25,
    pitCount: 2,
    elevation: 'elevation',
    pitCells: 'pitCells',
    burnDepth: 'burnDepth',
    validity: 'validity',
    output: {
      elevation: 'edited',
      cutDepth: 'cutDepth',
      modificationMask: 'modificationMask',
      pathCells: 'pathCells',
      pathLengths: 'pathLengths',
      pitStatus: 'pitStatus',
      unresolvedCount: 'unresolvedCount'
    }
  };
}
