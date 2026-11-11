// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import type {RasterFillHolesOperationProps} from './gpu-raster-fill-holes-operation';
import {
  validateRasterMorphologyOptions,
  visitRasterMorphologyNeighbors
} from './raster-morphology-operation-utils';

export type CPURasterFillHolesOptions = Pick<
  RasterFillHolesOperationProps,
  'grid' | 'connectivity'
>;
export type CPURasterFillHolesResult = {
  values: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
};

/**
 * Floods zero-valued background from the boundary and fills every unreached component. All
 * nonzero input values are foreground and become 1, as in the contributor's reconstruction.
 * Only grid.width * grid.height input rows are read; any surplus rows are ignored.
 *
 * What differs from the GPU contributor:
 * - Boundary flooding always finishes; maximumIterations and the GPU node budget are ignored.
 *   converged is always 1 and iterationLimitReached is always 0.
 * - iterationCount is maximum shortest boundary-background path length plus one verifying round
 *   (1 when no background propagates), so it can exceed the GPU's round budget.
 *
 * @throws Error naming input for an inconsistent length, or an invalid option.
 */
export function computeRasterFillHolesOnCPU(
  input: Uint32Array,
  options: CPURasterFillHolesOptions
): CPURasterFillHolesResult {
  const id = 'computeRasterFillHolesOnCPU';
  validateRasterMorphologyOptions(id, options);
  const {width, height} = options.grid;
  const length = width * height;
  if (input.length < length) throw new Error(`${id} input must have at least one row per cell`);
  const distances = new Float64Array(length).fill(-1);
  const queue: number[] = [];
  for (let cell = 0; cell < length; cell++) {
    const row = Math.floor(cell / width);
    const column = cell % width;
    if (
      input[cell] === 0 &&
      (row === 0 || column === 0 || row === height - 1 || column === width - 1)
    ) {
      distances[cell] = 0;
      queue.push(cell);
    }
  }
  let maximumDistance = 0;
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const cell = queue[cursor];
    visitRasterMorphologyNeighbors(cell, width, height, options.connectivity ?? 4, neighbor => {
      if (input[neighbor] !== 0 || distances[neighbor] !== -1) return;
      distances[neighbor] = distances[cell] + 1;
      maximumDistance = Math.max(maximumDistance, distances[neighbor]);
      queue.push(neighbor);
    });
  }
  return {
    values: Uint32Array.from(distances, distance => (distance < 0 ? 1 : 0)),
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    iterationCount: Uint32Array.of(maximumDistance + 1)
  };
}

/** CPU boundary flood, without the GPU command-node or convergence budget. */
export class CPURasterFillHolesOperation extends GPUCPUOperation {
  constructor(props: RasterFillHolesOperationProps) {
    const id = props.id ?? 'raster-fill-holes';
    validateRasterMorphologyOptions(id, props);
    super({
      id,
      inputs: {input: props.input},
      outputs: {
        [props.output.values]: gpuGraph.vector('uint32', {
          length: props.grid.width * props.grid.height
        }),
        ...Object.fromEntries(
          Object.values(props.output.status)
            .filter(value => value !== undefined)
            .map(value => [value, gpuGraph.vector('uint32', {length: 1})])
        )
      },
      execute({inputs}) {
        const result = computeRasterFillHolesOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'input'),
          props
        );
        return {
          [props.output.values]: result.values,
          [props.output.status.converged]: result.converged,
          ...(props.output.status.iterationCount !== undefined
            ? {[props.output.status.iterationCount]: result.iterationCount}
            : {}),
          ...(props.output.status.iterationLimitReached !== undefined
            ? {[props.output.status.iterationLimitReached]: result.iterationLimitReached}
            : {})
        };
      }
    });
  }
}
