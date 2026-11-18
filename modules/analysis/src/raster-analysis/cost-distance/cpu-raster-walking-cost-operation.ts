// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getRasterGroundScale} from '../../analysis-core/contracts/raster-grid';
import {GPU_RASTER_WALKING_COST_PARAMETER_LENGTH} from '../gpu-raster-walking-cost';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../raster-grid/raster-operation-band';
import type {RasterWalkingCostOperationProps} from './gpu-raster-walking-cost-operation';

export type CPURasterWalkingCostInput = {
  elevation: CPURasterOperationBand;
  friction: CPURasterOperationBand;
  sources: Uint32Array;
  parameters: Float32Array;
};
export type CPURasterWalkingCostOptions = Pick<
  RasterWalkingCostOperationProps,
  'grid' | 'cellSizeMode' | 'model'
>;
export type CPURasterWalkingCostResult = {
  costs: Float32Array;
  validity: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
  invalidCount: Uint32Array;
};

/**
 * Computes the directed least-cost fixed point, with the contributor's Tobler/GRASS transition,
 * midpoint-latitude ground lengths, impassable transitions and model-mismatch status. Sources
 * are a per-cell nonzero mask, not a list of cell indices. Nonfinite parameters follow the same
 * tests as the kernel: invalid model words move nothing; other nonfinite transition costs are
 * discarded. Band calibration is the shared raster operation calibration. Only the first
 * grid.width * grid.height sources rows are read; surplus source rows are ignored.
 *
 * What differs from the GPU contributor:
 * - Relaxation always reaches the exact fixed point: maximumIterations and the GPU command-node
 *   budget are ignored. converged is 1 and iterationLimitReached is 0, even with no sources.
 * - Calibration, ground scales, transitions and costs use float64 before one float32 output
 *   rounding. Compare costs with relative tolerance 3e-5; near-equal path choices and the number
 *   of rounds can differ. A float64-finite transition that overflows float32 remains usable.
 * - iterationCount counts synchronous float64 rounds including the first unchanged round; it
 *   can exceed the GPU budget. invalidCount keeps the GPU's model-layout mismatch meaning.
 * - Subnormal samples and costs are retained; GPU implementations may flush them to zero.
 *
 * @throws Error naming an input whose array length is inconsistent, or an invalid option.
 */
export function computeRasterWalkingCostOnCPU(
  input: CPURasterWalkingCostInput,
  options: CPURasterWalkingCostOptions
): CPURasterWalkingCostResult {
  const id = 'computeRasterWalkingCostOnCPU';
  validateRasterOperationBand(id, input.elevation, options.grid);
  validateRasterOperationBand(id, input.friction, options.grid);
  validateWalkingOptions(id, options);
  const {width, height} = options.grid;
  const length = width * height;
  if (input.sources.length < length)
    throw new Error(`${id} sources must have at least one row per cell`);
  if (input.parameters.length < GPU_RASTER_WALKING_COST_PARAMETER_LENGTH)
    throw new Error(`${id} parameters must contain the walking-cost parameter layout`);
  // Layout of getGPURasterWalkingCostParameterValues, including its padding words.
  const [
    columnX,
    rowX,
    columnY,
    rowY,
    northEdge,
    southEdge,
    frictionWeight,
    ,
    distanceCoefficient,
    uphillCoefficient,
    moderateDownhillCoefficient,
    steepDownhillCoefficient,
    downhillSlopeThreshold
  ] = input.parameters;
  const grass = options.model?.kind === 'grass';
  const modelWords = [
    distanceCoefficient,
    uphillCoefficient,
    moderateDownhillCoefficient,
    steepDownhillCoefficient,
    downhillSlopeThreshold
  ];
  const parametersMatchModel = grass
    ? downhillSlopeThreshold < 0 && modelWords.every(Number.isFinite)
    : modelWords.every(value => value === 0);
  const elevation = calibrateRasterOperationBand(input.elevation, length);
  const friction = calibrateRasterOperationBand(input.friction, length);
  const validity = Uint32Array.from(elevation, (value, cell) =>
    Number.isFinite(value) && Number.isFinite(friction[cell]) && friction[cell] >= 0 ? 1 : 0
  );
  let previous = Float64Array.from(validity, (valid, cell) =>
    valid !== 0 && input.sources[cell] !== 0 ? 0 : Infinity
  );
  let next = new Float64Array(length);
  // Cache the fixed directed graph once: each row has at most eight incoming transitions.
  const incoming: {neighbor: number; cost: number}[][] = Array.from({length}, () => []);
  if (parametersMatchModel) {
    for (let cell = 0; cell < length; cell++) {
      if (validity[cell] === 0) continue;
      const row = Math.floor(cell / width);
      const column = cell % width;
      for (let rowStep = -1; rowStep <= 1; rowStep++) {
        for (let columnStep = -1; columnStep <= 1; columnStep++) {
          if (rowStep === 0 && columnStep === 0) continue;
          const neighborRow = row + rowStep;
          const neighborColumn = column + columnStep;
          if (
            neighborRow < 0 ||
            neighborColumn < 0 ||
            neighborRow >= height ||
            neighborColumn >= width
          )
            continue;
          const neighbor = neighborRow * width + neighborColumn;
          if (validity[neighbor] === 0) continue;
          const rowFraction = (row + 0.5 + 0.5 * rowStep) / height;
          const [scaleX, scaleY] = getRasterGroundScale(
            options.cellSizeMode ?? 'uniform',
            northEdge + (southEdge - northEdge) * rowFraction
          );
          const distance = Math.hypot(
            (columnX * columnStep + rowX * rowStep) * scaleX,
            (columnY * columnStep + rowY * rowStep) * scaleY
          );
          if (!(distance > 0)) continue;
          const rise = elevation[cell] - elevation[neighbor];
          const slope = rise / distance;
          const energy = grass
            ? distanceCoefficient * distance +
              (rise > 0
                ? uphillCoefficient
                : slope < downhillSlopeThreshold
                  ? steepDownhillCoefficient
                  : moderateDownhillCoefficient) *
                rise
            : distance * 0.6 * Math.exp(3.5 * Math.abs(slope + 0.05));
          const cost =
            energy + frictionWeight * 0.5 * (friction[cell] + friction[neighbor]) * distance;
          if (cost >= 0 && Number.isFinite(cost)) incoming[cell].push({neighbor, cost});
        }
      }
    }
  }
  let iterationCount = 0;
  for (;;) {
    let changed = false;
    for (let cell = 0; cell < length; cell++) {
      let best = previous[cell];
      for (const {neighbor, cost} of incoming[cell])
        best = Math.min(best, previous[neighbor] + cost);
      next[cell] = best;
      if (best < previous[cell]) changed = true;
    }
    iterationCount++;
    [previous, next] = [next, previous];
    if (!changed) break;
  }
  return {
    costs: Float32Array.from(previous, (cost, cell) => (validity[cell] !== 0 ? cost : NaN)),
    validity,
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    iterationCount: Uint32Array.of(iterationCount),
    invalidCount: Uint32Array.of(parametersMatchModel ? 0 : 1)
  };
}

function validateWalkingOptions(
  id: string,
  options: CPURasterWalkingCostOptions & {maximumIterations?: number}
): void {
  const mode = options.cellSizeMode ?? 'uniform';
  validateRequirement(
    id,
    ['uniform', 'geographic', 'web-mercator'].includes(mode),
    'cellSizeMode must be uniform, geographic or web-mercator'
  );
  validateRequirement(
    id,
    mode === 'uniform' || (options.grid.affine[1] === 0 && options.grid.affine[3] === 0),
    'cellSizeMode requires an axis-aligned grid'
  );
  validateRequirement(
    id,
    options.model === undefined ||
      options.model.kind === 'grass' ||
      options.model.kind === 'tobler',
    'model.kind must be grass or tobler'
  );
  if (options.model?.kind === 'grass') {
    validateRequirement(
      id,
      options.model.coefficients.length === 4 && options.model.coefficients.every(Number.isFinite),
      'model.coefficients must be four finite numbers'
    );
    validateRequirement(
      id,
      Number.isFinite(options.model.downhillSlopeThreshold) &&
        options.model.downhillSlopeThreshold < 0,
      'model.downhillSlopeThreshold must be negative and finite'
    );
  }
  if (options.maximumIterations !== undefined)
    validateCapacity(id, 'maximumIterations', options.maximumIterations);
}

/** CPU directed walking cost, without the GPU command-node or convergence budget. */
export class CPURasterWalkingCostOperation extends GPUCPUOperation {
  constructor(props: RasterWalkingCostOperationProps) {
    const id = props.id ?? 'raster-walking-cost';
    validateRasterOperationBand(id, props.elevation, props.grid);
    validateRasterOperationBand(id, props.friction, props.grid);
    validateWalkingOptions(id, props);
    const length = props.grid.width * props.grid.height;
    super({
      id,
      inputs: {
        elevationValues: props.elevation.storage.values,
        frictionValues: props.friction.storage.values,
        ...(props.elevation.validity !== undefined
          ? {elevationValidity: props.elevation.validity}
          : {}),
        ...(props.friction.validity !== undefined
          ? {frictionValidity: props.friction.validity}
          : {}),
        sources: props.sources,
        parameters: props.parameters
      },
      outputs: {
        [props.output.costs]: gpuGraph.vector('float32', {length}),
        [props.output.validity]: gpuGraph.vector('uint32', {length}),
        ...Object.fromEntries(
          Object.values(props.output.status)
            .filter(value => value !== undefined)
            .map(value => [value, gpuGraph.vector('uint32', {length: 1})])
        )
      },
      execute({inputs}) {
        const result = computeRasterWalkingCostOnCPU(
          {
            elevation: readRasterOperationBand(
              id,
              {
                ...inputs,
                values: inputs['elevationValues']!,
                validity: inputs['elevationValidity']!
              },
              props.elevation
            ),
            friction: readRasterOperationBand(
              id,
              {...inputs, values: inputs['frictionValues']!, validity: inputs['frictionValidity']!},
              props.friction
            ),
            sources: getCPUOperationUint32Rows(id, inputs, 'sources'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          props
        );
        return {
          [props.output.costs]: result.costs,
          [props.output.validity]: result.validity,
          [props.output.status.converged]: result.converged,
          ...(props.output.status.iterationLimitReached !== undefined
            ? {[props.output.status.iterationLimitReached]: result.iterationLimitReached}
            : {}),
          ...(props.output.status.iterationCount !== undefined
            ? {[props.output.status.iterationCount]: result.iterationCount}
            : {}),
          ...(props.output.status.invalidCount !== undefined
            ? {[props.output.status.invalidCount]: result.invalidCount}
            : {})
        };
      }
    });
  }
}
