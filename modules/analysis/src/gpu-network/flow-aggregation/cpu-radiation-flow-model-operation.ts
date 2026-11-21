// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import type {GPURadiationFlowModelNormalization} from './gpu-radiation-flow-model';
import {resolveRadiationFlowModelNormalization} from './radiation-flow-model-normalization';
import {
  GPURadiationFlowModelOperation,
  snapshotRadiationFlowModelOperationProps,
  type RadiationFlowModelOperationProps
} from './gpu-radiation-flow-model-operation';

/** Array inputs and model choices for the pure CPU computation. */
export type CPURadiationFlowModelInput = {
  positions: Float32Array;
  populations: Float32Array;
  outflows: Float32Array;
  spatialContext?: GPUSpatialContext;
  normalization?: GPURadiationFlowModelNormalization;
};

/** Dense origin-major radiation flows. */
export type CPURadiationFlowModelResult = {flows: Float32Array};

/**
 * Computes radiation fluxes using one distance sort and grouped population prefix per origin.
 * Returns float32 flows at row origin * zoneCount + destination, with a zero diagonal.
 * Equal distances are processed together, preserving the strict intervening-distance rule.
 *
 * What differs from the GPU contributor:
 * - Distances, population sums and fluxes use float64; each output is rounded once. Near ties
 *   may therefore differ from float32 distance ties on the GPU.
 * - The GPU-only 1024-zone work cap is dropped; sorting costs O(zones squared log zones).
 *
 * @throws If positions, populations or outflows have inconsistent lengths.
 */
export function computeRadiationFlowModelOnCPU(
  input: CPURadiationFlowModelInput
): CPURadiationFlowModelResult {
  const {positions, populations, outflows} = input;
  const zoneCount = populations.length;
  if (positions.length !== zoneCount * 2) {
    throw new Error('positions must hold two components per populations row');
  }
  if (outflows.length !== zoneCount) {
    throw new Error('outflows length must equal populations length');
  }
  let total = 0;
  for (let zone = 0; zone < zoneCount; zone++) {
    total += populations[zone];
    validateRequirement(
      'radiation-flow-model',
      Number.isFinite(positions[zone * 2]) && Number.isFinite(positions[zone * 2 + 1]),
      `positions at zone ${zone} must be finite; received [${positions[zone * 2]}, ${positions[zone * 2 + 1]}]`
    );
  }
  const context = resolveGPUSpatialContext(
    'radiation-flow-model',
    input.spatialContext,
    GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
  );
  const sphereRadius = getGPUGreatCircleRadius(context);
  const normalization = resolveRadiationFlowModelNormalization(
    'radiation-flow-model',
    input.normalization
  );
  const flows = new Float32Array(zoneCount * zoneCount);
  for (let origin = 0; origin < zoneCount; origin++) {
    const destinations: {zone: number; distance: number}[] = [];
    let outside = 0;
    for (let destination = 0; destination < zoneCount; destination++) {
      if (destination === origin) continue;
      outside += populations[destination];
      let distance: number;
      if (sphereRadius === undefined) {
        distance = Math.hypot(
          positions[destination * 2] - positions[origin * 2],
          positions[destination * 2 + 1] - positions[origin * 2 + 1]
        );
      } else {
        const latitude = (positions[origin * 2 + 1] * Math.PI) / 180;
        const destinationLatitude = (positions[destination * 2 + 1] * Math.PI) / 180;
        const longitudeDegrees = positions[destination * 2] - positions[origin * 2];
        // Subtract whole turns rather than adding a large offset: adding 180 can erase
        // representable separations smaller than its float64 spacing.
        const longitudeDelta =
          ((longitudeDegrees - 360 * Math.round(longitudeDegrees / 360)) * Math.PI) / 180;
        // The spherical atan2 form stays conditioned near antipodes. Colatitude gives exact
        // zero cosines at the poles, preserving their longitude-independent distance ties.
        const sourceCosine = Math.sin(((90 - Math.abs(positions[origin * 2 + 1])) * Math.PI) / 180);
        const destinationCosine = Math.sin(
          ((90 - Math.abs(positions[destination * 2 + 1])) * Math.PI) / 180
        );
        const halfVersine = 2 * Math.sin(longitudeDelta / 2) ** 2;
        const east = destinationCosine * Math.sin(longitudeDelta);
        const north =
          Math.sin(destinationLatitude - latitude) +
          Math.sin(latitude) * destinationCosine * halfVersine;
        const up =
          Math.sin(latitude) * Math.sin(destinationLatitude) +
          sourceCosine * destinationCosine * (1 - halfVersine);
        distance = sphereRadius * Math.atan2(Math.hypot(east, north), up);
      }
      destinations.push({zone: destination, distance});
    }
    destinations.sort((left, right) => left.distance - right.distance);
    const sourcePopulation = populations[origin];
    const writeFlow = (destination: number, between: number) => {
      const targetPopulation = populations[destination];
      const denominator =
        (sourcePopulation + between) * (sourcePopulation + targetPopulation + between);
      let flow = 0;
      if (denominator > 0) {
        flow = (outflows[origin] * sourcePopulation * targetPopulation) / denominator;
        if (normalization === 'finite-size') {
          const remaining = outside / total;
          flow = remaining > 0 ? flow / remaining : 0;
        }
      }
      flows[origin * zoneCount + destination] = flow;
    };
    let between = 0;
    let first = 0;
    while (first < destinations.length) {
      let last = first + 1;
      while (
        last < destinations.length &&
        destinations[last].distance === destinations[first].distance
      )
        last++;
      for (let index = first; index < last; index++) writeFlow(destinations[index].zone, between);
      for (let index = first; index < last; index++)
        between += populations[destinations[index].zone];
      first = last;
    }
  }
  return {flows};
}

/** CPU twin, preferred for the serial reference model. The GPU zone cap is not retained. */
export class CPURadiationFlowModelOperation extends GPUCPUOperation {
  constructor(originalProps: RadiationFlowModelOperationProps) {
    const props = snapshotRadiationFlowModelOperationProps(originalProps);
    const id = props.id ?? 'radiation-flow-model';
    validateCapacity(id, 'zoneCount', props.zoneCount, 2);
    resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    const operation = new GPURadiationFlowModelOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: {
        [props.output.flows]: gpuGraph.vector('float32', {length: props.zoneCount ** 2}),
        ...(props.output.invalidInput !== undefined
          ? {
              [props.output.invalidInput]: gpuGraph.vector('uint32', {length: props.zoneCount})
            }
          : {})
      },
      execute({inputs}) {
        const populations = getCPUOperationFloat32Rows(id, inputs, 'populations');
        if (populations.length !== props.zoneCount)
          throw new Error(`${id} populations length must equal zoneCount`);
        return {
          [props.output.flows]: computeRadiationFlowModelOnCPU({
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            populations,
            outflows: getCPUOperationFloat32Rows(id, inputs, 'outflows'),
            spatialContext: props.spatialContext,
            normalization: props.normalization
          }).flows,
          ...(props.output.invalidInput !== undefined
            ? {
                [props.output.invalidInput]: new Uint32Array(props.zoneCount)
              }
            : {})
        };
      }
    });
  }
}
