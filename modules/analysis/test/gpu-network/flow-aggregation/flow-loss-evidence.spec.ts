// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUFlowAggregation} from '../../../src/gpu-network/flow-aggregation/gpu-flow-aggregation';
import {GPURadiationFlowModel} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model';
import {
  GPUKernelFlowSmoothing,
  getGPUKernelFlowSmoothingParameterValues
} from '../../../src/gpu-network/flow-aggregation/gpu-kernel-flow-smoothing';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';
import {getFlowLossEvidenceFrame} from './flow-loss-evidence-fixture';

it('DGN-05 marks all origin rows affected by an unknown centroid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [coordinate, expected] of [
    [0.5, [0, 2.5, 7.5]],
    [2, [0, 7.5, 2.5]],
    [NaN, []]
  ] as const) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const flows = output('flows', 'float32', 9);
    const invalidInput = output('invalid', 'uint32', 3);
    graph.add(
      new GPURadiationFlowModel({
        positions: input('positions', 'float32x2', [0, 0, 1, 0, coordinate, 0]),
        populations: input('population', 'float32', [1, 1, 1]),
        outflows: input('outflows', 'float32', [10, 0, 0]),
        normalization: 'finite-size',
        output: {flows, status: {invalidInput}}
      })
    );
    try {
      context.run();
      if (Number.isNaN(coordinate)) {
        expect(await read(invalidInput)).toEqual([1, 1, 1]);
        expect(await read(flows)).toEqual([0, NaN, NaN, NaN, 0, NaN, NaN, NaN, 0]);
      } else {
        expect((await read(flows)).slice(0, 3)).toEqual(expected);
        expect(await read(invalidInput)).toEqual([0, 0, 0]);
      }
    } finally {
      context.destroy();
    }
  }
});

it('CPL-09 reports independent loss flags and suppresses uncertified identity smoothing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withStatus of [false, true]) {
    const context = createOracleContext(device);
    const {graph, input, output, read, write} = context;
    const overflow = input('overflow', 'uint32', [0]);
    const candidateOverflow = input('candidate-overflow', 'uint32', [0]);
    const pairOverflow = input('pair-overflow', 'uint32', [0]);
    const count = input('count', 'uint32', [2]);
    const complete = withStatus ? output('complete', 'uint32', 1) : undefined;
    const smoothed = output('smoothed', 'float32', 4);
    graph.add(
      new GPUKernelFlowSmoothing({
        positions: input('positions', 'float32x2', [0, 0, 1, 0]),
        locationWeights: input('locations', 'float32', [1, 1]),
        parameters: input(
          'parameters',
          'float32',
          Array.from(getGPUKernelFlowSmoothingParameterValues({neighborhoodSize: 0}))
        ),
        flows: {
          kind: 'pairs',
          originZoneIds: input('origins', 'uint32', [0, 1]),
          destinationZoneIds: input('destinations', 'uint32', [1, 0]),
          weights: input('weights', 'float32', [2, 1]),
          status: {count, overflow, candidateOverflow, pairOverflow}
        },
        output: {smoothed, ...(complete ? {status: {complete}} : {})}
      })
    );
    try {
      // A complete prefix can be smaller than capacity.
      write(count, [1]);
      context.run();
      expect(await read(smoothed)).toEqual([0, 2, 0, 0]);
      if (complete) expect(await read(complete)).toEqual([1]);
      write(count, [2]);
      context.run();
      expect(await read(smoothed)).toEqual([0, 2, 1, 0]);
      for (const loss of [overflow, candidateOverflow, pairOverflow]) {
        // Hold the saturated two-row prefix fixed while independently changing each loss word.
        write(loss, [1]);
        context.run();
        if (complete) {
          expect(await read(complete)).toEqual([0]);
          expect(await read(smoothed)).toEqual([0, 2, 1, 0]);
        } else expect(await read(smoothed)).toEqual([NaN, NaN, NaN, NaN]);
        write(loss, [0]);
        context.run();
        expect(await read(smoothed)).toEqual([0, 2, 1, 0]);
        if (complete) expect(await read(complete)).toEqual([1]);
      }
    } finally {
      context.destroy();
    }
  }
});

it('radiation normalization preserves the only destination at extreme population ratios', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  const flows = output('flows', 'float32', 4);
  graph.add(
    new GPURadiationFlowModel({
      positions: input('positions', 'float32x2', [0, 0, 1, 0]),
      populations: input('populations', 'float32', [2 ** 64, 1]),
      outflows: input('outflows', 'float32', [8, 6]),
      normalization: 'finite-size',
      output: {flows}
    })
  );
  try {
    context.run();
    expect(await read(flows)).toEqual([0, 8, 6, 0]);
  } finally {
    context.destroy();
  }
});

it('CPL-09 composes real pair-table loss, clean prefixes and recovery into identity smoothing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withCandidateOverflow of [false, true]) {
    for (const withCertificate of [false, true]) {
      const context = createOracleContext(device);
      const {graph, input, output, read, write} = context;
      const mask = input('mask', 'uint32', [1, 0, 0]);
      const count = output('count', 'uint32');
      const overflow = output('overflow', 'uint32');
      const pairOverflow = output('pair-overflow', 'uint32');
      const candidateOverflow = withCandidateOverflow
        ? output('candidate-overflow', 'uint32')
        : undefined;
      const flowOrigins = output('flow-origins', 'uint32', 2);
      const flowDestinations = output('flow-destinations', 'uint32', 2);
      const flowWeights = output('flow-weights', 'float32', 2);
      const status = {
        count,
        overflow,
        pairOverflow,
        ...(candidateOverflow ? {candidateOverflow} : {})
      };
      const complete = withCertificate ? output('complete', 'uint32') : undefined;
      const smoothed = output('smoothed', 'float32', 4);
      graph.add(
        new GPUFlowAggregation({
          zones: {kind: 'ids', zoneCount: 2},
          originZoneIds: input('origins', 'uint32', [0, 1, 0]),
          destinationZoneIds: input('destinations', 'uint32', [1, 0, 0]),
          weights: input('weights', 'float32', [2, 1, 4]),
          mask,
          // One table slot loses a pair when two distinct pairs are enabled. Both source
          // pairs fit the two-row final list, isolating candidate loss from final truncation.
          pairCapacity: 1,
          output: {
            ids: output('ids', 'uint32', 2),
            status,
            flowOriginZoneIds: flowOrigins,
            flowDestinationZoneIds: flowDestinations,
            flowWeights
          }
        })
      );
      graph.add(
        new GPUKernelFlowSmoothing({
          positions: input('positions', 'float32x2', [0, 0, 1, 0]),
          locationWeights: input('location-weights', 'float32', [1, 1]),
          parameters: input(
            'parameters',
            'float32',
            Array.from(getGPUKernelFlowSmoothingParameterValues({neighborhoodSize: 0}))
          ),
          flows: {
            kind: 'pairs',
            originZoneIds: flowOrigins,
            destinationZoneIds: flowDestinations,
            weights: flowWeights,
            status
          },
          output: {smoothed, ...(complete ? {status: {complete}} : {})}
        })
      );
      try {
        for (const loss of [false, true, false]) {
          const frame = getFlowLossEvidenceFrame(loss);
          write(mask, frame.mask);
          context.run();
          expect(await read(count)).toEqual([1]); // complete partial prefix and incomplete prefix have identical counts
          expect(await read(overflow)).toEqual([0]);
          expect(await read(pairOverflow)).toEqual([Number(loss)]);
          if (candidateOverflow) expect(await read(candidateOverflow)).toEqual([Number(loss)]);
          const origins = await read(flowOrigins);
          const destinations = await read(flowDestinations);
          const retainedKey = origins[0] * 2 + destinations[0];
          expect(frame.retainedKeys).toContain(retainedKey);
          // Hand-worked identity: one retained pair contributes its literal input weight
          // at row origin*2+destination; all other cells are zero. The hash winner is unspecified.
          const literalWeights = [4, 2, 1, 0];
          expect(await read(flowWeights)).toEqual([literalWeights[retainedKey], 0]);
          const expected = [0, 0, 0, 0];
          expected[retainedKey] = literalWeights[retainedKey];
          if (complete) {
            expect(await read(complete)).toEqual([Number(!loss)]);
            expect(await read(smoothed)).toEqual(expected);
          } else {
            expect(await read(smoothed)).toEqual(loss ? [NaN, NaN, NaN, NaN] : expected);
          }
        }
      } finally {
        context.destroy();
      }
    }
  }
});
