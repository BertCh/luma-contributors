// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUGeodesicPairs,
  GPUGeodesicDestination
} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {GPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation';
import {CPUGeodesicPairsOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

// The adjacent independent Python equations give first-step changes >1e-6
// for an equatorial one-degree inverse and a 1000km northward direct from 45N.
// A one-step budget therefore cannot converge on either. Coincident/zero-distance
// rows have exactly zero change; equatorial antipodes are singular before iteration.
// Short inverse 0.01 degrees changes 5.85176e-7 > relative threshold 8.40567e-11;
// direct 2000m north from (0,0) changes 5.27660e-7 > threshold 2.4e-7.
// Both are below 1e-6, so they reject an incorrect fixed absolute stopping rule.
// At 0.001 degrees, change 5.85176e-8 > relative threshold 8.40567e-12
// but < constant 4.8e-7. The long direct row (85.75 degrees, 19500km from
// the equator) instead converges: float32 sigma 3.0675757 -> 3.067575 gives
// change 7.15256e-7 <= scaled threshold 7.36218e-7. Accepting that larger
// change while rejecting the 2000m change rules out any fixed direct tolerance.
it('geodesic status distinguishes an independently unfinished step, coincidence and singularity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const;
  const inverse = {
    converged: output('inverse-converged', 'uint32', 5),
    approximation: output('inverse-approximation', 'uint32', 5),
    iterationLimitReached: output('inverse-limit', 'uint32', 5)
  };
  const direct = {
    converged: output('direct-converged', 'uint32', 4),
    approximation: output('direct-approximation', 'uint32', 4),
    iterationLimitReached: output('direct-limit', 'uint32', 4)
  };
  graph.add(
    new GPUGeodesicPairs({
      spatialContext,
      maximumIterations: 1,
      origins: input('origins', 'float32x2', Float32Array.of(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
      targets: input(
        'targets',
        'float32x2',
        Float32Array.of(1, 0, 0, 0, 180, 0, 0.01, 0, 0.001, 0)
      ),
      output: {distances: output('distances', 'float32', 5), status: inverse}
    })
  );
  graph.add(
    new GPUGeodesicDestination({
      spatialContext,
      maximumIterations: 1,
      origins: input('direct-origins', 'float32x2', Float32Array.of(0, 45, 0, 45, 0, 0, 0, 0)),
      bearings: input('bearings', 'float32', Float32Array.of(0, 0, 0, 85.75)),
      distances: input('direct-distances', 'float32', Float32Array.of(1000000, 0, 2000, 19500000)),
      output: {destinations: output('destinations', 'float32x2', 4), status: direct}
    })
  );
  const compiled = graph.compile();
  try {
    const result = await fixture.submitTwice(compiled, async () => ({
      inverse: await fixture.readRows([
        inverse.converged,
        inverse.approximation,
        inverse.iterationLimitReached
      ]),
      direct: await fixture.readRows([
        direct.converged,
        direct.approximation,
        direct.iterationLimitReached
      ])
    }));
    expect(result.inverse).toEqual([0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1]);
    expect(result.direct).toEqual([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
  // The inverse also has GPU and CPU operation forms; require the same
  // independent complete status columns, including the new 0.001-degree row.
  for (const Operation of [GPUGeodesicPairsOperation, CPUGeodesicPairsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        origins: {format: 'float32x2', values: Float32Array.of(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)},
        targets: {
          format: 'float32x2',
          values: Float32Array.of(1, 0, 0, 0, 180, 0, 0.01, 0, 0.001, 0)
        }
      },
      operations: [
        new Operation({
          spatialContext,
          origins: 'origins',
          targets: 'targets',
          pairCount: 5,
          maximumIterations: 1,
          output: {
            status: {
              converged: 'converged',
              approximation: 'approximation',
              iterationLimitReached: 'limit'
            }
          }
        })
      ],
      outputs: ['converged', 'approximation', 'limit']
    });
    expect(Array.from(result['converged'])).toEqual([0, 1, 0, 0, 0]);
    expect(Array.from(result['approximation'])).toEqual([1, 0, 1, 1, 1]);
    expect(Array.from(result['limit'])).toEqual([1, 0, 0, 1, 1]);
  }
});
