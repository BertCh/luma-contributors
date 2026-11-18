// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it.each([
  'overflow',
  'candidateOverflow'
] as const)('suppresses REDCAP partition after weights %s and recovers on the next submission', async flagName => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'redcap-incomplete');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const loss = fixture.output('loss', 'uint32', 1);
    const offsets = fixture.output('offsets', 'uint32', 4);
    const neighbors = fixture.output('neighbors', 'uint32', 4);
    const labels = fixture.output('labels', 'uint32', 3);
    const regionCount = fixture.output('region-count', 'uint32', 1);
    const converged = fixture.output('converged', 'uint32', 1);
    const iterationLimitReached = fixture.output('limit', 'uint32', 1);
    const contributor = new GPURedcapSingleLinkage({
      componentIterations: 3,
      standardize: false,
      values: fixture.input('values', Float32Array.of(1, 2, 4), 'float32', 3),
      parameters: fixture.input('parameters', Uint32Array.of(1), 'uint32', 1),
      weights: {
        offsets: offsets.view,
        neighbors: neighbors.view,
        weights: fixture.input('weights', Float32Array.of(1, 1, 1, 1), 'float32', 4),
        status: {
          overflow: flagName === 'overflow' ? loss.view : undefined,
          candidateOverflow: flagName === 'candidateOverflow' ? loss.view : undefined
        }
      },
      output: {
        labels: labels.view,
        regionCount: regionCount.view,
        status: {converged: converged.view, iterationLimitReached: iterationLimitReached.view}
      }
    });
    fixture.graph.add(contributor);
    const baselineLabels = fixture.output('baseline-labels', 'uint32', 3);
    const baselineCount = fixture.output('baseline-count', 'uint32', 1);
    fixture.graph.add(
      new GPURedcapSingleLinkage({
        ...contributor.props,
        id: 'unflagged-redcap',
        weights: {...contributor.props.weights, status: undefined},
        output: {labels: baselineLabels.view, regionCount: baselineCount.view}
      })
    );
    compiled = fixture.graph.compile();
    for (const incomplete of [0, 1, 0]) {
      loss.buffer.write(Uint32Array.of(incomplete));
      offsets.buffer.write(Uint32Array.from(incomplete ? [0, 1, 2, 2] : [0, 1, 3, 4]));
      neighbors.buffer.write(Uint32Array.from(incomplete ? [1, 0, 0, 0] : [1, 0, 2, 1]));
      submitGraph(device, compiled, undefined);
      expect(await fixture.readUint32(labels, 3)).toEqual(
        incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
      );
      expect(await fixture.readUint32(regionCount, 1)).toEqual([incomplete ? 0 : 1]);
      expect(await fixture.readUint32(converged, 1)).toEqual([incomplete ? 0 : 1]);
      expect(await fixture.readUint32(iterationLimitReached, 1)).toEqual([0]);
      if (!incomplete) {
        expect(await fixture.readUint32(labels, 3)).toEqual(
          await fixture.readUint32(baselineLabels, 3)
        );
        expect(await fixture.readUint32(regionCount, 1)).toEqual(
          await fixture.readUint32(baselineCount, 1)
        );
      }
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it.each(
  Array.from({length: 8}, (_, combination) => combination)
)('writes REDCAP iterationCount with independent convergence outputs (combination=%s)', async combination => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'redcap-count');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const parameters = fixture.output('parameters', 'uint32', 1);
    const labels = fixture.output('labels', 'uint32', 2);
    const iterationCount =
      combination & 1 ? fixture.output('iteration-count', 'uint32', 1) : undefined;
    const converged = combination & 2 ? fixture.output('converged', 'uint32', 1) : undefined;
    const iterationLimitReached =
      combination & 4 ? fixture.output('limit', 'uint32', 1) : undefined;
    for (const output of [iterationCount, converged, iterationLimitReached])
      output?.buffer.write(Uint32Array.of(99));
    fixture.graph.add(
      new GPURedcapSingleLinkage({
        componentIterations: 3,
        standardize: false,
        values: fixture.input('values', Float32Array.of(1, 2), 'float32', 2),
        parameters: parameters.view,
        weights: {
          offsets: fixture.input('offsets', Uint32Array.of(0, 1, 2), 'uint32', 3),
          neighbors: fixture.input('neighbors', Uint32Array.of(1, 0), 'uint32', 2),
          weights: fixture.input('weights', Float32Array.of(1, 1), 'float32', 2)
        },
        output: {
          labels: labels.view,
          status: {
            iterationCount: iterationCount?.view,
            converged: converged?.view,
            iterationLimitReached: iterationLimitReached?.view
          }
        }
      })
    );
    compiled = fixture.graph.compile();
    for (const regionCount of [1, 2]) {
      parameters.buffer.write(Uint32Array.of(regionCount));
      submitGraph(device, compiled, undefined);
      expect(await fixture.readUint32(labels, 2)).toEqual(regionCount === 1 ? [0, 0] : [0, 1]);
      // Even an edgeless partition runs all three rounds; this count excludes the MST stage.
      if (iterationCount) expect(await fixture.readUint32(iterationCount, 1)).toEqual([3]);
      if (converged) expect(await fixture.readUint32(converged, 1)).toEqual([1]);
      if (iterationLimitReached)
        expect(await fixture.readUint32(iterationLimitReached, 1)).toEqual([0]);
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
