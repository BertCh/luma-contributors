// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUMapColoring} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {GPUMapColoringOperation} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {addOracleOperation, createOracleFixture} from '../global-coordinates/output-oracle-fixture';
import {MAP_COLORING_STATUS_CASES} from './map-coloring-status-oracles-fixture';

it('coloring operation overwrites zero conflict and completed singleton status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const operation = new GPUMapColoringOperation({
    rowCount: 1,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    parameters: 'parameters',
    maximumIterations: 1,
    output: {
      colors: 'colors',
      colorCount: 'colorCount',
      conflictCount: 'conflictCount',
      status: {converged: 'converged', iterationCount: 'iterationCount'}
    }
  });
  const outputs = addOracleOperation(fixture, operation, {
    offsets: fixture.input('uint32', [0, 0]),
    // Storage padding is outside the active CSR range and is never visited.
    neighbors: fixture.input('uint32', [0]),
    weights: fixture.input('float32', [1]),
    parameters: fixture.input('uint32', [0])
  });
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of Object.values(outputs)) output.poison();
      submitGraph(device, compiled, undefined);
      expect(await outputs['colors'].read()).toEqual([0]);
      expect(await outputs['colorCount'].read()).toEqual([1]);
      expect(await outputs['conflictCount'].read()).toEqual([0]);
      expect(await outputs['converged'].read()).toEqual([1]);
      expect(await outputs['iterationCount'].read()).toEqual([1]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it.each(
  MAP_COLORING_STATUS_CASES
)('chunked coloring overwrites full hand-derived status: $name', async scene => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const rows = scene.offsets.length - 1;
  const colors = fixture.output('uint32', rows);
  const outputs = {
    colorCount: fixture.output('uint32', 1),
    conflictCount: fixture.output('uint32', 1),
    converged: fixture.output('uint32', 1),
    iterationCount: fixture.output('uint32', 1)
  };
  // One dispatch per row fixes the scheduling order used in the independent
  // fixture derivation, including the named blocked row and asymmetric conflict.
  fixture.graph.add(
    new GPUMapColoring({
      weights: {
        rowCount: rows,
        chunks: Array.from({length: rows}, (_, row) => {
          const neighbors = scene.neighbors.slice(scene.offsets[row], scene.offsets[row + 1]);
          return {
            rowStart: row,
            rowCount: 1,
            weights: {
              offsets: fixture.input('uint32', [0, neighbors.length]),
              neighbors: fixture.input('uint32', neighbors.length ? neighbors : [0]),
              weights: fixture.input('float32', Array(Math.max(1, neighbors.length)).fill(1))
            }
          };
        })
      },
      parameters: fixture.input('uint32', [0]),
      priority: 'largest-degree-first',
      maximumIterations: scene.maximumIterations,
      output: {
        colors: colors.view,
        colorCount: outputs.colorCount.view,
        conflictCount: outputs.conflictCount.view,
        status: {converged: outputs.converged.view, iterationCount: outputs.iterationCount.view}
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of [colors, ...Object.values(outputs)]) output.poison();
      submitGraph(device, compiled, undefined);
      expect(await colors.read()).toEqual(scene.colors);
      for (const port of ['colorCount', 'conflictCount', 'converged', 'iterationCount'] as const)
        expect(await outputs[port].read()).toEqual([scene[port]]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
