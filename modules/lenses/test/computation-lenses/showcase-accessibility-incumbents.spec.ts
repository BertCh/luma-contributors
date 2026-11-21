// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkAccessibility,
  getGPUNetworkAccessibilityParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  readFloat32,
  readUint32,
  submitGraph
} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {SpatialAnalysisResources} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/resources';
import {importGraphBuffer} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/graph-buffers';
import jobSource from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/networks/job-accessibility.compute.ts?raw';
import explorerSource from '../../../../examples/deck/spatial-analysis-explorer/modes/accessibility-mode.ts?raw';
import {
  createApplicationContributor,
  createApplicationScorePublisher,
  readApplicationScorePacket
} from './showcase-accessibility-fixture';

it.each([
  'job',
  'explorer'
] as const)('NET-101 %s application keeps certified catchment incumbents in buffers while withholding presentation', async route => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new SpatialAnalysisResources(device, `incumbents-${route}`);
  try {
    const source = route === 'job' ? jobSource : explorerSource;
    const scoreGraph = new GPUCommandGraph<void>(device);
    const matrix = resources.createBuffer('matrix', Float32Array.of(2, 7));
    const matrixConverged = resources.createBuffer('converged', Uint32Array.of(1));
    const horizon = resources.createParameterBuffer('horizon', 'float32', 1, Float32Array.of(2));
    const scoring = resources.createParameterBuffer(
      'scoring',
      'float32',
      4,
      getGPUNetworkAccessibilityParameterValues({threshold: 5})
    );
    const supply = resources.createBuffer('supply', Float32Array.of(12));
    const demand = resources.createBuffer('demand', Float32Array.of(3, 4));
    const cumulative = resources.createBuffer('cumulative', 8);
    const gravity = resources.createBuffer('gravity', 8);
    const twoStep = resources.createBuffer('two-step', 8);
    const ratios = resources.createBuffer('ratios', 4);
    const complete = resources.createBuffer('complete', 4);
    const snapOverflow = resources.createBuffer('snap-overflow', Uint32Array.of(0));
    const view = (
      buffer: Parameters<typeof importGraphBuffer>[2],
      format: Parameters<typeof importGraphBuffer>[3],
      length: number
    ) => importGraphBuffer(scoreGraph, buffer.id, buffer, format, length);
    scoreGraph.add(
      createApplicationContributor(source, {
        GPUNetworkAccessibility,
        importGraphBuffer,
        scoreGraph,
        view,
        nodeCount: 2,
        rowCount: 1,
        matrixLength: 2,
        matrix,
        matrixBuffer: matrix,
        matrixConverged,
        costLimit: horizon,
        costLimitParameter: horizon,
        scoresComplete: complete,
        opportunityWeights: supply,
        scoring,
        demandBuffer: demand,
        cumulative,
        cumulativeBuffer: cumulative,
        gravity,
        gravityBuffer: gravity,
        twoStep,
        twoStepBuffer: twoStep,
        facilityRatios: ratios,
        ratiosBuffer: ratios
      })
    );
    const compiled = resources.track(await scoreGraph.compileAsync());
    const publisher = createApplicationScorePublisher(route, source);
    for (const [limit, converged, expectedComplete] of [
      [2, 1, 0],
      [5, 0, 0],
      [5, 1, 1]
    ]) {
      horizon.write(Float32Array.of(limit));
      matrixConverged.write(Uint32Array.of(converged));
      submitGraph(device, compiled, undefined);
      // Only demand 3 is within threshold 5, so retained supply / demand = 12/3 = 4.
      // The other node (cost 7) has zero score, regardless of the completion certificate.
      const scores = await readFloat32(twoStep, 2);
      expect(scores.every(Number.isFinite)).toBe(true);
      expect(scores).toEqual([4, 0]);
      expect(await readFloat32(ratios, 1)).toEqual([4]);
      expect(await readUint32(complete, 1)).toEqual([expectedComplete]);
      const bytes = await readApplicationScorePacket(
        route,
        source,
        {
          nodeCount: 2,
          cumulativeBuffer: cumulative,
          gravityBuffer: gravity,
          twoStepBuffer: twoStep,
          cumulative,
          gravity,
          twoStep,
          scoresComplete: complete,
          snapOverflow
        },
        async (buffer, size) => new Uint8Array(await (buffer as Buffer).readAsync(0, size))
      );
      publisher.publish(route === 'job' ? bytes : new Uint8Array(bytes));
      expect(publisher.ready()).toBe(Boolean(expectedComplete));
      if (!expectedComplete) expect(publisher.values.get('reached')).toMatch(/withheld/);
      else if (route === 'job')
        expect(publisher.values.get('top')).toBe('4000.0 maximum (fixed comparison classes)');
      else expect(publisher.values.get('range')).toBe('4.00 / 4.00');
    }
  } finally {
    resources.destroy();
  }
});
