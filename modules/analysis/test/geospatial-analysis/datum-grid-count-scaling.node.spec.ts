// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUDatumGridTransform} from '../../src/geospatial-analysis/datum-grid-transform';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {
  DATUM_SCALING_GRID_COUNTS,
  getDatumScalingGrids
} from '../gpu-spatial-analysis/global-coordinates/global-coordinate-scaling-fixtures';
import {getDatumSourceObservation} from '../gpu-spatial-analysis/global-coordinates/global-coordinate-output-assertions';

it('measures grid metadata scaling independently of sample and query counts (DGGS-111)', () => {
  const device = createNullWebGPUDevice();
  const measurements: Record<string, number>[] = [];
  try {
    for (const gridCount of DATUM_SCALING_GRID_COUNTS) {
      const graph = new GPUCommandGraph(device);
      const props = {
        positions: createTransientView(graph, 'positions', 'float32x3', 4),
        samples: createTransientView(graph, 'samples', 'float32x3', 4),
        mask: createTransientView(graph, 'mask', 'uint32', 4),
        grids: getDatumScalingGrids(gridCount),
        operation: 'horizontal' as const,
        output: {
          positions: createTransientView(graph, 'output', 'float32x3', 4),
          status: createTransientView(graph, 'status', 'uint32', 4)
        }
      };
      const start = performance.now();
      const contributor = new GPUDatumGridTransform(props);
      const constructionMilliseconds = performance.now() - start;
      const declarationStart = performance.now();
      const nodes = contributor.getCommandNodes(graph);
      const declarationMilliseconds = performance.now() - declarationStart;
      // The recorder uses a null device. This is source generation, never GPU compilation.
      const source = getGPUCommandNodeKernelSources(nodes)
        .map(kernel => kernel.source)
        .join('\n');
      const {generatedBytes} = getDatumSourceObservation(source, gridCount);
      // Source smoke checks only; these observations do not certify GPU compilation or results.
      expect(generatedBytes).toBeGreaterThan(0);
      expect(source.match(/let corners = vec4u/g)).toHaveLength(gridCount);
      measurements.push({
        gridCount,
        sampleCount: 4,
        queryCount: 4,
        generatedBytes,
        constructionMilliseconds,
        declarationMilliseconds
      });
      // A separate axis changes resident samples but leaves grid metadata unchanged.
      const largerSamples = new GPUDatumGridTransform({
        ...props,
        samples: createTransientView(graph, 'largerSamples', 'float32x3', 4096),
        mask: createTransientView(graph, 'largerMask', 'uint32', 4096)
      });
      const largerSource = getGPUCommandNodeKernelSources(largerSamples.getCommandNodes(graph))
        .map(kernel => kernel.source)
        .join('\n');
      expect(largerSource.match(/let corners = vec4u/g)).toHaveLength(gridCount);
      measurements.push({
        sampleCount: 4096,
        queryCount: 4,
        ...getDatumSourceObservation(largerSource, gridCount)
      });
    }
    expect(measurements).toHaveLength(DATUM_SCALING_GRID_COUNTS.length * 2);
    process.stdout.write(
      `DATUM_GRID_COUNT_HOST_OBSERVATIONS ${JSON.stringify({
        method:
          'one host observation per shape, no latency threshold; null device source recording, device compilation/execution unverified',
        measurements
      })}\n`
    );
  } finally {
    device.destroy();
  }
});
