// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUMapMatching,
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues
} from '../../src/gpu-network/map-matching/index';
import {
  GPUNetworkSnapping,
  GPU_NETWORK_SNAPPING_NONE
} from '../../src/gpu-network/network-accessibility/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

/**
 * NET-106: one directed road `start -> end` and one query point, translated by `offset` in both
 * coordinates. Every coordinate is exactly representable as float32 at each translation used.
 *
 * Expected values are closed forms in the untranslated frame:
 * - road (0,0)->(2,2), point (1,0): fraction 1/4, closest point (1/2,1/2), distance sqrt(1/2).
 * - road (0,0)->(4,1), point (2,1): fraction (2*4 + 1*1)/17 = 9/17, closest point (36/17, 9/17),
 *   residual (-2/17, 8/17), distance sqrt(68)/17 = 2/sqrt(17).
 * Before the fix the planar shader formed `start + fraction * (end - start)` in absolute float32
 * coordinates; at 1e7 (spacing 1) the first distance became 1 and the second 0, at 5e6 (spacing
 * 0.5) the second became 0.5 (Math.fround transcription of the WGSL expression).
 */
type TranslationCase = {
  start: [number, number];
  end: [number, number];
  point: [number, number];
  radius: number;
  fraction: number;
  distance: number;
};

const CASES: TranslationCase[] = [
  {start: [0, 0], end: [2, 2], point: [1, 0], radius: 0.8, fraction: 0.25, distance: Math.SQRT1_2},
  {
    start: [0, 0],
    end: [4, 1],
    point: [2, 1],
    radius: 0.49,
    fraction: 9 / 17,
    distance: 2 / Math.sqrt(17)
  }
];
const OFFSETS = [0, 5e6, 1e7];

function translate(values: readonly number[], offset: number): Float32Array {
  const translated = Float32Array.from(values, value => value + offset);
  // Guard the fixture: the translation must be exact in float32.
  translated.forEach((value, index) => expect(value - offset).toBe(values[index]));
  return translated;
}

type SegmentResult = {edge: number; fraction: number; distance: number};

function createGraphIO(device: Device, graph: GPUCommandGraph) {
  const buffers: Buffer[] = [];
  const outputs: Record<string, Buffer> = {};
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      id,
      buffer,
      format,
      format === 'float32x2' ? values.length / 2 : values.length
    );
  };
  const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, format === 'float32x2' ? length * 2 : length);
    buffers.push(buffer);
    outputs[id] = buffer;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  return {buffers, outputs, input, output};
}

async function runSnappingScan(
  device: Device,
  testCase: TranslationCase,
  offset: number
): Promise<SegmentResult> {
  const graph = new GPUCommandGraph(device, {id: 'snapping-translation'});
  const {buffers, outputs, input, output} = createGraphIO(device, graph);
  const limit = new GPUParameterBuffer(device, {
    id: 'snapping-translation-limit',
    format: 'float32',
    length: 1,
    values: Float32Array.of(testCase.radius)
  });
  // No candidateCapacity: the exhaustive scan route.
  graph.add(
    new GPUNetworkSnapping({
      id: 'snap',
      positions: input('points', translate(testCase.point, offset), 'float32x2'),
      topology: {
        offsets: input('offsets', Uint32Array.of(0, 1, 1), 'uint32'),
        neighbors: input('targets', Uint32Array.of(1), 'uint32'),
        nodePositions: input(
          'positions',
          translate([...testCase.start, ...testCase.end], offset),
          'float32x2'
        )
      },
      maximumSnapDistance: limit.importToGraph(graph),
      output: {
        snappedEdges: output('edges', 'uint32', 1),
        snapFractions: output('fractions', 'float32', 1),
        snapDistances: output('distances', 'float32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      edge: (await readUint32(outputs['edges'], 1))[0],
      fraction: (await readFloat32(outputs['fractions'], 1))[0],
      distance: (await readFloat32(outputs['distances'], 1))[0]
    };
  } finally {
    compiled.destroy();
    limit.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

async function runMapMatching(
  device: Device,
  testCase: TranslationCase,
  offset: number
): Promise<SegmentResult> {
  const graph = new GPUCommandGraph(device, {id: 'map-matching-translation'});
  const {buffers, outputs, input, output} = createGraphIO(device, graph);
  const parameters = new GPUParameterBuffer(device, {
    id: 'map-matching-translation-parameters',
    format: 'float32',
    length: 8,
    values: getGPUMapMatchingParameterValues({sigma: 1, beta: 1, searchRadius: testCase.radius})
  });
  graph.add(
    new GPUMapMatching({
      id: 'match',
      positions: input('points', translate(testCase.point, offset), 'float32x2'),
      trackOffsets: input('track-offsets', Uint32Array.of(0, 1), 'uint32'),
      topology: {
        offsets: input('offsets', Uint32Array.of(0, 1, 1), 'uint32'),
        neighbors: input('targets', Uint32Array.of(1), 'uint32'),
        nodePositions: input(
          'nodes',
          translate([...testCase.start, ...testCase.end], offset),
          'float32x2'
        )
      },
      parameters: parameters.importToGraph(graph),
      cellSize: 1,
      bounds: {minimum: [offset - 2, offset - 2], maximum: [offset + 6, offset + 6]},
      output: {
        matchedEdges: output('edges', 'uint32', 1),
        matchedFractions: output('fractions', 'float32', 1),
        snapDistances: output('distances', 'float32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      edge: (await readUint32(outputs['edges'], 1))[0],
      fraction: (await readFloat32(outputs['fractions'], 1))[0],
      distance: (await readFloat32(outputs['distances'], 1))[0]
    };
  } finally {
    compiled.destroy();
    parameters.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function expectAdmitted(result: SegmentResult, testCase: TranslationCase, label: string): void {
  expect(result.edge, `${label} edge`).toBe(0);
  expect(Number.isFinite(result.fraction), `${label} fraction finite`).toBe(true);
  expect(Number.isFinite(result.distance), `${label} distance finite`).toBe(true);
  expect(result.fraction, `${label} fraction`).toBeCloseTo(testCase.fraction, 5);
  expect(result.distance, `${label} distance`).toBeCloseTo(testCase.distance, 5);
}

it('GPUNetworkSnapping scan admission and distance do not change when the road is translated (NET-106)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of CASES) {
    for (const offset of OFFSETS) {
      const result = await runSnappingScan(device, testCase, offset);
      expect(GPU_NETWORK_SNAPPING_NONE).not.toBe(0);
      expectAdmitted(result, testCase, `snap road ${testCase.end} at +${offset}`);
    }
  }
});

it('GPUMapMatching candidate admission and distance do not change when the road is translated (NET-106)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of CASES) {
    for (const offset of OFFSETS) {
      const result = await runMapMatching(device, testCase, offset);
      expect(GPU_MAP_MATCHING_NONE).not.toBe(0);
      expectAdmitted(result, testCase, `match road ${testCase.end} at +${offset}`);
    }
  }
});
