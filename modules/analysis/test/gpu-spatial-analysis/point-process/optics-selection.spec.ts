// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const INVALID_ROW = 0xffffffff;

/** Independent sorted-distance reference on collinear integer coordinates. */
function getReference(
  coordinates: number[],
  minimumSamples: number,
  maximumDistance: number,
  extractionDistance: number,
  maximumIterations: number
) {
  const rowCount = coordinates.length;
  const coreDistances = coordinates.map(coordinate => {
    if (!Number.isFinite(coordinate)) return Infinity;
    const distances = coordinates
      .filter(Number.isFinite)
      .map(other => Math.abs(coordinate - other))
      .filter(distance => distance <= maximumDistance)
      .sort((first, second) => first - second);
    return distances[minimumSamples - 1] ?? Infinity;
  });
  const ordering = new Array<number>(rowCount).fill(INVALID_ROW);
  const predecessors = new Array<number>(rowCount).fill(INVALID_ROW);
  const reachability = new Array<number>(rowCount).fill(Infinity);
  const labels = new Array<number>(rowCount).fill(INVALID_ROW);
  const processed = coordinates.map(coordinate => !Number.isFinite(coordinate));
  const invalidCount = processed.filter(Boolean).length;
  let visitedCount = 0;
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    let focus = INVALID_ROW;
    for (let row = 0; row < rowCount; row++) {
      if (!processed[row] && (focus === INVALID_ROW || reachability[row] < reachability[focus])) {
        focus = row;
      }
    }
    if (focus === INVALID_ROW) break;
    ordering[visitedCount++] = focus;
    processed[focus] = true;
    if (!Number.isFinite(coreDistances[focus])) continue;
    for (let row = 0; row < rowCount; row++) {
      if (processed[row]) continue;
      const distance = Math.abs(coordinates[focus] - coordinates[row]);
      const candidate = Math.max(coreDistances[focus], distance);
      if (distance <= maximumDistance && candidate < reachability[row]) {
        reachability[row] = candidate;
        predecessors[row] = focus;
      }
    }
  }
  const complete = visitedCount + invalidCount === rowCount;
  if (complete) {
    let cluster = -1;
    for (const row of ordering.slice(0, visitedCount)) {
      if (reachability[row] > extractionDistance) {
        if (coreDistances[row] <= extractionDistance) labels[row] = ++cluster;
      } else {
        labels[row] = cluster < 0 ? INVALID_ROW : cluster;
      }
    }
  }
  return {
    ordering,
    predecessors,
    coreDistances,
    reachability,
    labels,
    status: [visitedCount, Number(complete), invalidCount]
  };
}

const mixedCoordinates = Array.from({length: 273}, (_, row) =>
  row % 61 === 0 ? NaN : ((row * 73) % 257) + (row % 3 === 0 ? 1024 : 0)
);
const cases = [
  ...[1, 2, 17, 32, 128, 256].map(minimumSamples => ({
    name: `rank ${minimumSamples}, ties and invalid rows across scan blocks`,
    coordinates: mixedCoordinates,
    minimumSamples,
    maximumDistance: 2048,
    maximumIterations: mixedCoordinates.length
  })),
  {
    name: 'inclusive radius, duplicate points, noise and separate clusters',
    coordinates: [0, 20, 1, 21, 0, 22, 100, NaN, Infinity],
    minimumSamples: 3,
    maximumDistance: 2,
    maximumIterations: 9
  },
  {
    name: 'too few neighbors for the largest supported rank',
    coordinates: [0, 1, 1, NaN, 2],
    minimumSamples: 256,
    maximumDistance: 1,
    maximumIterations: 5
  },
  {
    name: 'partial ordering across frontier batch boundaries',
    coordinates: mixedCoordinates,
    minimumSamples: 17,
    maximumDistance: 2048,
    maximumIterations: 129
  },
  {
    name: 'all invalid rows complete with only noise',
    coordinates: [NaN, Infinity, -Infinity],
    minimumSamples: 2,
    maximumDistance: 1,
    maximumIterations: 3
  }
];

for (const testCase of cases) {
  it(`OPTICS selection preserves every output: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const positionsBuffer = createInputBuffer(
      device,
      Float32Array.from(testCase.coordinates.flatMap(coordinate => [coordinate, 0]))
    );
    buffers.push(positionsBuffer);
    const rowCount = testCase.coordinates.length;
    const output = <Format extends 'float32' | 'uint32'>(format: Format, length = rowCount) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, `optics-selection-${buffers.length}`, buffer, format, length)
      };
    };
    const outputs = {
      ordering: output('uint32'),
      coreDistances: output('float32'),
      reachability: output('float32'),
      predecessors: output('uint32'),
      labels: output('uint32'),
      status: output('uint32', 3)
    };
    const referenceCoreDistances = output('float32');
    const positions = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount);
    // Preserve the original repeated-rank GPU algorithm as a test-only numerical regression oracle.
    // Both algorithms use WGSL distance(), whose sqrt can differ from the real integer distance.
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'optics-original-core-reference',
        operation: 'GPUOPTICS',
        variant: 'original-core-reference',
        invocationCount: rowCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'core', view: referenceCoreDistances.view, type: 'f32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_FLOAT_WGSL}
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn validPoint(row: u32) -> bool {
  let value = point(row);
  return isFiniteFloat(value.x) && isFiniteFloat(value.y);
}`,
        body: `let infinity = bitcast<f32>(0x7f800000u | (index & 0u));
  var previousDistance = -1.0;
  var previousRow = 0xffffffffu;
  var selectedDistance = infinity;
  if (validPoint(index)) {
    for (var rank = 0u; rank < ${testCase.minimumSamples}u; rank++) {
      selectedDistance = infinity;
      var selectedRow = 0xffffffffu;
      for (var row = 0u; row < ${rowCount}u; row++) {
        if (!validPoint(row)) { continue; }
        let pairDistance = distance(point(index), point(row));
        let after = pairDistance > previousDistance || (pairDistance == previousDistance && row > previousRow);
        if (after && pairDistance <= ${testCase.maximumDistance}.0 && (pairDistance < selectedDistance || (pairDistance == selectedDistance && row < selectedRow))) {
          selectedDistance = pairDistance;
          selectedRow = row;
        }
      }
      previousDistance = selectedDistance;
      previousRow = selectedRow;
    }
  }
  core[coreOffset + index] = selectedDistance;`
      })
    );
    const extractionDistance = testCase.maximumDistance / 2;
    graph.add(
      new GPUOPTICS({
        positions,
        minimumSamples: testCase.minimumSamples,
        maximumDistance: testCase.maximumDistance,
        extractionDistance,
        maximumIterations: testCase.maximumIterations,
        frontierBatchSize: 7,
        output: {
          ordering: outputs.ordering.view,
          coreDistances: outputs.coreDistances.view,
          reachability: outputs.reachability.view,
          predecessors: outputs.predecessors.view,
          labels: outputs.labels.view,
          status: outputs.status.view
        }
      })
    );
    const compiled = graph.compile();
    try {
      // Re-encoding changed inputs must clear old labels, invalid counts and frontier state.
      for (const coordinates of [testCase.coordinates, [...testCase.coordinates].reverse()]) {
        positionsBuffer.write(
          Float32Array.from(coordinates.flatMap(coordinate => [coordinate, 0]))
        );
        submitGraph(device, compiled, undefined);
        const reference = getReference(
          coordinates,
          testCase.minimumSamples,
          testCase.maximumDistance,
          extractionDistance,
          testCase.maximumIterations
        );
        for (const field of ['ordering', 'predecessors', 'labels', 'status'] as const) {
          expect(await readUint32(outputs[field].buffer, outputs[field].view.length)).toEqual(
            reference[field]
          );
        }
        expect(await readFloat32(outputs.coreDistances.buffer, rowCount)).toEqual(
          await readFloat32(referenceCoreDistances.buffer, rowCount)
        );
        for (const field of ['coreDistances', 'reachability'] as const) {
          const actual = await readFloat32(outputs[field].buffer, rowCount);
          for (let row = 0; row < rowCount; row++) {
            const expected = reference[field][row];
            if (!Number.isFinite(expected) || expected === 0) {
              expect(actual[row]).toBe(expected);
            } else {
              // Integer-coordinate squares are exact here, but Metal's distance/sqrt has a
              // one-ULP rounding difference at some integer roots (1010 -> 1010.0000610351562).
              // The original GPU selector above must still agree bit for bit; discrete outputs
              // retain exact assertions against the independent mathematical CPU reference.
              const bitPatterns = new Uint32Array(Float32Array.of(actual[row], expected).buffer);
              expect(Math.abs(bitPatterns[0] - bitPatterns[1])).toBeLessThanOrEqual(1);
            }
          }
        }
      }
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  });
}
