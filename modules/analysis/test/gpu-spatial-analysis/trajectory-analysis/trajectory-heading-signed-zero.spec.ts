// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// FX-17: the planar step heading of `GPUTrajectoryMetrics` is `atan2(deltaY, deltaX)` with
// `deltaX = x[i] - x[i - 1]`. A due north or south step from x = +0 to x = -0 makes `deltaX`
// exactly -0.0 (IEEE: -0 - +0 = -0), and the test GPU's `atan2(y, -0.0)` returns the opposite
// half-turn. Headings are compared with `Math.atan2`, with the zero `deltaX` treated as +0.

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** One track on the line x = 0 with signed zeros: north, north, south, then east and west steps. */
const POSITIONS: [number, number][] = [
  [0, 0],
  [-0, 1],
  [0, 2],
  [-0, 1],
  [1, 1],
  [0, 1]
];

/**
 * Rows after the first whose step heading is missing, extra, nonfinite or more than 1e-6 from
 * `Math.atan2` with a zero `deltaX` treated as +0. Written so that NaN fails (TOOL-109).
 */
function getHeadingFailures(positions: [number, number][], headings: ArrayLike<number>): string[] {
  const failures: string[] = [];
  if (headings.length !== positions.length) {
    failures.push(`length ${headings.length} != ${positions.length}`);
  }
  for (let row = 1; row < positions.length; row++) {
    const deltaX = Math.fround(positions[row][0]) - Math.fround(positions[row - 1][0]);
    const deltaY = Math.fround(positions[row][1]) - Math.fround(positions[row - 1][1]);
    const expected = Math.atan2(deltaY, deltaX === 0 ? 0 : deltaX);
    const actual = headings[row];
    if (!Number.isFinite(actual) || !(Math.abs(actual - expected) <= 1e-6)) {
      failures.push(`row ${row}: GPU ${actual}, CPU ${expected}`);
    }
  }
  return failures;
}

it('GPUTrajectoryMetrics planar heading is correct for due north and south steps across signed zeros', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = POSITIONS.length;
  const positionsBuffer = createInputBuffer(device, new Float32Array(POSITIONS.flat()));
  const timestampsBuffer = createInputBuffer(
    device,
    Float32Array.from({length: rowCount}, (_, index) => index)
  );
  const offsetsBuffer = createInputBuffer(device, Uint32Array.from([0, rowCount]));
  const parameters = new GPUParameterBuffer(device, {
    id: 'trajectory-heading-parameters',
    format: 'float32',
    length: GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
    values: getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 0, stopMinimumDuration: 0})
  });
  const headingsBuffer = createOutputBuffer(device, rowCount);
  const graph = new GPUCommandGraph<void>(device, {id: 'trajectory-heading-signed-zero'});
  graph.add(
    new GPUTrajectoryMetrics({
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      timestamps: importGraphBuffer(graph, 'timestamps', timestampsBuffer, 'float32', rowCount),
      trackOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', 2),
      parameters: parameters.importToGraph(graph),
      output: {
        stepHeadings: importGraphBuffer(graph, 'headings', headingsBuffer, 'float32', rowCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const headings = await readFloat32(headingsBuffer, rowCount);
  compiled.destroy();
  parameters.destroy();
  for (const buffer of [positionsBuffer, timestampsBuffer, offsetsBuffer, headingsBuffer]) {
    buffer.destroy();
  }
  const failures = getHeadingFailures(POSITIONS, headings);
  expect(failures).toEqual([]);
});
