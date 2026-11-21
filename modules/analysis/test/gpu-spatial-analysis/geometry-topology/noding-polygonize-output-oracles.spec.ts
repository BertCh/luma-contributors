// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createReviewNodingCase,
  REVIEW_NODING_CASES,
  matchesSnappedReviewParameters
} from './noding-review-output-oracles-fixtures';
import {createReviewPolygonizeCase} from './polygonize-review-output-oracles-fixtures';
import {
  createReview2OffsetCase,
  createReview2StackedSquaresCase,
  REVIEW2_OFFSET_OPTIONS,
  REVIEW2_OFFSET_ORIGINAL,
  REVIEW2_OFFSET_RAW
} from './review2-output-oracles-fixtures';
import {CPUOffsetCurveComponentsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';
import {GPUOffsetCurveComponentsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createNodingOracleCase,
  createPolygonizeOracleCase,
  getOracleView,
  type OracleAllocate
} from './noding-polygonize-output-oracles-fixtures';

async function runOracleCase(
  createFixture: (
    graph: GPUCommandGraph,
    allocate: OracleAllocate
  ) =>
    | ReturnType<typeof createNodingOracleCase>
    | ReturnType<typeof createPolygonizeOracleCase>
    | ReturnType<typeof createReviewNodingCase>
    | ReturnType<typeof createReviewPolygonizeCase>
    | ReturnType<typeof createReview2OffsetCase>
    | ReturnType<typeof createReview2StackedSquaresCase>,
  allowSnappedParameterRounding = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  const allocate: OracleAllocate = (name, format, length, values) => {
    const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    const data = format.startsWith('uint')
      ? Uint32Array.from(values ?? new Array(length * width).fill(777))
      : Float32Array.from(values ?? new Array(length * width).fill(-777));
    const buffer = values
      ? createInputBuffer(device, data)
      : createOutputBuffer(device, length * width);
    if (!values) buffer.write(data);
    const view = importGraphBuffer(graph, name, buffer, format, length);
    buffers.set(view, buffer);
    return view;
  };
  const fixture = createFixture(graph, allocate);
  for (const node of fixture.nodes) graph.add(node);
  const compiled = graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const path of Object.keys(fixture.expected)) {
        const view = getOracleView(fixture.output, path);
        const words = (view.length * view.rowByteLength) / 4;
        buffers
          .get(view)!
          .write(
            view.format.startsWith('uint')
              ? new Uint32Array(words).fill(777)
              : new Float32Array(words).fill(-777)
          );
      }
      submitGraph(device, compiled, undefined);
      for (const [path, expected] of Object.entries(fixture.expected)) {
        const view = getOracleView(fixture.output, path);
        const actual = view.format.startsWith('uint')
          ? await readUint32(buffers.get(view)!, expected.length)
          : await readFloat32(buffers.get(view)!, expected.length);
        if (
          allowSnappedParameterRounding &&
          (path === 'sourceStartParameters' || path === 'sourceEndParameters')
        ) {
          expect(matchesSnappedReviewParameters(actual, expected), path).toBe(true);
        } else {
          expect(actual, path).toEqual(expected);
        }
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers.values()) buffer.destroy();
    device.destroy();
  }
}

it.each(REVIEW_NODING_CASES)('review noding oracle %s', async subject => {
  await runOracleCase(
    (graph, allocate) => createReviewNodingCase(graph, allocate, subject),
    subject === 'snapped'
  );
});

it('review 2 GPU offset preserves surviving traversal row 2', async () => {
  await runOracleCase((graph, allocate) => createReview2OffsetCase(graph, allocate));
});

it.each([
  CPUOffsetCurveComponentsOperation,
  GPUOffsetCurveComponentsOperation
])('review 2 %s preserves surviving offset traversal row 2', async Operation => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const props = {
      ...REVIEW2_OFFSET_OPTIONS,
      lineCount: 1,
      original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
      raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
      output: {
        positions: 'positions',
        components: 'components',
        sourceEdges: 'sourceEdges',
        status: 'status'
      }
    };
    const result = await runOperationGraph(device, {
      inputs: {
        originalPositions: {
          format: 'float32x2',
          values: Float32Array.from(REVIEW2_OFFSET_ORIGINAL)
        },
        originalOffsets: {format: 'uint32', values: Uint32Array.of(0, 4)},
        rawPositions: {format: 'float32x2', values: Float32Array.from(REVIEW2_OFFSET_RAW)},
        rawOffsets: {format: 'uint32', values: Uint32Array.of(0, 4)}
      },
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    // The first two traversals cancel; only raw edge 2 survives (fixture derivation).
    expect(Array.from(result['positions'])).toEqual([0, 1, 2, 1]);
    expect(Array.from(result['components'])).toEqual([0, 2, 0, 0]);
    expect(Array.from(result['sourceEdges'])).toEqual([2, 2]);
    expect(Array.from(result['status'])).toEqual([1, 2, 1, 2, 0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
});

it.each([
  8, 4096
])('review 2 stacked %i squares bound each endpoint to two candidates', async squareCount => {
  await runOracleCase((graph, allocate) =>
    createReview2StackedSquaresCase(graph, allocate, squareCount)
  );
});

it.each([
  'spur',
  'chain',
  'short-spur',
  'overlap-square-first',
  'overlap-exterior-first'
] as const)('review polygonize recursively removes dangles: %s', async subject => {
  await runOracleCase((graph, allocate) => createReviewPolygonizeCase(graph, allocate, subject));
});

it('review uncertain counter omission still propagates incompleteness', async () => {
  await runOracleCase((graph, allocate) =>
    createReviewNodingCase(graph, allocate, 'uncertain', false)
  );
});

it.each([
  {mode: 'line', uncertainty: false},
  {mode: 'line', uncertainty: true},
  {mode: 'snap', uncertainty: false},
  {mode: 'snap', uncertainty: true}
] as const)('$mode noding pins quarter-crossing provenance with uncertainty=$uncertainty', async ({
  mode,
  uncertainty
}) => {
  await runOracleCase((graph, allocate) =>
    createNodingOracleCase(graph, allocate, uncertainty, undefined, mode)
  );
});

it.each([
  {uncertainty: 'none', snapped: false},
  {uncertainty: 'top', snapped: false},
  {uncertainty: 'noded', snapped: false},
  {uncertainty: 'directed', snapped: false},
  {uncertainty: 'all', snapped: false},
  {uncertainty: 'none', snapped: true},
  {uncertainty: 'all', snapped: true}
] as const)('polygonize pins bridge diagnostics and ownership with uncertainty=$uncertainty snap=$snapped', async ({
  uncertainty,
  snapped
}) => {
  await runOracleCase((graph, allocate) =>
    createPolygonizeOracleCase(graph, allocate, uncertainty, snapped)
  );
});
