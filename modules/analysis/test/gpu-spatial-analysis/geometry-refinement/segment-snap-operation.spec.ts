// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSegmentSnapOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';
import {
  GPUSegmentSnapOperation,
  type SegmentSnapOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const invalid of [false, true])
  it(`compares every segment-snap output (${invalid ? 'invalid reference' : 'two lines, ties, insertions and overflow'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SegmentSnapOperationProps = {
      positions: 'positions',
      lineOffsets: 'offsets',
      referencePositions: 'references',
      rowCount: 4,
      referenceCount: 7,
      lineCount: 2,
      tolerance: 0.6,
      maximumVerticesPerLine: 5,
      output: {
        positions: 'snapped',
        sourceEdges: 'edges',
        counts: 'counts',
        validity: 'validity',
        candidateOverflow: 'candidateOverflow'
      }
    };
    const references = new Float32Array([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0, 10, 0, 2, 0.5]);
    if (invalid) references[0] = NaN;
    const outputs = ['snapped', 'edges', 'counts', 'validity', 'candidateOverflow'];
    const results = [];
    for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {
              format: 'float32x2',
              values: new Float32Array([0.2, 0, 6.2, 0, 0.2, 0, 10.1, 0])
            },
            offsets: {format: 'uint32', values: new Uint32Array([0, 2, 4])},
            references: {format: 'float32x2', values: references}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    // All coordinates are copied from input/reference rows and all other fields are integer:
    // compare exactly, numerically (including zeros). No floating-point tolerance is needed.
    for (const port of outputs) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++)
        expect(results[0][port][index] === results[1][port][index]).toBe(true);
    }
  });

it('indexes a million references and pins inserted positions and original edge rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 1024;
  const references = new Float32Array(width * width * 2);
  for (let vertical = 0; vertical < width; vertical++)
    for (let horizontal = 0; horizontal < width; horizontal++) {
      const row = vertical * width + horizontal;
      references[row * 2] = horizontal;
      references[row * 2 + 1] = vertical;
    }
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.from({length: 8192 * 2}, (_, index) => (index === 2 ? 3 : 0))
      },
      offsets: {format: 'uint32', values: new Uint32Array([0, 2])},
      references: {format: 'float32x2', values: references}
    },
    operations: [
      new GPUSegmentSnapOperation({
        positions: 'positions',
        lineOffsets: 'offsets',
        referencePositions: 'references',
        rowCount: 8192,
        referenceCount: width * width,
        lineCount: 1,
        tolerance: 0,
        maximumVerticesPerLine: 4,
        candidateCapacity: 256,
        searchStructure: 'grid',
        output: {
          positions: 'snapped',
          sourceEdges: 'edges',
          counts: 'counts',
          status: {candidateOverflow: 'candidateOverflow'}
        }
      })
    ],
    outputs: ['snapped', 'edges', 'counts', 'candidateOverflow']
  });
  expect(Array.from(result['snapped'])).toEqual([0, 0, 1, 0, 2, 0, 3, 0]);
  expect(Array.from(result['edges'])).toEqual([0, 0, 0, 0]);
  expect(Array.from(result['counts'])).toEqual([4, 4, 0, 0]);
  expect(Array.from(result['candidateOverflow'])).toEqual([0]);
});

it.each([
  1, 16
])('keeps exact geometry when the candidate list capacity is %s', async candidateCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const base = {
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 2,
    referenceCount: 4,
    lineCount: 1,
    tolerance: 0,
    maximumVerticesPerLine: 4,
    candidateCapacity,
    output: {
      positions: 'snapped',
      sourceEdges: 'edges',
      counts: 'counts',
      status: {candidateOverflow: 'candidateOverflow'}
    }
  };
  for (const searchStructure of ['grid', 'brute-force'] as const) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array([0, 0, 3, 0])},
        offsets: {format: 'uint32', values: new Uint32Array([0, 2])},
        references: {format: 'float32x2', values: new Float32Array([3, 0, 1, 0, 0, 0, 2, 0])}
      },
      operations: [new GPUSegmentSnapOperation({...base, searchStructure})],
      outputs: ['snapped', 'counts', 'candidateOverflow']
    });
    expect(Array.from(result['snapped'])).toEqual([0, 0, 1, 0, 2, 0, 3, 0]);
    expect(Array.from(result['counts'])).toEqual([4, 4, 0, 0]);
    expect(Array.from(result['candidateOverflow'])).toEqual([
      Number(searchStructure === 'grid' && candidateCapacity === 1)
    ]);
  }
});

it.each([
  'grid',
  'brute-force'
] as const)('preserves translated collinearity and large finite coordinates (%s)', async searchStructure => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [positions, references, expected] of [
    [
      [-100, 0, -94, 0],
      [-99, 0, -99, 0, -99, 0],
      [-100, 0, -99, 0, -94, 0]
    ],
    [
      [0, 0, 4e20, 0],
      [3e20, 0, 1e20, 0, 2e20, 0],
      [0, 0, 1e20, 0, 2e20, 0, 3e20, 0, 4e20, 0]
    ]
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(positions)},
        offsets: {format: 'uint32', values: new Uint32Array([0, 2])},
        references: {format: 'float32x2', values: new Float32Array(references)}
      },
      operations: [
        new GPUSegmentSnapOperation({
          positions: 'positions',
          lineOffsets: 'offsets',
          referencePositions: 'references',
          rowCount: 2,
          referenceCount: 3,
          lineCount: 1,
          searchStructure,
          tolerance: 0,
          maximumVerticesPerLine: expected.length / 2,
          output: {positions: 'snapped', sourceEdges: 'edges', counts: 'counts'}
        })
      ],
      outputs: ['snapped', 'edges', 'counts']
    });
    expect(Array.from(result['snapped'])).toEqual(Array.from(new Float32Array(expected)));
    expect(Array.from(result['counts'])).toEqual([expected.length / 2, expected.length / 2, 0, 0]);
    expect(Array.from(result['edges'])).toEqual(Array(expected.length / 2).fill(0));
  }
});

it('keeps candidate-overflow parity when all default lists would exceed 128 MiB and bounds exhaustive work', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lineCount = 40_000;
  const references = new Float32Array(1025 * 2);
  const props: SegmentSnapOperationProps = {
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 2,
    referenceCount: 1025,
    lineCount,
    tolerance: 0,
    maximumVerticesPerLine: 2,
    output: {
      positions: 'snapped',
      sourceEdges: 'edges',
      counts: 'counts',
      validity: 'validity',
      status: {candidateOverflow: 'candidateOverflow'}
    }
  };
  for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(4)},
        offsets: {
          format: 'uint32',
          values: Uint32Array.from({length: lineCount + 1}, (_, index) => (index === 0 ? 0 : 2))
        },
        references: {format: 'float32x2', values: references}
      },
      operations: [new Operation(props)],
      outputs: ['counts', 'validity', 'candidateOverflow']
    });
    expect(Array.from(result['candidateOverflow'])).toEqual([1, ...Array(lineCount - 1).fill(0)]);
    expect(Array.from(result['counts'])).toEqual([
      0,
      0,
      1,
      0,
      ...Array((lineCount - 1) * 4).fill(0)
    ]);
    expect(Array.from(result['validity'])).toEqual([4, ...Array(lineCount - 1).fill(0)]);
  }
});

it('retains shuffled duplicates and equal-distance ties when bounded exhaustive overflow is allowed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array([0.2, 0, 6.2, 0])},
      offsets: {format: 'uint32', values: new Uint32Array([0, 2])},
      references: {
        format: 'float32x2',
        values: new Float32Array([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0, 10, 0, 2, 0.5])
      }
    },
    operations: [
      new GPUSegmentSnapOperation({
        positions: 'positions',
        lineOffsets: 'offsets',
        referencePositions: 'references',
        rowCount: 2,
        referenceCount: 7,
        lineCount: 1,
        tolerance: 0.6,
        maximumVerticesPerLine: 4,
        candidateCapacity: 1,
        candidateOverflowPolicy: 'bounded-brute-force',
        output: {
          positions: 'snapped',
          sourceEdges: 'edges',
          counts: 'counts',
          status: {candidateOverflow: 'candidateOverflow'}
        }
      })
    ],
    outputs: ['snapped', 'counts', 'candidateOverflow']
  });
  expect(Array.from(result['snapped'])).toEqual([0, 0, 2, 0.5, 2, -0.5, 6, 0]);
  expect(Array.from(result['counts'])).toEqual([4, 4, 0, 0]);
  expect(Array.from(result['candidateOverflow'])).toEqual([1]);
});
