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
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('refuses silent suppression with required outputs only and accepts a genuinely empty input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SegmentSnapOperationProps = {
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 2,
    referenceCount: 2,
    lineCount: 1,
    tolerance: 0,
    maximumVerticesPerLine: 8,
    candidateCapacity: 1,
    candidateOverflowPolicy: 'incomplete',
    output: {positions: 'snapped', sourceEdges: 'edges', counts: 'counts'}
  };
  for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation]) {
    expect(() => new Operation(props)).toThrow(/output.*(validity|issueCode|candidateOverflow)/);
    const empty = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array()},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 0)},
        references: {format: 'float32x2', values: Float32Array.of(1, 0, 2, 0)}
      },
      operations: [new Operation({...props, rowCount: 0})],
      outputs: ['snapped', 'edges', 'counts']
    });
    expect(Array.from(empty['counts'])).toEqual([0, 0, 0, 0]);
    expect(Array.from(empty['snapped'])).toEqual(Array(16).fill(0));
    expect(Array.from(empty['edges'])).toEqual(Array(8).fill(0));
  }
});

it.each([
  {candidateCapacity: 1, maximumVerticesPerLine: 8, counts: [0, 0, 0, 0], candidate: 1},
  {candidateCapacity: 2, maximumVerticesPerLine: 2, counts: [2, 4, 1, 0], candidate: 0}
])('operation forms separate candidate and vertex capacity: %o', async options => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, 3, 0)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
        references: {format: 'float32x2', values: Float32Array.of(1, 0, 2, 0)}
      },
      operations: [
        new Operation({
          positions: 'positions',
          lineOffsets: 'offsets',
          referencePositions: 'references',
          rowCount: 2,
          referenceCount: 2,
          lineCount: 1,
          tolerance: 0,
          candidateCapacity: options.candidateCapacity,
          maximumVerticesPerLine: options.maximumVerticesPerLine,
          candidateOverflowPolicy: 'incomplete',
          output: {
            positions: 'output',
            sourceEdges: 'edges',
            counts: 'counts',
            issueCode: 'issues',
            validity: 'validity',
            status: {candidateOverflow: 'candidateOverflow'}
          }
        })
      ],
      outputs: ['counts', 'issues', 'validity', 'candidateOverflow']
    });
    expect(Array.from(result['counts'])).toEqual(options.counts);
    expect(Array.from(result['candidateOverflow'])).toEqual([options.candidate]);
    expect(Array.from(result['issues'])).toEqual([4]);
    expect(Array.from(result['validity'])).toEqual([0]);
  }
});

it('NaN query lines agree on candidate capacity, counts, issue codes and usability', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SegmentSnapOperationProps = {
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 2,
    referenceCount: 3,
    lineCount: 1,
    tolerance: 0,
    candidateCapacity: 1,
    candidateOverflowPolicy: 'incomplete',
    maximumVerticesPerLine: 2,
    output: {
      positions: 'output',
      sourceEdges: 'edges',
      counts: 'counts',
      issueCode: 'issues',
      validity: 'validity',
      status: {candidateOverflow: 'candidateOverflow'}
    }
  };
  for (const Operation of [GPUSegmentSnapOperation, CPUSegmentSnapOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, NaN, 0)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
        references: {format: 'float32x2', values: Float32Array.of(0, 0, 4, 0, 6, 0)}
      },
      operations: [new Operation(props)],
      outputs: ['counts', 'issues', 'validity', 'candidateOverflow']
    });
    expect(Array.from(result['counts'])).toEqual([0, 0, 0, 1]);
    expect(Array.from(result['candidateOverflow'])).toEqual([0]);
    expect(Array.from(result['issues'])).toEqual([2]);
    expect(Array.from(result['validity'])).toEqual([0]);
  }
});

it('submits aligned counts, issue code and validity views in one buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(device, Float32Array.of(0, 0, 2, 0));
  const offsets = createInputBuffer(device, Uint32Array.of(0, 2));
  const references = createInputBuffer(device, Float32Array.of(0, 0));
  const output = createOutputBuffer(device, 4);
  const edges = createOutputBuffer(device, 2);
  const packed = createOutputBuffer(device, 129);
  const packedView = importGraphBuffer(graph, 'packed', packed, 'uint32', 129);
  graph.add(
    new GPUSegmentSnap({
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
      lineOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 2),
      referencePositions: importGraphBuffer(graph, 'references', references, 'float32x2', 0),
      tolerance: 0,
      maximumVerticesPerLine: 2,
      output: {
        positions: importGraphBuffer(graph, 'output', output, 'float32x2', 2),
        sourceEdges: importGraphBuffer(graph, 'edges', edges, 'uint32', 2),
        counts: graph.createDataView(packedView.buffer, {
          format: 'uint32x4',
          length: 1,
          byteOffset: 0
        }),
        validity: graph.createDataView(packedView.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 256
        }),
        issueCode: graph.createDataView(packedView.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: 512
        })
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const words = await readUint32(packed, 129);
    expect(words.slice(0, 4)).toEqual([2, 2, 0, 0]);
    expect(words[64]).toBe(1);
    expect(words[128]).toBe(0);
  } finally {
    compiled.destroy();
    for (const buffer of [positions, offsets, references, output, edges, packed]) buffer.destroy();
  }
});

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
        issueCode: 'issues',
        candidateOverflow: 'candidateOverflow'
      }
    };
    const references = new Float32Array([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0, 10, 0, 2, 0.5]);
    if (invalid) references[0] = NaN;
    const outputs = ['snapped', 'edges', 'counts', 'validity', 'issues', 'candidateOverflow'];
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
    expect(Array.from(results[0]['validity'])).toEqual(invalid ? [0, 0] : [1, 0]);
    expect(Array.from(results[0]['issues'])).toEqual(invalid ? [2, 2] : [0, 4]);
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
      issueCode: 'issues',
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
      outputs: ['counts', 'validity', 'issues', 'candidateOverflow']
    });
    expect(Array.from(result['candidateOverflow'])).toEqual([1, ...Array(lineCount - 1).fill(0)]);
    // Candidate loss suppresses the line without exhausting its final vertex slots.
    expect(Array.from(result['counts'])).toEqual([
      0,
      0,
      0,
      0,
      ...Array((lineCount - 1) * 4).fill(0)
    ]);
    expect(Array.from(result['issues'])).toEqual([4, ...Array(lineCount - 1).fill(0)]);
    expect(Array.from(result['validity'])).toEqual([0, ...Array(lineCount - 1).fill(1)]);
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
