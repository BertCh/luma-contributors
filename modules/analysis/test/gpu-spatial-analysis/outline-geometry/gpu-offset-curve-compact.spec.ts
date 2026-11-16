// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {expect, it} from 'vitest';
import {
  getGPUOffsetCurveParameterValues,
  getGPUOffsetCurveRowsPerVertex,
  GPUOffsetCurve
} from '../../../src/gpu-spatial-analysis/outline-geometry/index';
import {createGeometryFixture} from './geometry-fixture';
import {getCompactOffsetExpected} from './compact-offset-oracle';
import {OFFSET_CURVE_REFERENCE_CASES} from './offset-curve-reference';
import {requireTestDevice} from '../../utils/require-test-device';

for (const referenceCase of OFFSET_CURVE_REFERENCE_CASES) {
  it(`compact offset matches the pinned GEOS oracle: ${referenceCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {vertices, joinStyle, quadSegments, ring} = referenceCase;
    const capacity = vertices.length * getGPUOffsetCurveRowsPerVertex(joinStyle, quadSegments);
    const fixture = createGeometryFixture(device, {
      inputs: {
        positions: {values: new Float32Array(vertices.flat()), format: 'float32x2'},
        lines: {values: new Uint32Array([0, vertices.length]), format: 'uint32'},
        single: {values: new Uint32Array([0, 1]), format: 'uint32'}
      },
      outputs: {
        positions: {format: 'float32x2', length: capacity},
        offsets: {format: 'uint32', length: 2},
        sourceRows: {format: 'uint32', length: capacity},
        sourceLines: {format: 'uint32', length: 1},
        count: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        vertices: {format: 'uint32', length: 1},
        required: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUOffsetCurve({
          positions: inputs['positions'] as never,
          ...(ring
            ? {
                geometryType: 'polygons' as const,
                ringOffsets: inputs['lines'] as never,
                polygonOffsets: inputs['single'] as never,
                featureOffsets: inputs['single'] as never
              }
            : {lineOffsets: inputs['lines'] as never}),
          joinStyle,
          quadSegments,
          parameters,
          output: {
            layout: 'compact',
            positions: outputs['positions'] as never,
            lineOffsets: outputs['offsets'] as never,
            sourceRows: outputs['sourceRows'] as never,
            sourceLineIndices: outputs['sourceLines'] as never,
            status: {
              count: outputs['count'] as never,
              overflow: outputs['overflow'] as never,
              vertexCount: outputs['vertices'] as never,
              requiredVertexCount: outputs['required'] as never
            }
          }
        })
    });
    try {
      const result = await fixture.run(
        getGPUOffsetCurveParameterValues({
          distance: referenceCase.distance,
          mitreLimit: referenceCase.mitreLimit
        })
      );
      const written = result['vertices'][0];
      const actual = Array.from({length: written}, (_, index) =>
        result['positions'].slice(index * 2, index * 2 + 2)
      );
      const expected = getCompactOffsetExpected(referenceCase);
      expect(written).toBe(expected.length);
      expected.forEach((point, index) => {
        expect(Math.abs(actual[index][0] - point[0]), `x at ${index}`).toBeLessThan(2e-4);
        expect(Math.abs(actual[index][1] - point[1]), `y at ${index}`).toBeLessThan(2e-4);
      });
      expect(result['offsets']).toEqual([0, written]);
      expect(result['count']).toEqual([1]);
      expect(result['required']).toEqual([written]);
      expect(result['overflow']).toEqual([0]);
      expect(result['sourceLines']).toEqual([0]);
      const sourceRows = result['sourceRows'].slice(0, written);
      expect(
        sourceRows.every(
          (row, index) => row < vertices.length && (index === 0 || row >= sourceRows[index - 1])
        )
      ).toBe(true);
    } finally {
      fixture.destroy();
    }
  });
}

it('compacts two million straight-line vertices without the old 17-fold padded allocation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The old default padded output would require 272 MB in one binding (>128 MiB).
  const rowCount = 2_000_000;
  const positions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) positions[row * 2] = row;
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: positions, format: 'float32x2'},
      lines: {values: new Uint32Array([0, rowCount]), format: 'uint32'}
    },
    outputs: {
      // A small capacity also exercises exact required count and suffix truncation at scale.
      positions: {format: 'float32x2', length: 4},
      offsets: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      required: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUOffsetCurve({
        positions: inputs['positions'] as never,
        lineOffsets: inputs['lines'] as never,
        parameters,
        output: {
          layout: 'compact',
          positions: outputs['positions'] as never,
          lineOffsets: outputs['offsets'] as never,
          status: {
            count: outputs['count'] as never,
            overflow: outputs['overflow'] as never,
            requiredVertexCount: outputs['required'] as never
          }
        }
      })
  });
  try {
    const result = await fixture.run(getGPUOffsetCurveParameterValues({distance: 1}));
    expect(result['positions']).toEqual([0, 1, 1, 1, 2, 1, 3, 1]);
    expect(result['offsets']).toEqual([0, 4]);
    expect(result['required']).toEqual([rowCount]);
    expect(result['overflow']).toEqual([1]);
    expect(result['count']).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});

// Axis-aligned fixtures provide exact integer-valued expectations, independent of GPU routing.
function createCornerInput(rowCount: number) {
  const positions: number[] = [];
  const offsets = [0, 0]; // leading empty line
  const expected: number[] = [];
  const sourceRows: number[] = [];
  const outputOffsets = [0, 0];
  for (let first = 0; first < rowCount; first += 3) {
    const count = Math.min(3, rowCount - first);
    const origin = first * 4;
    const turnLeft = (first / 3) % 2 === 0;
    positions.push(origin, 0);
    if (count > 1) positions.push(origin + 1, 0);
    if (count > 2) positions.push(origin + 1, turnLeft ? 1 : -1);
    if (count === 1) {
      expected.push(origin, 0);
      sourceRows.push(first);
    } else if (count === 2) {
      expected.push(origin, 2, origin + 1, 2);
      sourceRows.push(first, first + 1);
    } else if (turnLeft) {
      expected.push(origin, 2, origin + 1, 2, origin + 1, 0, origin - 1, 0, origin - 1, 1);
      sourceRows.push(first, first + 1, first + 1, first + 1, first + 2);
    } else {
      expected.push(origin, 2, origin + 1, 2, origin + 3, 0, origin + 3, -1);
      sourceRows.push(first, first + 1, first + 1, first + 2);
    }
    offsets.push(first + count);
    outputOffsets.push(sourceRows.length);
    if (first === 0) {
      offsets.push(first + count);
      outputOffsets.push(sourceRows.length);
    }
  }
  offsets.push(rowCount);
  outputOffsets.push(sourceRows.length); // trailing empty line
  return {positions, offsets, expected, sourceRows, outputOffsets};
}

for (const rowCount of [255, 256, 257, 65_535, 65_536, 65_537]) {
  it(`checks every mixed corner and terminal total at ${rowCount} source rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createCornerInput(rowCount);
    const fixture = createGeometryFixture(device, {
      inputs: {
        positions: {values: new Float32Array(input.positions), format: 'float32x2'},
        lines: {values: new Uint32Array(input.offsets), format: 'uint32'}
      },
      outputs: {
        positions: {format: 'float32x2', length: input.sourceRows.length},
        sources: {format: 'uint32', length: input.sourceRows.length},
        offsets: {format: 'uint32', length: input.offsets.length},
        count: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        required: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUOffsetCurve({
          positions: inputs['positions'] as never,
          lineOffsets: inputs['lines'] as never,
          parameters,
          joinStyle: 'bevel',
          insideTurns: 'buffer',
          output: {
            layout: 'compact',
            positions: outputs['positions'] as never,
            sourceRows: outputs['sources'] as never,
            lineOffsets: outputs['offsets'] as never,
            status: {
              count: outputs['count'] as never,
              overflow: outputs['overflow'] as never,
              requiredVertexCount: outputs['required'] as never
            }
          }
        })
    });
    try {
      const result = await fixture.run(getGPUOffsetCurveParameterValues({distance: 2}));
      expect(result['positions']).toEqual(input.expected);
      expect(result['sources']).toEqual(input.sourceRows);
      expect(result['offsets']).toEqual(input.outputOffsets);
      expect(result['count']).toEqual([input.offsets.length - 1]);
      expect(result['required']).toEqual([input.sourceRows.length]);
      expect(result['overflow']).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  });
}

for (const capacity of [2, 5, 6]) {
  it(`packs separate-pass status, uses unaligned subviews and rewrites parameters at capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers = [
      createInputBuffer(device, new Float32Array([99, 0, 0, 1, 0, 1, 1, 10, 0, 12, 0])),
      createInputBuffer(device, new Uint32Array([99, 0, 0, 3, 3, 5, 5])),
      createOutputBuffer(device, capacity * 2 + 3),
      createOutputBuffer(device, capacity + 1),
      createOutputBuffer(device, 6 + 1),
      createOutputBuffer(device, 2),
      createOutputBuffer(device, 1)
    ];
    const imported = buffers.map((buffer, index) =>
      importGraphBuffer(
        graph,
        `packed-${index}`,
        buffer,
        index === 0 || index === 2 ? 'float32' : 'uint32',
        buffer.byteLength / 4
      )
    );
    const positions = graph.createDataView(imported[0].buffer, {
      format: 'float32x2',
      length: 5,
      byteOffset: 4
    });
    const lineOffsets = graph.createDataView(imported[1].buffer, {
      format: 'uint32',
      length: 6,
      byteOffset: 4
    });
    const outputPositions = graph.createDataView(imported[2].buffer, {
      format: 'float32x2',
      length: capacity,
      byteOffset: 12
    });
    const sourceRows = graph.createDataView(imported[3].buffer, {
      format: 'uint32',
      length: capacity,
      byteOffset: 4
    });
    const outputOffsets = graph.createDataView(imported[4].buffer, {
      format: 'uint32',
      length: 6,
      byteOffset: 4
    });
    const count = graph.createDataView(imported[5].buffer, {format: 'uint32', length: 1});
    const overflow = graph.createDataView(imported[5].buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const parameterBuffer = new GPUParameterBuffer(device, {
      id: 'offset-parameters',
      format: 'float32',
      length: 4
    });
    const parameters = parameterBuffer.importToGraph(graph);
    const output = {
      layout: 'compact' as const,
      positions: outputPositions,
      sourceRows,
      lineOffsets: outputOffsets,
      status: {count, overflow, requiredVertexCount: imported[6] as never}
    };
    // Co-bound generation outputs cannot share an aligned prefix, even with disjoint logical rows.
    expect(
      () =>
        new GPUOffsetCurve({
          positions,
          lineOffsets,
          parameters,
          joinStyle: 'bevel',
          output: {
            ...output,
            sourceRows: graph.createDataView(imported[2].buffer, {
              format: 'uint32',
              length: capacity
            })
          }
        })
    ).toThrow(/byte ranges must not overlap/);
    graph.add(
      new GPUOffsetCurve({
        positions,
        lineOffsets,
        parameters,
        joinStyle: 'bevel',
        insideTurns: 'buffer',
        output
      })
    );
    const compiled = graph.compile();
    try {
      for (const distance of [2, -2, 2]) {
        parameterBuffer.write(getGPUOffsetCurveParameterValues({distance}));
        submitGraph(device, compiled, undefined);
        const expected =
          distance > 0
            ? [0, 2, 1, 2, 1, 0, -1, 0, -1, 1, 10, 2, 12, 2]
            : [0, -2, 1, -2, 3, 0, 3, 1, 10, -2, 12, -2];
        const required = expected.length / 2;
        const firstLineCount = distance > 0 ? 5 : 4;
        const sources = distance > 0 ? [0, 1, 1, 1, 2, 3, 4] : [0, 1, 1, 2, 3, 4];
        expect((await readFloat32(buffers[2], capacity * 2 + 3)).slice(3)).toEqual(
          expected.slice(0, capacity * 2)
        );
        expect((await readUint32(buffers[3], capacity + 1)).slice(1)).toEqual(
          sources.slice(0, capacity)
        );
        expect((await readUint32(buffers[4], 7)).slice(1)).toEqual([
          0,
          0,
          Math.min(capacity, firstLineCount),
          Math.min(capacity, firstLineCount),
          Math.min(capacity, required),
          Math.min(capacity, required)
        ]);
        expect(await readUint32(buffers[5], 2)).toEqual([5, Number(required > capacity)]);
        expect(await readUint32(buffers[6], 1)).toEqual([required]);
      }
    } finally {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
