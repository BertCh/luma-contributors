// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUOffsetCurveComponentsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';
import {
  GPUOffsetCurveComponentsOperation,
  type OffsetCurveComponentsOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  expectOffsetCurveLineWithSubdivisions,
  OFFSET_CURVE_ENDPOINT_CASES
} from './offset-curve-components-test-utils';

it('multipart offset operations preserve feature and part provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // See multipart-offset-components.oracle.py: straight offsets from Shapely; source rows by hand.
  const props: OffsetCurveComponentsOperationProps = {
    original: {
      positions: 'originalPositions',
      lineOffsets: 'originalOffsets',
      featureOffsets: 'originalFeatures'
    },
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets', featureOffsets: 'rawFeatures'},
    lineCount: 3,
    distance: 1,
    tolerance: 0,
    intersectionCapacity: 8,
    segmentCapacity: 8,
    vertexCapacity: 6,
    componentCapacity: 3,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {
      format: 'float32x2' as const,
      values: Float32Array.of(0, 0, 2, 0, 5, 0, 7, 0, 10, 0, 12, 0)
    },
    originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2, 4, 6)},
    originalFeatures: {format: 'uint32' as const, values: Uint32Array.of(0, 0, 2, 3)},
    rawPositions: {
      format: 'float32x2' as const,
      values: Float32Array.of(0, 1, 2, 1, 5, 1, 7, 1, 10, 1, 12, 1)
    },
    rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2, 4, 6)},
    rawFeatures: {format: 'uint32' as const, values: Uint32Array.of(0, 0, 2, 3)}
  };
  for (const Operation of [GPUOffsetCurveComponentsOperation, CPUOffsetCurveComponentsOperation]) {
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    expect(Array.from(result['positions'])).toEqual([0, 1, 2, 1, 5, 1, 7, 1, 10, 1, 12, 1]);
    expect(Array.from(result['components'])).toEqual([0, 2, 0, 1, 2, 2, 1, 1, 4, 2, 2, 2]);
    expect(Array.from(result['sourceEdges'])).toEqual([0, 0, 2, 2, 4, 4]);
    expect(Array.from(result['status'])).toEqual([3, 6, 3, 6, 0, 0, 0, 0]);
  }
});

it.each([
  1, 8
])('CPU/GPU honour segmentCapacity=%s with two noded segments', async segmentCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: OffsetCurveComponentsOperationProps = {
    id: 'components-segment-budget',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 1,
    distance: 1,
    tolerance: 0,
    intersectionCapacity: 8,
    segmentCapacity,
    vertexCapacity: 8,
    componentCapacity: 4,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 0, 2, 0)},
    originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 3)},
    rawPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 1, 1, 1, 2, 1)},
    rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 3)}
  };
  const vertexCount = segmentCapacity === 1 ? 2 : 3;
  for (const Operation of [GPUOffsetCurveComponentsOperation, CPUOffsetCurveComponentsOperation]) {
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    expect(Array.from(result['status'])).toEqual([
      1,
      vertexCount,
      1,
      vertexCount,
      0,
      segmentCapacity === 1 ? 1 : 0,
      0,
      0
    ]);
    expect(Array.from(result['positions'].subarray(0, vertexCount * 2))).toEqual([
      0,
      1,
      1,
      1,
      ...(vertexCount === 3 ? [2, 1] : [])
    ]);
    expect(Array.from(result['components'].subarray(0, 4))).toEqual([0, vertexCount, 0, 0]);
    expect(Array.from(result['sourceEdges'].subarray(0, vertexCount))).toEqual(
      vertexCount === 2 ? [0, 0] : [0, 0, 1]
    );
  }
});

it.each([
  1, 3
])('CPU/GPU report intersectionCapacity=%s for three crossing pairs', async intersectionCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: OffsetCurveComponentsOperationProps = {
    id: 'components-intersection-budget',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 3,
    distance: 1,
    tolerance: 0,
    intersectionCapacity,
    segmentCapacity: 16,
    vertexCapacity: 16,
    componentCapacity: 4,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {
      format: 'float32x2' as const,
      values: Float32Array.of(-2, -10, 2, -10, -2, -10, 2, -10, -2, -10, 2, -10)
    },
    originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2, 4, 6)},
    rawPositions: {
      format: 'float32x2' as const,
      values: Float32Array.of(-2, 0, 2, 0, 0, -2, 0, 2, -2, -2, 2, 2)
    },
    rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2, 4, 6)}
  };
  for (const Operation of [GPUOffsetCurveComponentsOperation, CPUOffsetCurveComponentsOperation]) {
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    if (intersectionCapacity === 1) {
      expect(result['status'][4]).toBe(0);
      expect(result['status'][5]).toBe(1);
      if (Operation === CPUOffsetCurveComponentsOperation) {
        expect(Array.from(result['status'])).toEqual([0, 0, 0, 0, 0, 1, 0, 0]);
        for (const port of ['positions', 'components', 'sourceEdges'])
          expect(result[port].every(value => value === 0)).toBe(true);
      }
    } else {
      expect(Array.from(result['status'])).toEqual([3, 9, 3, 9, 0, 0, 0, 0]);
      expect(Array.from(result['positions'].subarray(0, 18))).toEqual([
        -2, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 2, -2, -2, 0, 0, 2, 2
      ]);
      expect(Array.from(result['components'].subarray(0, 12))).toEqual([
        0, 3, 0, 0, 3, 3, 1, 0, 6, 3, 2, 0
      ]);
      expect(Array.from(result['sourceEdges'].subarray(0, 9))).toEqual([0, 0, 0, 2, 2, 2, 4, 4, 4]);
    }
  }
});

it.each(OFFSET_CURVE_ENDPOINT_CASES)('CPU/GPU endpoint identity: $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: OffsetCurveComponentsOperationProps = {
    id: 'components-endpoint-identity',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 1,
    distance: fixture.distance,
    tolerance: 0,
    intersectionCapacity: 32,
    segmentCapacity: 32,
    vertexCapacity: 16,
    componentCapacity: 4,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {
      format: 'float32x2' as const,
      values: Float32Array.from(fixture.originalPositions)
    },
    originalOffsets: {
      format: 'uint32' as const,
      values: Uint32Array.of(0, fixture.originalPositions.length / 2)
    },
    rawPositions: {format: 'float32x2' as const, values: Float32Array.from(fixture.rawPositions)},
    rawOffsets: {
      format: 'uint32' as const,
      values: Uint32Array.of(0, fixture.rawPositions.length / 2)
    }
  };
  for (const Operation of [GPUOffsetCurveComponentsOperation, CPUOffsetCurveComponentsOperation]) {
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    const diagnostic = JSON.stringify(
      Object.fromEntries(Object.entries(result).map(([port, values]) => [port, Array.from(values)]))
    );
    expect(Array.from(result['status']), diagnostic).toEqual(fixture.status);
    expect(
      Array.from(result['positions'].subarray(0, fixture.positions.length)),
      diagnostic
    ).toEqual(fixture.positions);
    expect(Array.from(result['components']), diagnostic).toEqual([
      ...fixture.components,
      ...new Array(16 - fixture.components.length).fill(0)
    ]);
    expect(
      Array.from(result['sourceEdges'].subarray(0, fixture.sourceEdges.length)),
      diagnostic
    ).toEqual(fixture.sourceEdges);
  }
});

for (const scene of ['straight', 'spur', 'short-edges'] as const) {
  it.each([
    2 ** -20,
    1,
    2 ** 20
  ])(`preserves CPU/GPU offset topology and provenance for ${scene} at scale %s`, async scale => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const distance = (scene === 'short-edges' ? 10000 : 1) * scale;
    const originalPositions = Float32Array.from(
      scene === 'short-edges'
        ? [1000, 0, 1000.03125, 0, 1000.0625, 0, 11000, 0, 21000, 0]
        : scene === 'spur'
          ? [0, 0, 6, 0]
          : [0, 0, 1, 0, 2, 0],
      value => value * scale
    );
    const rawPositions =
      scene === 'spur'
        ? Float32Array.from([0, 1, 3, 1, 3, 2, 3, 1, 6, 1], value => value * scale)
        : originalPositions.map((value, index) => (index % 2 === 1 ? distance : value));
    const expectedPositions =
      scene === 'spur'
        ? Float32Array.from([0, 1, 3, 1, 6, 1], value => value * scale)
        : rawPositions;
    const expectedSourceEdges =
      scene === 'short-edges' ? [0, 0, 1, 2, 3] : scene === 'spur' ? [0, 0, 3] : [0, 0, 1];
    const vertexCount = expectedPositions.length / 2;
    const props: OffsetCurveComponentsOperationProps = {
      id: `components-scale-${scene}`,
      original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
      raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
      lineCount: 1,
      distance,
      tolerance: 0,
      intersectionCapacity: 32,
      segmentCapacity: 32,
      vertexCapacity: 16,
      componentCapacity: 4,
      output: {
        positions: 'positions',
        components: 'components',
        sourceEdges: 'sourceEdges',
        status: 'status'
      }
    };
    const inputs = {
      originalPositions: {format: 'float32x2' as const, values: originalPositions},
      originalOffsets: {
        format: 'uint32' as const,
        values: Uint32Array.of(0, originalPositions.length / 2)
      },
      rawPositions: {format: 'float32x2' as const, values: rawPositions},
      rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, rawPositions.length / 2)}
    };
    for (const Operation of [
      GPUOffsetCurveComponentsOperation,
      CPUOffsetCurveComponentsOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs,
        operations: [new Operation(props)],
        outputs: Object.values(props.output)
      });
      // Assert against the hand-computed line, so a shared CPU/GPU defect cannot pass by parity.
      if (scene === 'short-edges') {
        // Float32 circle tangencies can add collinear cuts on the long edges. These do not
        // change topology: require all original vertices/provenance, including the short prefix.
        expectOffsetCurveLineWithSubdivisions(result, expectedPositions, expectedSourceEdges);
        continue;
      }
      expect(Array.from(result['status'])).toEqual([1, vertexCount, 1, vertexCount, 0, 0, 0, 0]);
      expect(Array.from(result['components'])).toEqual([
        0,
        vertexCount,
        0,
        0,
        ...new Array(12).fill(0)
      ]);
      expect(Array.from(result['sourceEdges'].subarray(0, vertexCount))).toEqual(
        expectedSourceEdges
      );
      // All copied coordinates are exactly representable powers-of-two-scaled float32 values.
      // Compare only published vertices: unused GPU position rows are unspecified.
      expect(Array.from(result['positions'].subarray(0, vertexCount * 2))).toEqual(
        Array.from(expectedPositions)
      );
    }
  });
}

for (const scene of ['ordinary', 'bevel', 'circle', 'overflow', 'empty'] as const) {
  it(`offset components compare every CPU/GPU output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: OffsetCurveComponentsOperationProps = {
      id: `components-${scene}`,
      original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
      raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
      lineCount: 2,
      distance: 1,
      tolerance: 0,
      intersectionCapacity: 32,
      segmentCapacity: 32,
      vertexCapacity: scene === 'overflow' ? 4 : 16,
      componentCapacity: scene === 'overflow' ? 2 : 4,
      output: {
        positions: 'positions',
        components: 'components',
        sourceEdges: 'sourceEdges',
        status: 'status'
      }
    };
    const inputs = {
      originalPositions: {
        format: 'float32x2' as const,
        values: Float32Array.of(0, 0, 6, 0, 10, 0, 16, 0)
      },
      originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2, 4)},
      rawPositions: {
        format: 'float32x2' as const,
        values:
          scene === 'empty'
            ? Float32Array.of(0, 0.5, 6, 0.5, 10, 0.5, 16, 0.5)
            : Float32Array.of(
                0,
                1,
                3,
                1,
                3,
                2,
                3,
                1,
                6,
                1,
                10,
                1,
                12,
                1,
                12,
                0,
                14,
                0,
                14,
                1,
                16,
                1
              )
      },
      rawOffsets: {
        format: 'uint32' as const,
        values: scene === 'empty' ? Uint32Array.of(0, 2, 4) : Uint32Array.of(0, 5, 11)
      }
    };
    if (scene === 'bevel') {
      props.distance = -1;
      props.rowsPerVertex = 3;
      inputs.originalPositions.values = Float32Array.of(0, 0, 4, 0, 4, 4, 10, 0, 14, 0, 14, 4);
      inputs.originalOffsets.values = Uint32Array.of(0, 3, 6);
      // Exact padded raw GPUOffsetCurve bevels of two left turns, offset to their right.
      inputs.rawPositions.values = Float32Array.of(
        0,
        -1,
        0,
        -1,
        0,
        -1,
        4,
        -1,
        5,
        0,
        5,
        0,
        5,
        4,
        5,
        4,
        5,
        4,
        10,
        -1,
        10,
        -1,
        10,
        -1,
        14,
        -1,
        15,
        0,
        15,
        0,
        15,
        4,
        15,
        4,
        15,
        4
      );
      inputs.rawOffsets.values = Uint32Array.of(0, 9, 18);
    }
    if (scene === 'circle') {
      inputs.rawPositions.values = Float32Array.of(-0.75, -2, -0.75, 2, 9.25, -2, 9.25, 2);
      inputs.rawOffsets.values = Uint32Array.of(0, 2, 4);
    }
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUOffsetCurveComponentsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUOffsetCurveComponentsOperation(props)],
      outputs
    });
    for (const port of ['components', 'sourceEdges', 'status'])
      expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
    // One final f32 rounding vs shader intermediates: 2e-5 absolute planar-unit tolerance.
    expect(gpu['positions'].length).toBe(cpu['positions'].length);
    cpu['positions'].forEach((value, index) =>
      expect(Math.abs(gpu['positions'][index] - value)).toBeLessThanOrEqual(2e-5)
    );
    expect(Array.from(cpu['status'])).toEqual(
      scene === 'ordinary'
        ? [3, 7, 3, 7, 0, 0, 0, 0]
        : scene === 'bevel'
          ? [2, 8, 2, 8, 0, 0, 0, 0]
          : scene === 'circle'
            ? [4, 8, 4, 8, 0, 0, 0, 0]
            : scene === 'overflow'
              ? [1, 3, 3, 7, 1, 0, 0, 0]
              : new Array(8).fill(0)
    );
  });
}

it.each([
  [0, 0],
  [0, 1],
  [2, 0]
])('compares offset vertexCapacity=%i and componentCapacity=%i', async (vertexCapacity, componentCapacity) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: OffsetCurveComponentsOperationProps = {
    id: 'components-zero-capacity',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 1,
    distance: 1,
    tolerance: 0,
    intersectionCapacity: 4,
    segmentCapacity: 4,
    vertexCapacity,
    componentCapacity,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 0)},
    originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)},
    rawPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 1, 1, 1)},
    rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  // Copied unit coordinates and integer words are exact; compare numeric zeros without sign checks.
  for (const port of outputs) {
    expect(gpu[port].length).toBe(cpu[port].length);
    cpu[port].forEach((value, index) => expect(gpu[port][index] === value).toBe(true));
  }
  for (const result of [gpu, cpu]) {
    expect(Array.from(result['status'])).toEqual([0, 0, 1, 2, 1, 0, 0, 0]);
    expect(Array.from(result['positions'])).toEqual(vertexCapacity === 0 ? [] : [0, 1, 1, 1]);
    expect(Array.from(result['components'])).toEqual(new Array(componentCapacity * 4).fill(0));
    expect(Array.from(result['sourceEdges'])).toEqual(new Array(vertexCapacity).fill(0));
  }
});

it('compares the finite separation sentinel for a singleton original at distance 3.2e38', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: OffsetCurveComponentsOperationProps = {
    id: 'components-separation-sentinel',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 1,
    distance: 3.2e38,
    tolerance: 0,
    intersectionCapacity: 4,
    segmentCapacity: 4,
    vertexCapacity: 2,
    componentCapacity: 1,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
  const inputs = {
    originalPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0)},
    originalOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
    rawPositions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 0)},
    rawOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUOffsetCurveComponentsOperation(props)],
    outputs
  });
  // No original edge is visited, so the finite initializer is below the requested distance and
  // rejects the fragment. Every status word and fresh output row should therefore be zero.
  for (const port of outputs) {
    expect(gpu[port].length).toBe(cpu[port].length);
    for (const result of [gpu, cpu]) expect(result[port].every(value => value === 0)).toBe(true);
  }
});
