// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {validateCPULineFeatureLayout} from '../../../src/analysis-core/contracts/line-feature-provenance';
import {
  createGPUMultipartRowOwnersNode,
  GPU_MULTIPART_NO_FEATURE_ROW,
  type GPUFeatureGeometryPort,
  type GPUMultipartGeometryPort,
  type GPUPackedGeometryPort,
  validateGPUFeatureGeometryPort,
  validateGPUMultipartGeometryPort
} from '../../../src/analysis-core/contracts/index';
import {GPUAffineTransform} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../../gpu-spatial-analysis/segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();

it.each([
  [Uint32Array.of(0), 0],
  [Uint32Array.of(0, 0), 1]
] as const)('W3-06 accepts empty multipart layout %s with %s features', (featureOffsets, featureCount) => {
  expect(
    validateCPULineFeatureLayout(
      'empty-lines',
      new Float32Array(),
      Uint32Array.of(0),
      featureOffsets
    )
  ).toBe(0);
  // Offsets contain one terminal word in addition to the logical feature rows.
  expect(featureOffsets.length - 1).toBe(featureCount);
});

it.each([
  [Uint32Array.of(0), Uint32Array.of(0, 1)],
  [Uint32Array.of(1), Uint32Array.of(0)],
  [Uint32Array.of(0, 1), Uint32Array.of(0, 1)],
  [Uint32Array.of(0, 0), Uint32Array.of(0, 1, 0)]
] as const)('W3-06 rejects a nonzero or malformed empty hierarchy %s %s', (lineOffsets, featureOffsets) => {
  expect(() =>
    validateCPULineFeatureLayout('empty-lines', new Float32Array(), lineOffsets, featureOffsets)
  ).toThrow(/empty-lines/);
});

expectTypeOf<GPUPackedGeometryPort<GPUMultipartGeometryPort>>().toExtend<GPUFeatureGeometryPort>();

it('default line geometry requires a part-offset level for multipart affine inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    expect(
      () =>
        new GPUAffineTransform({
          positions: createTransientView(graph, 'positions', 'float32x2', 4),
          featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
          parameters: createTransientView(graph, 'parameters', 'float32', 12),
          output: {positions: createTransientView(graph, 'output', 'float32x2', 4)}
        })
    ).toThrow(/affine-transform featureOffsets requires lineOffsets/);
  } finally {
    device.destroy();
  }
});

for (const kind of ['points', 'lines'] as const) {
  it(`multipart ${kind} source IDs align with features rather than parts`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const geometry = {
      kind,
      positions: createTransientView(graph, 'positions', 'float32x2', kind === 'points' ? 2 : 4),
      lineOffsets: createTransientView(graph, 'parts', 'uint32', 3),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      sourceIds: createTransientView(graph, 'identities', 'uint32', 1)
    };
    try {
      expect(() => validateGPUFeatureGeometryPort('multipart', geometry)).not.toThrow();
      expect(() =>
        validateGPUFeatureGeometryPort('multipart', {
          ...geometry,
          sourceIds: createTransientView(graph, 'part-identities', 'uint32', 2)
        })
      ).toThrow(/multipart sourceIds length must equal the feature count/);
    } finally {
      device.destroy();
    }
  });

  it(`multipart ${kind} requires a uint32 feature-to-part level with a terminal entry`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const geometry = {
      kind,
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      lineOffsets: createTransientView(graph, 'parts', 'uint32', 3),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 0)
    };
    try {
      expect(() => validateGPUFeatureGeometryPort('multipart', geometry)).toThrow(
        /multipart featureOffsets must contain a terminal entry/
      );
    } finally {
      device.destroy();
    }
  });

  it(`multipart ${kind} validates formats and allows a terminal-only empty hierarchy`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const geometry = {
      kind,
      positions: createTransientView(graph, 'positions', 'float32x2', 0),
      lineOffsets: createTransientView(graph, 'parts', 'uint32', 1),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 1),
      sourceIds: createTransientView(graph, 'identities', 'uint32', 0)
    };
    try {
      expect(() => validateGPUFeatureGeometryPort('empty', geometry)).not.toThrow();
      expect(() => validateGPUMultipartGeometryPort('empty', geometry)).not.toThrow();
      const output = createTransientView(graph, 'empty-owners', 'uint32', 0);
      const memory = createCPUGraphMemory();
      memory.write(geometry.featureOffsets, Uint32Array.of(0));
      executeCommandNodesOnCPU(
        [createGPUMultipartRowOwnersNode(graph, 'empty-owners-node', geometry, output)],
        memory
      );
      expect(Array.from(memory.readUint32(output))).toEqual([]);
      expect(() =>
        validateGPUFeatureGeometryPort('bad-format', {
          ...geometry,
          featureOffsets: createTransientView(graph, 'bad-features', 'float32', 1)
        } as unknown as GPUMultipartGeometryPort)
      ).toThrow(/bad-format featureOffsets must use uint32/);
    } finally {
      device.destroy();
    }
  });

  it(`multipart ${kind} consumers retain feature counts and share terminal validation`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const geometry = {
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      lineOffsets: createTransientView(graph, 'parts', 'uint32', 3),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
    };
    const measures = (featureOffsets = geometry.featureOffsets) =>
      new GPUGeometryMeasures({
        ...geometry,
        geometryType: kind,
        featureOffsets,
        output: {
          bounds: createTransientView(graph, `bounds-${featureOffsets.length}`, 'float32x4', 1)
        }
      });
    const affine = (featureOffsets = geometry.featureOffsets) =>
      new GPUAffineTransform({
        ...geometry,
        geometryType: kind,
        featureOffsets,
        featureTransforms: createTransientView(
          graph,
          `transforms-${featureOffsets.length}`,
          'float32',
          6
        ),
        output: {
          positions: createTransientView(
            graph,
            `transformed-${featureOffsets.length}`,
            'float32x2',
            4
          )
        }
      });
    try {
      expect(measures().featureCount).toBe(1);
      expect(affine().featureCount).toBe(1);
      const emptyOffsets = createTransientView(graph, 'missing-terminal', 'uint32', 0);
      expect(() => measures(emptyOffsets)).toThrow(
        /geometry-measures featureOffsets must contain a terminal entry/
      );
      expect(() => affine(emptyOffsets)).toThrow(
        /affine-transform featureOffsets must contain a terminal entry/
      );
    } finally {
      device.destroy();
    }
  });
}

for (const kind of ['points', 'lines'] as const) {
  it(`multipart ${kind} owners skip empty features and reset on offset changes`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const featureOffsets = createTransientView(graph, 'features', 'uint32', 6);
    const featureRows = createTransientView(graph, 'owners', 'uint32', 4);
    const geometry = {
      kind,
      positions: createTransientView(graph, 'positions', 'float32x2', kind === 'points' ? 4 : 6),
      // Four line parts, including an empty part. Vertex count is independent of part count.
      lineOffsets: createTransientView(graph, 'parts', 'uint32', 5),
      featureOffsets,
      sourceIds: createTransientView(graph, 'identities', 'uint32', 5)
    };
    try {
      const node = createGPUMultipartRowOwnersNode(graph, 'owners-node', geometry, featureRows);
      const memory = createCPUGraphMemory();
      memory.write(geometry.lineOffsets, Uint32Array.of(0, 2, 2, 4, 6));
      memory.write(geometry.sourceIds, Uint32Array.of(99, 99, 99, 99, 99));
      memory.write(featureOffsets, Uint32Array.of(0, 0, 2, 2, 4, 4));
      executeCommandNodesOnCPU([node], memory);
      expect(Array.from(memory.readUint32(featureRows))).toEqual([1, 1, 3, 3]);
      memory.write(featureOffsets, Uint32Array.of(0, 1, 1, 1, 4, 4));
      executeCommandNodesOnCPU([node], memory);
      expect(Array.from(memory.readUint32(featureRows))).toEqual([0, 3, 3, 3]);
      // Incomplete input ranges are not empty features. They have no owner, not feature row 0.
      memory.write(featureOffsets, Uint32Array.of(1, 1, 2, 2, 3, 3));
      executeCommandNodesOnCPU([node], memory);
      expect(Array.from(memory.readUint32(featureRows))).toEqual([
        GPU_MULTIPART_NO_FEATURE_ROW,
        1,
        3,
        GPU_MULTIPART_NO_FEATURE_ROW
      ]);
    } finally {
      device.destroy();
    }
  });
}

it('multipart owners reject short outputs and aliased read/write buffers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const featureOffsets = createTransientView(graph, 'features', 'uint32', 2);
  const geometry = {
    kind: 'points' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    featureOffsets
  };
  try {
    expect(() =>
      createGPUMultipartRowOwnersNode(
        graph,
        'owners',
        geometry,
        createTransientView(graph, 'short-output', 'uint32', 1)
      )
    ).toThrow(/owners featureRows length must equal the part count/);
    expect(() =>
      createGPUMultipartRowOwnersNode(graph, 'owners', geometry, featureOffsets)
    ).toThrow(/featureRows.*featureOffsets/);
    const malformedLines = {
      ...geometry,
      kind: 'lines' as const,
      lineOffsets: createTransientView(graph, 'missing-part-terminal', 'uint32', 0)
    };
    expect(() => validateGPUMultipartGeometryPort('lines', malformedLines)).toThrow(
      /lines lineOffsets must contain a terminal entry/
    );
    const emptyFeatures = {
      ...geometry,
      featureOffsets: createTransientView(graph, 'no-features', 'uint32', 1)
    };
    const output = createTransientView(graph, 'unowned-parts', 'uint32', 2);
    const memory = createCPUGraphMemory();
    memory.write(emptyFeatures.featureOffsets, Uint32Array.of(0));
    executeCommandNodesOnCPU(
      [createGPUMultipartRowOwnersNode(graph, 'empty-owners', emptyFeatures, output)],
      memory
    );
    expect(Array.from(memory.readUint32(output))).toEqual([
      GPU_MULTIPART_NO_FEATURE_ROW,
      GPU_MULTIPART_NO_FEATURE_ROW
    ]);
  } finally {
    device.destroy();
  }
});

it('multipart owners honor packed subview offsets and binding-size limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'feature-backing', 'uint32', 4);
  const featureOffsets = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 3
  });
  const outputBacking = createTransientView(graph, 'owner-backing', 'uint32', 4);
  const output = graph.createDataView(outputBacking.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 3
  });
  const geometry = {
    kind: 'points' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    featureOffsets
  };
  try {
    const memory = createCPUGraphMemory();
    memory.write(backing, Uint32Array.of(99, 0, 2, 3));
    memory.write(outputBacking, Uint32Array.of(99, 99, 99, 99));
    executeCommandNodesOnCPU(
      [createGPUMultipartRowOwnersNode(graph, 'offset-owners', geometry, output)],
      memory
    );
    expect(Array.from(memory.readUint32(outputBacking))).toEqual([99, 0, 0, 1]);
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 12},
      configurable: true
    });
    expect(() => createGPUMultipartRowOwnersNode(graph, 'limited', geometry, output)).toThrow(
      /limited featureOffsets needs 16 bytes/
    );
  } finally {
    device.destroy();
  }
});
