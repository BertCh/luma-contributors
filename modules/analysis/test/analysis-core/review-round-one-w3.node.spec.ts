// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../src/testing/gpu-contributor-test-utils';
import {
  createFixedPointGroupedScaleNodes,
  createFixedPointDecodeNode
} from '../../src/analysis-core/kernels/fixed-point-accumulator';
import {
  GPUFittingCoordinateEncoder,
  type GPUFittingCoordinateEncoderProps
} from '../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {validateDisjointWritableViews} from '../../src/analysis-core/kernels/graph-chunk-dispatch';
import {createGraphChunkGroupKernelNode} from '../../src/analysis-core/kernels/graph-chunk-groups';
import * as kernels from '../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('B1 rejects every strided grouped-scale port before creating nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const port of ['values', 'cellRows', 'scale'] as const) {
      const graph = new GPUCommandGraph(device);
      kernels.recordWGSLKernelNodes(graph);
      const values = createTransientView(graph, 'values', 'float32', 3);
      const cellRows = createTransientView(graph, 'cells', 'uint32', 3);
      const scale = createTransientView(graph, 'scale', 'uint32', 3);
      const props = {
        id: 'strided',
        operation: 'Strided',
        values: graph.createDataView(values.buffer, {format: 'float32', length: 2}),
        cellRows: graph.createDataView(cellRows.buffer, {format: 'uint32', length: 2}),
        scale: graph.createDataView(scale.buffer, {format: 'uint32', length: 2})
      };
      // Logical float terms [2,2^100] occupy words 0 and 2; word 1 is padding.
      if (port === 'values')
        props.values = graph.createDataView(values.buffer, {
          format: 'float32',
          length: 2,
          byteStride: 8
        });
      if (port === 'cellRows')
        props.cellRows = graph.createDataView(cellRows.buffer, {
          format: 'uint32',
          length: 2,
          byteStride: 8
        });
      if (port === 'scale')
        props.scale = graph.createDataView(scale.buffer, {
          format: 'uint32',
          length: 2,
          byteStride: 8
        });
      expect(() => createFixedPointGroupedScaleNodes(graph, props)).toThrow(/strided.*packed/);
      expect(kernels.getWGSLKernelNodeRecord(graph, 'strided-clear')).toBeUndefined();
    }
  } finally {
    device.destroy();
  }
});

it('B4 rejects entry-point shadowing names in both builders', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = createTransientView(graph, 'values', 'uint32', 1);
    for (const builder of [kernels.createWGSLKernelNode, createGraphChunkGroupKernelNode]) {
      for (const name of [
        'workgroupId',
        'localInvocationIndex',
        'index',
        'workgroupIndex',
        ...(builder === createGraphChunkGroupKernelNode ? ['workgroupCount'] : [])
      ]) {
        expect(() =>
          builder(graph, {
            id: 'shadow',
            operation: 'Shadow',
            invocationCount: 1,
            bindings: [{name, view, type: 'u32', access: 'read'}],
            body: ''
          })
        ).toThrow(new RegExp(`shadow.*${name}`));
      }
    }
  } finally {
    device.destroy();
  }
});

it('B5 rejects an inherited misspelled getter without evaluating it and captures inherited mask', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const mask = createTransientView(graph, 'mask', 'uint32', 1);
    const props = {
      encoding: 'fourier' as const,
      rowCount: 1,
      coordinates: createTransientView(graph, 'coordinates', 'float32', 2),
      dimensions: 2,
      frequencyCount: 1,
      frequencies: createTransientView(graph, 'frequency', 'float32', 2),
      output: {
        features: createTransientView(graph, 'features', 'float32', 2),
        validity: createTransientView(graph, 'validity', 'uint32', 1)
      }
    };
    let typoReads = 0;
    class TypoOptions {
      get maks() {
        typoReads++;
        return mask;
      }
    }
    const typo = Object.assign(new TypoOptions(), props);
    expect(() => new GPUFittingCoordinateEncoder(typo as GPUFittingCoordinateEncoderProps)).toThrow(
      /fitting-coordinate-encoder.*maks/
    );
    expect(typoReads).toBe(0);
    let maskReads = 0;
    class MaskOptions {
      get mask() {
        maskReads++;
        return mask;
      }
    }
    const encoder = new GPUFittingCoordinateEncoder(Object.assign(new MaskOptions(), props));
    expect(encoder.props.mask).toBe(mask);
    expect(maskReads).toBe(1);
  } finally {
    device.destroy();
  }
});

it('S1 inspects chunk shapes only linearly for disjoint outputs', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let inspections = 0;
    const count = 128;
    const chunks = Array.from({length: count}, (_, index) => {
      const chunk = createTransientView(graph, `chunk-${index}`, 'uint32', 1);
      Object.defineProperty(chunk, 'format', {
        get() {
          inspections++;
          return 'uint32';
        }
      });
      return chunk;
    });
    const vector = new GraphVectorView({
      id: 'vector',
      name: 'vector',
      format: 'uint32',
      length: count,
      valueLength: count,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: chunks
    });
    inspections = 0;
    validateDisjointWritableViews('linear', {output: vector});
    expect(inspections).toBeLessThanOrEqual(count * 4);
  } finally {
    device.destroy();
  }
});

it('S3 rejects options from the other encoding branch', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      encoding: 'spherical-harmonics' as const,
      spatialContext: {coordinateSpace: 'longitude-latitude' as const, metric: 'none' as const},
      rowCount: 1,
      maximumDegree: 0,
      coordinates: createTransientView(graph, 'coordinates', 'float32', 2),
      output: {
        features: createTransientView(graph, 'features', 'float32', 1),
        validity: createTransientView(graph, 'validity', 'uint32', 1)
      },
      frequencyCount: 8
    };
    expect(() => new GPUFittingCoordinateEncoder(props)).toThrow(
      /frequencyCount.*spherical-harmonics/
    );
    expect(
      () =>
        new GPUFittingCoordinateEncoder({
          encoding: 'spherical-harmonics',
          spatialContext: props.spatialContext,
          rowCount: props.rowCount,
          maximumDegree: props.maximumDegree,
          coordinates: props.coordinates,
          output: props.output
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('S4 accepts zero terms and zero cells consistently', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'values', 'float32', 0);
    const cellRows = createTransientView(graph, 'cells', 'uint32', 0);
    const scale = createTransientView(graph, 'scale', 'uint32', 0);
    expect(
      createFixedPointGroupedScaleNodes(graph, {
        id: 'empty',
        operation: 'Empty',
        values,
        cellRows,
        scale
      })
    ).toEqual([]);
    expect(() =>
      createFixedPointDecodeNode(graph, {
        id: 'decode-empty',
        operation: 'Empty',
        scale,
        words: cellRows,
        maximumTermCount: 0,
        output: values,
        scaleMode: 'per-cell'
      })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('B2 keeps selected compiled provenance separate from a later candidate', async () => {
  vi.resetModules();
  const other = await import('../../src/analysis-core/kernels/wgsl-kernel-nodes');
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const compiledSources: string[] = [];
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string; shader: {source: string}}) => {
      compiledSources.push(props.shader.source);
      return {id: props.id ?? '', hash: '', destroy: () => {}};
    }
  });
  try {
    for (const [selectedBuilder, candidateBuilder] of [
      [kernels, other],
      [other, kernels]
    ]) {
      const graph = new GPUCommandGraph(device);
      selectedBuilder.recordWGSLKernelNodes(graph);
      const props = {
        id: 'same',
        operation: 'Selected',
        invocationCount: 257,
        workgroupSize: 256,
        bindings: [],
        body: 'let executed = index;'
      };
      const selected = selectedBuilder.createWGSLKernelNode(graph, props);
      graph.add(selected);
      const candidate = candidateBuilder.createWGSLKernelNode(graph, {
        ...props,
        invocationCount: 1,
        body: 'let unused = index;'
      });
      getGPUCommandNodeKernelSources([selected]);
      // Standalone inspection invokes the original callback. The graph selects it at compilation.
      expect(candidateBuilder.getCompiledWGSLKernelNodeRecord(graph, 'same')).toBeUndefined();
      compiledSources.length = 0;
      const compiled = graph.compile();
      try {
        expect(compiledSources).toHaveLength(1);
        const [compiledSource] = compiledSources;
        expect(compiledSource).toContain('let executed = index;');
        const expectedRecord = {
          source: compiledSource,
          body: 'let executed = index;',
          workgroups: [2, 1, 1]
        };
        expect(candidateBuilder.getCompiledWGSLKernelNodeRecord(graph, 'same')).toMatchObject(
          expectedRecord
        );
        getGPUCommandNodeKernelSources([candidate]);
        expect(selectedBuilder.getCompiledWGSLKernelNodeRecord(graph, 'same')).toMatchObject(
          expectedRecord
        );
      } finally {
        compiled.destroy();
      }
      const unrecorded = new GPUCommandGraph(device);
      const unrecordedNode = selectedBuilder.createWGSLKernelNode(unrecorded, props);
      getGPUCommandNodeKernelSources([unrecordedNode]);
      expect(candidateBuilder.getCompiledWGSLKernelNodeRecord(unrecorded, 'same')).toBeUndefined();
      expect(selectedBuilder.getWGSLKernelNodeRecord(graph, 'same')!.body).toBe(
        'let unused = index;'
      );
    }
  } finally {
    device.destroy();
  }
});
