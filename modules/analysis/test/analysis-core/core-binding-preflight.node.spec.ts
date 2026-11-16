// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {validateFittingWritableViews} from '../../src/analysis-core/kernels/fitting-utils';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUFittingMinibatchOrder} from '../../src/analysis-core/kernels/gpu-fitting-minibatch-order';
import {GPUFittingCoordinateEncoder} from '../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {GPUFittingGradientSums} from '../../src/analysis-core/kernels/gpu-fitting-gradient-sums';
import {GPUAdamStep} from '../../src/analysis-core/kernels/gpu-adam-step';
import {GPUPolygonTriangulation} from '../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {GPUSegmentRingAssembly} from '../../src/analysis-core/ring-assembly/gpu-segment-ring-assembly';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';
import {createKernelAuditContext} from './kernel-audit/kernel-audit-context';
import {createAdamAuditProps} from './kernel-audit/cases-fitting-kernels';
import {validateDisjointWritableViews} from '../../src/analysis-core/kernels/graph-chunk-dispatch';

function expectDisjointStorageUsages(nodes: readonly GPUCommandNode[]): void {
  for (const node of nodes) {
    const usages = new Map<object, Set<string>>();
    for (const resource of node.resources ?? []) {
      if (!('buffer' in resource) || !resource.usage.startsWith('storage-')) continue;
      const buffer = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
      const bufferUsages = usages.get(buffer) ?? new Set<string>();
      bufferUsages.add(resource.usage);
      usages.set(buffer, bufferUsages);
    }
    for (const bufferUsages of usages.values()) {
      expect(
        bufferUsages.has('storage-read') && bufferUsages.has('storage-read-write'),
        node.id
      ).toBe(false);
    }
  }
}

function createRingFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    endpoints: createTransientView(graph, 'endpoints', 'float32x4', 1),
    output: {
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  };
  return {device, graph, props};
}

it('never lets the separate-dispatch option bypass an explicit co-bound alignment check', () => {
  const {device, graph, props} = createRingFixture();
  try {
    const shared = props.output.positions.buffer;
    const views = {
      first: graph.createDataView(shared, {format: 'uint32', length: 1}),
      second: graph.createDataView(shared, {format: 'uint32', length: 1, byteOffset: 4})
    };
    expect(() =>
      validateDisjointWritableViews('writes', views, {separateDispatches: true})
    ).not.toThrow();
    expect(() =>
      validateDisjointWritableViews('writes', views, {
        separateDispatches: true,
        strictAlignedBindings: true
      })
    ).toThrow(/aligned binding/);
  } finally {
    device.destroy();
  }
});

it('fitting checks aligned input prefixes before allocating sort scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'backing', 'uint32', 32);
  const parameters = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 3,
    byteOffset: 20
  });
  const order = createTransientView(graph, 'order', 'uint32', 2);
  const contributor = new GPUFittingMinibatchOrder({rowCount: 2, parameters, output: {order}});
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 16});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/view 0 chunk 0 needs 32 bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('triangulation checks output bindings before declaring its scratch and layout', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUPolygonTriangulation({
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2),
    output: {
      indices: createTransientView(graph, 'indices', 'uint32', 100),
      valid: createTransientView(graph, 'valid', 'uint32', 1)
    }
  });
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 396});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 400 bytes/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('ring assembly checks output bindings before declaring any scratch', () => {
  const {device, graph, props} = createRingFixture();
  const contributor = new GPUSegmentRingAssembly(props);
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 16});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 32 bytes/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects nonfinite or float32-zero ring tolerances at construction', () => {
  const {device, props} = createRingFixture();
  try {
    for (const vertexTolerance of [NaN, Infinity, 1e40, 1e-100, 0, -1]) {
      expect(() => new GPUSegmentRingAssembly({...props, vertexTolerance})).toThrow(
        /^segment-ring-assembly vertexTolerance must be a finite number/
      );
    }
  } finally {
    device.destroy();
  }
});

it('fitting rejects copy-only parameter bindings before allocating sort scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = graph.createDataView(
    graph.createTransientBuffer({id: 'parameters', byteLength: 12, usage: Buffer.COPY_SRC}),
    {format: 'uint32', length: 3}
  );
  const order = createTransientView(graph, 'order', 'uint32', 2);
  const contributor = new GPUFittingMinibatchOrder({rowCount: 2, parameters, output: {order}});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /view 0 chunk 0 buffer usage must include Buffer.STORAGE/
    );
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects aliased ring outputs, including nested status and polygon fields', () => {
  const {device, graph, props} = createRingFixture();
  try {
    const scalar = props.output.status.count;
    expect(
      () =>
        new GPUSegmentRingAssembly({
          ...props,
          output: {...props.output, ringIsHole: scalar}
        })
    ).toThrow(
      /output.ringIsHole.*output.status.count.*overlap|output.status.count.*output.ringIsHole.*overlap/
    );
    const twoRows = createTransientView(graph, 'two-rows', 'uint32', 2);
    expect(
      () =>
        new GPUSegmentRingAssembly({
          ...props,
          output: {
            ...props.output,
            polygons: {
              kind: 'polygons',
              positions: createTransientView(graph, 'polygon-positions', 'float32x2', 4),
              ringOffsets: twoRows,
              polygonOffsets: twoRows,
              featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
            }
          }
        })
    ).toThrow(/output.polygons.*overlap/);
    // Distinct logical scalar rows with overlapping aligned bindings are also unsafe when co-bound.
    const count = graph.createDataView(twoRows.buffer, {format: 'uint32', length: 1});
    const overflow = graph.createDataView(twoRows.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    expect(
      () =>
        new GPUSegmentRingAssembly({
          ...props,
          output: {...props.output, status: {count, overflow}}
        })
    ).toThrow(/aligned binding.*overlap/);
  } finally {
    device.destroy();
  }
});

it('rejects excessive explicit monotone capacity before declaring scratch, views or nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUPolygonTriangulation({
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    monotoneMaximumVertices: 11,
    output: {
      indices: createTransientView(graph, 'indices', 'uint32', 3),
      valid: createTransientView(graph, 'valid', 'uint32', 1)
    }
  });
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 400});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/monotoneMaximumVertices.*10/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('triangulation rejects overlapping validity and triangle-count outputs at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const valid = createTransientView(graph, 'valid', 'uint32', 1);
  try {
    expect(
      () =>
        new GPUPolygonTriangulation({
          positions: createTransientView(graph, 'positions', 'float32x2', 3),
          featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
          polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
          ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
          output: {
            indices: createTransientView(graph, 'indices', 'uint32', 3),
            valid,
            triangleCount: valid
          }
        })
    ).toThrow(/valid.*triangleCount.*overlap|triangleCount.*valid.*overlap/);
  } finally {
    device.destroy();
  }
});

it('fitting rejects overlapping offset views of one physical buffer imported once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE});
  try {
    const first = importGraphBuffer(graph, 'first', buffer, 'uint32', 2);
    const second = graph.createDataView(first.buffer, {format: 'uint32', length: 2});
    expect(() => validateFittingWritableViews('fitting', [first, second], [])).toThrow(
      /^fitting output 0 chunk 0 and output 1 chunk 0 aligned binding byte ranges must not overlap$/
    );
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('accepts packed ring fields published in different dispatches and rejects mixed storage usage', () => {
  const {device, graph, props} = createRingFixture();
  try {
    const buffer = graph.createTransientBuffer({
      id: 'ring-columns',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const output = {
      ...props.output,
      ringIsHole: graph.createDataView(buffer, {format: 'uint32', length: 1}),
      status: {
        count: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 4}),
        overflow: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 256})
      }
    };
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const contributor = new GPUSegmentRingAssembly({...props, output});
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expectDisjointStorageUsages(contributor.getCommandNodes(graph));
    // Use genuinely disjoint bytes: packing read-only positions and writable areas is still invalid.
    const shared = graph.createTransientBuffer({
      id: 'mixed-ring-columns',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    expect(
      () =>
        new GPUSegmentRingAssembly({
          ...props,
          output: {
            ...props.output,
            positions: graph.createDataView(shared, {format: 'float32x2', length: 4}),
            ringAreas: graph.createDataView(shared, {format: 'float32', length: 1, byteOffset: 256})
          }
        })
    ).toThrow(/share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

it('accepts triangle offsets sharing an alignment window with separately published indices', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const buffer = graph.createTransientBuffer({
      id: 'triangulation-columns',
      byteLength: 768,
      usage: Buffer.STORAGE
    });
    const contributor = new GPUPolygonTriangulation({
      positions: createTransientView(graph, 'positions', 'float32x2', 3),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      output: {
        indices: graph.createDataView(buffer, {format: 'uint32', length: 3}),
        triangleOffsets: graph.createDataView(buffer, {
          format: 'uint32',
          length: 2,
          byteOffset: 12
        }),
        valid: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 256}),
        triangleCount: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 512})
      }
    });
    expectDisjointStorageUsages(contributor.getCommandNodes(graph));
  } finally {
    device.destroy();
  }
});

it('does not bind or reject unused validity/count outputs when there are no polygons', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', 0);
    const ignored = graph.createTransientBuffer({
      id: 'ignored-count',
      byteLength: 4,
      usage: Buffer.COPY_SRC
    });
    const contributor = new GPUPolygonTriangulation({
      positions,
      featureOffsets: createTransientView(graph, 'features', 'uint32', 1),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 1),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 1),
      output: {
        indices: createTransientView(graph, 'indices', 'uint32', 0),
        valid: graph.createDataView(positions.buffer, {format: 'uint32', length: 0}),
        triangleCount: graph.createDataView(ignored, {format: 'uint32', length: 0})
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(
      nodes
        .flatMap(node => node.resources ?? [])
        .some(
          resource =>
            'buffer' in resource && resource.buffer === contributor.props.output.triangleCount
        )
    ).toBe(false);
    expectDisjointStorageUsages(nodes);
  } finally {
    device.destroy();
  }
});

it('accepts packed fitting features and validity while ignoring frequencies outside Fourier mode', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const output = graph.createTransientBuffer({
      id: 'encoder-columns',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const ignored = graph.createTransientBuffer({
      id: 'ignored-frequencies',
      byteLength: 4,
      usage: Buffer.COPY_SRC
    });
    const contributor = new GPUFittingCoordinateEncoder({
      encoding: 'spherical-harmonics',
      maximumDegree: 0,
      rowCount: 2,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
      coordinates: createTransientView(graph, 'coordinates', 'float32', 4),
      frequencies: graph.createDataView(ignored, {format: 'float32', length: 1}),
      output: {
        features: graph.createDataView(output, {format: 'float32', length: 2}),
        validity: graph.createDataView(output, {format: 'uint32', length: 2, byteOffset: 256})
      }
    });
    expectDisjointStorageUsages(contributor.getCommandNodes(graph));
  } finally {
    device.destroy();
  }
});

it('accepts packed Adam writable state but rejects iteration status read/write buffer conflicts', () => {
  const context = createKernelAuditContext('adam-packed-state');
  const {device, graph} = context;
  try {
    const props = createAdamAuditProps(context, 3, 8);
    const buffer = context.buffer(1024);
    const output = {
      ...props.output,
      weights: graph.createDataView(buffer, {format: 'float32', length: 3}),
      firstMoment: graph.createDataView(buffer, {format: 'float32', length: 3, byteOffset: 256}),
      secondMoment: graph.createDataView(buffer, {format: 'float32', length: 3, byteOffset: 512}),
      lossHistory: graph.createDataView(buffer, {format: 'float32', length: 9, byteOffset: 12})
    };
    expectDisjointStorageUsages(new GPUAdamStep({...props, output}).getCommandNodes(graph));
    expect(
      () =>
        new GPUAdamStep({
          ...props,
          output: {
            ...output,
            status: {
              ...output.status,
              iterationCount: graph.createDataView(buffer, {
                format: 'uint32',
                length: 1,
                byteOffset: 768
              })
            }
          }
        })
    ).toThrow(/share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

it('prechecks nested fitting, ring-shell and polygon allocation sizes without declaring resources', () => {
  for (const family of ['fitting', 'rings', 'polygons'] as const) {
    const {device, graph, props} = createRingFixture();
    try {
      const contributor =
        family === 'fitting'
          ? new GPUFittingGradientSums({
              rowCount: 769,
              parameterCount: 64,
              gradients: createTransientView(graph, 'gradients', 'float32', 769 * 64),
              output: {sums: createTransientView(graph, 'sums', 'float32', 64)}
            })
          : family === 'rings'
            ? new GPUSegmentRingAssembly({
                ...props,
                output: {
                  ...props.output,
                  ringOffsets: createTransientView(graph, 'wide-offsets', 'uint32', 65),
                  ringShells: createTransientView(graph, 'shells', 'uint32', 64)
                }
              })
            : new GPUPolygonTriangulation({
                positions: createTransientView(graph, 'polygon-positions', 'float32x2', 3),
                featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
                polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
                ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
                output: {
                  indices: createTransientView(graph, 'indices', 'uint32', 3),
                  valid: createTransientView(graph, 'valid', 'uint32', 1)
                }
              });
      device.limits.maxBufferSize = family === 'polygons' ? 256 : 512;
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      const add = vi.spyOn(graph, 'add');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        family === 'fitting'
          ? /gradient partials byteLength/
          : family === 'rings'
            ? /shell records byteLength/
            : /scratch byteLength/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});
