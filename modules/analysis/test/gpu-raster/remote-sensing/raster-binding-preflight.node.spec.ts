// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPURasterConservativeRemapping} from '../../../src/raster-analysis/remote-sensing/gpu-raster-conservative-remapping';
import {GPURasterRegistration} from '../../../src/raster-analysis/remote-sensing/gpu-raster-registration';
import {GPURasterSpectralUnmixing} from '../../../src/raster-analysis/remote-sensing/gpu-raster-spectral-unmixing';
import {GPURasterTexture} from '../../../src/raster-analysis/remote-sensing/gpu-raster-texture';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

const CASES = [
  'texture',
  'unmixing',
  'remapping-internal',
  'remapping-prepare',
  'remapping-reuse',
  'periodic',
  'masked'
] as const;
type CaseName = (typeof CASES)[number];
type Format = 'float32' | 'float32x2' | 'uint32';
let serial = 0;
function createCase(
  graph: GPUCommandGraph,
  name: CaseName,
  replace?: (name: string, format: Format, length: number) => GraphDataView | undefined,
  dimension = 4,
  upsampleFactor = 1
) {
  const views = new Map<string, GraphDataView>();
  const view = <ViewFormat extends Format>(field: string, format: ViewFormat, length: number) => {
    const result =
      replace?.(field, format, length) ??
      createTransientView(graph, `binding-${serial++}`, format, length);
    views.set(field, result);
    return result as GraphDataView<ViewFormat>;
  };
  let contributor: GPUCommandNodeProducer;
  if (name === 'texture') {
    contributor = new GPURasterTexture({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      values: view('values', 'float32', 4),
      mask: view('mask', 'uint32', 4),
      levels: 4,
      range: [0, 4],
      radius: 1,
      offset: [1, 0],
      features: view('features', 'float32', 28),
      pairCount: view('pairCount', 'uint32', 4)
    });
  } else if (name === 'unmixing') {
    contributor = new GPURasterSpectralUnmixing({
      cellCount: 2,
      bandCount: 2,
      endmemberCount: 2,
      values: view('values', 'float32', 4),
      endmembers: view('endmembers', 'float32', 4),
      mask: view('mask', 'uint32', 4),
      output: {
        abundances: view('abundances', 'float32', 4),
        residual: view('residual', 'float32', 2),
        status: view('status', 'uint32', 2),
        iterationCounts: view('iterationCounts', 'uint32', 2)
      }
    });
  } else if (name.startsWith('remapping')) {
    const grid = {width: 2, height: 2, origin: [0, 0] as const, spacing: [1, 1] as const};
    contributor = new GPURasterConservativeRemapping({
      source: grid,
      destination: grid,
      normalization: 'covered-area',
      values: view('values', 'float32', 4),
      mask: view('mask', 'uint32', 4),
      ...(name === 'remapping-internal'
        ? {}
        : {
            weightMode: name === 'remapping-reuse' ? ('reuse' as const) : ('prepare' as const),
            preparedWeights: {
              indices: view('indices', 'uint32', 16),
              areas: view('areas', 'float32', 16)
            }
          }),
      output: {
        values: view('result', 'float32', 4),
        coveredArea: view('coveredArea', 'float32', 4),
        status: view('status', 'uint32', 4)
      }
    });
  } else {
    contributor = new GPURasterRegistration({
      mode: name === 'masked' ? 'masked-linear' : 'periodic',
      grid: makeGPURasterMetadata({width: dimension, height: dimension}),
      reference: view('reference', 'float32', dimension ** 2),
      moving: view('moving', 'float32', dimension ** 2),
      upsampleFactor,
      ...(name === 'masked'
        ? {
            referenceMask: view('referenceMask', 'uint32', dimension ** 2),
            movingMask: view('movingMask', 'uint32', dimension ** 2)
          }
        : {}),
      output: {
        shift: view('shift', 'float32x2', 1),
        peak: view('peak', 'float32', 1),
        status: view('status', 'uint32', 1),
        secondPeakRatio: view('secondPeakRatio', 'float32', 1),
        ...(name === 'masked' ? {overlap: view('overlap', 'uint32', 1)} : {})
      }
    });
  }
  return {contributor, views};
}

function expectUnchanged(
  graph: GPUCommandGraph,
  contributor: GPUCommandNodeProducer,
  message: RegExp
) {
  const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(message);
  expect(createBuffer).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
}

for (const name of CASES) {
  it(`${name}: preflights each caller's aligned binding independently before mutation`, () => {
    const device = createNullWebGPUDevice();
    try {
      const registration = name === 'periodic' || name === 'masked';
      const limit = registration ? 4096 : 192;
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: limit});
      const baseline = createCase(new GPUCommandGraph(device), name);
      for (const field of baseline.views.keys()) {
        const graph = new GPUCommandGraph(device);
        const {contributor} = createCase(graph, name, (candidate, format, length) => {
          if (candidate !== field) return;
          const byteOffset = registration ? 4 : 196;
          const badLength = registration ? 1025 : length;
          const bytes = badLength * (format === 'float32x2' ? 8 : 4) + byteOffset;
          const backing = createTransientView(
            graph,
            `oversized-${serial++}`,
            'uint32',
            Math.ceil(bytes / 4)
          );
          return graph.createDataView(backing.buffer, {format, length: badLength, byteOffset});
        });
        expectUnchanged(graph, contributor, new RegExp(`limit ${limit}`));
      }
    } finally {
      device.destroy();
    }
  });
  it(`${name}: rejects missing STORAGE on each caller field before mutation`, () => {
    const device = createNullWebGPUDevice();
    try {
      const baseline = createCase(new GPUCommandGraph(device), name);
      for (const field of baseline.views.keys()) {
        const graph = new GPUCommandGraph(device);
        const {contributor, views} = createCase(graph, name);
        Object.defineProperty(views.get(field)!.buffer, 'usage', {value: Buffer.COPY_SRC});
        expectUnchanged(graph, contributor, /Buffer.STORAGE/);
      }
    } finally {
      device.destroy();
    }
  });
  it(`${name}: accepts valid packed writable subviews with one physical import`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      // All outputs in the single-dispatch classes share an import. Registration's status is
      // read-only in periodic finish, so it has its own buffer; ratio is a separate output stage.
      const buffer = device.createBuffer({byteLength: 4096, usage: Buffer.STORAGE});
      const handle = graph.importBuffer(
        {id: `packed-${serial++}`, byteLength: 4096, usage: Buffer.STORAGE},
        buffer
      );
      let byteOffset = 0;
      const writableFields = new Set([
        'features',
        'pairCount',
        'abundances',
        'residual',
        'iterationCounts',
        'result',
        'coveredArea',
        'shift',
        'peak',
        'overlap',
        ...(name === 'periodic' ? [] : ['status'])
      ]);
      try {
        const {contributor} = createCase(graph, name, (field, format, length) => {
          if (!writableFields.has(field)) return;
          const result = graph.createDataView(handle, {format, length, byteOffset});
          byteOffset += 256;
          return result;
        });
        expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
      } finally {
        buffer.destroy();
      }
    } finally {
      device.destroy();
    }
  });
}

it('remapping preflights derived weight scratch even when caller bindings fit', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 48});
    const graph = new GPUCommandGraph(device);
    expectUnchanged(
      graph,
      createCase(graph, 'remapping-internal').contributor,
      /prepared indices and areas scratch.*64.*limit 48/
    );
  } finally {
    device.destroy();
  }
});

for (const mode of ['periodic', 'masked'] as const) {
  it(`${mode}: preflights nested FFT and refinement scratch before declarations`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const {contributor} = createCase(graph, mode, undefined, 4, 32);
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 4096});
      expectUnchanged(graph, contributor, /refin.*limit 4096/);
    } finally {
      device.destroy();
    }
    const fftDevice = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(fftDevice);
      const {contributor} = createCase(graph, mode);
      Object.defineProperty(fftDevice.limits, 'maxUniformBufferBindingSize', {value: 16});
      expectUnchanged(graph, contributor, /GPUFFT2D parameter binding/);
    } finally {
      fftDevice.destroy();
    }
  });
}

it('GPU-01 uses fixed-order parallel means and paired moments, with full WGSL identifier audits', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {contributor} = createCase(graph, 'masked', undefined, 129, 32);
    const nodes = contributor.getCommandNodes(graph);
    const reductionNodes = nodes.filter(node =>
      /masked-(mean-groups|statistics|refine-groups|refine)$/.test(node.id)
    );
    expect(reductionNodes).toHaveLength(4);
    const kernels = getGPUCommandNodeKernelSources(reductionNodes);
    expect(kernels).toHaveLength(4);
    for (const kernel of kernels) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      expect(kernel.source).toContain('@workgroup_size(256)');
      expect(kernel.source).toContain('workgroupBarrier()');
      expect(kernel.source).not.toMatch(/for\s*\(var cell\s*=/);
    }
    const groupKernel = kernels.find(kernel => kernel.id.endsWith('refine-groups'))!;
    expect(groupKernel.constants['INVOCATION_COUNT']).toBe(1089 * Math.ceil(129 ** 2 / 256) * 256);
    expect(groupKernel.source).toContain('if (workgroupIndex >= 71874u) { return; }');
    // Candidate and tile IDs are computed in the kernel; the graph never unrolls candidates.
    expect(new Set(kernels.map(kernel => kernel.source)).size).toBe(4);
  } finally {
    device.destroy();
  }
});

for (const mode of ['periodic', 'masked'] as const) {
  it(`${mode}: accepts a shared physical buffer used by read and write in separate dispatches`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const backing = createTransientView(graph, `separate-phases-${serial++}`, 'uint32', 128);
      const {contributor} = createCase(graph, mode, (field, format, length) => {
        if (field !== 'reference' && field !== 'shift') return;
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: field === 'reference' ? 0 : 256
        });
      });
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
}

it('rejects read-only and writable ranges of one buffer in a single dispatch, even when aligned ranges are disjoint', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const backing = createTransientView(graph, 'texture-mixed-usage', 'uint32', 128);
    expect(() =>
      createCase(graph, 'texture', (field, format, length) => {
        if (field !== 'values' && field !== 'features') return;
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: field === 'values' ? 0 : 256
        });
      })
    ).toThrow(/same physical buffer/);
  } finally {
    device.destroy();
  }
});

it('requires all views of one physical buffer to share one imported graph handle', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 128, usage: Buffer.STORAGE});
  try {
    const graph = new GPUCommandGraph(device);
    expect(() =>
      createCase(graph, 'texture', (field, format, length) => {
        if (field !== 'values' && field !== 'mask') return;
        const handle = graph.importBuffer(
          {id: `duplicate-import-${serial++}`, byteLength: 128, usage: Buffer.STORAGE},
          buffer
        );
        return graph.createDataView(handle, {
          format,
          length,
          byteOffset: field === 'values' ? 0 : 64
        });
      })
    ).toThrow(/physical buffer must be imported once/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('audits every registration kernel and large numeric literals in unmixing and texture', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const name of CASES) {
      const graph = new GPUCommandGraph(device);
      const original = createCase(graph, name).contributor;
      const contributor =
        original instanceof GPURasterSpectralUnmixing
          ? new GPURasterSpectralUnmixing({
              ...original.props,
              constraint: 'simplex',
              tolerance: 1e20
            })
          : original instanceof GPURasterTexture
            ? new GPURasterTexture({...original.props, range: [1e20, 2e20]})
            : original;
      const nodes = contributor.getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(
        nodes.filter(node => node.workload?.operation !== 'GPUFFT2D')
      );
      expect(kernels.length).toBeGreaterThan(0);
      for (const kernel of kernels) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        expect(kernel.source, kernel.id).not.toMatch(/\b100000000000000000000\b/);
      }
    }
  } finally {
    device.destroy();
  }
});

it('normalizes texture ranges with exponent-scaled division instead of a flushed reciprocal', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const original = createCase(graph, 'texture').contributor as GPURasterTexture;
    const contributor = new GPURasterTexture({
      ...original.props,
      range: [0, Math.fround(3.402823466e38)]
    });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels).toHaveLength(1);
    const source = kernels[0].source;
    expect(source).toContain('getScaledQuotient(value - ');
    expect(auditWGSLModule(source, kernels[0].constants)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('rejects texture ranges whose endpoints collapse to one float32 value', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const original = createCase(graph, 'texture').contributor as GPURasterTexture;
    for (const range of [
      [0, 1e-50],
      [1, 1 + Number.EPSILON]
    ] as const)
      expect(() => new GPURasterTexture({...original.props, range})).toThrow(
        /^raster-texture range/
      );
  } finally {
    device.destroy();
  }
});
