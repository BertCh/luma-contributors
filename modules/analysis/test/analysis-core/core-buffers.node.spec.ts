// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {runInNewContext} from 'node:vm';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, expectTypeOf, it, vi} from 'vitest';
import {
  defineGPUSpatialParameterSchema,
  packGPUSpatialParameterValues
} from '../../src/analysis-core/contracts/parameter-schema';
import {validateGPUSpatialContext} from '../../src/analysis-core/contracts/spatial-context';
import {getGPUSpatialFrameContextUnits} from '../../src/analysis-core/contracts/spatial-frame';
import {createGPUSpatialFrameFromCRS} from '../../src/analysis-core/contracts/spatial-frame-crs';
import {sliceGraphChunkGroupView} from '../../src/analysis-core/kernels/graph-chunk-groups';
import {
  GPUParameterBuffer,
  type GPUParameterBufferProps,
  type GPUParameterFormat,
  validateFiniteNumber
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  getWGSLKernelBinding
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getGPUTrajectoryMetricsParameterValues} from '../../src/gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-parameters';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';

function createRecordingDevice() {
  const write = vi.fn();
  const createBuffer = vi.fn(() => ({write, destroy: vi.fn()}) as unknown as Buffer);
  return {device: {createBuffer} as unknown as Device, createBuffer, write};
}

describe('chunk subview bounds', () => {
  it('slices nested packed byte ranges without imposing storage-word alignment', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const buffer = graph.createTransientBuffer({
        id: 'bytes',
        byteLength: 4,
        usage: Buffer.STORAGE
      });
      const parent = graph.createDataView(buffer, {format: 'uint8', length: 4});
      const child = sliceGraphChunkGroupView(graph, parent, 1, 2);
      const nested = sliceGraphChunkGroupView(graph, child, 0, 1);
      expect(nested.byteOffset).toBe(1);
      expect(nested.length).toBe(1);
      expect(nested.byteStride).toBe(1);
    } finally {
      device.destroy();
    }
  });

  it('keeps a middle-of-buffer parent inside its own rows', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const buffer = graph.createTransientBuffer({
        id: 'parent',
        byteLength: 32,
        usage: Buffer.STORAGE
      });
      const parent = graph.createDataView(buffer, {format: 'uint32', length: 2, byteOffset: 8});
      expect(sliceGraphChunkGroupView(graph, parent, 1, 1).byteOffset).toBe(12);
      expect(sliceGraphChunkGroupView(graph, parent, 2, 0).length).toBe(0);
      expect(() => sliceGraphChunkGroupView(graph, parent, 2, 1)).toThrow(/rowCount/);
      expect(() => sliceGraphChunkGroupView(graph, parent, -1, 1)).toThrow(/rowStart/);
    } finally {
      device.destroy();
    }
  });

  it('rejects fractional bounds and strided parents', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const buffer = graph.createTransientBuffer({
        id: 'strided',
        byteLength: 32,
        usage: Buffer.STORAGE
      });
      const parent = graph.createDataView(buffer, {format: 'uint32', length: 2, byteStride: 8});
      expect(() => sliceGraphChunkGroupView(graph, parent, 1, 1)).toThrow(/packed/);
      const packed = graph.createDataView(buffer, {format: 'uint32', length: 2});
      expect(() => sliceGraphChunkGroupView(graph, packed, 0.5, 1)).toThrow(/rowStart/);
      expect(() => sliceGraphChunkGroupView(graph, packed, 0, 0.5)).toThrow(/rowCount/);
    } finally {
      device.destroy();
    }
  });
});

describe('parameter buffer uploads', () => {
  it('accepts matching foreign-realm arrays for initialization and updates', () => {
    const {device, write} = createRecordingDevice();
    const foreignValues = runInNewContext('Float32Array.of(1)') as Float32Array;
    expect(foreignValues instanceof Float32Array).toBe(false);
    const parameters = new GPUParameterBuffer(device, {
      id: 'foreign-parameters',
      format: 'float32',
      length: 1,
      values: foreignValues
    });
    expect(write).toHaveBeenLastCalledWith(foreignValues, 0);
    parameters.write(foreignValues);
    expect(write).toHaveBeenCalledTimes(2);
  });

  it('rejects mismatched foreign-realm arrays on initialization and update', () => {
    const {device, createBuffer, write} = createRecordingDevice();
    const foreignValues = runInNewContext('Uint32Array.of(1)') as Float32Array;
    expect(
      () =>
        new GPUParameterBuffer(device, {
          id: 'foreign-parameters',
          format: 'float32',
          length: 1,
          values: foreignValues
        })
    ).toThrow(/values.*format/);
    expect(createBuffer).not.toHaveBeenCalled();
    const parameters = new GPUParameterBuffer(device, {
      id: 'foreign-parameters',
      format: 'float32',
      length: 1
    });
    expect(() => parameters.write(foreignValues)).toThrow(/values.*format/);
    expect(write).not.toHaveBeenCalled();
  });

  it('does not accept a spoofed typed-array kind', () => {
    const {device, createBuffer} = createRecordingDevice();
    const values = Uint32Array.of(1);
    Object.defineProperty(values, Symbol.toStringTag, {value: 'Float32Array'});
    expect(
      () =>
        new GPUParameterBuffer(device, {
          id: 'spoofed-parameters',
          format: 'float32',
          length: 1,
          values: values as unknown as Float32Array
        })
    ).toThrow(/values.*format/);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it.each([
    null,
    false,
    0
  ])('rejects a supplied non-array initializer %s before allocation', values => {
    const {device, createBuffer} = createRecordingDevice();
    expect(
      () =>
        new GPUParameterBuffer(device, {
          id: 'parameters',
          format: 'float32',
          length: 1,
          values: values as unknown as Float32Array
        })
    ).toThrow(/values.*format/);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('infers the constructor format independently of initializer values', () => {
    const {device} = createRecordingDevice();
    const createMismatchedBuffer = () =>
      new GPUParameterBuffer(device, {
        id: 'parameters',
        format: 'float32',
        length: 1,
        // @ts-expect-error float32 initializers must not widen the inferred format to an array union.
        values: Uint32Array.of(1)
      });
    expectTypeOf(createMismatchedBuffer).returns.toEqualTypeOf<GPUParameterBuffer<'float32'>>();
  });

  it('maps formats to the matching initializer and update array types', () => {
    expectTypeOf<GPUParameterBufferProps<'float32'>['values']>().toEqualTypeOf<
      Float32Array | undefined
    >();
    expectTypeOf<GPUParameterBufferProps<'uint32'>['values']>().toEqualTypeOf<
      Uint32Array | undefined
    >();
    expectTypeOf<GPUParameterBufferProps<'sint32'>['values']>().toEqualTypeOf<
      Int32Array | undefined
    >();
    expectTypeOf<
      Parameters<GPUParameterBuffer<'float32'>['write']>[0]
    >().toEqualTypeOf<Float32Array>();
    expectTypeOf<
      Parameters<GPUParameterBuffer<'uint32'>['write']>[0]
    >().toEqualTypeOf<Uint32Array>();
    expectTypeOf<
      Parameters<GPUParameterBuffer<'sint32'>['write']>[0]
    >().toEqualTypeOf<Int32Array>();
  });

  it.each([
    -1,
    0.5,
    NaN,
    Infinity
  ])('rejects invalid logical length %s before allocation', length => {
    const {device, createBuffer} = createRecordingDevice();
    expect(
      () => new GPUParameterBuffer(device, {id: 'parameters', format: 'uint32', length})
    ).toThrow(/length/);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it.each([
    ['float32', Uint32Array.of(1)],
    ['uint32', Float32Array.of(1)],
    ['sint32', Uint32Array.of(1)]
  ] as const)('rejects mismatched %s initializers before allocation', (format, values) => {
    const {device, createBuffer} = createRecordingDevice();
    const props = {id: 'parameters', format, length: 1, values};
    expect(
      () => new GPUParameterBuffer(device, props as GPUParameterBufferProps<typeof format>)
    ).toThrow(/values.*format/);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('rejects mismatched updates without forwarding their bytes', () => {
    const {device, write} = createRecordingDevice();
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'uint32',
      length: 2
    });
    // Simulate JavaScript callers bypassing the format-specific TypeScript contract.
    const unchecked = parameters as GPUParameterBuffer<GPUParameterFormat>;
    expect(() => unchecked.write(Float32Array.of(1))).toThrow(/values.*format/);
    expect(write).not.toHaveBeenCalled();
  });

  it('checks initial logical length before allocation, including zero-length buffers', () => {
    const {device, createBuffer} = createRecordingDevice();
    expect(
      () =>
        new GPUParameterBuffer(device, {
          id: 'parameters',
          format: 'uint32',
          length: 0,
          values: Uint32Array.of(1)
        })
    ).toThrow(/values/);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('uses integer update offsets and preserves valid byte offsets', () => {
    const {device, write} = createRecordingDevice();
    const initial = Uint32Array.of(7);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'uint32',
      length: 2,
      values: initial
    });
    expect(write.mock.calls[0][0]).toBe(initial);
    write.mockClear();
    expect(() => parameters.write(Uint32Array.of(9), 0.5)).toThrow(/firstElement/);
    expect(() => parameters.write(Uint32Array.of(9), 2)).toThrow(/values/);
    expect(write).not.toHaveBeenCalled();
    const update = Uint32Array.of(9);
    parameters.write(update, 1);
    expect(write).toHaveBeenCalledWith(update, 4);
    parameters.write(new Uint32Array(0), 2);
    expect(() => parameters.write(new Uint32Array(0), NaN)).toThrow(/firstElement/);
    expect(() => parameters.write(new Uint32Array(0), Infinity)).toThrow(/firstElement/);
  });
});

describe('encoded parameter domains', () => {
  const schema = (field = {}) =>
    defineGPUSpatialParameterSchema({
      id: 'encoded',
      format: 'float32',
      wordLength: 1,
      fields: [
        {
          name: 'threshold',
          format: 'float32',
          wordOffset: 0,
          defaultValue: 0,
          dynamic: true,
          ...field
        }
      ]
    });

  it('refuses a trajectory threshold that overflows float32', () => {
    expect(() =>
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: 1e39,
        stopMinimumDuration: 0
      })
    ).toThrow(/stopSpeedThreshold.*finite/);
  });

  it('validates encoded defaults as well as overrides', () => {
    expect(() => schema({defaultValue: 1e39})).toThrow(/threshold.*finite/);
    expect(() => schema({defaultValue: 0.1, maximum: 0.1})).toThrow(/maximum/);
  });

  it('rejects rounding across either bound without changing representable endpoints', () => {
    expect(() => packGPUSpatialParameterValues(schema({maximum: 0.1}), {threshold: 0.1})).toThrow(
      /maximum/
    );
    expect(() =>
      packGPUSpatialParameterValues(schema({defaultValue: 1, minimum: 0.7}), {threshold: 0.7})
    ).toThrow(/minimum/);
    expect(packGPUSpatialParameterValues(schema({maximum: 0.5}), {threshold: 0.5})[0]).toBe(0.5);
    expect(() => validateFiniteNumber('encoded', 'threshold', 0.1, 0, 0.1)).toThrow(/finite/);
  });

  it('preserves explicit nonfinite fields and rejects integer wrapping', () => {
    expect(
      packGPUSpatialParameterValues(schema({allowNonFinite: true}), {threshold: Infinity})[0]
    ).toBe(Infinity);
    expect(
      packGPUSpatialParameterValues(schema({allowNonFinite: true}), {threshold: NaN})[0]
    ).toBeNaN();
    const integers = defineGPUSpatialParameterSchema({
      id: 'integers',
      format: 'uint32',
      wordLength: 1,
      fields: [{name: 'count', format: 'uint32', wordOffset: 0, defaultValue: 0, dynamic: true}]
    });
    expect(() => packGPUSpatialParameterValues(integers, {count: 2 ** 32})).toThrow(/uint32/);
  });
});

describe('word-oriented helpers', () => {
  it('encodes a zero-component fill at the backing-buffer end without binding its target', () => {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device, 'props', {
      value: {...device.props, _cacheShaders: false, _cachePipelines: false}
    });
    vi.spyOn(device, 'createComputePipeline').mockReturnValue({
      hash: '',
      destroy: vi.fn()
    } as unknown as ReturnType<Device['createComputePipeline']>);
    const backing = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE});
    try {
      const graph = new GPUCommandGraph(device);
      const handle = graph.importBuffer(
        {id: 'empty-end', byteLength: 8, usage: backing.usage},
        backing
      );
      const parent = graph.createDataView(handle, {format: 'uint32', length: 2});
      const empty = sliceGraphChunkGroupView(graph, parent, 2, 0);
      const node = createFillNode(graph, {
        id: 'empty-fill',
        operation: 'fill',
        view: empty,
        type: 'u32',
        value: '9u'
      });
      expect(node.type).toBe('compute');
      if (node.type !== 'compute') throw new Error('expected a compute node');
      const executable = node.compile({device});
      const getBuffer = vi.fn(() => backing);
      try {
        expect(() =>
          executable.encode({getBuffer} as unknown as Parameters<typeof executable.encode>[0])
        ).not.toThrow();
        expect(getBuffer).not.toHaveBeenCalled();
        expect(node.resources).toEqual([]);
      } finally {
        executable.destroy?.();
      }
    } finally {
      backing.destroy();
      device.destroy();
    }
  });

  it('rejects fill scalar representations that conflict with the view format', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const buffer = graph.createTransientBuffer({
        id: 'float-fill',
        byteLength: 4,
        usage: Buffer.STORAGE
      });
      const view = graph.createDataView(buffer, {format: 'float32', length: 1});
      const props = {id: 'float-fill', operation: 'fill', view};
      expect(() => createFillNode(graph, {...props, type: 'u32', value: '1u'})).toThrow(
        /type.*format/
      );
      expect(() => createFillNode(graph, {...props, type: 'f32', value: '1.0'})).not.toThrow();
    } finally {
      device.destroy();
    }
  });

  it('fills all vector components by default and bounds explicit prefixes', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const buffer = graph.createTransientBuffer({
        id: 'vectors',
        byteLength: 48,
        usage: Buffer.STORAGE
      });
      const view = graph.createDataView(buffer, {format: 'float32x4', length: 2, byteOffset: 8});
      const props = {id: 'fill', operation: 'fill', view, type: 'f32' as const, value: '3.0'};
      expect(createFillNode(graph, props).workload?.maximumInvocationCount).toBe(8);
      expect(
        createFillNode(graph, {...props, componentCount: 3}).workload?.maximumInvocationCount
      ).toBe(3);
      expect(() => createFillNode(graph, {...props, componentCount: 9})).toThrow(/componentCount/);
      expect(() => createFillNode(graph, {...props, componentCount: 0.5})).toThrow(
        /componentCount/
      );
    } finally {
      device.destroy();
    }
  });

  it('rejects unsupported inferred formats and strided layouts', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const buffer = graph.createTransientBuffer({
        id: 'words',
        byteLength: 32,
        usage: Buffer.STORAGE
      });
      const narrow = graph.createDataView(buffer, {format: 'uint16x2', length: 2});
      expect(() => getWGSLKernelBinding('words', narrow)).toThrow(/format/);
      const strided = graph.createDataView(buffer, {format: 'uint32', length: 2, byteStride: 8});
      expect(() => getWGSLKernelBinding('words', strided)).toThrow(/packed/);
      const fill = {id: 'fill', operation: 'fill', type: 'u32' as const, value: '0u'};
      expect(() => createFillNode(graph, {...fill, view: narrow})).toThrow(/format/);
      expect(() => createFillNode(graph, {...fill, view: strided})).toThrow(/packed/);
      const signed = graph.createDataView(buffer, {format: 'sint32x2', length: 2});
      expect(getWGSLKernelBinding('words', signed, 'atomic').type).toBe('atomic<i32>');
    } finally {
      device.destroy();
    }
  });
});

it('round-trips kilometer frames through the native planar context validator', () => {
  const frame = createGPUSpatialFrameFromCRS('+proj=utm +zone=33 +units=km');
  expect(() =>
    validateGPUSpatialContext('kilometer-frame', {
      coordinateSpace: 'planar',
      metric: 'native',
      frame,
      units: getGPUSpatialFrameContextUnits(frame)
    })
  ).not.toThrow();
  expect(frame.unit?.metersPerUnit).toBe(1000);
});
