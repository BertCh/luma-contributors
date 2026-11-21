// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUVector} from '@luma.gl/graph/gpu-data';
import {Buffer} from '@luma.gl/core';
import {createRequire} from 'node:module';
import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {describe, expect, expectTypeOf, test} from 'vitest';
import {createGPULocalFrame} from '../../src/analysis-core/contracts/local-frame';
import {
  defineGPUSpatialParameterSchema,
  packGPUSpatialParameterValues
} from '../../src/analysis-core/contracts/parameter-schema';
import type {RasterGridAffine, PublishNodeProps, WGSLDispatchGate} from '../../src/analysis-core';
import {
  GPUParameterBuffer,
  AnalysisValidationError,
  getGraphViewChunks,
  isGraphVectorView,
  isGraphDataView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewTopology,
  validateSinglePackedView,
  validateGraphPorts,
  validateCompactOutput,
  validateFiniteNumber,
  validateNonZeroFiniteNumber,
  validateCapacity,
  validateMaximumCount,
  validateViewLength,
  validateOption
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUCompactOutput} from '../../src/analysis-core/kernels/gpu-contributor-types';
import {getSortedSegmentSumNodes} from '../../src/analysis-core/kernels/sorted-segment-sums';
import {createStorageTopKNode} from '../../src/analysis-core/kernels/storage-backed-top-k';
import {getGPUSpatialKMeansParameterValues} from '../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans';
import {getGPUDBSCANParameterValues} from '../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters';
import {createShorterGradientProducer} from './fitting/shorter-gradient-producer';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

// AEQD on the equator maps a longitude offset to the arc R * radians. Reference (float32 output):
//   /usr/bin/python3 -c "import numpy as np, math; print(float(np.float32(6371008.8 * math.radians(0.001))))"
//   -> 111.19508361816406
const PROJECTED_X = 111.19508361816406;
// Inverse reference: math.degrees(111.19508361816406 / 6371008.8) -> 0.0010000000304386772; the
// math.gl inverse differs by about 6e-15 degrees, so the check uses 1e-12.
const UNPROJECTED_LONGITUDE = 0.0010000000304386772;

describe('CORE-101: local-frame array conversion checks a supplied target', () => {
  const frame = createGPULocalFrame({origin: [0, 0]});

  test('projectArray rejects a short target before writing, naming both lengths', () => {
    const target = new Float32Array([-1, -1]);
    expect(() => frame.projectArray([0, 0, 0.001, 0], 2, target)).toThrow(
      /projectArray target length 2 is shorter than the 4 required/
    );
    expect(Array.from(target)).toEqual([-1, -1]);
  });

  test('projectArray keeps the row layout in exact and padded targets', () => {
    const exact = frame.projectArray([0, 0, 0.001, 0], 2, new Float32Array(4));
    expect(Array.from(exact)).toEqual([0, 0, PROJECTED_X, 0]);
    const padded = frame.projectArray([0, 0, 0.001, 0], 2, new Float32Array([9, 9, 9, 9, 9, 9]));
    expect(Array.from(padded)).toEqual([0, 0, PROJECTED_X, 0, 9, 9]);
  });

  test('unprojectArray rejects a short target before writing, naming both lengths', () => {
    const target = new Float64Array([-1, -1]);
    expect(() => frame.unprojectArray([0, 0, PROJECTED_X, 0], target)).toThrow(
      /unprojectArray target length 2 is shorter than the 4 required/
    );
    expect(Array.from(target)).toEqual([-1, -1]);
  });

  test('unprojectArray keeps the row layout in exact and padded targets', () => {
    for (const target of [new Float64Array(4), new Float64Array([9, 9, 9, 9, 9])]) {
      const output = frame.unprojectArray([0, 0, PROJECTED_X, 0], target);
      expect(output).toBe(target);
      expect(output.slice(0, 4).every(Number.isFinite)).toBe(true);
      expect(output[0]).toBe(0);
      expect(output[1]).toBe(0);
      expect(output[2]).toBeCloseTo(UNPROJECTED_LONGITUDE, 12);
      expect(output[3]).toBe(0);
      if (target.length > 4) expect(output[4]).toBe(9);
    }
  });
});

describe('CORE-107: clustering packers validate the encoded float32 word', () => {
  const settings = {bounds: [0, 0, 1, 1] as const, epsilon: 4, minimumSamples: 1};

  test('a finite radius that encodes as 0 or infinity is rejected', () => {
    expect(() => getGPUDBSCANParameterValues({...settings, epsilon: 1e-50})).toThrow(/epsilon/);
    expect(() => getGPUDBSCANParameterValues({...settings, epsilon: 1e39})).toThrow(/epsilon/);
    expect(Array.from(getGPUDBSCANParameterValues(settings))).toEqual([0, 0, 1, 1, 4, 1, 0, 0]);
  });

  test('bounds and timeEpsilon are validated after float32 encoding', () => {
    expect(() => getGPUDBSCANParameterValues({...settings, bounds: [0, 0, 1e39, 1]})).toThrow(
      /bounds/
    );
    expect(() => getGPUDBSCANParameterValues({...settings, timeEpsilon: 1e-50})).toThrow(
      /timeEpsilon/
    );
    expect(() => getGPUDBSCANParameterValues({...settings, timeEpsilon: 1e39})).toThrow(
      /timeEpsilon/
    );
    expect(getGPUDBSCANParameterValues({...settings, timeEpsilon: 2})[6]).toBe(2);
  });

  test('the k-means tolerance cannot encode as infinity', () => {
    expect(() => getGPUSpatialKMeansParameterValues({tolerance: 1e39})).toThrow(/tolerance/);
    const values = getGPUSpatialKMeansParameterValues({tolerance: 0.5});
    expect(new Float32Array(values.buffer)[1]).toBe(0.5);
  });
});

describe('CORE-108: parameter schema reads only own overrides', () => {
  const PROTOTYPE_NAMES = ['constructor', 'toString', '__proto__', 'hasOwnProperty', 'valueOf'];

  test.each(PROTOTYPE_NAMES)('field "%s" uses its default and an own override', name => {
    const schema = defineGPUSpatialParameterSchema({
      id: 'prototype-schema',
      format: 'float32',
      wordLength: 1,
      fields: [{name, format: 'float32', wordOffset: 0, defaultValue: 7, dynamic: true}]
    });
    expect(Array.from(packGPUSpatialParameterValues(schema))).toEqual([7]);
    expect(Array.from(packGPUSpatialParameterValues(schema, {}))).toEqual([7]);
    expect(Array.from(packGPUSpatialParameterValues(schema, Object.create(null)))).toEqual([7]);
    // A computed key defines an own property, including `__proto__`.
    expect(Array.from(packGPUSpatialParameterValues(schema, {[name]: 3}))).toEqual([3]);
    const nullPrototypeValues = Object.create(null) as Record<string, number>;
    nullPrototypeValues[name] = 5;
    expect(Array.from(packGPUSpatialParameterValues(schema, nullPrototypeValues))).toEqual([5]);
  });
});

describe('CORE-109: sorted segment sums validate before declaring scratch', () => {
  test.each([
    'rows',
    'segments'
  ] as const)('a %s dispatch refusal leaves the resource set unchanged', subject => {
    const device = createNullWebGPUDevice();
    Object.assign(device.limits, {maxComputeWorkgroupsPerDimension: 1});
    try {
      const graph = new GPUCommandGraph(device);
      const length = subject === 'rows' ? 257 : 2;
      const segmentKeys = createTransientView(graph, 'keys', 'uint32', length);
      const sumContributions = createTransientView(graph, 'contributions', 'float32', length);
      const sums = createTransientView(graph, 'sums', 'float32', 2);
      const props = {
        id: 'reduce',
        operation: 'GPUDemo',
        segmentCount: subject === 'segments' ? 2 : 1,
        segmentKeys,
        sumContributions,
        sums
      };
      const resources = Reflect.get(graph, 'buffers') as Map<string, unknown>;
      const before = Array.from(resources.keys());
      expect(() => getSortedSegmentSumNodes(graph, props)).toThrow(/reduce.*3D dispatch limit/);
      expect(Array.from(resources.keys())).toEqual(before);
      expect(
        getSortedSegmentSumNodes(graph, {
          ...props,
          segmentCount: 1,
          segmentKeys: graph.createDataView(segmentKeys.buffer, {format: 'uint32', length: 2}),
          sumContributions: graph.createDataView(sumContributions.buffer, {
            format: 'float32',
            length: 2
          })
        }).map(node => node.id)
      ).toContain('reduce-reduce-sums');
    } finally {
      device.destroy();
    }
  });

  test.each([
    'contributions',
    'output',
    'keys',
    'offsets'
  ] as const)('a device binding refusal for %s leaves the resource set unchanged', subject => {
    const device = createNullWebGPUDevice();
    Object.assign(device.limits, {maxStorageBufferBindingSize: 16});
    try {
      const graph = new GPUCommandGraph(device);
      const keyStorage = createTransientView(graph, 'keys', 'uint32', subject === 'keys' ? 5 : 2);
      const segmentKeys =
        subject === 'keys'
          ? graph.createDataView(keyStorage.buffer, {format: 'uint32', length: 2, byteOffset: 12})
          : keyStorage;
      const contributions = createTransientView(
        graph,
        'contributions',
        'float32',
        subject === 'contributions' ? 5 : 2
      );
      const output = createTransientView(graph, 'output', 'float32', subject === 'output' ? 5 : 4);
      const props = {
        id: 'reduce',
        operation: 'GPUDemo',
        segmentCount: subject === 'offsets' ? 4 : 1,
        segmentKeys,
        sumContributions: contributions,
        sums: output
      };
      // Real graph handles, without allocating a device buffer. Observe declaration state only.
      const resources = Reflect.get(graph, 'buffers') as Map<string, unknown>;
      const before = Array.from(resources.keys());
      expect(() => getSortedSegmentSumNodes(graph, props)).toThrow(
        subject === 'keys' ? /segmentKeys.*limit 16/ : /limit 16/
      );
      expect(Array.from(resources.keys())).toEqual(before);
      // Correct views borrow existing storage; the rejected reduction ID must be reusable.
      const corrected = getSortedSegmentSumNodes(graph, {
        ...props,
        segmentCount: 1,
        segmentKeys: graph.createDataView(segmentKeys.buffer, {format: 'uint32', length: 2}),
        sumContributions: graph.createDataView(contributions.buffer, {
          format: 'float32',
          length: 2
        }),
        sums: graph.createDataView(output.buffer, {format: 'float32', length: 1})
      });
      expect(corrected.map(node => node.id)).toContain('reduce-reduce-sums');
    } finally {
      device.destroy();
    }
  });

  test('a rejected call reserves no IDs, so a corrected call with the same ID succeeds', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const segmentKeys = createTransientView(graph, 'keys', 'uint32', 2);
    const sumContributions = createTransientView(graph, 'contributions', 'float32', 2);
    const props = {
      id: 'reduce',
      operation: 'GPUDemo',
      segmentCount: 2,
      segmentKeys,
      sumContributions
    };
    expect(() =>
      getSortedSegmentSumNodes(graph, {
        ...props,
        sums: createTransientView(graph, 'short-sums', 'float32', 1)
      })
    ).toThrow(/^reduce-reduce-sums segmentCount must be at most 1/);
    const nodes = getSortedSegmentSumNodes(graph, {
      ...props,
      sums: createTransientView(graph, 'sums', 'float32', 2)
    });
    expect(nodes.map(node => node.id)).toContain('reduce-reduce-sums');
  });

  test('duplicate reduction names are rejected before any declaration', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const segmentKeys = createTransientView(graph, 'keys', 'uint32', 2);
    const contributions = createTransientView(graph, 'contributions', 'float32', 2);
    const output = createTransientView(graph, 'output', 'float32', 2);
    const props = {id: 'reduce', operation: 'GPUDemo', segmentCount: 2, segmentKeys};
    expect(() =>
      getSortedSegmentSumNodes(graph, {
        ...props,
        reductions: [
          {name: 'x', contributions, output},
          {name: 'x', contributions, output}
        ]
      })
    ).toThrow(/^reduce reduction name "x" must be unique/);
    expect(getSortedSegmentSumNodes(graph, props).length).toBeGreaterThan(0);
  });
});

describe('CORE-113: GPUParameterBuffer allocates from the validated length', () => {
  test('a stateful length getter is read once', () => {
    const device = createNullWebGPUDevice();
    let reads = 0;
    const props = {
      id: 'stateful-length',
      format: 'float32' as const,
      get length() {
        reads++;
        return reads === 1 ? 2 : 0;
      }
    };
    const parameters = new GPUParameterBuffer(device, props);
    try {
      expect(parameters.length).toBe(2);
      expect(parameters.buffer.byteLength).toBe(8);
      expect(reads).toBe(1);
    } finally {
      parameters.destroy();
    }
  });
});

describe('CORE-121: storage top-K releases its uniform when the kernel is refused', () => {
  test('the node-owned selection-settings buffer is destroyed on a compile refusal', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const node = createStorageTopKNode(graph, {
      id: 'topK',
      operation: 'GPUDemo',
      bindings: [
        {
          name: 'values',
          view: createTransientView(graph, 'values', 'uint32', 4),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: 4,
      body: 'values[valuesOffset + index] = 0u;'
    });
    const created: {id: string; destroyed: boolean}[] = [];
    // A non-WebGPU device makes the published Kernel constructor refuse after the uniform exists.
    const refusingDevice = {
      type: 'webgl',
      createBuffer: (props: {id: string}) => {
        const buffer = {
          id: props.id,
          destroyed: false,
          destroy() {
            buffer.destroyed = true;
          }
        };
        created.push(buffer);
        return buffer;
      }
    } as unknown as Device;
    expect(() => node.compile({device: refusingDevice})).toThrow(/WebGPU/);
    expect(created.map(buffer => [buffer.id, buffer.destroyed])).toEqual([
      ['topK-selection-settings', true]
    ]);
  });
});

describe('review round 1: shared kernel construction contracts', () => {
  test('B1 the shorter fitting fixture lowers over the same padded storage and destination', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const storage = createTransientView(graph, 'gradient-storage', 'float32', 260);
      const gradients = graph.createDataView(storage.buffer, {
        format: 'float32',
        length: 258,
        byteOffset: 4
      });
      const sums = createTransientView(graph, 'sums', 'float32', 1);
      const producer = createShorterGradientProducer(graph, gradients, sums);
      expect(producer.props.gradients.buffer).toBe(gradients.buffer);
      expect(producer.props.gradients.byteOffset).toBe(4);
      expect(producer.props.gradients.length).toBe(256);
      expect(producer.props.output.sums).toBe(sums);
      expect(producer.getCommandNodes(graph).length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });

  test.each([
    'oversized',
    'offset empty'
  ] as const)('B3 empty reductions accept an unbound %s contribution view', subject => {
    const device = createNullWebGPUDevice();
    Object.assign(device.limits, {maxStorageBufferBindingSize: 16});
    try {
      const graph = new GPUCommandGraph(device);
      const storage = createTransientView(graph, 'source-storage', 'float32', 8);
      const contributions =
        subject === 'oversized'
          ? storage
          : graph.createDataView(storage.buffer, {format: 'float32', length: 0, byteOffset: 16});
      const nodes = getSortedSegmentSumNodes(graph, {
        id: 'empty',
        operation: 'GPUDemo',
        segmentCount: 2,
        segmentKeys: createTransientView(graph, 'keys', 'uint32', 0),
        sumContributions: contributions,
        sums: createTransientView(graph, 'sums', 'float32', 2)
      });
      expect(nodes.map(node => node.id)).toEqual([
        'empty-sort-prepare',
        'empty-segment-offsets',
        'empty-reduce-sums'
      ]);
    } finally {
      device.destroy();
    }
  });

  test.each([
    'keys',
    'output',
    'contributions'
  ] as const)('B4 rejects unsupported %s layout before reserving scratch', subject => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const keyStorage = createTransientView(graph, 'key-storage', 'uint32', 5);
      const sourceStorage = createTransientView(graph, 'source-storage', 'float32', 5);
      const outputStorage = createTransientView(graph, 'output-storage', 'float32', 3);
      const props = {
        id: 'layout',
        operation: 'GPUDemo',
        segmentCount: 2,
        segmentKeys: graph.createDataView(keyStorage.buffer, {
          format: 'uint32',
          length: 3,
          byteStride: subject === 'keys' ? 8 : 4
        }),
        sumContributions: graph.createDataView(sourceStorage.buffer, {
          format: 'float32',
          length: 3,
          byteStride: subject === 'contributions' ? 6 : 4
        }),
        sums: graph.createDataView(outputStorage.buffer, {
          format: 'float32',
          length: 2,
          byteStride: subject === 'output' ? 8 : 4
        })
      };
      const resources = Reflect.get(graph, 'buffers') as Map<string, unknown>;
      const before = Array.from(resources.keys());
      expect(() => getSortedSegmentSumNodes(graph, props)).toThrow();
      expect(Array.from(resources.keys())).toEqual(before);
      expect(
        getSortedSegmentSumNodes(graph, {
          ...props,
          segmentKeys: graph.createDataView(keyStorage.buffer, {format: 'uint32', length: 3}),
          sumContributions: graph.createDataView(sourceStorage.buffer, {
            format: 'float32',
            length: 3
          }),
          sums: graph.createDataView(outputStorage.buffer, {format: 'float32', length: 2})
        }).length
      ).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });

  test('B4 aligned strided contributions remain supported', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const source = createTransientView(graph, 'source', 'float32', 5);
      expect(
        getSortedSegmentSumNodes(graph, {
          id: 'strided',
          operation: 'GPUDemo',
          segmentCount: 2,
          segmentKeys: createTransientView(graph, 'keys', 'uint32', 3),
          sumContributions: graph.createDataView(source.buffer, {
            format: 'float32',
            length: 3,
            byteStride: 8
          }),
          sums: createTransientView(graph, 'sums', 'float32', 2)
        }).length
      ).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });

  test.each(['keys', 'indices'] as const)('B5 a reduction named %s has separate scratch', name => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const nodes = getSortedSegmentSumNodes(graph, {
        id: 'names',
        operation: 'GPUDemo',
        segmentCount: 2,
        segmentKeys: createTransientView(graph, 'keys', 'uint32', 3),
        reductions: [
          {
            name,
            contributions: createTransientView(graph, 'source', 'float32', 3),
            output: createTransientView(graph, 'output', 'float32', 2)
          }
        ]
      });
      expect(nodes.map(node => node.id)).toContain(`names-reduce-${name}`);
    } finally {
      device.destroy();
    }
  });

  test('S2 storage binding count refuses before scratch declaration and names the instance', () => {
    const device = createNullWebGPUDevice();
    Object.assign(device.limits, {maxStorageBuffersPerShaderStage: 2});
    try {
      const graph = new GPUCommandGraph(device);
      const props = {
        id: 'limited',
        operation: 'GPUDemo',
        segmentCount: 1,
        segmentKeys: createTransientView(graph, 'keys', 'uint32', 2)
      };
      const resources = Reflect.get(graph, 'buffers') as Map<string, unknown>;
      const before = Array.from(resources.keys());
      expect(() => getSortedSegmentSumNodes(graph, props)).toThrow(/limited/);
      expect(Array.from(resources.keys())).toEqual(before);
      Object.assign(device.limits, {maxStorageBuffersPerShaderStage: 8});
      expect(getSortedSegmentSumNodes(graph, props).length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });

  test('earlier-fix note: missing contribution rows refuse before declaration', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const storage = createTransientView(graph, 'source', 'float32', 3);
      const props = {
        id: 'incomplete',
        operation: 'GPUDemo',
        segmentCount: 2,
        segmentKeys: createTransientView(graph, 'keys', 'uint32', 3),
        sumContributions: graph.createDataView(storage.buffer, {format: 'float32', length: 2}),
        sums: createTransientView(graph, 'sums', 'float32', 2)
      };
      const resources = Reflect.get(graph, 'buffers') as Map<string, unknown>;
      const before = Array.from(resources.keys());
      expect(() => getSortedSegmentSumNodes(graph, props)).toThrow(/contributions/);
      expect(Array.from(resources.keys())).toEqual(before);
      expect(
        getSortedSegmentSumNodes(graph, {...props, sumContributions: storage}).length
      ).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
});

test('ENT-08: public core functions expose their named signature types', () => {
  expectTypeOf<RasterGridAffine>().toEqualTypeOf<
    import('../../src/analysis-core/contracts/raster-grid').RasterGridAffine
  >();
  expectTypeOf<PublishNodeProps>().toEqualTypeOf<
    import('../../src/analysis-core/kernels/wgsl-kernel-nodes').PublishNodeProps
  >();
  expectTypeOf<WGSLDispatchGate<unknown>>().toEqualTypeOf<
    import('../../src/analysis-core/kernels/wgsl-kernel-nodes').WGSLDispatchGate<unknown>
  >();
});

describe('ENT-02: graph views across module conditions', () => {
  test('preserves foreign vectors, empty vectors, identity, topology and ownership', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const otherGraph = new GPUCommandGraph(device);
      const first = createTransientView(graph, 'first', 'float32', 1);
      const second = createTransientView(graph, 'second', 'float32', 2);
      const {GraphVectorView: ForeignVector} = createRequire(import.meta.url)(
        '@luma.gl/graph/gpu-core'
      ) as {
        GraphVectorView: new (
          props: ConstructorParameters<typeof GraphVectorView>[0]
        ) => GraphVectorView;
      };
      const props = {
        id: 'foreign',
        name: 'foreign',
        format: 'float32' as const,
        length: 3,
        valueLength: 3,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: [first, second]
      };
      const vector = new ForeignVector(props);
      const empty = new ForeignVector({...props, length: 0, valueLength: 0, data: []});
      expect(getGraphViewChunks(vector)).toEqual([first, second]);
      expect(getGraphViewChunks(vector)[0]).toBe(first);
      expect(getGraphViewChunks(first)[0]).toBe(first);
      expect(getGraphViewChunks(empty)).toEqual([]);
      expect(isGraphVectorView(vector)).toBe(true);
      expect(isGraphVectorView(first)).toBe(false);
      expect(() => validateGraphViewsBelongToGraph('mixed', graph, [vector, empty])).not.toThrow();
      expect(() => validateGraphViewsBelongToGraph('mixed', otherGraph, [vector])).toThrow(
        /target graph/
      );
      expect(() => validateGraphOutputsDisjointFromInputs('mixed', [first], [vector])).toThrow(
        /conflicts/
      );
      expect(() =>
        validateGraphViewTopology('mixed', 'values', vector, new ForeignVector(props))
      ).not.toThrow();
      expect(() => validateGraphViewTopology('mixed', 'values', vector, first)).toThrow(/topology/);
      expect(() =>
        validateGraphPorts(
          'mixed',
          {vector},
          {vector: {required: true, viewType: 'data', formats: ['float32']}}
        )
      ).toThrow(/vector.*data graph view/);
      expect(() =>
        validateGraphPorts(
          'mixed',
          {vector: empty},
          {vector: {required: true, viewType: 'vector', formats: ['float32']}}
        )
      ).not.toThrow();
      expect(() => validateSinglePackedView(vector, ['float32'], 'mixed values')).toThrow(
        /one packed view/
      );
    } finally {
      device.destroy();
    }
  });
});

describe('VAL-01 / VAL-10: named port preflight', () => {
  test('refuses missing, wrong-kind and wrong-format ports before use', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const values = createTransientView(graph, 'values', 'uint32', 3);
      const descriptor = {required: true, viewType: 'data' as const, formats: ['float32'] as const};
      expect(() => validateGraphPorts('lidar', {values: undefined}, {values: descriptor})).toThrow(
        /lidar values/
      );
      try {
        validateGraphPorts(
          {id: 'lidar', contributor: 'GPUPointCloudExample'},
          {values},
          {values: descriptor}
        );
        expect.unreachable('wrong format accepted');
      } catch (error) {
        expect(error).toMatchObject({
          code: 'VIEW_FORMAT',
          id: 'lidar',
          contributor: 'GPUPointCloudExample',
          path: 'values',
          received: 'uint32',
          expected: {viewType: 'data', formats: ['float32']}
        });
      }
      expect(() => validateGraphPorts('lidar', {values}, {values: descriptor})).toThrow(
        /received uint32/
      );
      expect(() =>
        validateGraphPorts('lidar', {}, {optional: {...descriptor, required: false}})
      ).not.toThrow();
      expect(() =>
        validateSinglePackedView(undefined as never, ['float32'], 'lidar values')
      ).toThrow(/lidar values/);
    } finally {
      device.destroy();
    }
  });
});

describe('VAL-09: compact status preflight', () => {
  test('requires count and overflow but keeps optional loss words optional', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const ids = createTransientView(graph, 'ids', 'uint32', 4);
      const count = createTransientView(graph, 'count', 'uint32', 1);
      const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
      for (const [property, path] of [
        ['status', 'status'],
        ['count', 'status.count'],
        ['overflow', 'status.overflow']
      ] as const) {
        const output: GPUCompactOutput = {ids, status: {count, overflow}};
        // Remove a required field as an untyped caller can, retaining the public input type.
        if (property === 'status') Reflect.deleteProperty(output, property);
        else Reflect.deleteProperty(output.status, property);
        expect(() => validateCompactOutput('pick', output, 'output.matches')).toThrow(
          `pick output.matches.${path}`
        );
      }
      expect(() => validateCompactOutput('pick', {ids, status: {count, overflow}})).not.toThrow();
      const short = createTransientView(graph, 'short', 'uint32', 0);
      expect(() => validateCompactOutput('pick', {ids, status: {count: short, overflow}})).toThrow(
        /count.*one uint32 row/
      );
    } finally {
      device.destroy();
    }
  });
});

describe('VAL-14: numeric validation does not coerce', () => {
  test.each([true, '2', null, 1e39])('rejects %s and preserves its original value', value => {
    for (const validate of [validateFiniteNumber, validateNonZeroFiniteNumber]) {
      try {
        validate('lidar', 'threshold', value as number);
        expect.unreachable('invalid input accepted');
      } catch (error) {
        expect(error).toBeInstanceOf(AnalysisValidationError);
        expect((error as AnalysisValidationError).received).toBe(value);
        expect((error as AnalysisValidationError).code).toBe('NUMBER_RANGE');
      }
    }
  });
});

describe('VAL-12: structured construction errors', () => {
  test('keeps stable fields, derived-limit sources and readable instance prefixes', () => {
    const identity = {id: 'lidar', contributor: 'GPUPointCloudExample'};
    const cases = [
      [
        () => validateCapacity(identity, 'capacity', 1.5),
        'INTEGER_RANGE',
        'capacity',
        1.5,
        {minimum: 1, maximum: 0xffffffff}
      ],
      [
        () => validateMaximumCount(identity, 'cells', 5, 4, 'gridSize[0] * gridSize[1]'),
        'INTEGER_RANGE',
        'cells',
        5,
        {maximum: 4, source: 'gridSize[0] * gridSize[1]'}
      ],
      [
        () => validateViewLength(identity, 'values', {length: 3}, 4, 'counts.length * capacity'),
        'VIEW_LENGTH',
        'values',
        3,
        {rows: 4, source: 'counts.length * capacity'}
      ],
      [
        () => validateOption(identity, 'mode', 'mystery', ['nearest', 'linear']),
        'INVALID_OPTION',
        'mode',
        'mystery',
        {values: ['nearest', 'linear']}
      ],
      [
        () =>
          validateGraphPorts(
            identity,
            {},
            {values: {required: true, viewType: 'data', formats: ['float32']}}
          ),
        'MISSING_PROP',
        'values',
        undefined,
        {viewType: 'data', formats: ['float32']}
      ]
    ] as const;
    for (const [run, code, path, received, expected] of cases) {
      try {
        run();
        expect.unreachable('invalid input accepted');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect(error).toBeInstanceOf(AnalysisValidationError);
        expect(error).toMatchObject({
          code,
          contributor: 'GPUPointCloudExample',
          id: 'lidar',
          path,
          received,
          expected
        });
        expect((error as Error).message).toMatch(/^lidar /);
      }
    }
    expect(() => validateCapacity('lidar', 'capacity', 1.5)).toThrow(
      /lidar capacity must be an integer/
    );
  });
});

describe('review round 1 shared host boundaries', () => {
  test('B1 captures canonical inherited status getters once', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const ids = createTransientView(graph, 'getter-ids', 'uint32', 2);
      const count = createTransientView(graph, 'getter-count', 'uint32', 1);
      const overflow = createTransientView(graph, 'getter-overflow', 'uint32', 1);
      const reads = {count: 0, overflow: 0, candidateOverflow: 0, requiredCount: 0};
      class Status {
        get count() {
          reads.count++;
          return count;
        }
        get overflow() {
          reads.overflow++;
          return overflow;
        }
        get candidateOverflow() {
          reads.candidateOverflow++;
          return undefined;
        }
        get requiredCount() {
          reads.requiredCount++;
          return undefined;
        }
      }
      expect(() => validateCompactOutput('getter', {ids, status: new Status()})).not.toThrow();
      expect(reads).toEqual({count: 1, overflow: 1, candidateOverflow: 1, requiredCount: 1});
    } finally {
      device.destroy();
    }
  });

  test('B3 rejects physical upstream vectors and data, including empty vectors', () => {
    const device = createNullWebGPUDevice();
    const buffer = device.createBuffer({byteLength: 4, usage: Buffer.STORAGE});
    const vector = new GPUVector({
      type: 'buffer',
      name: 'physical',
      buffer,
      format: 'float32',
      length: 1
    });
    const empty = new GPUVector({
      type: 'data',
      name: 'empty-physical',
      format: 'float32',
      data: []
    });
    try {
      expect(isGraphVectorView(vector)).toBe(false);
      expect(isGraphVectorView(empty)).toBe(false);
      expect(isGraphDataView(vector.data[0])).toBe(false);
      for (const [value, viewType] of [
        [vector, 'vector'],
        [empty, 'vector'],
        [vector.data[0], 'data']
      ] as const) {
        expect(() =>
          validateGraphPorts(
            'physical',
            {value},
            {value: {required: true, viewType, formats: ['float32']}}
          )
        ).toThrow(/graph view/);
      }
    } finally {
      vector.destroy();
      empty.destroy();
      buffer.destroy();
      device.destroy();
    }
  });

  test('S1 discriminates graph vectors without visiting chunks, preflight checks each chunk', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const chunk = createTransientView(graph, 'constant-shape', 'float32', 1);
      const {GraphVectorView: Vector} = createRequire(import.meta.url)(
        '@luma.gl/graph/gpu-core'
      ) as {
        GraphVectorView: new (
          props: ConstructorParameters<typeof GraphVectorView>[0]
        ) => GraphVectorView;
      };
      const vector = new Vector({
        id: 'constant-shape',
        name: 'constant-shape',
        format: 'float32',
        length: 1,
        valueLength: 1,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: [chunk]
      });
      const data = new Proxy(vector.data, {
        get(target, key, receiver) {
          if (key === '0') throw new Error('discriminator traversed chunks');
          return Reflect.get(target, key, receiver);
        }
      });
      const candidate = {...vector, data};
      expect(isGraphVectorView(candidate)).toBe(true);
      expect(getGraphViewChunks(candidate)).toBe(data);
      expect(() =>
        validateGraphPorts(
          'constant',
          {candidate},
          {candidate: {required: true, viewType: 'vector', formats: ['float32']}}
        )
      ).toThrow('discriminator traversed chunks');
    } finally {
      device.destroy();
    }
  });

  test('S3 unknown vectors do not permit a caller-selected scalar-format narrowing', () => {
    const assertUnknownFormat = (value: unknown) => {
      // @ts-expect-error A format parameter requires a correspondingly typed graph-view input.
      isGraphVectorView<'uint32'>(value);
      if (isGraphVectorView(value)) expectTypeOf(value.format).not.toEqualTypeOf<'uint32'>();
    };
    expectTypeOf(assertUnknownFormat).toBeFunction();
  });

  test('B6 uses separate identities and paths through compact and parameter callers', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const wrong = createTransientView(graph, 'wrong-chunk', 'uint32', 1);
      try {
        validateSinglePackedView(
          wrong,
          ['float32'],
          {id: 'survey A', contributor: 'ChunkReader'},
          'readings chunk 2'
        );
        expect.unreachable('wrong format accepted');
      } catch (error) {
        expect(error).toMatchObject({
          id: 'survey A',
          contributor: 'ChunkReader',
          path: 'readings chunk 2'
        });
        expect((error as Error).message).toContain('survey A readings chunk 2 must be packed');
      }
      try {
        new GPUParameterBuffer(device, {id: 'abc', format: 'uint32', length: -1});
        expect.unreachable('invalid length accepted');
      } catch (error) {
        expect(error).toMatchObject({id: 'abc', contributor: 'GPUParameterBuffer', path: 'length'});
      }
      expect(() =>
        validateSinglePackedView(wrong, ['float32'], 'survey A readings chunk 2', 'survey A')
      ).toThrowError(expect.objectContaining({id: 'survey A', path: 'readings chunk 2'}));
      const floatCount = createTransientView(graph, 'wrong-compact-count', 'float32', 1);
      expect(() =>
        validateCompactOutput(
          {id: 'survey A', contributor: 'CompactProducer'},
          {ids: wrong, status: {count: floatCount as never, overflow: wrong}},
          'output.matches'
        )
      ).toThrowError(
        expect.objectContaining({
          id: 'survey A',
          contributor: 'CompactProducer',
          path: 'output.matches.status.count'
        })
      );
      const parameters = new GPUParameterBuffer(device, {id: 'abc', format: 'uint32', length: 1});
      try {
        expect(() => parameters.write(Uint32Array.of(1), 2)).toThrowError(
          expect.objectContaining({
            id: 'abc',
            contributor: 'GPUParameterBuffer',
            path: 'firstElement'
          })
        );
      } finally {
        parameters.destroy();
      }
    } finally {
      device.destroy();
    }
  });

  test('B8 preserves established packed error prefixes and adds received formats', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const wrong = createTransientView(graph, 'wrong-format-prefix', 'uint32', 1);
      expect(() => validateSinglePackedView(wrong, ['float32'], 'lidar distances')).toThrow(
        'lidar distances must be packed, uint32-aligned float32 GPU data; received uint32'
      );
    } finally {
      device.destroy();
    }
  });
});
