// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUSpaceTimeKriging,
  GPU_SPACE_TIME_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT,
  GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT,
  type GPUSpaceTimeKrigingProps,
  type SpaceTimeKrigingRowViews
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createProps(sampleCount = 8, chunkLengths = [5, 3]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number,
    prefix = 0
  ) => {
    const storage = createTransientView(graph, `view-${serial++}`, format, length + prefix + 1);
    return graph.createDataView(storage.buffer, {
      format,
      length,
      byteOffset: prefix * storage.rowByteLength
    });
  };
  const vector = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format
  ) =>
    createVectorView(
      `vector-${serial++}`,
      format,
      chunkLengths.map((length, chunkIndex) => view(format, length, chunkIndex + 1))
    );
  const props = {
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    },
    positions: view('float32x2', sampleCount),
    times: view('float32', sampleCount),
    observations: view('float32x4', sampleCount),
    queries: vector('float32x2'),
    queryTimes: vector('float32'),
    parameters: view('float32', 28),
    mode: 'ordinary',
    output: {values: vector('float32x4'), status: vector('uint32')}
  } satisfies GPUSpaceTimeKrigingProps<SpaceTimeKrigingRowViews>;
  return {device, graph, props, view};
}

it('defaults to the index beyond the named cut-over and preserves the explicit scan ceiling', () => {
  for (const sampleCount of [
    GPU_SPACE_TIME_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT,
    GPU_SPACE_TIME_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT + 1,
    8 * 1024 * 1024
  ]) {
    const {graph, props} = createProps(sampleCount);
    const contributor = new GPUSpaceTimeKriging(props);
    expect(contributor.sampleIndex).toBe(
      sampleCount > GPU_SPACE_TIME_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT ? 'grid' : 'brute-force'
    );
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    if (sampleCount > GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT) {
      expect(() => new GPUSpaceTimeKriging({...props, sampleIndex: 'brute-force'})).toThrow(
        `positions.length must be an integer in [1, ${GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT}]`
      );
    }
  }
});

it('keeps the distinct shader and pipeline count constant as uneven, unaligned query chunks grow', () => {
  for (const mode of ['simple', 'ordinary'] as const) {
    for (const sampleIndex of ['grid', 'brute-force'] as const) {
      for (const neighborSearch of ['metric', 'octant', 'staged'] as const) {
        const pipelineKeys = (chunkLengths: number[]) => {
          const {graph, props} = createProps(4097, chunkLengths);
          const contributor = new GPUSpaceTimeKriging({
            ...props,
            mode,
            sampleIndex,
            neighborSearch
          });
          const nodes = contributor.getCommandNodes(graph);
          expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
          const sources = getParsedKernelSources(nodes);
          expect(sources.size).toBeGreaterThan(0);
          const kernels = getGPUCommandNodeKernelSources(nodes);
          for (const {id, source} of kernels) {
            const code = stripWGSLComments(source);
            expect(findReservedIdentifiers(code), id).toEqual([]);
            expect(
              (code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
                WGSL_RESERVED_WORDS.has(identifier)
              ),
              id
            ).toEqual([]);
          }
          return {
            sources: [...new Set(kernels.map(kernel => kernel.source))].sort(),
            pipelines: [...getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true})].sort()
          };
        };
        expect(pipelineKeys([1, 0, 31, 32, 33, 255, 256, 257])).toEqual(pipelineKeys([2, 35, 256]));
      }
    }
  }
});

it('rejects overlapping writable ranges within and across output fields at construction', () => {
  const {props, view, graph: overlapGraph} = createProps();
  const values = view('float32x4', 8);
  const repeated = createVectorView('overlapping', 'float32x4', [
    overlapGraph.createDataView(values.buffer, {format: 'float32x4', length: 5, byteOffset: 0}),
    overlapGraph.createDataView(values.buffer, {format: 'float32x4', length: 3, byteOffset: 16})
  ]);
  expect(
    () => new GPUSpaceTimeKriging({...props, output: {...props.output, values: repeated}})
  ).toThrow('byte ranges must not overlap');
  const {props: packed, graph, view: packedView} = createProps(8, [2]);
  const outputValues = packedView('float32x4', 2);
  const outputStatus = graph.createDataView(outputValues.buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: outputValues.byteOffset + 4
  });
  expect(
    () => new GPUSpaceTimeKriging({...packed, output: {values: outputValues, status: outputStatus}})
  ).toThrow('byte ranges must not overlap');
});

it('rejects mismatched query chunk boundaries and invalid packed formats', () => {
  const {props, view} = createProps();
  expect(
    () =>
      new GPUSpaceTimeKriging({
        ...props,
        queryTimes: createVectorView('mismatched', 'float32', [
          view('float32', 4),
          view('float32', 4)
        ])
      })
  ).toThrow('queryTimes chunk lengths');
  expect(() => new GPUSpaceTimeKriging({...props, positions: view('float32', 8) as never})).toThrow(
    'positions must be packed'
  );
});

it('checks aligned view ranges and all derived scratch before declaring graph resources', () => {
  const {device, props, graph} = createProps(400, [2]);
  // The sample observation binding alone exceeds this deliberately tiny device limit.
  const contributor = new GPUSpaceTimeKriging({...props, sampleIndex: 'grid'});
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 5000}
  });
  const createTransientBuffer = vi.spyOn(graph, 'createTransientBuffer');
  const createDataView = vi.spyOn(graph, 'createDataView');
  expect(() => contributor.getCommandNodes(graph)).toThrow('observations chunk 0 needs 6400 bytes');
  expect(createTransientBuffer).not.toHaveBeenCalled();
  expect(createDataView).not.toHaveBeenCalled();
  createTransientBuffer.mockRestore();
  createDataView.mockRestore();

  const second = createProps(8, [2]);
  Object.defineProperty(second.device, 'limits', {
    value: {...second.device.limits, maxStorageBufferBindingSize: 140}
  });
  const secondContributor = new GPUSpaceTimeKriging({...second.props, sampleIndex: 'grid'});
  const allocate = vi.spyOn(second.graph, 'createTransientBuffer');
  expect(() => secondContributor.getCommandNodes(second.graph)).toThrow(
    'private parameters needs 160 bytes'
  );
  expect(allocate).not.toHaveBeenCalled();
  allocate.mockRestore();
});

it('accepts empty chunks and emits no work for no queries', () => {
  const {graph, props} = createProps(8, [0, 0]);
  const contributor = new GPUSpaceTimeKriging(props);
  expect(contributor.getCommandNodes(graph)).toEqual([]);
  expect(contributor.estimatedCommandNodeCount).toBe(0);
});

it('checks the aligned binding prefix when the logical input bytes alone fit', () => {
  const {device, graph, props} = createProps(7, [1]);
  const observations = graph.createDataView(props.observations.buffer, {
    format: 'float32x4',
    length: 7,
    byteOffset: 4
  });
  // 112 logical bytes fit; the actual aligned range includes the four-byte prefix.
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 114}
  });
  const contributor = new GPUSpaceTimeKriging({...props, observations});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createDataView = vi.spyOn(graph, 'createDataView');
  expect(() => contributor.getCommandNodes(graph)).toThrow('observations chunk 0 needs 116 bytes');
  expect(allocate).not.toHaveBeenCalled();
  expect(createDataView).not.toHaveBeenCalled();
  allocate.mockRestore();
  createDataView.mockRestore();
});

it('dispatches each chunk at its own workgroup count with no chunk-dependent pipeline overrides', () => {
  const lengths = [1, 0, 31, 32, 33, 255, 256, 257];
  const {graph, props} = createProps(8, lengths);
  const nodes = new GPUSpaceTimeKriging({...props, sampleIndex: 'grid'}).getCommandNodes(graph);
  const nonempty = lengths.filter(length => length > 0);
  const chunkNodes = nodes.filter(node => /-chunk-\d+-/.test(node.id));
  expect(chunkNodes.length).toBe(nonempty.length * 4);
  for (const node of chunkNodes) {
    const chunkIndex = Number(node.id.match(/-chunk-(\d+)-/)![1]);
    expect(node.workload?.maximumInvocationCount).toBe(nonempty[chunkIndex]);
    expect(node.workload?.maximumWorkgroupCount).toBe(Math.ceil(nonempty[chunkIndex] / 32));
  }
  for (const kernel of getGPUCommandNodeKernelSources(chunkNodes)) {
    expect(kernel.constants).toEqual({});
    expect(kernel.source).toContain('@builtin(num_workgroups)');
    expect(kernel.source).not.toMatch(/\boverride\b/);
  }
});

it('allows adjacent shared-buffer output chunks and ignores empty chunk bindings', () => {
  const {graph, props, view} = createProps(8, [1, 0, 1]);
  const values = view('float32x4', 2);
  const status = view('uint32', 2);
  const chunks = <Format extends 'float32x4' | 'uint32'>(
    storage: import('@luma.gl/graph/gpu-core').GraphDataView<Format>
  ) =>
    createVectorView(storage.buffer.id, storage.format, [
      graph.createDataView(storage.buffer, {format: storage.format, length: 1, byteOffset: 0}),
      graph.createDataView(storage.buffer, {
        format: storage.format,
        length: 0,
        byteOffset: storage.rowByteLength
      }),
      graph.createDataView(storage.buffer, {
        format: storage.format,
        length: 1,
        byteOffset: storage.rowByteLength
      })
    ]);
  const contributor = new GPUSpaceTimeKriging({
    ...props,
    output: {values: chunks(values), status: chunks(status)}
  });
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
});

it('rejects non-final output spare capacity and permits it only in the final chunk', () => {
  const {props, view} = createProps(8, [1, 1]);
  const values = (lengths: number[]) =>
    createVectorView(
      'values',
      'float32x4',
      lengths.map(length => view('float32x4', length))
    );
  expect(
    () => new GPUSpaceTimeKriging({...props, output: {...props.output, values: values([2, 1])}})
  ).toThrow('non-final output chunk lengths must match queries');
  expect(
    () => new GPUSpaceTimeKriging({...props, output: {...props.output, values: values([1, 2])}})
  ).not.toThrow();
});

it('checks runtime uniform support before declaring any graph resources or nodes', () => {
  const {device, graph, props} = createProps();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxUniformBufferBindingSize: 128}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createDataView = vi.spyOn(graph, 'createDataView');
  expect(() => new GPUSpaceTimeKriging(props).getCommandNodes(graph)).toThrow(
    'chunk uniform byteLength'
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createDataView).not.toHaveBeenCalled();
  allocate.mockRestore();
  createDataView.mockRestore();
});

it('uses the full WGSL reserved grammar and allows the existing sample identifier', () => {
  expect(WGSL_RESERVED_WORDS.size).toBeGreaterThan(100);
  for (const identifier of ['target', 'active', 'filter', 'module', 'meta'])
    expect(WGSL_RESERVED_WORDS.has(identifier)).toBe(true);
  expect(WGSL_RESERVED_WORDS.has('sample')).toBe(false);
  for (const neighborSearch of ['metric', 'octant', 'staged'] as const) {
    const {graph, props} = createProps(8, [1]);
    const nodes = new GPUSpaceTimeKriging({
      ...props,
      sampleIndex: 'brute-force',
      neighborSearch
    }).getCommandNodes(graph);
    const prediction = getGPUCommandNodeKernelSources(nodes).find(({source}) =>
      source.includes('fn insertCandidate(')
    )?.source;
    expect(prediction).toBeDefined();
    expect(prediction).toContain('for (var sample = 0u;');
    expect(prediction).toContain('let observation = read_observations(sample);');
    if (neighborSearch === 'staged')
      expect(prediction).toContain('let sample = candidateRow[slot];');
    expect(prediction).not.toMatch(/\bsampleRow\b/);
  }
});

it('accepts logically adjacent output fields because no publish pass co-binds them', () => {
  const {graph, props, view} = createProps(8, [1]);
  const storage = view('uint32', 5);
  const values = graph.createDataView(storage.buffer, {
    format: 'float32x4',
    length: 1,
    byteOffset: 0
  });
  const status = graph.createDataView(storage.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 16
  });
  const nodes = new GPUSpaceTimeKriging({...props, output: {values, status}}).getCommandNodes(
    graph
  );
  for (const node of nodes.filter(node => node.id.includes('-publish-')))
    expect(
      (node.resources ?? []).filter(
        resource => resource.usage === 'storage-read-write' || resource.usage === 'storage-write'
      )
    ).toHaveLength(1);
});

it('rejects overlapping output aliases imported under different graph handles', () => {
  const {device, graph, props} = createProps(8, [1]);
  const physical = device.createBuffer({byteLength: 128, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer(
      {id: 'first-alias', byteLength: 128, usage: Buffer.STORAGE},
      physical
    );
    const second = graph.importBuffer(
      {id: 'second-alias', byteLength: 128, usage: Buffer.STORAGE},
      physical
    );
    const values = graph.createDataView(first, {format: 'float32x4', length: 1, byteOffset: 0});
    const status = graph.createDataView(second, {format: 'uint32', length: 1, byteOffset: 4});
    expect(() => new GPUSpaceTimeKriging({...props, output: {values, status}})).toThrow(
      'byte ranges must not overlap'
    );
  } finally {
    physical.destroy();
  }
});

it('does not bind a trailing empty shared-buffer chunk with no backing row', () => {
  const {graph, props, view} = createProps(8, [1, 0]);
  const source = view('float32x2', 1);
  const queries = createVectorView('trailing', 'float32x2', [
    source,
    graph.createDataView(source.buffer, {
      format: 'float32x2',
      length: 0,
      byteOffset: source.buffer.byteLength
    })
  ]);
  expect(() => new GPUSpaceTimeKriging({...props, queries}).getCommandNodes(graph)).not.toThrow();
});
