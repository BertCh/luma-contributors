// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture, type CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createGPURenderCommandNode,
  createTransientView
} from '@luma.gl/graph/gpu-core';
import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../src/testing/index';
import {
  createGPUComputationRecord,
  getGPUComputationRunId,
  validateGPUComputationRecord,
  GPUComputationRecorder,
  describeGPUCommandGraph,
  GPUSnapshotArena,
  GPUSnapshotTap,
  printGPUComputationGraph
} from '../../src/computation-record/index';

const identity = {id: 'review', graphs: [], domains: [], adapter: {backend: 'webgpu'}};

test('R3 both tap encodings retain the validated handle and offset after in-place view mutation', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const storage = createTransientView(
    graph,
    'storage',
    'uint32',
    4,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const view = graph.createDataView(storage.buffer, {format: 'uint32', length: 2});
  const arena = new GPUSnapshotArena(device, {
    slots: [{name: 'value', byteLength: 8, elementType: 'u32'}],
    capacity: 2
  });
  try {
    const node = new GPUSnapshotTap({
      id: 'review-tap',
      arena,
      views: {value: view}
    }).getCommandNodes(graph)[0];
    const executable = await node.compile({device} as never);
    const copyBufferToBuffer = vi.fn();
    const getBuffer = vi.fn((_handle: unknown) => ({byteLength: 16}));
    const context = {commandEncoder: {copyBufferToBuffer} as unknown as CommandEncoder, getBuffer};
    executable.encode(context as never);
    Object.assign(view, {byteOffset: 8, length: 1});
    executable.encode(context as never);
    expect(copyBufferToBuffer.mock.calls.map(([copy]) => [copy.sourceOffset, copy.size])).toEqual([
      [0, 8],
      [0, 8]
    ]);
    expect(getBuffer.mock.calls.every(([handle]) => handle === storage.buffer)).toBe(true);
  } finally {
    arena.destroy();
    device.destroy();
  }
});

test('R4 texture estimates use descriptors and recorded external textures are imported with unavailable size', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recorder = new GPUComputationRecorder();
  const texture = graph.createTransientTexture({
    id: 'image',
    format: 'rgba8unorm',
    width: 4,
    height: 4,
    usage: Texture.SAMPLE
  });
  const externalTexture = graph.importExternalTexture({id: 'external', width: 8, height: 8});
  recorder.addNode(
    graph,
    createGPURenderCommandNode({
      id: 'sample',
      resources: [
        {texture, usage: 'sampled'},
        {externalTexture, usage: 'sampled'}
      ],
      compile: () => ({encode: () => {}})
    })
  );
  const compiled = graph.compile();
  try {
    const recorded = describeGPUCommandGraph(compiled, {recorder}).graph.resources!;
    const plain = describeGPUCommandGraph(compiled).graph.resources!;
    expect(recorded.find(resource => resource.id === 'image')).toMatchObject({
      bytes: 64,
      bytesKind: 'estimated',
      ownership: 'transient'
    });
    expect(plain.find(resource => resource.id === 'image')).toMatchObject({
      bytes: 64,
      bytesKind: 'lower-bound',
      ownership: 'unknown'
    });
    expect(recorded.find(resource => resource.id === 'external')).toMatchObject({
      bytesKind: 'unavailable',
      ownership: 'imported'
    });
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

test.each(
  [NaN, Infinity, -Infinity, {}, [], null].map(value => [value])
)('R7 rejects non-scalar or non-finite identity value %j before hashing and on load', value => {
  const parameters = {x: value} as never;
  expect(() => getGPUComputationRunId({...identity, parameters})).toThrow('parameters.x');
  expect(() => createGPUComputationRecord({...identity, runId: 'explicit', parameters})).toThrow(
    'parameters.x'
  );
  expect(
    validateGPUComputationRecord({...createGPUComputationRecord(identity), parameters}).valid
  ).toBe(false);
});

test.each([
  NaN,
  Infinity,
  -Infinity
])('R7 numeric seed and choice values must be finite: %j', value => {
  expect(() => getGPUComputationRunId({...identity, seed: value})).toThrow('seed');
  expect(() =>
    getGPUComputationRunId({
      ...identity,
      choices: [{stage: 'fit', name: 'threshold', value, sweepable: true}]
    })
  ).toThrow('choices');
  expect(
    validateGPUComputationRecord({
      ...createGPUComputationRecord(identity),
      choices: [{stage: 'fit', name: 'threshold', value, sweepable: true}]
    }).valid
  ).toBe(false);
  expect(() =>
    getGPUComputationRunId({...identity, adapter: {backend: 'webgpu', description: value as never}})
  ).toThrow('adapter.description');
});

test('R8 records snapshot caller choices and adapter before the next execution edits them', () => {
  const adapter = {backend: 'webgpu', description: 'first'};
  const choices = [
    {stage: 'fit', name: 'threshold', value: 1, sweepable: true, alternatives: [1, 2]}
  ];
  const first = createGPUComputationRecord({...identity, adapter, choices});
  const originalId = first.runId;
  choices[0].value = 2;
  choices[0].alternatives[0] = 9;
  adapter.description = 'second';
  const second = createGPUComputationRecord({...identity, adapter, choices});
  expect(first.choices[0]).toMatchObject({value: 1, alternatives: [1, 2]});
  expect(first.adapter.description).toBe('first');
  expect(first.runId).toBe(originalId);
  expect(second.runId).not.toBe(originalId);
});

test('R10 legacy and explicitly logical imported resources print different certainty', () => {
  const text = printGPUComputationGraph({
    id: 'compiled',
    form: 'compiled',
    domain: 'test',
    nodes: [],
    edges: [],
    resources: [
      {id: 'legacy', kind: 'buffer', bytes: 8, ownership: 'imported'},
      {id: 'known', kind: 'buffer', bytes: 8, bytesKind: 'logical', ownership: 'imported'}
    ]
  });
  expect(text).toContain('legacy unknown 8B (precision unknown)');
  expect(text).toContain('known imported 8B');
});

// Literal fixtures below use the two disjoint graphs in the hand-built record.
import {computationRecordFixture, computationCapturesFixture} from './computation-record-fixture';
import {validateGPUComputationSidecars} from '../../src/computation-record/validate-computation-record';
import {createGPUComputationMetrics} from '../../src/computation-record/computation-metrics';
import {toChromeTraceEvents} from '../../src/computation-record/export-chrome-trace';

test('UDF-04 rejects null legacy lanes instead of restoring them as zero', () => {
  const captures = structuredClone(computationCapturesFixture);
  captures.snapshots[0].values = [null, 0] as never;
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
});

test('UDF-09 intersects local references and requires unambiguous inference and unique ids', () => {
  const captures = structuredClone(computationCapturesFixture);
  Object.assign(captures.snapshots[0], {graph: undefined, resource: 'first', afterNode: 'prepare'});
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
  Object.assign(captures.snapshots[0], {resource: undefined, afterNode: undefined});
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
  captures.snapshots[0].resource = 'first';
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(true);
  captures.replays = [{...captures.snapshots[0], provenance: 'gpu-replay'}];
  expect(validateGPUComputationSidecars(computationRecordFixture, {captures}).valid).toBe(false);
});

test('UDF-05 snapshots all plain metadata after mixed graph source merging', () => {
  const input = structuredClone(computationRecordFixture);
  const record = createGPUComputationRecord({
    ...input,
    graphs: [input.graphs[0], {graph: input.graphs[1], sources: input.refs.sources}],
    formulas: input.refs.formulas
  });
  // Literal expectations come from the fixture, before any mutation (not from the result).
  expect(record.graphs[1].nodes[1].attrs).toEqual({variant: 'portable'});
  expect(record.graphs[1].resources![0].bytes).toBe(16);
  expect(record.graphs[1].edges[0]).toEqual({
    from: 'init',
    to: 'step:0',
    kind: 'raw',
    resource: 'first'
  });
  expect(record.axes.axes[0].label).toBe('Lane');
  expect(record.refs.formulas[0].ascii).toBe('x');
  expect(record.decisions[0].chosen).toBe('portable');
  expect(record.workflow).toEqual({
    id: 'solve',
    invocations: [{graph: 'compiled', reason: 'Solve input'}]
  });
  const before = structuredClone(record);
  (input.graphs[1].nodes[0].attrs as Record<string, number>)['radius'] = 9;
  Object.assign(input.graphs[1].resources![0], {bytes: 999});
  Object.assign(input.graphs[1].edges[0], {kind: 'explicit'});
  Object.assign(input.axes.axes[0], {label: 'changed'});
  Object.assign(input.refs.formulas[0], {ascii: 'changed'});
  Object.assign(input.decisions[0], {chosen: 'changed'});
  Object.assign(input.workflow!.invocations[0], {reason: 'changed'});
  expect(record).toEqual(before);
});

test('UDF-12 invalid endpoints cannot poison the independent 3-to-4 ms node or measured zero', () => {
  const metrics = createGPUComputationMetrics({
    runId: computationRecordFixture.runId,
    graph: 'compiled',
    encoding: {coalesced: false, timestamped: true, taps: 0, lensTwin: false},
    timing: {
      nodes: [
        {
          id: 'init',
          cpuEncodeTimeMilliseconds: 0,
          gpuStartTimeMilliseconds: NaN,
          gpuEndTimeMilliseconds: 2,
          gpuTimeMilliseconds: 0
        },
        {
          id: 'step:0',
          cpuEncodeTimeMilliseconds: 0,
          gpuStartTimeMilliseconds: 3,
          gpuEndTimeMilliseconds: 4
        }
      ]
    }
  });
  expect(metrics.nodes['step:0']).toMatchObject({gpuBegin: 0, gpuEnd: 1});
  expect(metrics.nodes['init'].gpuTimeMs).toBe(0);
  expect(metrics.nodes['init'].gpuBegin).toBeUndefined();
  const trace = toChromeTraceEvents(computationRecordFixture, [metrics]);
  expect(trace.traceEvents.filter(event => event.ph === 'X')).toMatchObject([{ts: 0, dur: 1000}]);
  expect(trace.traceEvents.every(event => Number.isFinite(event.ts))).toBe(true);
  const invalid = {...metrics, nodes: {init: {cpuEncodeMs: 0, gpuBegin: NaN, gpuEnd: 2}}};
  expect(
    validateGPUComputationSidecars(computationRecordFixture, {metrics: [invalid]}).issues.some(
      issue => issue.path.endsWith('.gpuBegin')
    )
  ).toBe(true);
  expect(
    toChromeTraceEvents(computationRecordFixture, [invalid]).traceEvents.filter(
      event => event.ph === 'X'
    )
  ).toEqual([]);
});

test('UDF-13 saved identity reproduces run id including zero seed', () => {
  const record = createGPUComputationRecord({...identity, datasetVersion: 'table-v4', seed: 0});
  const saved = JSON.parse(JSON.stringify(record));
  expect(saved.datasetVersion).toBe('table-v4');
  expect(saved.seed).toBe(0);
  expect(getGPUComputationRunId(saved)).toBe(record.runId);
});

import * as captureValidation from '../../src/computation-record/validate-computation-record';
import * as rasterValidation from '../../src/computation-record/capture-raster-validation';

test('B5: legacy southward omitted origin keeps the top outer edge at height times cell size', () => {
  const geometry = {kind: 'raster' as const, shape: [1, 2] as const, cellSize: 3};
  // Existing raster support [0,3] x [0,6]; first sample is (1.5,4.5).
  expect(rasterValidation.getGPUComputationRasterSamplePosition(geometry, 0, 0)).toEqual([
    1.5, 4.5
  ]);
  expect(
    rasterValidation.getGPUComputationRasterSamplePosition({...geometry, origin: [0, 6]}, 0, 0)
  ).toEqual([1.5, 4.5]);
  expect(
    rasterValidation.getGPUComputationRasterSamplePosition(
      {...geometry, rowDirection: 'north'},
      0,
      0
    )
  ).toEqual([1.5, 1.5]);
});

test('B6: graph resolver intersects references without mutating the capture', () => {
  const record = structuredClone(computationRecordFixture);
  const capture = {
    ...computationCapturesFixture.snapshots[0],
    graph: undefined,
    resource: 'first',
    afterNode: 'step:0'
  };
  record.graphs[0].nodes = [...record.graphs[0].nodes, {...record.graphs[1].nodes[1]}];
  const before = structuredClone(capture);
  expect(captureValidation.resolveGPUComputationCaptureGraph(record, capture)?.id).toBe('compiled');
  expect(capture).toEqual(before);
  expect(
    captureValidation.resolveGPUComputationCaptureGraph(record, {...capture, resource: undefined})
  ).toBeUndefined();
  expect(
    captureValidation.resolveGPUComputationCaptureGraph(record, {...capture, graph: 'intent'})
  ).toBeUndefined();
});

test.each([
  [1, 16],
  [512, 16],
  [256, 0]
])('B7: binding window %s/%s must be aligned, positive and within logical resource', (bindingByteOffset, bindingByteLength) => {
  const record = structuredClone(computationRecordFixture);
  Object.assign(record.graphs[1].resources![0], {bytes: 512, bytesKind: 'logical'});
  Object.assign(record.graphs[1].nodes[1].bindings![0], {
    bindingByteOffset,
    bindingByteLength,
    elementType: 'u32',
    length: 4
  });
  expect(validateGPUComputationRecord(record).valid).toBe(false);
});

test('S4: rejected endpoint pairs cannot change duration or counter placement', () => {
  const metrics = createGPUComputationMetrics({
    runId: computationRecordFixture.runId,
    graph: 'compiled',
    encoding: {coalesced: false, timestamped: true, taps: 0, lensTwin: false},
    timing: {
      nodes: [
        {
          id: 'init',
          cpuEncodeTimeMilliseconds: 0,
          gpuStartTimeMilliseconds: -1e20,
          gpuEndTimeMilliseconds: NaN
        },
        {
          id: 'step:0',
          cpuEncodeTimeMilliseconds: 0,
          gpuStartTimeMilliseconds: 7,
          gpuEndTimeMilliseconds: 9
        }
      ]
    }
  });
  expect(metrics.nodes['step:0']).toMatchObject({gpuBegin: 0, gpuEnd: 2});
  const direct = {
    ...metrics,
    nodes: {
      'step:0': {cpuEncodeMs: 0, gpuBegin: -100, gpuEnd: NaN},
      'step:1': {cpuEncodeMs: 0, gpuBegin: 7, gpuEnd: 9}
    },
    series: [{name: 'residual', axis: 'iteration', values: [1, 2]}]
  };
  const record = structuredClone(computationRecordFixture);
  Object.assign(record.graphs[1].nodes[2].loop!, {iteration: 0});
  const trace = toChromeTraceEvents(record, [direct]);
  expect(trace.traceEvents.find(event => event.ph === 'C')?.ts).toBe(7000);
});

test.each([
  false,
  true
])('N1: source merging never invents an unknown entry point (reverse=%s)', reverse => {
  const source = {
    id: 'multi-entry',
    language: 'wgsl' as const,
    moduleId: 'multi-entry',
    hash: 'same',
    text: '@compute @workgroup_size(1) fn dry() {} @compute @workgroup_size(1) fn wet() {}'
  };
  const graph = {
    ...computationRecordFixture.graphs[0],
    nodes: [{...computationRecordFixture.graphs[0].nodes[0], source: source.id}],
    edges: []
  };
  const unknown = {graph: {...graph, id: 'unknown'}, sources: [source]};
  const known = {graph: {...graph, id: 'known'}, sources: [{...source, entryPoint: 'wet'}]};
  const record = createGPUComputationRecord({
    ...identity,
    graphs: reverse ? [known, unknown] : [unknown, known]
  });
  const effective = (id: string) => {
    const node = record.graphs.find(candidate => candidate.id === id)!.nodes[0];
    return (
      node.entryPoint ??
      record.refs.sources.find(candidate => candidate.id === node.source)?.entryPoint
    );
  };
  expect(effective('unknown')).toBeUndefined();
  expect(effective('known')).toBe('wet');
});
