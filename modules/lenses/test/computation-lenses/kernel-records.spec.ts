// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {commands} from 'vitest/browser';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Buffer, type Device} from '@luma.gl/core';
import {
  createGPUComputationRecord,
  describeGPUCommandGraph,
  submitGPUCommandGraphWithTimings,
  validateGPUComputationRecord,
  type GPUComputationMetrics,
  type GPUComputationRecord
} from '@luma-contributors/analysis/computation-record';
import {createFillNode} from '@luma-contributors/analysis/analysis-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  GPUPointCloudFeatures,
  GPUPointCloudNeighborhoods,
  GPUPointCloudStatisticalFilter
} from '@luma-contributors/analysis/pointcloud-analysis';
import {getLensScenario, toRecordedLensFile} from '../../src/computation-lenses/index';
import {runLensScenario} from '../../src/computation-lenses/run-lens-scenario';
import {
  LENS_RECORD_SOURCE_TEXT,
  LensCommandGraph,
  readLensRunMetrics,
  submitTimedLensGraph
} from '../../src/computation-lenses/lens-view-snapshot';
// Register the tested families without importing unrelated scenario examples into this suite.
import '../../src/computation-views/packs/gpu-core/index';
import '../../src/gpu-spatial-analysis/computation-pack/index';
import '../../src/gpu-network/computation-pack/index';
import '../../src/gpu-raster/computation-pack/index';
import '../../src/gpu-terrain/computation-pack/index';
import '../../src/gpu-dataframe/computation-pack/index';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/local-moran.lens';
import '../../src/computation-lenses/scenarios/gpu-network/register-structure';
import '../../src/computation-lenses/scenarios/gpu-raster/expansion.scenarios';
import '../../src/computation-lenses/scenarios/gpu-terrain/geomorphons.lens';
import '../../src/computation-lenses/scenarios/gpu-dataframe/class-breaks.lens';

/** One real lens scenario per entry point family; point clouds have no lens and are built below. */
const FAMILY_SCENARIOS = {
  spatial: 'gpu-spatial-analysis.local-moran',
  network: 'gpu-network.network-centrality',
  raster: 'gpu-raster.raster-rank-filter',
  terrain: 'gpu-terrain.geomorphons',
  dataframe: 'gpu-dataframe.class-breaks'
} as const;

/** Families whose lens harness writes a metrics sidecar from a timed submission. */
const TIMED_FAMILIES = new Set(['spatial', 'network', 'raster', 'terrain', 'point cloud']);

/**
 * Compute kernels that still carry no kernel facts, matched on `<op>:<nodeId>` (`unnamed` when the
 * node publishes no operation), and why. They are nodes of published `@luma.gl/graph` primitives,
 * not of `createWGSLKernelNode`, and luma.gl publishes no per-node shader description for them
 * (the hook a record needs: source, entry point, workgroup size and bindings on
 * `compiled.preflight.nodes[]`, or a public `describe()` on the command node). Every other compute
 * node must carry source text, bindings, workgroup size and dispatch. Copy nodes (the snapshot tap)
 * have no kernel and are not counted.
 */
const KERNELS_WITHOUT_FACTS: readonly {pattern: RegExp; reason: string}[] = [
  {pattern: /^GPUScan:/, reason: 'published GPUScan (also inside published GPUSort)'},
  {
    pattern: /^(GPUGather|GPUUint32Gather):/,
    reason: 'published gathers (permutation gathers between sort passes, segment-sum inputs)'
  },
  {
    pattern: /^unnamed:.*-radix-digit-\d+-(histogram|scatter)$/,
    reason: 'published GPUSort radix passes; they publish no operation name either'
  },
  {
    pattern: /^unnamed:.*-(level-\d+|finalize)$/,
    reason: 'published GPUReduction stages (point cloud statistical filter)'
  },
  {
    pattern: /^unnamed:.*-elevation$/,
    reason:
      'published GPURasterNeighborhood / GPURasterTextureToBuffer band canonicalization (raster-band-utils)'
  }
];

type KernelCoverage = {
  family: string;
  scenario: string;
  computeNodes: number;
  withFacts: number;
  sources: number;
  sourceTextBytes: number;
  /** `<op>:<nodeId>` of every compute node without kernel facts. */
  missing: string[];
  timedNodes: number;
  /** Size of the committed record file, before this wave's kernel facts and timings. */
  bytesBefore?: number;
  /** Size the same record file would have now (same compact encoding). */
  bytesAfter?: number;
};

function measureKernelCoverage(
  family: string,
  scenario: string,
  record: GPUComputationRecord
): KernelCoverage {
  const graph = record.graphs[0];
  const sources = new Map(record.refs.sources.map(source => [source.id, source]));
  const compute = graph.nodes.filter(node => node.kind === 'compute');
  const missing: string[] = [];
  let withFacts = 0;
  for (const node of compute) {
    const source = node.source ? sources.get(node.source) : undefined;
    if (!source) {
      missing.push(`${node.op ?? node.workload?.operation ?? 'unnamed'}:${node.id}`);
      continue;
    }
    withFacts++;
    expect(source.language, node.id).toBe('wgsl');
    expect(source.entryPoint, node.id).toBe('main');
    expect(typeof source.text, node.id).toBe(LENS_RECORD_SOURCE_TEXT ? 'string' : 'undefined');
    expect(node.bindings?.length, node.id).toBeGreaterThan(0);
    expect(node.workgroupSize?.[0], node.id).toBeGreaterThan(0);
    expect(node.dispatch?.workgroups, node.id).toHaveLength(3);
    for (const binding of node.bindings ?? []) {
      expect(graph.resources?.some(resource => resource.id === binding.resource)).toBe(true);
    }
  }
  return {
    family,
    scenario,
    computeNodes: compute.length,
    withFacts,
    sources: record.refs.sources.length,
    sourceTextBytes: record.refs.sources.reduce(
      (total, source) => total + (source.text?.length ?? 0),
      0
    ),
    missing,
    timedNodes: 0
  };
}

function expectTimings(device: Device, metrics: readonly GPUComputationMetrics[] | undefined) {
  expect(metrics?.length).toBeGreaterThan(0);
  const nodes = Object.values(metrics![0].nodes);
  const timed = nodes.filter(node => node.gpuBegin !== undefined && node.gpuEnd !== undefined);
  if (device.features.has('timestamp-query')) {
    expect(metrics![0].encoding.timestamped).toBe(true);
    expect(timed.length).toBeGreaterThan(0);
    for (const node of timed) expect(node.gpuEnd!).toBeGreaterThanOrEqual(node.gpuBegin!);
  } else {
    // Absent, not zero: no node carries a GPU begin or end.
    expect(metrics![0].encoding.timestamped).toBe(false);
    expect(nodes.every(node => node.gpuBegin === undefined && node.gpuEnd === undefined)).toBe(
      true
    );
  }
  return timed.length;
}

function checkMissing(coverage: KernelCoverage) {
  const unexplained = coverage.missing.filter(
    entry => !KERNELS_WITHOUT_FACTS.some(({pattern}) => pattern.test(entry))
  );
  expect(unexplained, `${coverage.scenario}: kernels without facts and no listed reason`).toEqual(
    []
  );
}

const coverages: KernelCoverage[] = [];

for (const [family, scenarioId] of Object.entries(FAMILY_SCENARIOS)) {
  test(`${family} lens record carries kernel facts (${scenarioId})`, async () => {
    const device = await getWebGPUTestDevice();
    expect(device, 'Kernel records require WebGPU').toBeTruthy();
    requireTestDevice(device);
    const scenario = getLensScenario(scenarioId);
    expect(scenario).toBeTruthy();
    const result = await runLensScenario(scenario!, device);
    expect(validateGPUComputationRecord(result).issues.filter(i => i.level === 'error')).toEqual(
      []
    );
    const coverage = measureKernelCoverage(family, scenarioId, result);
    expect(coverage.withFacts).toBeGreaterThan(0);
    if (TIMED_FAMILIES.has(family)) coverage.timedNodes = expectTimings(device, result.metrics);
    const recorded = toRecordedLensFile(result, {
      scenarioId,
      recordedAt: '2026-11-17T00:00:00.000Z',
      adapterLabel: 'measure'
    });
    const before = await commands.readFile(`docs/lens-records/${scenarioId}.json`).catch(() => '');
    coverages.push({
      ...coverage,
      bytesBefore: before.length,
      bytesAfter: JSON.stringify(recorded).length + 1
    });
    checkMissing(coverage);
  });
}

test('point cloud graph record carries kernel facts and timings', async () => {
  const device = await getWebGPUTestDevice();
  expect(device, 'Kernel records require WebGPU').toBeTruthy();
  requireTestDevice(device);
  const graph = new LensCommandGraph(device, {id: 'pointcloud-kernel-record'});
  const owned: Buffer[] = [];
  const upload = <Format extends 'float32' | 'float32x3' | 'uint32'>(
    id: string,
    format: Format,
    data: Float32Array | Uint32Array,
    components = 1
  ) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    owned.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length: data.length / components});
  };
  const positions = upload(
    'positions',
    'float32x3',
    new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0.1, 50, 50, 50]),
    3
  );
  const indices = upload('indices', 'uint32', new Uint32Array(5 * 4));
  const distances = upload('distances', 'float32', new Float32Array(5 * 4));
  const counts = upload('counts', 'uint32', new Uint32Array(5));
  graph.add(
    new GPUPointCloudNeighborhoods({
      positions,
      radius: 3,
      capacity: 4,
      output: {indices, squaredDistances: distances, counts}
    })
  );
  graph.add(
    new GPUPointCloudFeatures({
      positions,
      indices,
      counts,
      capacity: 4,
      output: {
        normals: upload('normals', 'float32x3', new Float32Array(15), 3),
        eigenvalues: upload('eigenvalues', 'float32x3', new Float32Array(15), 3),
        shapes: upload('shapes', 'float32x3', new Float32Array(15), 3),
        status: upload('status', 'uint32', new Uint32Array(5))
      }
    })
  );
  graph.add(
    new GPUPointCloudStatisticalFilter({
      squaredDistances: distances,
      counts,
      capacity: 4,
      standardDeviations: 1,
      output: {
        meanDistances: upload('means', 'float32', new Float32Array(5)),
        validity: upload('mask', 'uint32', new Uint32Array(5))
      }
    })
  );
  const compiled = await graph.compileAsync();
  try {
    const description = describeGPUCommandGraph(compiled, {
      graph,
      domain: 'gpu-pointcloud',
      sourceText: LENS_RECORD_SOURCE_TEXT
    });
    const record = createGPUComputationRecord({
      id: 'pointcloud-kernel-record',
      adapter: {backend: device.type},
      domains: ['gpu-pointcloud'],
      graphs: [description]
    });
    expect(validateGPUComputationRecord(record).issues.filter(i => i.level === 'error')).toEqual(
      []
    );
    const submission = submitTimedLensGraph(device, compiled, {parameters: undefined});
    const metrics = await readLensRunMetrics(submission, {
      runId: record.runId,
      executionId: record.executionId,
      graphId: description.graph.id,
      taps: 0
    });
    const coverage = measureKernelCoverage('point cloud', 'pointcloud-kernel-record', record);
    expect(coverage.withFacts).toBeGreaterThan(0);
    coverage.timedNodes = expectTimings(device, [metrics]);
    coverages.push(coverage);
    checkMissing(coverage);
  } finally {
    compiled.destroy();
    for (const buffer of owned) buffer.destroy();
  }
});

test('a run without timestamps records no timing, not zero', async () => {
  const device = await getWebGPUTestDevice();
  expect(device, 'Kernel records require WebGPU').toBeTruthy();
  requireTestDevice(device);
  const graph = new LensCommandGraph(device, {id: 'untimed-kernel-record'});
  const view = createTransientView(graph, 'filled', 'uint32', 64);
  graph.add(createFillNode(graph, {id: 'fill', operation: 'Fill', view, type: 'u32', value: '1u'}));
  const compiled = await graph.compileAsync();
  try {
    const submission = submitGPUCommandGraphWithTimings(
      device,
      compiled,
      {parameters: undefined},
      {timestamps: false}
    );
    expect(submission.timestamped).toBe(false);
    const metrics = await readLensRunMetrics(submission, {runId: 'run', graphId: 'g', taps: 0});
    expect(metrics.encoding.timestamped).toBe(false);
    expect(metrics.nodes).toEqual({});
    // The kernel facts do not depend on timing.
    const {graph: described} = describeGPUCommandGraph(compiled, {graph});
    expect(described.nodes[0].bindings?.[0]).toMatchObject({
      name: 'fillTarget',
      access: 'read_write'
    });
  } finally {
    compiled.destroy();
  }
});

test('kernel coverage summary', async () => {
  // Written for the LN-REC report when KERNEL_RECORD_SUMMARY names a file: what a record holds now.
  const summaryPath = (import.meta as ImportMeta & {env?: Record<string, string | undefined>})
    .env?.['VITE_KERNEL_RECORD_SUMMARY'];
  if (summaryPath) await commands.writeFile(summaryPath, JSON.stringify(coverages, null, 1));
  expect(coverages.length).toBeGreaterThan(0);
});
