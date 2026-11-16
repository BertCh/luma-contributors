// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  findStatisticsNonFiniteConstants,
  findStatisticsBarrierRisks
} from '../spatial-weights/statistics-wgsl-audit';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  WgslReflect,
  WgslParser,
  Function as WGSLFunction,
  Var,
  Let,
  If,
  Return,
  Call,
  VariableExpr,
  Node as WGSLNode
} from 'wgsl_reflect';
import {GPUGlobalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {GPUSpatialPearson} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-additional-spatial-statistics';
import {GPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-statistics';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createScene(sizes: readonly number[]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `global-tile-${serial++}`, format, length);
  const rows = sizes.reduce((sum, size) => sum + size, 0);
  let rowStart = 0;
  const chunks = sizes.map(rowCount => {
    const result = {
      rowStart,
      rowCount,
      weights: {
        offsets: view('uint32', rowCount + 1),
        neighbors: view('uint32', rowCount),
        weights: view('float32', rowCount)
      }
    };
    rowStart += rowCount;
    return result;
  });
  const props = {
    weights: {rowCount: rows, chunks},
    values: view('float32', rows),
    secondValues: view('float32', rows),
    statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'] as const,
    output: {summary: view('float32', 52), counts: view('uint32', 3), joinCounts: view('uint32', 3)}
  };
  return {device, graph, view, props};
}

it('30M-row global analytic statistics use separate bindings below the baseline device limit', () => {
  const {device, graph, props} = createScene([30_000_000]);
  try {
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const nodes = new GPUGlobalSpatialStatistics({
      ...props,
      symmetryStrategy: 'pairwise'
    }).getCommandNodes(graph);
    for (const [descriptor] of allocate.mock.calls)
      expect(descriptor.byteLength).toBeLessThanOrEqual(134217728);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
      expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
      if (/workgroupBarrier|storageBarrier/.test(source))
        expect(findStatisticsBarrierRisks(source)).toEqual([]);
      expect(() => new WgslReflect(source)).not.toThrow();
    }
    expect(() => new GPUGlobalSpatialStatistics({...props, mode: 'reference'})).toThrow(
      /REFERENCE_MAXIMUM_ROWS/
    );
  } finally {
    device.destroy();
  }
});

it('global analysis-owned shader and pipeline cardinality stays constant across unequal CSR chunks', () => {
  const record = (sizes: readonly number[]) => {
    const {device, graph, props} = createScene(sizes);
    try {
      const nodes = new GPUGlobalSpatialStatistics({
        ...props,
        symmetryStrategy: 'pairwise'
      }).getCommandNodes(graph);
      const analysisNodes = nodes.filter(node => !node.id.includes('-sort-'));
      const sources = getGPUCommandNodeKernelSources(analysisNodes);
      for (const {source} of sources) {
        expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
        expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
        if (/workgroupBarrier|storageBarrier/.test(source))
          expect(findStatisticsBarrierRisks(source)).toEqual([]);
        expect(() => new WgslReflect(source)).not.toThrow();
      }
      return {
        records: sources,
        sources: new Set(sources.map(item => item.source)),
        pipelines: getGPUCommandNodePipelineKeys(analysisNodes),
        islandRows: analysisNodes
          .filter(node => /-islands-\d+$/.test(node.id))
          .map(node => node.workload?.maximumInvocationCount),
        islandWorkgroups: analysisNodes
          .filter(node => /-islands-\d+$/.test(node.id))
          .map(node => node.workload?.maximumWorkgroupCount)
      };
    } finally {
      device.destroy();
    }
  };
  const packed = record([769]);
  const chunked = record([1, 255, 256, 257]);
  expect(chunked.sources.size).toBe(packed.sources.size);
  expect(chunked.pipelines.size).toBe(packed.pipelines.size);
  expect(packed.islandRows).toEqual([769]);
  expect(chunked.islandRows).toEqual([1, 255, 256, 257]);
  expect(chunked.islandWorkgroups).toEqual([1, 1, 1, 2]);
});

it('global preflight rejects caller usage before declaring resources, including a later CSR chunk', () => {
  const {device, graph, props} = createScene([1, 4]);
  const buffer = device.createBuffer({byteLength: 16, usage: Buffer.COPY_DST});
  try {
    props.weights.chunks[1].weights.neighbors = importGraphBuffer(
      graph,
      'missing-storage',
      buffer,
      'uint32',
      4
    );
    const contributor = new GPUGlobalSpatialStatistics(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const views = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
    expect(views).not.toHaveBeenCalled();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('global outputs reject co-bound aligned aliases and allow separate-dispatch packed counts', () => {
  const {device, graph, props} = createScene([5]);
  try {
    const shared = createTransientView(graph, 'global-shared', 'uint32', 128);
    props.output.summary = graph.createDataView(shared.buffer, {format: 'float32', length: 52});
    props.output.joinCounts = graph.createDataView(shared.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 208
    });
    expect(() => new GPUGlobalSpatialStatistics(props)).toThrow(/overlap/);
    props.output.summary = graph.createDataView(shared.buffer, {
      format: 'float32',
      length: 52,
      byteOffset: 12
    });
    props.output.joinCounts = graph.createDataView(shared.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 256
    });
    props.output.counts = graph.createDataView(shared.buffer, {format: 'uint32', length: 3});
    expect(() => new GPUGlobalSpatialStatistics(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

for (const mode of ['tiled', 'reference'] as const)
  it(`${mode} global kernels declare every function and struct exactly once`, () => {
    const {device, graph, props} = createScene([4]);
    try {
      const sources = getGPUCommandNodeKernelSources(
        new GPUGlobalSpatialStatistics({...props, mode}).getCommandNodes(graph)
      );
      for (const {id, source} of sources) {
        const names = [...stripWGSLComments(source).matchAll(/\b(?:fn|struct|alias)\s+(\w+)/g)].map(
          match => match[1]
        );
        expect(
          names.filter((name, index) => names.indexOf(name) !== index),
          id
        ).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  });

it('statistics preflight rejects duplicate physical imports before graph mutation and accepts one import with offset views', () => {
  const {device, graph, props} = createScene([4]);
  const buffer = device.createBuffer({byteLength: 32, usage: Buffer.STORAGE});
  try {
    props.values = importGraphBuffer(graph, 'physical-values', buffer, 'float32', 4);
    props.weights.chunks[0].weights.neighbors = importGraphBuffer(
      graph,
      'physical-neighbors',
      buffer,
      'uint32',
      4
    );
    const contributor = new GPUGlobalSpatialStatistics(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const declare = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/import it once/);
    expect(allocate).not.toHaveBeenCalled();
    expect(declare).not.toHaveBeenCalled();
    props.weights.chunks[0].weights.neighbors = graph.createDataView(props.values.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 16
    });
    expect(() => new GPUGlobalSpatialStatistics(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

/** Include statement containers: wgsl_reflect's Node.search visits their expressions only. */
function visitStatisticsWGSLNodes(root: WGSLNode, callback: (node: WGSLNode) => void): void {
  const visited = new Set<WGSLNode>();
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
    } else if (value instanceof WGSLNode && !visited.has(value)) {
      visited.add(value);
      callback(value);
      for (const child of Object.values(value)) visit(child);
    }
  };
  visit(root);
}

/** A focused guard audit, not a replacement for WebGPU's full uniformity analysis. Module-private
 * loads and invocation builtins cannot control an early return that precedes a barrier, even when
 * a private was just assigned a uniform value. Follow immutable local aliases of those values. */
function findStatisticsBarrierGuardRisks(source: string): string[] {
  const statements = new WgslParser().parse(source);
  const entry = statements.find(
    (statement): statement is WGSLFunction =>
      statement instanceof WGSLFunction && statement.name === 'main'
  );
  if (!entry) throw new Error('Statistics source has no main entry point');
  const nonuniformNames = new Set(
    statements
      .filter(
        (statement): statement is Var => statement instanceof Var && statement.storage !== 'uniform'
      )
      .map(statement => statement.name)
  );
  for (const argument of entry.args) {
    const builtin = argument.attributes?.find(attribute => attribute.name === 'builtin')?.value;
    if (builtin !== 'workgroup_id' && builtin !== 'num_workgroups')
      nonuniformNames.add(argument.name);
  }
  const aliases = new Map<string, WGSLNode>();
  let lastBarrierLine = 0;
  visitStatisticsWGSLNodes(entry, statement => {
    if (statement instanceof Let && statement.value) aliases.set(statement.name, statement.value);
    if (statement instanceof Var) nonuniformNames.add(statement.name);
    if (
      statement instanceof Call &&
      ['workgroupBarrier', 'storageBarrier'].includes(statement.name)
    )
      lastBarrierLine = Math.max(lastBarrierLine, statement.line);
  });
  const dependencies = (expression: WGSLNode, visited = new Set<string>()): Set<string> => {
    const names = new Set<string>();
    visitStatisticsWGSLNodes(expression, node => {
      if (!(node instanceof VariableExpr)) return;
      if (nonuniformNames.has(node.name)) names.add(node.name);
      else if (aliases.has(node.name) && !visited.has(node.name)) {
        const next = new Set(visited);
        next.add(node.name);
        for (const name of dependencies(aliases.get(node.name)!, next)) names.add(name);
      }
    });
    return names;
  };
  const findings: string[] = [];
  visitStatisticsWGSLNodes(entry, statement => {
    if (!(statement instanceof If) || statement.line > lastBarrierLine) return;
    let returns = false;
    for (const branch of statement.body)
      visitStatisticsWGSLNodes(branch, node => {
        if (node instanceof Return) returns = true;
      });
    if (returns)
      for (const name of dependencies(statement.condition))
        findings.push(`${statement.line}: nonuniform return depends on ${name}`);
  });
  return findings;
}

it('barrier guard audit catches private loads and invocation aliases but permits immutable uniform bounds', () => {
  const module = `struct Parameters { count: u32 }
@group(0) @binding(0) var<uniform> parameters: Parameters;`;
  const entry = (setup: string, predicate: string) => `
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_index) localInvocationIndex: u32) {
${setup}
if (${predicate}) { return; }
workgroupBarrier();
}`;
  const privateSource = `${module}\nvar<private> invocationCount: u32;${entry('invocationCount = parameters.count;', 'workgroupId.x >= invocationCount')}`;
  expect(findStatisticsBarrierGuardRisks(privateSource)).toEqual([
    expect.stringContaining('invocationCount')
  ]);
  const invocationSource = `${module}${entry('let row = localInvocationIndex;', 'row >= parameters.count')}`;
  expect(findStatisticsBarrierGuardRisks(invocationSource)).toEqual([
    expect.stringContaining('localInvocationIndex')
  ]);
  const uniformSource = `${module}${entry('let invocationCount = parameters.count;', 'workgroupId.x >= invocationCount')}`;
  expect(findStatisticsBarrierGuardRisks(uniformSource)).toEqual([]);
  for (const source of [privateSource, invocationSource, uniformSource]) {
    expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
    expect(findStatisticsNonFiniteConstants(source)).toEqual([]);
  }
});

for (const family of [
  'global',
  'inline-permutation',
  'materialized-permutation',
  'pearson',
  'region'
] as const)
  it(`${family} cooperative shaders preserve uniform return guards and every reflected binding`, () => {
    const {device, graph, props, view} = createScene([1, 255, 256, 257]);
    try {
      const contributor =
        family === 'global'
          ? new GPUGlobalSpatialStatistics({...props, symmetryStrategy: 'pairwise'})
          : family === 'pearson'
            ? new GPUSpatialPearson({
                weights: props.weights,
                values: props.values,
                secondValues: props.secondValues,
                output: {statistic: view('float32', 1)}
              })
            : family === 'region'
              ? new GPURegionStatistics({
                  region: {kind: 'mask', mask: view('uint32', props.values.length)},
                  values: props.values,
                  output: {summary: view('uint32', 8)}
                })
              : new GPUGlobalPermutationTest({
                  weights: props.weights,
                  values: props.values,
                  statistic: 'moran',
                  parameters: view('uint32', 4),
                  maximumPermutations: 3,
                  ...(family === 'materialized-permutation' ? {permutationBatchSize: 2} : {}),
                  output: {summary: view('float32', 12)}
                });
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).filter(
        record => record.source.includes('struct StatisticsDispatchParameters')
      );
      expect(sources.length).toBeGreaterThan(0);
      for (const {id, source, shaderLayout} of sources) {
        expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
        let risks: string[];
        try {
          risks = findStatisticsBarrierGuardRisks(source);
        } catch (error) {
          const line = Number(
            String((error as {message?: unknown}).message ?? error).match(/Line:(\d+)/)?.[1] ?? 0
          );
          throw new Error(
            `${id}: ${String((error as {message?: unknown}).message ?? error)}\n${source
              .split('\n')
              .slice(Math.max(0, line - 3), line + 2)
              .join('\n')}`
          );
        }
        expect(risks, id).toEqual([]);
        // WebGPU infers layouts from resources statically accessed by the entry point, not every
        // module declaration. A uniform accessed only by unreachable helpers disappears.
        const resources = new WgslReflect(source).entry.compute[0].resources
          .map(resource => `${resource.group}:${resource.binding}:${resource.name}`)
          .sort();
        const bindings = (
          shaderLayout as {bindings: {name: string; location: number; group?: number}[]}
        ).bindings
          .map(binding => `${binding.group ?? 0}:${binding.location}:${binding.name}`)
          .sort();
        expect(resources, id).toEqual(bindings);
      }
    } finally {
      device.destroy();
    }
  });
