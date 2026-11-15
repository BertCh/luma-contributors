// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  createSaturatedCountScanNodes,
  createSaturatedScanTotalNodes,
  SCAN_TOTAL_WGSL
} from '../../src/analysis-core/kernels/scan-total';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

it('scan arithmetic WGSL matches BigInt at uint32 and resize boundaries', () => {
  const values = [
    0, 1, 2, 765, 768, 771, 1536, 0x1fffffff, 0x20000000, 0x80000000, 0xfffffffe, 0xffffffff
  ];
  const multipliers = [0, 1, 2, 3, 7, 8, 31, 0x80000000, 0xffffffff];
  const cases = values.flatMap(offset =>
    values.flatMap(count => multipliers.map(multiplier => [offset, count, multiplier]))
  );
  const input = Uint32Array.from(cases.flat());
  const output = new Uint32Array(2 * cases.length);
  const source = `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
${SCAN_TOTAL_WGSL}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x;
  let sum = getSaturatedScanSum(input[3u * row], input[3u * row + 1u]);
  output[2u * row] = sum;
  output[2u * row + 1u] = getSaturatedScanProduct(sum, input[3u * row + 2u]);
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [cases.length, 1, 1], {
    0: {0: input, 1: output}
  });
  const saturate = (value: bigint) => Number(value > 0xffffffffn ? 0xffffffffn : value);
  expect(Array.from(output)).toEqual(
    cases.flatMap(([offset, count, multiplier]) => {
      const sum = BigInt(offset) + BigInt(count);
      return [saturate(sum), saturate(BigInt(saturate(sum)) * BigInt(multiplier))];
    })
  );
});

/** WGSL source and constants of every pipeline that `nodes` compile. */
function getPipelineKeys(nodes: readonly unknown[]): string[] {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const keys: string[] = [];
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {shader: {source: string}; constants?: object}) => {
      new WgslReflect(props.shader.source);
      keys.push(`${props.shader.source}\n${JSON.stringify(props.constants ?? null)}`);
      return {id: '', hash: '', destroy: () => {}};
    }
  });
  for (const node of nodes as {compile?: (context: {device: unknown}) => unknown}[]) {
    node.compile?.({device});
  }
  device.destroy();
  return keys;
}

function buildTotalNodes(rowCount: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `scan-total-${rowCount}`});
  return createSaturatedScanTotalNodes(graph, {
    id: 'pairs',
    operation: 'ScanTotalTest',
    counts: createTransientView(graph, 'counts', 'uint32', rowCount),
    offsets: createTransientView(graph, 'offsets', 'uint32', rowCount),
    rowCount,
    total: createTransientView(graph, 'state', 'uint32', 4),
    totalRow: 2
  });
}

it('saturated scan total: a total node then a carry node over every row', () => {
  const nodes = buildTotalNodes(1000);
  expect(nodes.map(node => node.id)).toEqual(['pairs-total', 'pairs-carry']);
  const sources = getPipelineKeys(nodes);
  expect(sources).toHaveLength(2);
  expect(sources[1]).toMatch(/var<storage, read_write> total: array<atomic<u32>>/);
  expect(sources[1]).toMatch(/offset \+ counts\[countsOffset \+ index\] < offset/);
});

it('saturated scan total shares its WGSL modules across row counts', () => {
  const sources = (rowCount: number) =>
    getPipelineKeys(buildTotalNodes(rowCount)).map(key => key.slice(0, key.lastIndexOf('\n')));
  expect(sources(1000)).toEqual(sources(1001));
});

it('GPUSegmentIntersection saturates its pair total before the write and finalize nodes', async () => {
  const {GEOMETRY_CASES} = await import('../analysis-core/kernel-audit/cases-geometry');
  const {createKernelAuditContext, getDeclaredNodes} = await import(
    '../analysis-core/kernel-audit/kernel-audit-context'
  );
  const auditCase = GEOMETRY_CASES.find(
    candidate =>
      candidate.exportName === 'GPUSegmentIntersection' &&
      candidate.name === 'lines x self, spatialSort false'
  );
  expect(auditCase).toBeDefined();
  const context = createKernelAuditContext('scan-total');
  auditCase?.build(context);
  const ids = (getDeclaredNodes(context) as {id: string}[]).map(node => node.id);
  context.device.destroy();
  const total = ids.indexOf('segment-intersection-total');
  const carry = ids.indexOf('segment-intersection-carry');
  expect(total).toBeGreaterThanOrEqual(0);
  expect(carry).toBe(total + 1);
  expect(ids.indexOf('segment-intersection-write')).toBeGreaterThan(carry);
});

it('saturated scan total rejects a rowCount past its views and a totalRow past the total view', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'scan-total-limits'});
  const props = {
    id: 'pairs',
    operation: 'ScanTotalTest',
    counts: createTransientView(graph, 'counts', 'uint32', 10),
    offsets: createTransientView(graph, 'offsets', 'uint32', 10),
    rowCount: 10,
    total: createTransientView(graph, 'state', 'uint32', 4),
    totalRow: 3
  };
  expect(() => createSaturatedScanTotalNodes(graph, {...props, rowCount: 11})).toThrow(
    /^pairs rowCount must be at most 10/
  );
  expect(() => createSaturatedScanTotalNodes(graph, {...props, totalRow: 4})).toThrow(
    /^pairs totalRow must be at most 3/
  );
});

for (const rowCount of [0, 1, 5000]) {
  it(`saturated count scan shifts a uint64 prefix and rewrites its total (${rowCount} rows)`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device, {id: 'saturated-count-scan'});
    try {
      const nodes = createSaturatedCountScanNodes(graph, {
        id: 'counts',
        operation: 'ScanTotalTest',
        counts: createTransientView(graph, 'input', 'uint32', rowCount),
        offsets: createTransientView(graph, 'offsets', 'uint32', rowCount + 1),
        rowCount,
        total: createTransientView(graph, 'total', 'uint32', 3),
        totalRow: 1
      });
      const sources = getPipelineKeys(nodes);
      expect(nodes.at(-1)?.id).toBe('counts-saturate');
      const source = sources.at(-1)!;
      expect(source).toContain('high[highOffset + index - 1u] != 0u');
      expect(source).toContain('high[highOffset + ROW_COUNT - 1u] != 0u');
      expect(source).toContain('total[totalOffset + TOTAL_ROW] = sum;');
      expect(source).toContain('"TOTAL_ROW":1');
    } finally {
      device.destroy();
    }
  });
}
