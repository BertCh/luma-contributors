// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync} from 'node:fs';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {createStatisticsKernelNode} from '../../../src/gpu-spatial-analysis/spatial-weights/statistics-kernel-node';
import {createStatisticsPaddingProbe} from '../global-spatial-statistics/statistics-padding-probe';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {QUANTILE_LISA_AUDIT_CASES} from '../../analysis-core/kernel-audit/cases-quantile-lisa';
import {BACKLOG_CASES} from '../../analysis-core/kernel-audit/cases-backlog';
import {STATISTICS_CASES} from '../../analysis-core/kernel-audit/cases-statistics';
import {POINTS_AND_REGIONS_CASES} from '../../analysis-core/kernel-audit/cases-points-and-regions';
import {COVERAGE_STATISTICS_CASES} from '../../analysis-core/kernel-audit/cases-coverage-statistics';
import {EXPANSION_CASES} from '../../analysis-core/kernel-audit/cases-expansion';
import {STX06_CHANGE_POINTS_STX07_SCAN_AUDIT_CASES} from '../../analysis-core/kernel-audit/cases-stx06-change-points-stx07-scan';
import {STX08_SPATIAL_METHODS_AUDIT_CASES} from '../../analysis-core/kernel-audit/cases-stx08-spatial-methods';
import {STX08_SECOND_GROUP_AUDIT_CASES} from '../../analysis-core/kernel-audit/cases-stx08-colocation-licd-adaptive-density';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  recordShaderModules
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  findStatisticsBarrierRisks,
  findStatisticsNonFiniteConstants
} from './statistics-wgsl-audit';

const families = [
  'spatial-autocorrelation',
  'permutation-inference',
  'emerging-hot-spots',
  'global-spatial-statistics',
  'multiple-testing',
  'rate-smoothing',
  'scan-statistics',
  'spatial-entropy',
  'segregation',
  'categorical-association',
  'geographical-detector',
  'region-statistics',
  'neighborhood-summary',
  'outlier-detection',
  'change-points',
  'similar-locations',
  'spatial-weights'
];
const root = 'modules/analysis/src/gpu-spatial-analysis';
const ownedNames = new Set<string>();
const classFamilies = new Map<string, string>();
for (const family of families) {
  for (const name of readdirSync(join(root, family), {recursive: true})) {
    if (
      typeof name !== 'string' ||
      !name.endsWith('.ts') ||
      name === 'gpu-analysis-contiguity-statistics.ts'
    )
      continue;
    const source = readFileSync(join(root, family, name), 'utf8');
    for (const match of source.matchAll(/export class (GPU\w+)/g)) {
      ownedNames.add(match[1]);
      classFamilies.set(match[1], family);
    }
  }
}
const cases = [
  ...QUANTILE_LISA_AUDIT_CASES,
  ...BACKLOG_CASES,
  ...STATISTICS_CASES,
  ...POINTS_AND_REGIONS_CASES,
  ...COVERAGE_STATISTICS_CASES,
  ...EXPANSION_CASES,
  ...STX06_CHANGE_POINTS_STX07_SCAN_AUDIT_CASES,
  ...STX08_SPATIAL_METHODS_AUDIT_CASES,
  ...STX08_SECOND_GROUP_AUDIT_CASES
];
// Each contributor is built with an existing maximal-options case. Other route-specific node
// specs supplement these configurations; this is restricted to the F-STATS ownership list.
const selected = cases.filter(
  (candidate, index) =>
    ownedNames.has(candidate.exportName) &&
    cases.findIndex(previous => previous.exportName === candidate.exportName) === index
);

it('nonfinite audit rejects literal and constant-expression NaN/Infinity but permits runtime NaN', () => {
  for (const expression of [
    'bitcast<f32>(0x7fc00000u)',
    'bitcast<f32>(0x7f800000u)',
    'NaN',
    'Infinity',
    '1e999',
    '0.0 / 0.0'
  ])
    expect(
      findStatisticsNonFiniteConstants(`fn missing() -> f32 { return ${expression}; }`),
      expression
    ).not.toEqual([]);
  expect(
    findStatisticsNonFiniteConstants(
      `${GET_NAN_WGSL} fn missing() -> f32 { let nan = getNaN(); return nan; }`
    )
  ).toEqual([]);
});

it('barrier audit follows helper calls and rejects varying branches, loop bounds and exits', () => {
  const entry = (body: string) => `fn collective() { workgroupBarrier(); }
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane: u32) { ${body} }`;
  for (const body of [
    'if (lane < 4u) { collective(); }',
    'for (var row = 0u; row < lane; row++) { workgroupBarrier(); }',
    'var bound = 2u; if (lane == 0u) { bound = 4u; } for (var row = 0u; row < bound; row++) { collective(); }',
    'let row = lane; if (row > 0u) { return; } collective();',
    'for (var row = 0u; row < 4u; row++) { if (lane == row) { break; } collective(); }'
  ])
    expect(findStatisticsBarrierRisks(entry(body)), body).not.toEqual([]);
  expect(
    findStatisticsBarrierRisks(
      entry('for (var row = 0u; row < 4u; row++) { if (lane == row) { _ = lane; } collective(); }')
    )
  ).toEqual([]);
});

it('statistics shader inventory covers all seventeen owned families', () => {
  expect(selected.length).toBeGreaterThan(40);
  for (const family of families)
    expect(
      selected.some(candidate => classFamilies.get(candidate.exportName) === family),
      family
    ).toBe(true);
});

for (const candidate of selected)
  it(`${candidate.exportName} generated WGSL has finite constants, legal identifiers and uniform barrier guards`, () => {
    const context = createKernelAuditContext(`f-stats-${candidate.exportName}`);
    try {
      candidate.build(context);
      const nodes = getDeclaredNodes(context);
      expect(nodes.length).toBeGreaterThan(0);
      const seen = new Set<string>();
      const findings: string[] = [];
      recordShaderModules(nodes, module => {
        if (seen.has(module.source)) return;
        seen.add(module.source);
        // wgsl_reflect mistakes legal contextual local names for language keywords.
        const reflection = new WgslReflect(
          module.source
            .replace(/\bblock\b/g, 'block_')
            .replace(/(?<!<\s*)\bworkgroup\b(?!\s*>)/g, 'workgroup_')
        );
        const usedResources = new Set(
          reflection.entry.compute.flatMap(entry => entry.resources.map(resource => resource.name))
        );
        for (const resource of [...reflection.storage, ...reflection.uniforms])
          if (!usedResources.has(resource.name))
            findings.push(
              `${module.nodeId}: binding ${resource.name} is absent from the entry point's inferred layout`
            );
        findings.push(
          ...findStatisticsNonFiniteConstants(module.source).map(
            finding => `${module.nodeId}: ${finding}`
          )
        );
        findings.push(
          ...findReservedIdentifiers(stripWGSLComments(module.source)).map(
            finding => `${module.nodeId}: ${finding.detail}`
          )
        );
        if (/workgroupBarrier|storageBarrier/.test(module.source))
          findings.push(
            ...findStatisticsBarrierRisks(module.source).map(
              finding => `${module.nodeId}: ${finding}`
            )
          );
      });
      expect(seen.size).toBeGreaterThan(0);
      expect(findings).toEqual([]);
    } finally {
      context.device.destroy();
    }
  }, 60_000);

it('the headless padding probe constructs a legal two-lane padded dispatch', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxComputeWorkgroupsPerDimension: 7}
  });
  const graph = new GPUCommandGraph(device);
  try {
    const {node, groupCount} = createStatisticsPaddingProbe(
      graph,
      createTransientView(graph, 'total', 'uint32', 1)
    );
    expect(groupCount).toBe(9);
    expect(node.workload?.maximumInvocationCount).toBe(18);
    expect(node.workload?.maximumWorkgroupCount).toBeGreaterThan(groupCount);
    const [record] = getGPUCommandNodeKernelSources([node]);
    expect(record.source).toContain('@workgroup_size(2)');
    expect(record.source).toContain('if (localInvocationIndex == 0u) { value = 1u; }');
    expect(record.source).toContain('if (localInvocationIndex == 0u) { atomicAdd(');
    expect(findStatisticsBarrierRisks(record.source)).toEqual([]);
    expect(findStatisticsNonFiniteConstants(record.source)).toEqual([]);
    expect(findReservedIdentifiers(stripWGSLComments(record.source))).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('cooperative statistics discard extra 2D workgroups before barriers or scratch writes', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxComputeWorkgroupsPerDimension: 7}
  });
  const graph = new GPUCommandGraph(device);
  try {
    const node = createStatisticsKernelNode(graph, {
      id: 'cooperative-grid-padding',
      operation: 'test',
      invocationCount: 32,
      workgroupSize: 4,
      guardIndex: false,
      bindings: [
        {
          name: 'total',
          view: createTransientView(graph, 'total', 'uint32', 1),
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      declarations: '',
      body: `workgroupBarrier();
if (localInvocationIndex == 0u) { atomicAdd(&total[totalOffset], 4u); }`
    });
    const [record] = getGPUCommandNodeKernelSources([node]);
    expect(node.workload?.maximumWorkgroupCount).toBeGreaterThan(8);
    expect(findStatisticsBarrierRisks(record.source)).toEqual([]);
    const source = record.source
      .replace(/_ = [^;]+;/g, '')
      .replace(
        /override (\w+): u32;/g,
        (_, name: string) => `const ${name}: u32 = ${record.constants[name]}u;`
      );
    const total = new Uint32Array(1);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
      'main',
      [record.constants['WGSL_KERNEL_DISPATCH_X'], record.constants['WGSL_KERNEL_DISPATCH_Y'], 1],
      {0: {0: total}}
    );
    expect(Array.from(total)).toEqual([32]);
    total.fill(0);
    const unguarded = source.replace(
      /if \(workgroupIndex >= \(INVOCATION_COUNT - 1u\) \/ 4u \+ 1u\) \{ return; \}/,
      ''
    );
    new WgslExec(new WgslParser().parse(unguarded)).dispatchWorkgroups(
      'main',
      [record.constants['WGSL_KERNEL_DISPATCH_X'], record.constants['WGSL_KERNEL_DISPATCH_Y'], 1],
      {0: {0: total}}
    );
    expect(total[0]).toBeGreaterThan(32);
  } finally {
    device.destroy();
  }
});
