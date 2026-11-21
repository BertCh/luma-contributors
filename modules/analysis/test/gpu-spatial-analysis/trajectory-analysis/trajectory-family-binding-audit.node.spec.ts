// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {
  getExpectedRejection,
  getKernelAuditRefusalFindings
} from '../../analysis-core/kernel-audit/kernel-audit-runner';
import {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {getTrajectoryBindingViews} from '../../../src/gpu-spatial-analysis/trajectory-analysis/trajectory-binding-validation';
import {KERNEL_AUDIT_CASES} from '../../analysis-core/kernel-audit/kernel-audit-cases';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';

const CONTRIBUTORS = [
  'GPUTrajectoryMetrics',
  'GPUTrajectoryEncounters',
  'GPUTrajectoryClosestApproach',
  'GPUTrajectoryKalman',
  'GPUBrownianBridgeUtilization',
  'GPUUtilizationVolumeIsopleths',
  'GPUTrajectoryPlayhead',
  'GPUTrajectoryResample',
  'GPUMobilityIndicators',
  'GPUModalSplit',
  'GPUTrajectoryRecursions',
  'GPUTrajectoryRestriction',
  'GPUTrajectoryStaypoints',
  'GPUTrajectorySegments',
  'GPUTrajectoryLocations',
  'GPUZoneEvents',
  'GPUTrajectoryHampel',
  'GPUTrajectoryOutlierFlags',
  'GPUTrajectoryTrackRelativeTimes',
  'GPUTrajectoryThinning',
  'GPUTrajectorySplit',
  'GPUTrackSimilarity',
  'GPUTrackSimilaritySearch',
  'GPUSpaceTimeCrossCorrelation',
  'GPUSpaceTimeLocalMoran',
  'GPUKnoxTest',
  'GPUMantelTest',
  'GPUJacquezTest',
  'GPULocalKnox',
  'GPUNearRepeatKnox',
  'GPUSpaceTimeHawkes'
];

for (const name of CONTRIBUTORS) {
  it(`${name} rejects invalid storage usage before any graph mutation`, () => {
    const auditCase = Object.values(KERNEL_AUDIT_CASES)
      .flat()
      .find(entry => entry.exportName === name);
    expect(auditCase, `${name} needs an audit fixture`).toBeDefined();
    const context = createKernelAuditContext(`trajectory-usage-${name}`);
    let checked = false;
    try {
      Object.defineProperty(context.graph, 'add', {
        value: (producer: {
          props: object;
          getCommandNodes: (graph: typeof context.graph) => unknown;
        }) => {
          const first = Object.values(getTrajectoryBindingViews(producer.props))[0]!;
          const view = 'data' in first ? first.data[0] : first;
          const originalUsage = view.buffer.usage;
          const declarations = context.graph as unknown as {
            buffers: Map<string, unknown>;
            nodes: unknown[];
          };
          const before = [declarations.buffers.size, declarations.nodes.length];
          Object.defineProperty(view.buffer, 'usage', {configurable: true, value: Buffer.COPY_DST});
          try {
            expect(() => producer.getCommandNodes(context.graph)).toThrow(/Buffer.STORAGE/);
            expect([declarations.buffers.size, declarations.nodes.length]).toEqual(before);
            checked = true;
          } finally {
            Object.defineProperty(view.buffer, 'usage', {configurable: true, value: originalUsage});
          }
        }
      });
      auditCase!.build(context);
      expect(checked).toBe(true);
    } finally {
      context.device.destroy();
    }
  });
}

it('parses generated WGSL and checks every identifier against the complete W3C reserved-word table across the family', () => {
  const syntaxErrors: string[] = [];
  for (const name of CONTRIBUTORS) {
    const auditCase = Object.values(KERNEL_AUDIT_CASES)
      .flat()
      .find(entry => entry.exportName === name)!;
    const context = createKernelAuditContext(`trajectory-identifiers-${name}`);
    try {
      Object.defineProperty(context.graph, 'add', {
        value: (producer: {
          getCommandNodes: (graph: typeof context.graph) => readonly unknown[];
        }) => {
          const kernels = getGPUCommandNodeKernelSources(producer.getCommandNodes(context.graph));
          for (const kernel of kernels) {
            const source = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
            const identifiers = new Set(source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []);
            expect(
              [...identifiers].filter(identifier => WGSL_RESERVED_WORDS.has(identifier)),
              `${name}: ${kernel.id}`
            ).toEqual([]);
            expect(source, `${name}: ${kernel.id}`).not.toMatch(/\b(?:Infinity|NaN)\b/);
            try {
              // Match the kernel audit's parser-only normalization for valid WGSL
              // identifiers that wgsl_reflect still treats as legacy keywords.
              new WgslReflect(
                source
                  .replace(/\bblock\b/g, 'block_')
                  .replace(/(?<!<\s*)\bworkgroup\b(?!\s*>)/g, 'workgroup_')
              );
            } catch (error) {
              // Its constant evaluator rejects some valid u32 arithmetic after parsing.
              if (!/Invalid cast/.test(String(error))) {
                syntaxErrors.push(`${name}: ${kernel.id}: ${String(error)}`);
              }
            }
            if (
              ['GPUTrajectoryEncounters', 'GPUTrajectoryClosestApproach', 'GPUZoneEvents'].includes(
                name
              )
            )
              expect(source).not.toContain('atomicAdd(&state[stateOffset], 1u)');
          }
        }
      });
      auditCase.build(context);
    } finally {
      context.device.destroy();
    }
  }
  expect(syntaxErrors).toEqual([]);
});

it('rejects logical output aliases at construction across every multi-output contributor', () => {
  for (const name of CONTRIBUTORS) {
    const auditCase = Object.values(KERNEL_AUDIT_CASES)
      .flat()
      .find(entry => entry.exportName === name)!;
    const context = createKernelAuditContext(`trajectory-alias-${name}`);
    try {
      Object.defineProperty(context.graph, 'add', {
        value: (producer: {
          props: {output: object};
          constructor: new (props: unknown) => unknown;
        }) => {
          const fields = Object.entries(getTrajectoryBindingViews(producer.props.output));
          if (fields.length < 2) return;
          const first = fields[0][1]!;
          const second = fields[1][1]!;
          if (!('buffer' in first) || !('buffer' in second)) return;
          const byteLength = Math.max(
            first.length * first.rowByteLength,
            second.length * second.rowByteLength
          );
          const buffer = context.buffer(Math.max(4, byteLength));
          const replacements = new Map([
            [
              first,
              context.graph.createDataView(buffer, {format: first.format, length: first.length})
            ],
            [
              second,
              context.graph.createDataView(buffer, {format: second.format, length: second.length})
            ]
          ]);
          const replace = (value: unknown): unknown => {
            if (replacements.has(value as typeof first))
              return replacements.get(value as typeof first);
            if (
              value &&
              typeof value === 'object' &&
              Object.getPrototypeOf(value) === Object.prototype
            )
              return Object.fromEntries(
                Object.entries(value).map(([key, child]) => [key, replace(child)])
              );
            return value;
          };
          expect(
            () =>
              new producer.constructor({...producer.props, output: replace(producer.props.output)}),
            name
          ).toThrow(/overlap|separate buffers/);
        }
      });
      auditCase.build(context);
    } finally {
      context.device.destroy();
    }
  }
});

it('never accepts mixed read-only/writable storage usage when outputs share a physical buffer', () => {
  const violations: string[] = [];
  for (const name of CONTRIBUTORS) {
    const auditCase = Object.values(KERNEL_AUDIT_CASES)
      .flat()
      .find(entry => entry.exportName === name)!;
    const context = createKernelAuditContext(`trajectory-packed-scope-${name}`);
    try {
      Object.defineProperty(context.graph, 'add', {
        value: (producer: {
          props: {output: object};
          constructor: new (
            props: unknown
          ) => {getCommandNodes: (graph: typeof context.graph) => readonly GPUCommandNode[]};
        }) => {
          // Fixtures may choose their row count from the original limits. Enlarge only
          // after that choice, allowing their separate outputs to fit one metadata buffer.
          Object.defineProperty(context.device, 'limits', {
            value: {
              ...context.device.limits,
              maxBufferSize: 2 ** 34,
              maxStorageBufferBindingSize: 2 ** 34
            }
          });
          const fields = Object.values(getTrajectoryBindingViews(producer.props.output)).filter(
            view => view !== undefined
          );
          let byteLength = 0;
          const offsets = fields.map(view => {
            const offset = byteLength;
            byteLength += Math.ceil(Math.max(4, view.length * view.rowByteLength) / 256) * 256;
            return offset;
          });
          const buffer = context.buffer(Math.max(4, byteLength));
          const replacements = new Map(
            fields.map((view, index) => [
              view,
              context.graph.createDataView(buffer, {
                format: view.format,
                length: view.length,
                byteOffset: offsets[index]
              })
            ])
          );
          const replace = (value: unknown): unknown => {
            if (replacements.has(value as (typeof fields)[number]))
              return replacements.get(value as (typeof fields)[number]);
            if (
              value &&
              typeof value === 'object' &&
              Object.getPrototypeOf(value) === Object.prototype
            )
              return Object.fromEntries(
                Object.entries(value).map(([key, child]) => [key, replace(child)])
              );
            return value;
          };
          recordWGSLKernelNodes(context.graph);
          let nodes: readonly GPUCommandNode[];
          try {
            nodes = new producer.constructor({
              ...producer.props,
              output: replace(producer.props.output)
            }).getCommandNodes(context.graph);
          } catch (error) {
            expect(String(error), name).toMatch(
              /outputs must not share buffers with inputs|separate buffers/
            );
            return;
          }
          for (const node of nodes) {
            const record = getWGSLKernelNodeRecord(context.graph, node.id);
            if (!record) continue;
            for (const writable of record.bindings.filter(
              binding => binding.access === 'read_write'
            )) {
              for (const readable of record.bindings.filter(binding => binding.access === 'read')) {
                if (writable.view.buffer === readable.view.buffer)
                  violations.push(`${name}: ${node.id}: ${writable.name}/${readable.name}`);
              }
            }
          }
        }
      });
      auditCase.build(context);
    } finally {
      context.device.destroy();
    }
  }
  expect(violations).toEqual([]);
});

it('checks aligned writable ranges for every recorded co-bound public output pair', () => {
  const violations: string[] = [];
  for (const name of CONTRIBUTORS) {
    const auditCase = Object.values(KERNEL_AUDIT_CASES)
      .flat()
      .find(entry => entry.exportName === name)!;
    const context = createKernelAuditContext(`trajectory-aligned-scope-${name}`);
    try {
      Object.defineProperty(context.graph, 'add', {
        value: (producer: {
          props: {output: object};
          constructor: new (props: unknown) => unknown;
          getCommandNodes: (graph: typeof context.graph) => readonly GPUCommandNode[];
        }) => {
          recordWGSLKernelNodes(context.graph);
          const fields = new Set(Object.values(getTrajectoryBindingViews(producer.props.output)));
          const nodes = producer.getCommandNodes(context.graph);
          const records = nodes
            .map(node => getWGSLKernelNodeRecord(context.graph, node.id))
            .filter(record => record !== undefined);
          Object.defineProperty(context.device, 'limits', {
            value: {
              ...context.device.limits,
              maxBufferSize: 2 ** 34,
              maxStorageBufferBindingSize: 2 ** 34
            }
          });
          for (const node of nodes) {
            const record = getWGSLKernelNodeRecord(context.graph, node.id);
            const bindings =
              record?.bindings.filter(
                binding => binding.access === 'read_write' && fields.has(binding.view)
              ) ?? [];
            for (let firstIndex = 0; firstIndex < bindings.length; firstIndex++) {
              for (const second of bindings.slice(firstIndex + 1)) {
                const first = bindings[firstIndex];
                if (first.view === second.view) continue;
                const secondOffset = 4 + first.view.length * first.view.rowByteLength;
                const buffer = context.buffer(
                  secondOffset + Math.max(4, second.view.length * second.view.rowByteLength)
                );
                const replacements = new Map([
                  [
                    first.view,
                    context.graph.createDataView(buffer, {
                      format: first.view.format,
                      length: first.view.length,
                      byteOffset: 4
                    })
                  ],
                  [
                    second.view,
                    context.graph.createDataView(buffer, {
                      format: second.view.format,
                      length: second.view.length,
                      byteOffset: secondOffset
                    })
                  ]
                ]);
                try {
                  new producer.constructor({
                    ...producer.props,
                    output: replaceViews(producer.props.output, replacements)
                  });
                  violations.push(`${name}: ${node.id}: ${first.name}/${second.name}`);
                } catch (error) {
                  expect(String(error), `${name}: ${node.id}`).toMatch(
                    /overlap|share buffers|separate buffers/
                  );
                }
                const hasMixedUsage = records.some(
                  candidate =>
                    (candidate.bindings.some(
                      binding => binding.access === 'read' && binding.view === first.view
                    ) &&
                      candidate.bindings.some(
                        binding => binding.access === 'read_write' && binding.view === second.view
                      )) ||
                    (candidate.bindings.some(
                      binding => binding.access === 'read' && binding.view === second.view
                    ) &&
                      candidate.bindings.some(
                        binding => binding.access === 'read_write' && binding.view === first.view
                      ))
                );
                if (!hasMixedUsage) {
                  const alignedOffset = Math.ceil(secondOffset / 256) * 256;
                  const alignedBuffer = context.buffer(
                    alignedOffset + Math.max(4, second.view.length * second.view.rowByteLength)
                  );
                  const alignedReplacements = new Map([
                    [
                      first.view,
                      context.graph.createDataView(alignedBuffer, {
                        format: first.view.format,
                        length: first.view.length,
                        byteOffset: 4
                      })
                    ],
                    [
                      second.view,
                      context.graph.createDataView(alignedBuffer, {
                        format: second.view.format,
                        length: second.view.length,
                        byteOffset: alignedOffset
                      })
                    ]
                  ]);
                  expect(
                    () =>
                      new producer.constructor({
                        ...producer.props,
                        output: replaceViews(producer.props.output, alignedReplacements)
                      }),
                    `${name}: ${node.id}: valid ${first.name}/${second.name}`
                  ).not.toThrow();
                }
              }
            }
          }
        }
      });
      auditCase.build(context);
    } finally {
      context.device.destroy();
    }
  }
  expect(violations).toEqual([]);
});

function replaceViews(value: unknown, replacements: ReadonlyMap<unknown, unknown>): unknown {
  if (replacements.has(value)) return replacements.get(value);
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, replaceViews(child, replacements)])
    );
  return value;
}

it('rejects spatial and time-unit props that cannot be represented in float32', () => {
  for (const name of [
    'GPUTrajectoryEncounters',
    'GPUTrajectoryClosestApproach',
    'GPUTrajectoryKalman'
  ]) {
    const auditCase = Object.values(KERNEL_AUDIT_CASES)
      .flat()
      .find(entry => entry.exportName === name)!;
    const context = createKernelAuditContext(`trajectory-cell-parameters-${name}`);
    try {
      Object.defineProperty(context.graph, 'add', {
        value: (producer: {props: object; constructor: new (props: unknown) => unknown}) => {
          if (name === 'GPUTrajectoryKalman') {
            const props = producer.props as {positions: {length: number}};
            for (const timeUnitsPerSecond of [1e40, 1e-50])
              expect(
                () =>
                  new producer.constructor({
                    ...producer.props,
                    temporalContext: undefined,
                    times: context.view('uint32x2', props.positions.length),
                    timeUnitsPerSecond
                  })
              ).toThrow(/timeUnitsPerSecond.*finite/);
          } else {
            for (const cellSize of [1e40, 1e-50])
              expect(() => new producer.constructor({...producer.props, cellSize})).toThrow(
                /cellSize.*finite/
              );
            if (name === 'GPUTrajectoryEncounters')
              expect(
                () =>
                  new producer.constructor({
                    ...producer.props,
                    spatialContext: undefined,
                    cellSize: 1e38,
                    bounds: [0, 0, 1e40, 1]
                  })
              ).toThrow(/bounds\[2\].*finite/);
          }
        }
      });
      auditCase.build(context);
    } finally {
      context.device.destroy();
    }
  }
});

it('preserves every existing registered rejection in the owned families', () => {
  const cases = Object.values(KERNEL_AUDIT_CASES)
    .flat()
    .filter(entry => CONTRIBUTORS.includes(entry.exportName) && getExpectedRejection(entry));
  expect(cases.length).toBeGreaterThan(10);
  for (const auditCase of cases) {
    const context = createKernelAuditContext(`trajectory-rejection-${auditCase.exportName}`);
    try {
      let caughtError: unknown;
      try {
        auditCase.build(context);
      } catch (error) {
        caughtError = error;
      }
      expect(caughtError, `${auditCase.exportName}: ${auditCase.name}`).toBeDefined();
      expect(
        getKernelAuditRefusalFindings(
          caughtError,
          getExpectedRejection(auditCase),
          context.rejectionObservation
        ),
        `${auditCase.exportName}: ${auditCase.name}`
      ).toEqual([]);
    } finally {
      context.device.destroy();
    }
  }
});

it('reserves a count sentinel above every accepted trajectory candidate capacity', () => {
  for (const [name, fields] of [
    ['GPUTrajectoryEncounters', ['hitCapacity']],
    ['GPUTrajectoryClosestApproach', ['entryCapacity', 'hitCapacity']],
    ['GPUZoneEvents', ['candidateCapacity']]
  ] as const) {
    const auditCase = Object.values(KERNEL_AUDIT_CASES)
      .flat()
      .find(entry => entry.exportName === name)!;
    const context = createKernelAuditContext(`trajectory-count-limit-${name}`);
    try {
      Object.defineProperty(context.graph, 'add', {
        value: (producer: {props: object; constructor: new (props: unknown) => unknown}) => {
          for (const field of fields)
            expect(
              () => new producer.constructor({...producer.props, [field]: 0xffffffff})
            ).toThrow(`${field} must be an integer in [1, 4294967294]`);
        }
      });
      auditCase.build(context);
    } finally {
      context.device.destroy();
    }
  }
});
