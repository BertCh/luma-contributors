// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  auditWGSLModule,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUIslandAttachment,
  type GPUIslandAttachmentNearestSearch
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {
  ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
  createIslandAttachmentFallbackNodes
} from '../../../src/gpu-spatial-analysis/spatial-weights/island-attachment-fallback';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeKernelSources,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

import {getIslandAttachmentCSRReference} from './island-attachment-test-utils';

// Check the full W3C reserved-word table separately from the interpreter's contextual keywords.
const ISLAND_ATTACHMENT_FORBIDDEN_IDENTIFIERS = new Set(WGSL_RESERVED_WORDS);
function expectNoReservedIdentifiers(source: string): void {
  const identifiers =
    source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/\b[A-Za-z_][A-Za-z0-9_]*\b/g) ?? [];
  expect(
    identifiers.filter(identifier => ISLAND_ATTACHMENT_FORBIDDEN_IDENTIFIERS.has(identifier))
  ).toEqual([]);
}

function createAttachment(
  rows: number,
  nearestSearch: GPUIslandAttachmentNearestSearch = 'grid',
  uncertainCount = true
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const contributor = new GPUIslandAttachment({
    id: 'bounded-islands',
    nearestSearch,
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', rows),
      weights: view('float32', rows)
    },
    positions: view('float32x2', rows),
    output: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', 3 * rows),
      weights: view('float32', 3 * rows),
      attachment: view('uint32', rows),
      added: view('uint32', 3 * rows),
      status: {
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1),
        ...(uncertainCount ? {uncertainCount: view('uint32', 1)} : {})
      }
    }
  });
  return {device, graph, contributor, view};
}

describe('GPUIslandAttachment batched cooperative fallback', () => {
  for (const rows of [17, 65535, 65537, 131073, 1_000_000]) {
    for (const uncertainCount of [false, true]) {
      it(`${rows} rows, uncertainty port ${uncertainCount}: bounded lanes and constant pipelines`, () => {
        const {device, graph, contributor} = createAttachment(rows, 'grid', uncertainCount);
        try {
          const nodes = contributor.getCommandNodes(graph);
          expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
          const nearest = getWGSLKernelNodeRecord(graph, 'bounded-islands-nearest')!;
          expect(nearest.body).not.toContain(`other < ${rows}u`);
          expect(nearest.body).toContain('unresolved[unresolvedOffset + index] = 1u');
          const batchCount = Math.ceil(rows / 65536);
          const sources = new Set<string>();
          const pipelineKeys = new Set<string>();
          for (let batch = 0; batch < batchCount; batch++) {
            const fallback = getWGSLKernelNodeRecord(
              graph,
              `bounded-islands-resolve-unproven-${batch}`
            )!;
            expect(fallback.workgroupSize).toEqual([256, 1, 1]);
            expect(fallback.invocationCount).toBe(rows * 256);
            expect(fallback.indirect).toBe(true);
            expect(fallback.body).toContain('slot < representativeCount');
            expect(fallback.body).toContain('other < nearest');
            expect(fallback.body).toContain('let previous = attachment');
            sources.add(fallback.source);
            pipelineKeys.add(JSON.stringify([fallback.source, fallback.constants]));
            expect(new WgslReflect(fallback.source).entry.compute.map(entry => entry.name)).toEqual(
              ['main']
            );
            expect(getWorkgroupMemoryBytes(fallback.source, fallback.constants)).toBe(3072);
            expect(auditWGSLModule(fallback.source)).toEqual([]);
            expect(fallback.bindings.length).toBeLessThanOrEqual(8);
          }
          expect(sources.size).toBe(1);
          expect(pipelineKeys.size).toBe(1);
          // Record actual pipelines too: every fallback, advance and dispatch-updater family
          // has one pipeline despite growing batch counts. Audit every generated identifier.
          expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: /resolve-unproven/}).size).toBe(
            1
          );
          expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: /fallback-advance/}).size).toBe(
            1
          );
          expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: /fallback-dispatch/}).size).toBe(
            1
          );
          for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
            expect(auditWGSLModule(kernel.source), kernel.id).toEqual([]);
            expectNoReservedIdentifiers(kernel.source);
          }
        } finally {
          device.destroy();
        }
      });
    }
  }

  it('checks the full W3C reserved table while permitting the contextual name sample', () => {
    for (const identifier of ['target', 'active', 'filter', 'module', 'meta']) {
      expect(ISLAND_ATTACHMENT_FORBIDDEN_IDENTIFIERS.has(identifier)).toBe(true);
    }
    expect(ISLAND_ATTACHMENT_FORBIDDEN_IDENTIFIERS.has('sample')).toBe(false);
  });

  it('rejects an excessive internal test batch topology before resource declaration', () => {
    const {device, graph, contributor, view} = createAttachment(300);
    try {
      const unresolved = view('uint32', 300);
      const transient = vi.spyOn(graph, 'createTransientBuffer');
      expect(() =>
        createIslandAttachmentFallbackNodes(graph, {
          id: 'excessive-batches',
          positions: contributor.props.positions,
          attachment: contributor.props.output.attachment,
          unresolved,
          nearestDeclarations: ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
          queryBatchSize: 1,
          representativeBatchSize: 1
        })
      ).toThrow(/estimated command node count/);
      expect(transient).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });

  it('small-batch rollover keeps one pipeline per repeated kernel as batch count grows', () => {
    for (const queryBatchSize of [3, 7, 17]) {
      const {device, graph, contributor, view} = createAttachment(33);
      try {
        const nodes = createIslandAttachmentFallbackNodes(graph, {
          id: 'small-batches',
          positions: contributor.props.positions,
          unresolved: view('uint32', 33),
          attachment: contributor.props.output.attachment,
          nearestDeclarations: ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
          queryBatchSize,
          representativeBatchSize: 7
        });
        for (const pattern of [/resolve-unproven/, /fallback-advance/, /fallback-dispatch/]) {
          expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: pattern}).size).toBe(1);
        }
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          expectNoReservedIdentifiers(kernel.source);
          expect(auditWGSLModule(kernel.source), kernel.id).toEqual([]);
        }
      } finally {
        device.destroy();
      }
    }
  });

  it('map-and-sort CSR oracle matches a hand-computed reciprocal truncation', () => {
    const scene = {
      positions: new Float32Array(8),
      offsets: new Uint32Array([0, 0, 1, 3, 3]),
      neighbors: new Uint32Array([2, 1, 3]),
      weights: new Float32Array([2.5, 3.5, 4.5])
    };
    expect(getIslandAttachmentCSRReference(scene, [1, 0xffffffff, 0xffffffff, 2], true, 4)).toEqual(
      {
        offsets: [0, 1, 3, 4, 4],
        neighbors: [1, 0, 2, 1],
        weights: [1, 1, 2.5, 3.5],
        added: [1, 1, 0, 0],
        requiredCount: 6,
        overflow: 1
      }
    );
    expect(
      getIslandAttachmentCSRReference(scene, [1, 0xffffffff, 0xffffffff, 2], false, 10)
    ).toEqual({
      offsets: [0, 1, 2, 4, 5],
      neighbors: [1, 2, 1, 3, 2],
      weights: [1, 2.5, 3.5, 4.5, 1],
      added: [1, 0, 0, 0, 1],
      requiredCount: 5,
      overflow: 0
    });
  });

  it('splits cooperative query dispatches before the uint32 invocation boundary', () => {
    const small = createAttachment(17);
    const large = createAttachment(16_777_216);
    try {
      small.contributor.getCommandNodes(small.graph);
      const nearestDeclarations = getWGSLKernelNodeRecord(small.graph, 'bounded-islands-nearest')!
        .declarations!;
      const nodes = createIslandAttachmentFallbackNodes(large.graph, {
        id: 'query-batches',
        positions: large.contributor.props.positions,
        unresolved: large.view('uint32', 16_777_216),
        attachment: large.contributor.props.output.attachment,
        nearestDeclarations
      });
      // 256 representative batches times two query batches. Logical views allocate no GPU
      // memory on the null device; even this 16M-query wiring test stays small on the CPU.
      const sources = new Set<string>();
      const pipelines = new Set<string>();
      for (let batch = 0; batch < 512; batch++) {
        const kernel = getWGSLKernelNodeRecord(
          large.graph,
          `query-batches-resolve-unproven-${batch}`
        )!;
        expect(kernel.invocationCount).toBe(16_777_215 * 256);
        expect(kernel.invocationCount).toBeLessThanOrEqual(0xffffffff);
        sources.add(kernel.source);
        pipelines.add(JSON.stringify([kernel.source, kernel.constants]));
      }
      expect(sources.size).toBe(1);
      expect(pipelines.size).toBe(1);
      expect(nodes.length).toBeLessThanOrEqual(large.contributor.estimatedCommandNodeCount);
      expect(
        getWGSLKernelNodeRecord(large.graph, 'query-batches-fallback-advance-511')!.body
      ).toContain('count[countOffset] - queryBase');
    } finally {
      small.device.destroy();
      large.device.destroy();
    }
  });

  it('keeps the explicit scan reference cap while allowing the indexed route above it', () => {
    expect(() => createAttachment(65536, 'scan')).toThrow(
      /^bounded-islands rows for nearestSearch scan must be.*65535/
    );
    const {device, contributor} = createAttachment(65536);
    expect(contributor.props.positions.length).toBe(65536);
    device.destroy();
  });

  it('auto uses the indexed path above the cut-over; explicit scan retains its path', () => {
    for (const nearestSearch of ['auto', 'scan'] as const) {
      const {device, graph, contributor} = createAttachment(4097, nearestSearch);
      try {
        contributor.getCommandNodes(graph);
        expect(Boolean(getWGSLKernelNodeRecord(graph, 'bounded-islands-resolve-unproven-0'))).toBe(
          nearestSearch === 'auto'
        );
      } finally {
        device.destroy();
      }
    }
  });

  it('rejects overlap across every writable output field at construction', () => {
    const {device, graph, contributor} = createAttachment(17);
    try {
      const props = contributor.props;
      const outputs = [
        props.output.offsets,
        props.output.neighbors,
        props.output.weights,
        props.output.attachment,
        props.output.added,
        props.output.status.overflow,
        props.output.status.requiredCount!
      ];
      for (const output of outputs) {
        expect(
          () =>
            new GPUIslandAttachment({
              ...props,
              output: {
                ...props.output,
                status: {
                  ...props.output.status,
                  uncertainCount: graph.createDataView(output.buffer, {
                    format: 'uint32',
                    byteOffset: output.byteOffset,
                    length: 1
                  })
                }
              }
            })
        ).toThrow(/output writable byte ranges must not overlap/);
      }
    } finally {
      device.destroy();
    }
  });

  it('rejects aligned status aliases before declaring graph resources', () => {
    const {device, graph, contributor, view} = createAttachment(17);
    try {
      const words = view('uint32', 2);
      const transient = vi.spyOn(graph, 'createTransientBuffer');
      const compute = vi.spyOn(graph, 'addComputePass');
      expect(
        () =>
          new GPUIslandAttachment({
            ...contributor.props,
            output: {
              ...contributor.props.output,
              status: {
                ...contributor.props.output.status,
                overflow: graph.createDataView(words.buffer, {
                  format: 'uint32',
                  byteOffset: 0,
                  length: 1
                }),
                requiredCount: graph.createDataView(words.buffer, {
                  format: 'uint32',
                  byteOffset: 4,
                  length: 1
                })
              }
            }
          })
      ).toThrow(/aligned binding byte ranges must not overlap/);
      expect(transient).not.toHaveBeenCalled();
      expect(compute).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });

  it('rejects aligned aliases between read outputs and an emit destination', () => {
    const {device, graph, contributor, view} = createAttachment(17);
    try {
      const props = contributor.props;
      for (const readField of ['offsets', 'attachment'] as const) {
        for (const writeField of ['neighbors', 'weights', 'added'] as const) {
          const words = view(
            'uint32',
            props.output[readField].length + props.output[writeField].length
          );
          const first = graph.createDataView(words.buffer, {
            format: 'uint32',
            length: props.output[readField].length
          });
          const second = graph.createDataView(words.buffer, {
            format: writeField === 'weights' ? 'float32' : 'uint32',
            byteOffset: first.length * 4,
            length: props.output[writeField].length
          });
          expect(
            () =>
              new GPUIslandAttachment({
                ...props,
                output: {...props.output, [readField]: first, [writeField]: second}
              })
          ).toThrow(
            readField === 'offsets' && writeField !== 'added'
              ? new RegExp(
                  `offsets chunk 0 and ${writeField} chunk 0 aligned binding byte ranges must not overlap`
                )
              : /output read-only and writable bindings must use separate physical buffers/
          );
        }
      }
    } finally {
      device.destroy();
    }
  });

  it('rejects disjoint aligned read-only and writable ranges of one physical buffer', () => {
    const {device, graph, contributor, view} = createAttachment(17);
    try {
      for (const readField of ['offsets', 'attachment'] as const) {
        for (const writeField of ['neighbors', 'weights', 'added'] as const) {
          const words = view('uint32', 64 + contributor.props.output[writeField].length);
          const output = {
            ...contributor.props.output,
            [readField]: graph.createDataView(words.buffer, {
              format: 'uint32',
              length: contributor.props.output[readField].length
            }),
            [writeField]: graph.createDataView(words.buffer, {
              format: writeField === 'weights' ? 'float32' : 'uint32',
              byteOffset: 256,
              length: contributor.props.output[writeField].length
            })
          };
          const transient = vi.spyOn(graph, 'createTransientBuffer');
          expect(() => new GPUIslandAttachment({...contributor.props, output})).toThrow(
            /read-only and writable bindings must use separate physical buffers/
          );
          expect(transient).not.toHaveBeenCalled();
          transient.mockRestore();
        }
      }
    } finally {
      device.destroy();
    }
  });

  it('accepts disjoint packed attributes written in separate emit passes', () => {
    const {device, graph, contributor, view} = createAttachment(17);
    try {
      const words = view('uint32', 192);
      const output = {
        ...contributor.props.output,
        neighbors: graph.createDataView(words.buffer, {format: 'uint32', length: 51}),
        weights: graph.createDataView(words.buffer, {
          format: 'float32',
          byteOffset: 256,
          length: 51
        }),
        added: graph.createDataView(words.buffer, {format: 'uint32', byteOffset: 512, length: 51})
      };
      expect(() =>
        graph.add(new GPUIslandAttachment({...contributor.props, output}))
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  });

  it('accepts aligned adjacent status words and schedules their shared pass', () => {
    const {device, graph, contributor, view} = createAttachment(17);
    try {
      const words = view('uint32', 65);
      const props = contributor.props;
      const producer = new GPUIslandAttachment({
        ...props,
        output: {
          ...props.output,
          status: {
            ...props.output.status,
            overflow: graph.createDataView(words.buffer, {
              format: 'uint32',
              byteOffset: 0,
              length: 1
            }),
            requiredCount: graph.createDataView(words.buffer, {
              format: 'uint32',
              byteOffset: 256,
              length: 1
            })
          }
        }
      });
      expect(() => graph.add(producer)).not.toThrow();
    } finally {
      device.destroy();
    }
  });

  it('accepts disjoint writable ranges in one buffer', () => {
    const {device, graph, contributor, view} = createAttachment(17);
    try {
      const words = view('uint32', 2);
      const props = contributor.props;
      expect(() =>
        graph.add(
          new GPUIslandAttachment({
            ...props,
            output: {
              ...props.output,
              status: {
                overflow: graph.createDataView(words.buffer, {
                  format: 'uint32',
                  byteOffset: 0,
                  length: 1
                }),
                uncertainCount: graph.createDataView(words.buffer, {
                  format: 'uint32',
                  byteOffset: 4,
                  length: 1
                })
              }
            }
          })
        )
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  });

  it('preflights candidate scratch before adding any graph resources or nodes', () => {
    const {device, graph, contributor} = createAttachment(65537);
    try {
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
        value: 65537 * 64 - 4,
        configurable: true
      });
      const transient = vi.spyOn(graph, 'createTransientBuffer');
      const compute = vi.spyOn(graph, 'addComputePass');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /candidate lists and outputs needs.*limit/
      );
      expect(transient).not.toHaveBeenCalled();
      expect(compute).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });

  it('preflights the aligned binding prefix as well as logical bytes', () => {
    const {device, graph, contributor, view} = createAttachment(17, 'scan');
    try {
      const storage = view('uint32', 300);
      const offsets = graph.createDataView(storage.buffer, {
        format: 'uint32',
        byteOffset: 252,
        length: 18
      });
      const props = contributor.props;
      const shifted = new GPUIslandAttachment({...props, weights: {...props.weights, offsets}});
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
        value: 320,
        configurable: true
      });
      const transient = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => shifted.getCommandNodes(graph)).toThrow(/needs 324 bytes/);
      expect(transient).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });
});
