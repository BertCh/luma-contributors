// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import * as kernelNodes from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());

function createProps(graph: GPUCommandGraph, capacity = 8) {
  return {
    id: 'snap-contract',
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'lines', 'uint32', 2),
    referencePositions: createTransientView(graph, 'references', 'float32x2', 3),
    tolerance: 0.5,
    maximumVerticesPerLine: capacity,
    output: {
      positions: createTransientView(graph, 'result', 'float32x2', capacity),
      sourceEdges: createTransientView(graph, 'edges', 'uint32', capacity),
      counts: createTransientView(graph, 'counts', 'uint32x4', 1),
      validity: createTransientView(graph, 'validity', 'uint32', 1)
    }
  };
}

it.each([
  Number.NaN,
  Infinity,
  -1,
  1e100
])('rejects a non-f32 tolerance %s at construction', tolerance => {
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUSegmentSnap({...createProps(graph), tolerance})).toThrow(
    /^snap-contract tolerance must be/
  );
});

it('rejects scalar coordinate address overflow even when the row count fits uint32', () => {
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUSegmentSnap(createProps(graph, 0x80000000))).toThrow(
    /^snap-contract maximumVerticesPerLine.*2147483647/
  );
});

it('rejects overlapping writable output ranges and disjoint rows sharing an aligned prefix', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUSegmentSnap({
        ...props,
        output: {
          ...props.output,
          sourceEdges: graph.createDataView(props.output.positions.buffer, {
            format: 'uint32',
            length: 8
          })
        }
      })
  ).toThrow(/byte ranges must not overlap/);
  const backing = createTransientView(graph, 'shared-counts', 'uint32', 65);
  const counts = graph.createDataView(backing.buffer, {format: 'uint32x4', length: 1});
  const validity = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 16
  });
  expect(() => new GPUSegmentSnap({...props, output: {...props.output, counts, validity}})).toThrow(
    /aligned binding byte ranges must not overlap/
  );
});

it('checks actual aligned binding sizes before declaring any node', () => {
  const graph = new GPUCommandGraph(device);
  const capacity = 16_777_216;
  const props = createProps(graph, capacity);
  const backing = createTransientView(graph, 'unaligned-output', 'float32', capacity * 2 + 1);
  props.output.positions = graph.createDataView(backing.buffer, {
    format: 'float32x2',
    length: capacity,
    byteOffset: 4
  });
  const contributor = new GPUSegmentSnap(props);
  const createKernel = vi.spyOn(kernelNodes, 'createWGSLKernelNode');
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^snap-contract output.positions chunk 0 needs 134217732 bytes/
  );
  expect(createKernel).not.toHaveBeenCalled();
  expect(allocate).not.toHaveBeenCalled();
});

it('rejects foreign graphs and emits legal WGSL identifiers for a finite tolerance', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUSegmentSnap(createProps(graph));
  expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(/target graph/);
  for (const {source} of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
    const code = stripWGSLComments(source);
    expect(findReservedIdentifiers(code)).toEqual([]);
    expect(code).not.toMatch(/\b(?:nan|inf)\b/i);
  }
});

function createBoundaryProps(
  graph: GPUCommandGraph,
  rows: number,
  references: number,
  lines: number,
  capacity = 1
) {
  const props = createProps(graph, capacity);
  return {
    ...props,
    positions: createTransientView(graph, 'boundary-positions', 'float32x2', rows),
    referencePositions: createTransientView(graph, 'boundary-references', 'float32x2', references),
    lineOffsets: createTransientView(graph, 'boundary-lines', 'uint32', lines + 1),
    output: {
      positions: createTransientView(graph, 'boundary-result', 'float32x2', lines * capacity),
      sourceEdges: createTransientView(graph, 'boundary-edges', 'uint32', lines * capacity),
      counts: createTransientView(graph, 'boundary-counts', 'uint32x4', lines),
      validity: createTransientView(graph, 'boundary-validity', 'uint32', lines)
    }
  };
}

it('independently bounds aggregate output coordinates across lines', () => {
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUSegmentSnap(createBoundaryProps(graph, 0, 0, 2, 0x40000000))).toThrow(
    /^snap-contract lineCount \* maximumVerticesPerLine.*2147483647/
  );
  const accepted = new GPUCommandGraph(device);
  expect(
    () => new GPUSegmentSnap(createBoundaryProps(accepted, 0, 0, 1, 0x7fffffff))
  ).not.toThrow();
});

it.each([
  ['positions.length', 0x80000000, 0, 1],
  ['referencePositions.length', 0, 0x80000000, 1],
  ['lineOffsets.length', 0, 0, 0x40000000]
] as const)('independently bounds %s', (name, rows, references, lines) => {
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUSegmentSnap(createBoundaryProps(graph, rows, references, lines))).toThrow(
    new RegExp(`^snap-contract ${name.replaceAll('.', '\\.')} must be`)
  );
});

it.each([
  [0x7fffffff, 0, 1],
  [0, 0x7fffffff, 1],
  [0, 0, 0x3fffffff]
])('accepts addressable metadata boundary rows=%s references=%s lines=%s', (rows, references, lines) => {
  const graph = new GPUCommandGraph(device);
  expect(
    () => new GPUSegmentSnap(createBoundaryProps(graph, rows, references, lines))
  ).not.toThrow();
});

it('accepts one buffer containing disjoint aligned writable fields', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'packed-output', 'uint32', 257);
  const output = {
    positions: graph.createDataView(packed.buffer, {format: 'float32x2', length: 8}),
    sourceEdges: graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 256
    }),
    counts: graph.createDataView(packed.buffer, {format: 'uint32x4', length: 1, byteOffset: 512}),
    validity: graph.createDataView(packed.buffer, {format: 'uint32', length: 1, byteOffset: 768}),
    status: {
      candidateOverflow: graph.createDataView(packed.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 1024
      })
    }
  };
  expect(() => new GPUSegmentSnap({...props, output}).getCommandNodes(graph)).not.toThrow();
});

it('counts empty-line reference validation in the exhaustive work bound', () => {
  const graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPUSegmentSnap({
        ...createBoundaryProps(graph, 0, 1025, 40_000),
        searchStructure: 'brute-force'
      })
  ).toThrow(/MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK/);
});

it('rejects conflicting candidate status aliases and validates the canonical length', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const candidateOverflow = createTransientView(graph, 'candidate-status', 'uint32', 1);
  const different = createTransientView(graph, 'other-candidate-status', 'uint32', 1);
  expect(
    () =>
      new GPUSegmentSnap({
        ...props,
        output: {
          ...props.output,
          status: {candidateOverflow},
          candidateOverflow: different
        }
      })
  ).toThrow(/^snap-contract output.status.candidateOverflow.*deprecated alias/);
  const wrongLength = createTransientView(graph, 'wrong-candidate-status', 'uint32', 2);
  expect(
    () =>
      new GPUSegmentSnap({
        ...props,
        output: {
          ...props.output,
          status: {candidateOverflow: wrongLength}
        }
      })
  ).toThrow(/^snap-contract output.status.candidateOverflow.length must equal lineCount/);
});

it('preflights STORAGE usage on the nested canonical status before graph mutation', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const diagnostic = graph.createDataView(
    graph.createTransientBuffer({
      id: 'non-storage-diagnostic',
      byteLength: 4,
      usage: Buffer.COPY_DST
    }),
    {format: 'uint32', length: 1}
  );
  const contributor = new GPUSegmentSnap({
    ...props,
    output: {...props.output, status: {candidateOverflow: diagnostic}}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('allows a separate-pass diagnostic beside input references and protects their logical words', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'references-and-diagnostic', 'uint32', 7);
  const referencePositions = graph.createDataView(packed.buffer, {format: 'float32x2', length: 3});
  const diagnostic = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 24
  });
  expect(() =>
    graph.add(
      new GPUSegmentSnap({
        ...props,
        referencePositions,
        output: {
          ...props.output,
          status: {candidateOverflow: diagnostic}
        }
      })
    )
  ).not.toThrow();
  const overlapping = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 20
  });
  expect(
    () =>
      new GPUSegmentSnap({
        ...props,
        referencePositions,
        output: {
          ...props.output,
          status: {candidateOverflow: overlapping}
        }
      })
  ).toThrow(/logical byte ranges/);
});
