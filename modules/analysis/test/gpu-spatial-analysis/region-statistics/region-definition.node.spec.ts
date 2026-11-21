// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength
} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-readback';
import {REGION_DEFINITION, REGION_MASK_COMPOSITION_GRID} from './region-definition-fixtures';
import {GPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-statistics';
import {GPURegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-mask';
import {GPUPickRegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-pick-region-mask';
import {
  GPUCommandGraph,
  GPUGridIndex,
  GraphDataView,
  createTransientView
} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectRegionDefinitionSummary} from './region-definition-assertions';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {createExampleMaskSelection, LASSO_EXAMPLE_SOURCES} from './review-example-wiring';

test('R14: the recovery query fits candidate capacity after conservative cell padding', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `padded-query-${serial++}`, format, length);
  const fixture = REGION_MASK_COMPOSITION_GRID;
  try {
    const positions = view('float32x2', 2);
    const grid = new GPUGridIndex({
      positions,
      gridSize: [fixture.width, 1],
      bounds: fixture.bounds,
      cellOffsets: view('uint32', fixture.width + 1),
      objectIds: view('uint32', 2),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    });
    const kernels = getGPUCommandNodeKernelSources(
      new GPURegionStatistics({
        positions,
        region: {kind: 'rectangle', bounds: view('float32', 4)},
        spatialIndex: {kind: 'grid', index: grid, candidateCapacity: 1},
        output: {summary: view('uint32', 8)}
      }).getCommandNodes(graph)
    );
    const rangeKernel = kernels.find(value => value.id.endsWith('-cell-ranges'))!;
    const scanKernels = kernels.filter(value => value.id.includes('-range-scan-'));
    const totalKernel = kernels.find(value => value.id.endsWith('-candidate-total'))!;
    expect(rangeKernel).toBeDefined();
    // One grid row plus its terminal zero makes a two-word scan: one 256-thread block.
    expect(scanKernels).toHaveLength(1);
    expect(totalKernel).toBeDefined();
    // Independent uniform-grid enumeration, including the documented one-cell halo:
    // the broad query has both rows; the recovery query must have only row zero.
    const candidateCount = Uint32Array.of(99),
      candidateOverflow = Uint32Array.of(99);
    for (const [query, expectedCount, expectedLoss] of [
      [fixture.initialQuery, 2, 1],
      [fixture.completeQuery, 1, 0]
    ] as const) {
      const rangeLength = new Uint32Array(2);
      const rangePrefix = new Uint32Array(2);
      const data: Record<string, ArrayBufferView> = {
        query,
        cellOffsets: fixture.cellOffsets,
        indexOverflow: Uint32Array.of(0),
        rangeStart: new Uint32Array(1),
        rangeLength,
        rangePrefix,
        inputValues: rangeLength,
        outputValues: rangePrefix,
        candidateCount,
        candidateOverflow
      };
      for (const kernel of [rangeKernel, ...scanKernels, totalKernel]) {
        const reflection = new WgslReflect(kernel.source);
        const bindings: Record<number, Record<number, ArrayBuffer>> = {};
        for (const storage of reflection.storage) {
          expect(data[storage.name], `${kernel.id} storage ${storage.name}`).toBeDefined();
          bindings[storage.group] ??= {};
          bindings[storage.group][storage.binding] = data[storage.name].buffer as ArrayBuffer;
        }
        for (const uniform of reflection.uniforms) {
          bindings[uniform.group] ??= {};
          bindings[uniform.group][uniform.binding] = new ArrayBuffer(uniform.size);
        }
        new WgslExec(
          new WgslParser().parse(kernel.source.replace(/^\s*_\s*=.*;\s*$/gm, ''))
        ).dispatchWorkgroups(reflection.entry.compute[0].name, 1, bindings, {
          constants: kernel.constants
        });
      }
      // The broad query spans both occupied columns; the recovery query plus its halo
      // includes column 1 but excludes column 3. The terminal length is always zero.
      expect(Array.from(rangeLength)).toEqual([expectedCount, 0]);
      expect(Array.from(rangePrefix)).toEqual([0, expectedCount]);
      expect.soft(Array.from(candidateCount)).toEqual([expectedCount]);
      expect.soft(Array.from(candidateOverflow)).toEqual([expectedLoss]);
    }
  } finally {
    device.destroy();
  }
});

for (const {name, source} of LASSO_EXAMPLE_SOURCES) {
  for (const kind of ['pick', 'shape']) {
    test(`R19: ${name} ${kind} construction forwards the producer loss view`, () => {
      const device = createNullWebGPUDevice(),
        graph = new GPUCommandGraph(device);
      const fixture = createExampleMaskSelection(
        device,
        graph,
        source,
        createTransientView(graph, 'example-picks', 'uint32', 4),
        kind === 'shape'
          ? {
              positions: createTransientView(graph, 'example-positions', 'float32x2', 2),
              region: {
                kind: 'polygon',
                vertices: createTransientView(graph, 'example-vertices', 'float32x2', 3),
                vertexCount: createTransientView(graph, 'example-vertex-count', 'uint32', 1)
              }
            }
          : undefined
      );
      try {
        expect(
          fixture.region.status?.overflow === fixture.producer.props.output.status.overflow
        ).toBe(true);
        const consumer = new GPURegionStatistics({
          region: fixture.region,
          output: {summary: createTransientView(graph, 'example-summary', 'uint32', 8)}
        });
        expect(
          getGPUCommandNodeKernelSources(consumer.getCommandNodes(graph)).some(kernel =>
            kernel.source.includes('upstreamOverflow[upstreamOverflowOffset]')
          )
        ).toBe(true);
      } finally {
        fixture.destroy();
        device.destroy();
      }
    });
  }
}

test('R17: mask status getter is captured once and preserves the first overflow view', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const first = {overflow: createTransientView(graph, 'getter-loss', 'uint32', 1)};
    let reads = 0;
    const region = {
      kind: 'mask' as const,
      mask: createTransientView(graph, 'getter-mask', 'uint32', 2),
      get status() {
        reads++;
        return reads === 1 ? first : ({} as typeof first);
      }
    };
    const contributor = new GPURegionStatistics({
      region,
      output: {summary: createTransientView(graph, 'getter-summary', 'uint32', 8)}
    });
    expect(reads).toBe(1);
    expect(contributor.props.region).toMatchObject({status: first});
  } finally {
    device.destroy();
  }
});

for (const indexed of [false, true]) {
  test(`R16: strided polygon positions are rejected before allocation (${indexed ? 'grid' : 'direct'})`, () => {
    const device = createNullWebGPUDevice(),
      graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32x2' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `strided-${serial++}`, format, length);
    try {
      const packed = view('float32x2', 3),
        backing = view('float32x2', 6);
      const positions = new GraphDataView(backing.buffer, {
        format: 'float32x2',
        length: 3,
        byteStride: 16,
        rowByteLength: 8,
        byteOffset: 0
      });
      const region = {
        kind: 'polygon' as const,
        vertices: view('float32x2', 3),
        vertexCount: view('uint32', 1)
      };
      const grid = indexed
        ? new GPUGridIndex({
            positions: packed,
            gridSize: [2, 2],
            bounds: [0, 0, 2, 2],
            cellOffsets: view('uint32', 5),
            objectIds: view('uint32', 3),
            count: view('uint32', 1),
            overflow: view('uint32', 1)
          })
        : undefined;
      const props = {
        id: 'strided-shape',
        region,
        positions,
        output: {summary: view('uint32', 8)},
        ...(grid ? {spatialIndex: {kind: 'grid' as const, index: grid, candidateCapacity: 3}} : {})
      };
      let allocations = 0;
      const allocate = graph.createTransientBuffer.bind(graph);
      graph.createTransientBuffer = descriptor => {
        allocations++;
        return allocate(descriptor);
      };
      expect(() => new GPURegionStatistics(props)).toThrow(/positions.*packed/);
      expect(allocations).toBe(0);
      expect(() =>
        new GPURegionStatistics({...props, positions: packed}).getCommandNodes(graph)
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

test('R14: composed statistics mask uses membership flags, excluding compact-ID loss', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const upstreamSummary = createTransientView(graph, 'membership-summary', 'uint32', 8);
    const producerOutput = {
      mask: createTransientView(graph, 'membership-mask', 'uint32', 2),
      summary: upstreamSummary,
      status: {
        overflow: createTransientView(graph, 'aggregate-loss', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'candidate-loss', 'uint32', 1)
      }
    };
    const contributor = new GPURegionStatistics({
      region: {kind: 'mask', ...producerOutput},
      output: {summary: createTransientView(graph, 'consumer-summary', 'uint32', 8)}
    });
    expect(contributor.props.region).toMatchObject({summary: upstreamSummary});
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    for (const flags of [1, 2, 4, 6]) {
      const input = new Uint32Array(8);
      input[7] = flags;
      const selected = flags & 4 ? 1 : 2,
        summary = new Uint32Array(8);
      const data: Record<string, ArrayBufferView> = {
        upstreamSummary: input,
        regionOverflow: new Uint32Array(1),
        counts: Uint32Array.of(selected, 0),
        summary
      };
      for (const kernel of [
        kernels.find(value => value.id.endsWith('-region-overflow'))!,
        kernels.find(value => value.id.endsWith('-summary'))!
      ]) {
        const reflection = new WgslReflect(kernel.source);
        const bindings: Record<number, Record<number, ArrayBuffer>> = {};
        for (const storage of reflection.storage) {
          bindings[storage.group] ??= {};
          bindings[storage.group][storage.binding] = data[storage.name].buffer as ArrayBuffer;
        }
        for (const uniform of reflection.uniforms) {
          bindings[uniform.group] ??= {};
          bindings[uniform.group][uniform.binding] = new ArrayBuffer(uniform.size);
        }
        new WgslExec(
          new WgslParser().parse(kernel.source.replace(/^\s*_\s*=.*;\s*$/gm, ''))
        ).dispatchWorkgroups(reflection.entry.compute[0].name, 1, bindings, {
          constants: kernel.constants
        });
      }
      expect(Array.from(summary)).toEqual([selected, 0, 0, 0, 0, 0, 0, flags & 6]);
    }
  } finally {
    device.destroy();
  }
});

test('R3: mask composition captures and consumes the producer overflow', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const overflow = createTransientView(graph, 'review-mask-loss', 'uint32', 1);
    const output = {
      mask: createTransientView(graph, 'review-mask', 'uint32', 3),
      status: {overflow}
    };
    const contributor = new GPURegionStatistics({
      region: {kind: 'mask', ...output},
      output: {summary: createTransientView(graph, 'review-summary', 'uint32', 8)}
    });
    output.status.overflow = createTransientView(graph, 'replacement-loss', 'uint32', 1);
    expect(contributor.props.region).toMatchObject({status: {overflow}});
    expect(
      getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).some(kernel =>
        kernel.source.includes('upstreamOverflow[upstreamOverflowOffset] != 0u')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

test('R4: reject an interleaved mask before graph allocation; accept a packed offset mask', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const backing = createTransientView(graph, 'interleaved-mask', 'uint32', 6);
    const mask = new GraphDataView(backing.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 4,
      byteStride: 8,
      rowByteLength: 4
    });
    const summary = createTransientView(graph, 'interleaved-summary', 'uint32', 8);
    expect(
      () => new GPURegionStatistics({region: {kind: 'mask', mask}, output: {summary}})
    ).toThrow(/mask.*packed/);
    const packed = new GraphDataView(backing.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 4,
      byteStride: 4,
      rowByteLength: 4
    });
    expect(
      () => new GPURegionStatistics({region: {kind: 'mask', mask: packed}, output: {summary}})
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

test('R10: rectangle, polygon and pick shapes fail in the outer constructor before allocating graph scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'shape-positions', 'float32x2', 3);
    const summary = createTransientView(graph, 'shape-summary', 'uint32', 8);
    const regions = [
      {kind: 'rectangle' as const, bounds: createTransientView(graph, 'bad-bounds', 'float32', 3)},
      {
        kind: 'polygon' as const,
        vertices: createTransientView(graph, 'bad-vertices', 'float32x2', 2),
        vertexCount: createTransientView(graph, 'bad-vertex-count', 'uint32', 1)
      },
      {kind: 'pick-region' as const, picks: createTransientView(graph, 'bad-picks', 'uint32', 2)}
    ];
    for (const region of regions) {
      expect
        .soft(
          () =>
            new GPURegionStatistics({id: 'shape-validation', positions, region, output: {summary}}),
          region.kind
        )
        .toThrow(/shape-validation/);
    }
    const valid = new GPURegionStatistics({
      id: 'shape-validation',
      positions,
      region: {kind: 'rectangle', bounds: createTransientView(graph, 'valid-bounds', 'float32', 4)},
      output: {summary}
    });
    expect(valid.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

test('region GPU summary comparator accepts the reported Metal division result', () => {
  const fixture = REGION_DEFINITION.square;
  // Lead's real GPU output: 21/7 differs from the NumPy encoding by one
  // representable float32 step. No implementation output was used as an oracle.
  const actual: number[] = [...fixture.words];
  actual[3] = 1077936129;
  expectRegionDefinitionSummary(actual, fixture);
  actual[3] = 1077936130;
  expectRegionDefinitionSummary(actual, fixture);
});

test('region GPU summary comparator rejects NaN, excessive error and the selected-row divisor', () => {
  const fixture = REGION_DEFINITION.square;
  const actual: number[] = [...fixture.words];
  for (const meanBits of [
    0x7fc00000,
    1077936131,
    new Uint32Array(Float32Array.of(2.625).buffer)[0]
  ]) {
    // 2.625 = 21/8 uses selectedCount rather than finite valueCount=7.
    actual[3] = meanBits;
    expect(() => expectRegionDefinitionSummary(actual, fixture)).toThrow();
  }
});

test('region GPU summary comparator keeps every other field and empty means exact', () => {
  const fixture = REGION_DEFINITION.square;
  fixture.words.forEach((_, word) => {
    if (word === 3) return;
    const actual: number[] = [...fixture.words];
    actual[word]++;
    expect(() => expectRegionDefinitionSummary(actual, fixture)).toThrow();
  });
  const empty: number[] = [...REGION_DEFINITION.empty.words];
  empty[3] = 1;
  expect(() => expectRegionDefinitionSummary(empty, REGION_DEFINITION.empty)).toThrow();
});

test('region contributors retain captured shapes, histogram domains, output ports and pick filters', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `region-capture-${serial++}`, format, length);
  try {
    const positions = view('float32x2', 12),
      bounds = view('float32', 4),
      summary = view('uint32', 12),
      overflow = view('uint32', 1);
    const region = {kind: 'rectangle' as const, bounds};
    const domain: [number, number] = [0, 8];
    const props = {
      region,
      positions,
      values: view('float32', 12),
      histogram: {binCount: 4, domain},
      output: {summary, status: {overflow}}
    };
    const statistics = new GPURegionStatistics(props);
    const maskOutput = {mask: view('uint32', 12), status: {overflow: view('uint32', 1)}};
    const mask = new GPURegionMask({positions, region, output: maskOutput});
    const pickProps = {
      picks: view('uint32', 8),
      batchIndex: 2,
      output: {mask: view('uint32', 12), status: {overflow: view('uint32', 1)}}
    };
    const pick = new GPUPickRegionMask(pickProps);
    region.bounds = view('float32', 4);
    domain[0] = 100;
    props.histogram.binCount = 1;
    props.output.summary = view('uint32', 9);
    props.output.status.overflow = view('uint32', 1);
    maskOutput.status.overflow = view('uint32', 1);
    pickProps.batchIndex = 7;
    expect(statistics.props.output.summary).toBe(summary);
    expect(statistics.props.output.status?.overflow).toBe(overflow);
    expect(statistics.props.histogram?.domain).toEqual([0, 8]);
    expect(statistics.props.histogram?.binCount).toBe(4);
    expect(mask.props.region).toEqual({kind: 'rectangle', bounds});
    expect(pick.props.batchIndex).toBe(2);
    expect(Object.isFrozen(mask.props.output.status)).toBe(true);
  } finally {
    device.destroy();
  }
});

for (const [name, fixture] of Object.entries(REGION_DEFINITION)) {
  test(`region decoder reads independent NumPy IEEE words: ${name}`, () => {
    // A byte-offset view checks decoding without depending on TypedArray alignment.
    const bytes = new Uint8Array(49);
    new DataView(bytes.buffer).setUint8(0, 99);
    const view = new DataView(bytes.buffer, 1);
    fixture.words.forEach((word, index) => view.setUint32(4 * index, word, true));
    const actual = decodeGPURegionStatistics(bytes.subarray(1));
    expect(actual).toEqual({
      selectedCount: fixture.words[0],
      valueCount: fixture.words[1],
      sum: fixture.sum,
      mean: fixture.mean,
      minimum: fixture.minimum,
      maximum: fixture.maximum,
      histogram: Uint32Array.from(fixture.words.slice(8)),
      histogramOutsideCount: fixture.words[6],
      selectionTruncated: Boolean(fixture.overflow),
      regionTruncated: false,
      candidatesTruncated: false
    });
    expect(getGPURegionStatisticsSummaryLength(4)).toBe(12);
  });
}
