// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {RunBuilder, RunFormat} from '../../../../../website/src/gpu/runs/define-run';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {GPUSpatialWeightsSummary} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-summary';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {
  getRowLocalAuditExpected,
  ROW_LOCAL_RULE_AUDIT_CAPACITY
} from './weights-helpers-audit-fixtures';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  getTransformAuditTolerance,
  expectTransformAuditResult
} from './weights-transform-audit-fixture';

it.each([
  'cpu-tiny',
  'tail-first',
  'tail-last'
] as const)('transform comparison rejects an erased value (%s)', kind => {
  if (kind === 'cpu-tiny')
    expect(() =>
      expectNumericArrayClose(
        [0],
        [2e-30],
        getTransformAuditTolerance('average-reciprocal', 1e-30, 'cpu')
      )
    ).toThrow();
  else
    expect(() =>
      expectTransformAuditResult(
        kind === 'tail-first' ? [2e20, 0, -99] : [2e20, -99, 0],
        [2e20],
        'average-reciprocal',
        1e20
      )
    ).toThrow();
});

it('retained packed union literals distinguish min/max from either operand', () => {
  const weights = (rule: 'left' | 'right' | 'min' | 'max') =>
    getRowLocalAuditExpected('union', rule).weights.slice(0, ROW_LOCAL_RULE_AUDIT_CAPACITY);
  for (const rule of ['min', 'max'] as const)
    for (const operand of ['left', 'right'] as const)
      expect(weights(rule)).not.toEqual(weights(operand));
  expect(ROW_LOCAL_RULE_AUDIT_CAPACITY).toBeLessThan(15);
});

it.each([
  {name: 'GPU', Operation: GPUSpatialWeightsValidationOperation},
  {name: 'CPU', Operation: CPUSpatialWeightsValidationOperation}
])('named validator captures one coherent getter configuration ($name)', ({Operation}) => {
  let weightReads = 0,
    outputReads = 0,
    statusReads = 0;
  const operation = new Operation({
    rowCount: 2,
    get weights() {
      weightReads++;
      return {
        offsets: `ofs${weightReads}`,
        neighbors: `ng${weightReads}`,
        weights: `wt${weightReads}`
      };
    },
    get output() {
      outputReads++;
      const version = outputReads;
      return {
        flags: `flags${version}`,
        get status() {
          statusReads++;
          return {invalidCount: `invalid${version}`};
        }
      };
    }
  });
  expect({weightReads, outputReads, statusReads}).toEqual({
    weightReads: 1,
    outputReads: 1,
    statusReads: 1
  });
  expect(operation.inputs).toEqual({offsets: 'ofs1', neighbors: 'ng1', weights: 'wt1'});
  const outputs = operation.contract.outputs;
  expect(typeof outputs).toBe('object');
  expect(Object.keys(outputs ?? {}).sort()).toEqual(['flags1', 'invalid1']);
});

it('contiguity derived pair growth saturates at uint32 and can be fed back without allocation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      criterion: 'queen' as const,
      polygons: {
        kind: 'polygons' as const,
        positions: createTransientView(graph, 'polygon-positions', 'float32x2', 4),
        ringOffsets: createTransientView(graph, 'polygon-rings', 'uint32', 2),
        polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 2),
        featureOffsets: createTransientView(graph, 'polygon-features', 'uint32', 3)
      },
      output: {
        offsets: createTransientView(graph, 'capacity-offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'capacity-neighbors', 'uint32', 1),
        weights: createTransientView(graph, 'capacity-values', 'float32', 1),
        status: {overflow: createTransientView(graph, 'capacity-overflow', 'uint32', 1)}
      }
    };
    const plan = new GPUContiguityWeights(props).getCapacityPlan({requiredCount: 0xffffffff});
    expect(plan.slots.capacity).toBe(0xffffffff);
    expect(plan.pairs.capacity).toBe(0xffffffff);
    expect(plan.pairs.estimatedRequiredCount).toBe(17179869180);
    expect(plan.pairs.limited).toBe(true);
    const retry = new GPUContiguityWeights({
      ...props,
      id: 'retry',
      pairCapacity: plan.pairs.capacity
    });
    expect(retry.getPairCapacity()).toBe(0xffffffff);
    const doubled = retry.getCapacityPlan({candidateOverflow: true});
    expect(doubled.pairs.capacity).toBe(0xffffffff);
    expect(doubled.pairs.estimatedRequiredCount).toBe(8589934590);
    expect(doubled.pairs.limited).toBe(true);
  } finally {
    device.destroy();
  }
});

it('separate-buffer chunk transform construction scales linearly in view inspections', () => {
  const inspections = (count: number) => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let reads = 0;
      const view = <Format extends 'uint32' | 'float32'>(
        id: string,
        format: Format,
        length: number
      ) => {
        const port = createTransientView(graph, id, format, length),
          handle = port.buffer;
        Object.defineProperty(port, 'buffer', {
          get() {
            reads++;
            return handle;
          }
        });
        return port;
      };
      const chunks = Array.from({length: count}, (_, index) => ({
        rowStart: index,
        rowCount: 1,
        weights: {
          offsets: view(`offset-${index}`, 'uint32', 2),
          neighbors: view(`neighbor-${index}`, 'uint32', 1),
          weights: view(`weight-${index}`, 'float32', 1)
        }
      }));
      const output = chunks.map((_, index) => view(`output-${index}`, 'float32', 1));
      reads = 0;
      const started = performance.now();
      new GPUChunkedSpatialWeightsTransform({
        operation: 'binary',
        weights: {rowCount: count, chunks},
        output
      });
      return {reads, milliseconds: performance.now() - started};
    } finally {
      device.destroy();
    }
  };
  const first = inspections(32),
    second = inspections(64);
  expect(second.reads).toBeLessThanOrEqual(first.reads * 2.5 + 256);
  const sizes = [256, 512, 1024];
  const measured = sizes.map(count => ({chunks: count, ...inspections(count)}));
  for (let index = 1; index < measured.length; index++)
    expect(measured[index].reads).toBeLessThanOrEqual(measured[index - 1].reads * 2.1);
  console.info('weights transform construction scaling', JSON.stringify(measured));
});

it('chunk transform still rejects a destination overwriting another chunk input', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const chunks = Array.from({length: 2}, (_, index) => ({
      rowStart: index,
      rowCount: 1,
      weights: {
        offsets: createTransientView(graph, `cross-offset-${index}`, 'uint32', 2),
        neighbors: createTransientView(graph, `cross-neighbor-${index}`, 'uint32', 1),
        weights: createTransientView(graph, `cross-weight-${index}`, 'float32', 1)
      }
    }));
    expect(
      () =>
        new GPUChunkedSpatialWeightsTransform({
          operation: 'binary',
          weights: {rowCount: 2, chunks},
          output: [
            chunks[1].weights.weights,
            createTransientView(graph, 'other-destination', 'float32', 1)
          ]
        })
    ).toThrow(/overlap|share|buffer/);
  } finally {
    device.destroy();
  }
});

it.each([
  [false, 'binary'],
  [true, 'binary'],
  [false, 'row'],
  [true, 'row']
] as const)('%s chunked %s transform allows an aligned disjoint distance/output buffer', (chunked, operation) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const packed = createTransientView(graph, 'packed-distance', 'float32', 66);
    const distances = graph.createDataView(packed.buffer, {format: 'float32', length: 2}),
      output = graph.createDataView(packed.buffer, {format: 'float32', byteOffset: 256, length: 2});
    const weights = {
      offsets: createTransientView(graph, 'binary-offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'binary-neighbors', 'uint32', 2),
      weights: createTransientView(graph, 'binary-weights', 'float32', 2),
      distances
    };
    expect(() => {
      const transform = chunked
        ? new GPUChunkedSpatialWeightsTransform({
            operation,
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            output: [output]
          })
        : new GPUSpatialWeightsTransform({operation, weights, output});
      transform.getCommandNodes(graph);
    }).not.toThrow();
    expect(() => new GPUSpatialWeightsTransform({operation, weights, output: distances})).toThrow(
      /overlap|share|buffer/
    );
    expect(
      () => new GPUSpatialWeightsTransform({operation: 'kernel', bandwidth: 1, weights, output})
    ).toThrow(/share|buffer/);
  } finally {
    device.destroy();
  }
});

it.each([
  'transpose',
  'summary',
  'validation'
] as const)('%s refuses aliases with other published data or input counts', kind => {
  const device = createNullWebGPUDevice();
  let physical: Buffer | undefined;
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `published-${serial++}`, format, length);
    const backing = view('uint32', 2),
      count = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
    const weights = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2),
      status: {count}
    };
    if (kind === 'summary')
      expect(
        () =>
          new GPUSpatialWeightsSummary({
            weights,
            output: {
              statistics: view('float32', 3),
              counts: view('uint32', 6),
              cardinality: backing
            }
          })
      ).toThrow(/share|overlap|buffer/);
    else if (kind === 'validation')
      expect(
        () => new GPUSpatialWeightsValidation({weights, output: {status: {invalidCount: count}}})
      ).toThrow(/share|overlap|buffer/);
    else {
      physical = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
      const offsets = importGraphBuffer(graph, 'first-import', physical, 'uint32', 3),
        neighbors = importGraphBuffer(graph, 'second-import', physical, 'uint32', 2);
      expect(
        () =>
          new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            output: {
              chunks: [
                {
                  rowStart: 0,
                  rowCount: 2,
                  weights: {
                    offsets,
                    neighbors,
                    weights: view('float32', 2),
                    status: {
                      count: view('uint32', 1),
                      requiredCount: view('uint32', 1),
                      overflow: view('uint32', 1)
                    }
                  }
                }
              ]
            }
          })
      ).toThrow(/share|overlap|buffer/);
    }
  } finally {
    physical?.destroy();
    device.destroy();
  }
});

it.each([
  false,
  true
])('rectangular transpose clears double normalization after dropping mass (chunked=%s)', chunked => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `mass-${serial++}`, format, length);
    const information = {rowCount: 2, transform: 'double' as const, symmetric: true as const};
    const weights = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    const output = {
      offsets: view('uint32', 2),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    const transpose = chunked
      ? new GPUChunkedSpatialWeightsTranspose({
          weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}], information},
          columnCount: 1,
          output
        })
      : new GPUSpatialWeightsTranspose({
          weights: {...weights, information},
          columnCount: 1,
          output
        });
    expect(transpose.outputWeights.information?.transform).toBe('unknown');
  } finally {
    device.destroy();
  }
});

it.each([
  'plain',
  'getter',
  'chunk'
] as const)('validation preserves subtype views in both public descriptions (%s)', kind => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let reads = 0;
    const neighborCounts = createTransientView(graph, 'counts', 'uint32', 2);
    const base = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
      weights: createTransientView(graph, 'weights', 'float32', 2)
    };
    const context = {
      coordinateSpace: 'planar' as const,
      metric: 'native' as const,
      tag: 'captured',
      frame: {
        id: 'w5-frame',
        kind: 'local-azimuthal-equidistant' as const,
        origin: [0, 0] as [number, number],
        tag: 'frame-captured'
      }
    };
    type CountedWeights = GPUSpatialWeights & {
      neighborCounts: GraphDataView<'uint32'>;
      spatialContext: typeof context;
    };
    const counted = (
      kind === 'getter'
        ? Object.assign(
            Object.create({
              get neighborCounts() {
                reads++;
                return neighborCounts;
              }
            }),
            base
          )
        : {...base, neighborCounts}
    ) as CountedWeights;
    counted.spatialContext = context;
    const weights =
      kind === 'chunk'
        ? {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: counted}]}
        : counted;
    const validation = new GPUSpatialWeightsValidation({
      weights,
      output: {status: {invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)}}
    });
    context.tag = 'mutated';
    context.frame.tag = 'mutated';
    context.frame.origin[0] = 99;
    for (const publicWeights of [validation.props.weights, validation.outputWeights]) {
      const port = 'chunks' in publicWeights ? publicWeights.chunks[0].weights : publicWeights;
      expect(port.neighborCounts === neighborCounts).toBe(true);
      expect(port.spatialContext.tag).toBe('captured');
      expect(port.spatialContext.frame.tag).toBe('frame-captured');
      expect(port.spatialContext.frame.origin).toEqual([0, 0]);
    }
    if (kind === 'getter') expect(reads).toBe(1);
    expect(() => validation.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

vi.mock('../../../../../website/src/gpu/runs/define-run', () => ({
  defineOperationRun: (specification: unknown) => specification
}));

it.each([
  'spatial-weights-transpose',
  'chunked-weights-transpose'
])('website %s builds its actual transpose graph', async name => {
  const modulePath = `../../../../../website/src/gpu/runs/entries/regression-weights/${name}`;
  const scene = (await import(modulePath)).default as {
    createInputs(context: unknown): unknown;
    build(builder: RunBuilder, inputs: unknown, values: Record<string, number>): unknown;
  };
  const device = createNullWebGPUDevice();
  const buffers: Buffer[] = [];
  try {
    const graph = new GPUCommandGraph(device);
    const add = vi.spyOn(graph, 'add');
    const allocate = <Format extends RunFormat>(id: string, format: Format, length: number) => {
      const components = Number(format.split('x')[1] ?? 1);
      const buffer = device.createBuffer({
        byteLength: Math.max(length * components * 4, 4),
        usage: Buffer.STORAGE
      });
      buffers.push(buffer);
      return {
        id,
        format,
        length,
        buffer,
        view: createTransientView(graph, id, format, length),
        write: () => {}
      };
    };
    const builder: RunBuilder = {
      device,
      graph,
      allocate,
      upload: (id, format, values) =>
        allocate(id, format, values.length / Number(format.split('x')[1] ?? 1)),
      own: resource => resource
    };
    const inputs = scene.createInputs({
      seed: 17,
      random: () => 0.375,
      world: {width: 1, height: 1}
    });
    expect(() => scene.build(builder, inputs, {scatter: 0.4, k: 3, chunkCount: 4})).not.toThrow();
    const transpose = add.mock.calls
      .map(([producer]) => producer)
      .find(
        producer =>
          typeof producer === 'object' &&
          producer &&
          'id' in producer &&
          String(producer.id).includes('weights-transpose')
      ) as {props: {output: {status?: unknown; distances?: unknown}}} | undefined;
    expect(transpose).toBeDefined();
    expect(transpose?.props.output.status).toBeUndefined();
    expect(transpose?.props.output.distances).toBeDefined();
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
});
