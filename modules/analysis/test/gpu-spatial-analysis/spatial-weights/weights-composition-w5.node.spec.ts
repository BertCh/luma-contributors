// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createReviewSetup} from './audit-weights-a-review.test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {executeCommandNodesOnCPU} from '../segment-intersection/wgsl-graph-executor';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import {snapshotWeightsSpatialContext} from '../../../src/gpu-spatial-analysis/spatial-weights/weights-construction-snapshot';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';

useCPUExecutorTestHooks();

for (const route of [
  'whole-transform',
  'chunk-transform',
  'whole-transpose',
  'packed-transpose',
  'chunk-transpose'
] as const)
  for (const flags of [false, true])
    for (const child of route.startsWith('whole') ? [false] : [false, true])
      it(`CPL-03: ${route} forwards ${child ? 'child' : 'root'} completeness with negative flags=${flags}`, () => {
        const setup = createReviewSetup();
        try {
          const complete = setup.input(Uint32Array.of(1), 'uint32');
          const status = {
            complete,
            ...(flags
              ? {
                  overflow: setup.input(Uint32Array.of(0), 'uint32'),
                  candidateOverflow: setup.input(Uint32Array.of(0), 'uint32')
                }
              : {})
          };
          const base = {
            offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
            weights: setup.input(Float32Array.of(2, 4), 'float32')
          };
          const whole = {...base, status};
          const chunks = {
            rowCount: 2,
            ...(child ? {} : {status}),
            chunks: [{rowStart: 0, rowCount: 2, weights: {...base, ...(child ? {status} : {})}}]
          };
          const output = setup.output();
          const producer =
            route === 'whole-transform'
              ? new GPUSpatialWeightsTransform({
                  operation: 'binary',
                  weights: whole,
                  output: setup.view('float32', 2)
                })
              : route === 'chunk-transform'
                ? new GPUChunkedSpatialWeightsTransform({
                    operation: 'binary',
                    weights: chunks,
                    output: [setup.view('float32', 2)]
                  })
                : route === 'whole-transpose'
                  ? new GPUSpatialWeightsTranspose({weights: whole, output: setup.weights()})
                  : route === 'packed-transpose'
                    ? new GPUChunkedSpatialWeightsTranspose({
                        weights: chunks,
                        output: setup.weights()
                      })
                    : new GPUChunkedSpatialWeightsTranspose({
                        weights: chunks,
                        output: {
                          chunks: [
                            {
                              rowStart: 0,
                              rowCount: 2,
                              weights: {
                                ...output,
                                status: {...output.status, count: setup.view('uint32', 1)}
                              }
                            }
                          ]
                        }
                      });
          const weights = producer.outputWeights;
          const consumer =
            'chunks' in weights
              ? new GPUChunkedSpatialWeightsAlgebra({
                  operation: 'selfWeight',
                  selfWeight: 1,
                  weights,
                  output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4)}]}
                })
              : new GPUSpatialWeightsAlgebra({
                  operation: 'selfWeight',
                  selfWeight: 1,
                  weights,
                  output: setup.output(2, 4)
                });
          const nodes = [
            ...producer.getCommandNodes(setup.graph),
            ...consumer.getCommandNodes(setup.graph)
          ];
          // A positive certificate alone proves loss at zero; both negative flags stay zero.
          for (const value of [1, 0, 1]) {
            setup.memory.write(complete, Uint32Array.of(value));
            executeCommandNodesOnCPU(nodes, setup.memory);
            expect(
              Array.from(setup.memory.readUint32(consumer.outputWeights.status!.candidateOverflow!))
            ).toEqual([1 - value]);
          }
        } finally {
          setup.device.destroy();
        }
      });

it.each([
  false,
  true
])('W5-02: distance algebra rejects implicit meters mixed with kilometers in either order (reverse=%s)', reverse => {
  const setup = createReviewSetup();
  try {
    const contexts: GPUSpatialContext[] = [
      {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'kilometers'}
    ];
    if (reverse) contexts.reverse();
    const operands = contexts.map((spatialContext, index) => ({
      rowCount: 2,
      spatialContext,
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: {
            ...setup.weights(2, 2, true),
            distances: setup.input(Float32Array.of(index ? 0.007 : 7, 0), 'float32')
          }
        }
      ]
    }));
    expect(
      () =>
        new GPUChunkedSpatialWeightsAlgebra({
          operation: 'union',
          left: operands[0],
          right: operands[1],
          output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4, true)}]}
        })
    ).toThrow(/units/);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  false,
  true
])('W5-02: resolved contexts cannot assemble ellipsoidal linear edges (reverse=%s)', reverse => {
  const setup = createReviewSetup();
  try {
    const contexts: GPUSpatialContext[] = [
      {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel: 'linear'},
      {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'}
    ];
    if (reverse) contexts.reverse();
    const operands = contexts.map(spatialContext => ({
      rowCount: 2,
      spatialContext,
      chunks: [{rowStart: 0, rowCount: 2, weights: setup.weights()}]
    }));
    expect(
      () =>
        new GPUChunkedSpatialWeightsAlgebra({
          operation: 'union',
          left: operands[0],
          right: operands[1],
          output: {chunks: [{rowStart: 0, rowCount: 2, weights: setup.output(2, 4)}]}
        })
    ).toThrow(/metric|ellipsoidal/);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  false,
  true
])('W5-02: kernel bandwidth uses metric units; binary remains coordinate-only (chunked=%s)', chunked => {
  const setup = createReviewSetup();
  try {
    const weights = {
      ...setup.weights(2, 2, true),
      spatialContext: {
        coordinateSpace: 'longitude-latitude' as const,
        metric: 'great-circle' as const
      }
    };
    const spatialContext: GPUSpatialContext = {...weights.spatialContext, units: 'kilometers'};
    const construct = (operation: 'binary' | 'kernel') =>
      chunked
        ? new GPUChunkedSpatialWeightsTransform({
            operation,
            bandwidth: 1,
            spatialContext,
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]}
          })
        : new GPUSpatialWeightsTransform({operation, bandwidth: 1, spatialContext, weights});
    expect(() => construct('binary')).not.toThrow();
    expect(() => construct('kernel')).toThrow(/units/);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  'parent',
  'child',
  'undefined',
  'override'
] as const)('W5-03: binary to adaptive merges defined provenance (%s)', kind => {
  const setup = createReviewSetup();
  try {
    const information = {rowCount: 2, transform: 'raw' as const, symmetric: 'unknown' as const};
    const weights = {
      offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
      neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
      weights: setup.input(Float32Array.of(1, 1), 'float32'),
      distances: setup.input(Float32Array.of(7, 11), 'float32'),
      ...(kind === 'parent'
        ? {}
        : {
            information: {
              ...information,
              provenance: {source: 'review', criterion: kind === 'undefined' ? undefined : 'knn'}
            }
          })
    };
    const binary = new GPUChunkedSpatialWeightsTransform({
      operation: 'binary',
      weights: {
        rowCount: 2,
        ...(kind === 'child'
          ? {}
          : {
              information: {
                ...information,
                provenance: {source: 'review', criterion: kind === 'override' ? 'distance' : 'knn'}
              }
            }),
        chunks: [{rowStart: 0, rowCount: 2, weights}]
      }
    });
    const adaptive = new GPUChunkedSpatialWeightsTransform({
      operation: 'kernel',
      kernel: 'uniform',
      bandwidth: 'adaptive',
      weights: binary.outputWeights
    });
    executeCommandNodesOnCPU(
      [...binary.getCommandNodes(setup.graph), ...adaptive.getCommandNodes(setup.graph)],
      setup.memory
    );
    // Uniform K(d/h)=1/2 at the sole (k-th) neighbor, independently of d.
    expect(
      Array.from(setup.memory.readFloat32(adaptive.outputWeights.chunks[0].weights.weights))
    ).toEqual([0.5, 0.5]);
  } finally {
    setup.device.destroy();
  }
});

it('W5-04: symmetrization refuses disjoint neighbor/output views sharing physical storage', () => {
  const setup = createReviewSetup();
  try {
    const backing = setup.view('uint32', 66);
    const neighbors = setup.graph.createDataView(backing.buffer, {format: 'uint32', length: 2});
    const output = setup.graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 256,
      length: 2
    });
    expect(
      () =>
        new GPUChunkedSpatialWeightsTransform({
          operation: 'average-reciprocal',
          weights: {
            rowCount: 2,
            chunks: [{rowStart: 0, rowCount: 2, weights: {...setup.weights(), neighbors}}]
          },
          output: [output]
        })
    ).toThrow(/buffer|share/);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  false,
  true
])('W5-05: validation protects published neighborCounts before allocation (chunked=%s)', chunked => {
  const setup = createReviewSetup();
  try {
    const neighborCounts = setup.input(Uint32Array.of(1, 1), 'uint32');
    const base = {...setup.weights(), neighborCounts};
    const weights = chunked
      ? {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: base}]}
      : base;
    expect(
      () =>
        new GPUSpatialWeightsValidation({
          weights,
          output: {flags: neighborCounts, status: {invalidCount: setup.view('uint32', 1)}}
        })
    ).toThrow(/overlap|buffer|share/);
    expect(Array.from(setup.memory.readUint32(neighborCounts))).toEqual([1, 1]);
  } finally {
    setup.device.destroy();
  }
});

it.each([
  false,
  true
])('W5-06: square row count alone cannot prove full-domain double normalization (chunked=%s)', chunked => {
  const setup = createReviewSetup();
  try {
    const base = {
      ...setup.weights(),
      information: {rowCount: 2, transform: 'double' as const, symmetric: 'unknown' as const}
    };
    const operation = chunked
      ? new GPUChunkedSpatialWeightsTranspose({
          weights: {
            rowCount: 2,
            information: base.information,
            chunks: [{rowStart: 0, rowCount: 2, weights: base}]
          },
          columnCount: 2,
          output: setup.weights()
        })
      : new GPUSpatialWeightsTranspose({weights: base, columnCount: 2, output: setup.weights()});
    expect(operation.outputWeights.information?.transform).toBe('unknown');
  } finally {
    setup.device.destroy();
  }
});

it.each([
  false,
  true
])('W5-05: validation retains legal disjoint neighborCounts/flags storage (chunked=%s)', chunked => {
  const setup = createReviewSetup();
  try {
    const backing = setup.input(new Uint32Array(66).fill(1), 'uint32');
    const neighborCounts = setup.graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 2
    });
    const flags = setup.graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 2
    });
    const base = {
      offsets: setup.input(Uint32Array.of(0, 1, 2), 'uint32'),
      neighbors: setup.input(Uint32Array.of(1, 0), 'uint32'),
      weights: setup.input(Float32Array.of(1, 1), 'float32'),
      neighborCounts
    };
    const weights = chunked
      ? {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: base}]}
      : base;
    const operation = new GPUSpatialWeightsValidation({
      weights,
      output: {flags, status: {invalidCount: setup.view('uint32', 1)}}
    });
    // Row-group parameter binding is outside this interpreter; the GPU companion pins writes.
    expect(() => operation.getCommandNodes(setup.graph)).not.toThrow();
    expect(Array.from(setup.memory.readUint32(neighborCounts))).toEqual([1, 1]);
  } finally {
    setup.device.destroy();
  }
});

it('W5-03: a defined non-knn child overrides a knn parent', () => {
  const setup = createReviewSetup();
  try {
    const information = {rowCount: 2, transform: 'raw' as const, symmetric: 'unknown' as const};
    const binary = new GPUChunkedSpatialWeightsTransform({
      operation: 'binary',
      weights: {
        rowCount: 2,
        information: {...information, provenance: {source: 'review', criterion: 'knn'}},
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              ...setup.weights(2, 2, true),
              information: {...information, provenance: {source: 'review', criterion: 'distance'}}
            }
          }
        ]
      }
    });
    expect(binary.outputWeights.chunks[0].weights.information?.provenance?.criterion).toBe(
      'distance'
    );
    expect(
      () =>
        new GPUChunkedSpatialWeightsTransform({
          operation: 'kernel',
          bandwidth: 'adaptive',
          weights: binary.outputWeights
        })
    ).toThrow(/needs kNN/);
  } finally {
    setup.device.destroy();
  }
});

it('W5-07: context captures inherited extensions and nested getters once without cloning GPU views', () => {
  const setup = createReviewSetup();
  try {
    const view = setup.view('uint32', 1);
    let tag = 'original',
      tagReads = 0,
      axisReads = 0,
      unitReads = 0;
    const ellipsoid = {
      get semiMajorAxis() {
        axisReads++;
        return 6378137;
      },
      flattening: 0
    };
    const unit = {
      get name() {
        unitReads++;
        return 'meter';
      },
      metersPerUnit: 1
    };
    const context = Object.assign(
      Object.create({
        get tag() {
          tagReads++;
          return tag;
        }
      }),
      {
        coordinateSpace: 'planar',
        metric: 'native',
        view,
        ellipsoid,
        frame: {id: 'frame', kind: 'crs', crs: 'EPSG:3857', unit, origin: [0, 0]}
      }
    ) as GPUSpatialContext & {tag: string; view: typeof view};
    const captured = snapshotWeightsSpatialContext(context) as typeof context;
    tag = 'changed';
    context.frame!.origin![0] = 42;
    expect(captured.tag).toBe('original');
    expect(captured.view).toBe(view);
    expect(captured.frame?.origin).toEqual([0, 0]);
    expect({tagReads, axisReads, unitReads}).toEqual({tagReads: 1, axisReads: 1, unitReads: 1});
    expect(Object.isFrozen(captured.frame?.unit)).toBe(true);
    expect(Object.isFrozen(captured.ellipsoid)).toBe(true);
  } finally {
    setup.device.destroy();
  }
});
