// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {GPUAssuncaoReisStandardization} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-rate-standardization';
import {GPULOSHResidualPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-losh-residual-permutation-test';
import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect, WgslParser, WgslExec} from 'wgsl_reflect';
import {getLocalVariabilityMomentNodes} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/local-variability-moments';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test';
import {
  GPULocalSpatialHeteroscedasticity,
  GPUMultivariateLocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {GPURegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-mask';
import {GPUPickRegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-pick-region-mask';
import {GPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-statistics';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPULatticeWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-lattice-weights';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createScene(sizes: readonly number[] = [5]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `continuation-${serial++}`, format, length);
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
  return {device, graph, rows, view, weights: chunks[0].weights, chunked: {rowCount: rows, chunks}};
}

for (const Contributor of [GPULocalSpatialHeteroscedasticity, GPUMultivariateLocalGeary]) {
  it(`${Contributor.name} accepts 30M rows, has fixed chunk pipelines, and passes the full reserved-word audit`, () => {
    const record = (sizes: readonly number[]) => {
      const {device, graph, rows, view, chunked} = createScene(sizes);
      try {
        const nodes = new Contributor({
          weights: chunked,
          values: view('float32', rows),
          variableCount: 1,
          output: {
            statistic: view('float32', rows),
            status: view('uint32', rows),
            residualCount: view('uint32', 1),
            count: view('uint32', 1)
          }
        }).getCommandNodes(graph);
        const sources = getGPUCommandNodeKernelSources(nodes);
        for (const {source} of sources) {
          expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
          expect(() => new WgslReflect(source)).not.toThrow();
        }
        return {
          sources: new Set(sources.map(item => item.source)),
          pipelines: getGPUCommandNodePipelineKeys(nodes)
        };
      } finally {
        device.destroy();
      }
    };
    record([30_000_000]);
    const packed = record([769]);
    const chunked = record([1, 255, 256, 257]);
    expect(chunked.sources.size).toBe(packed.sources.size);
    expect(chunked.pipelines.size).toBe(packed.pipelines.size);
  });
  it(`${Contributor.name} rejects overlapping outputs`, () => {
    const {device, graph, rows, view, weights} = createScene();
    try {
      const result = view('float32', rows);
      const status = graph.createDataView(result.buffer, {format: 'uint32', length: rows});
      expect(
        () =>
          new Contributor({
            weights,
            values: view('float32', rows),
            variableCount: 1,
            output: {statistic: result, status}
          })
      ).toThrow(/overlap/);
    } finally {
      device.destroy();
    }
  });
}

for (const kind of [
  'losh',
  'multivariate',
  'region-mask',
  'region-statistics',
  'pick-mask',
  'lag',
  'transform',
  'chunked-transform',
  'algebra',
  'validation',
  'lattice'
] as const) {
  it(`${kind} checks caller STORAGE usage before any graph declaration`, () => {
    const {device, graph, rows, view, weights, chunked} = createScene();
    const buffer = device.createBuffer({byteLength: rows * 8, usage: Buffer.COPY_DST});
    try {
      const invalidFloat = importGraphBuffer(graph, 'invalid-float', buffer, 'float32', rows);
      const invalidInteger = importGraphBuffer(graph, 'invalid-integer', buffer, 'uint32', rows);
      const invalidPositions = importGraphBuffer(
        graph,
        'invalid-positions',
        buffer,
        'float32x2',
        rows
      );
      const outputWeights = {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows),
        status: {overflow: view('uint32', 1)}
      };
      const outputMask = {mask: view('uint32', rows), status: {overflow: view('uint32', 1)}};
      const variability = {
        weights,
        values: invalidFloat,
        output: {statistic: view('float32', rows), status: view('uint32', rows)}
      };
      const contributor =
        kind === 'losh'
          ? new GPULocalSpatialHeteroscedasticity(variability)
          : kind === 'multivariate'
            ? new GPUMultivariateLocalGeary({...variability, variableCount: 1})
            : kind === 'region-mask'
              ? new GPURegionMask({
                  positions: invalidPositions,
                  region: {kind: 'rectangle', bounds: view('float32', 4)},
                  output: outputMask
                })
              : kind === 'region-statistics'
                ? new GPURegionStatistics({
                    region: {kind: 'mask', mask: invalidInteger},
                    values: view('float32', rows),
                    output: {summary: view('uint32', 8)}
                  })
                : kind === 'pick-mask'
                  ? new GPUPickRegionMask({picks: invalidInteger, output: outputMask})
                  : kind === 'lag'
                    ? new GPUSpatialLag({
                        weights,
                        values: invalidFloat,
                        output: view('float32', rows)
                      })
                    : kind === 'transform'
                      ? new GPUSpatialWeightsTransform({
                          weights: {...weights, weights: invalidFloat},
                          operation: 'double',
                          output: view('float32', rows)
                        })
                      : kind === 'chunked-transform'
                        ? new GPUChunkedSpatialWeightsTransform({
                            weights: {
                              ...chunked,
                              chunks: [
                                {...chunked.chunks[0], weights: {...weights, weights: invalidFloat}}
                              ]
                            },
                            operation: 'double',
                            output: [view('float32', rows)]
                          })
                        : kind === 'algebra'
                          ? new GPUSpatialWeightsAlgebra({
                              operation: 'union',
                              left: {...weights, weights: invalidFloat},
                              right: weights,
                              output: outputWeights
                            })
                          : kind === 'validation'
                            ? new GPUSpatialWeightsValidation({
                                weights: {...weights, weights: invalidFloat},
                                output: {status: {invalidCount: view('uint32', 1)}}
                              })
                            : new GPULatticeWeights({
                                width: rows,
                                height: 1,
                                criterion: 'rook',
                                mask: invalidInteger,
                                output: outputWeights
                              });
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const declare = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(declare).not.toHaveBeenCalled();
    } finally {
      buffer.destroy();
      device.destroy();
    }
  });
}

it('region histogram domains reject nonfinite float32 literals at construction', () => {
  const {device, view, rows} = createScene();
  try {
    expect(
      () =>
        new GPURegionStatistics({
          region: {kind: 'mask', mask: view('uint32', rows)},
          values: view('float32', rows),
          histogram: {binCount: 2, domain: [0, 1e300]},
          output: {summary: view('uint32', 10)}
        })
    ).toThrow(/finite float32/);
  } finally {
    device.destroy();
  }
});

it('multivariate conditional inference keeps chunk, row-tile and replicate-tile pipelines constant', () => {
  const record = (sizes: readonly number[], rowTileSize: number, permutationTileSize: number) => {
    const {device, graph, rows, view, chunked} = createScene(sizes);
    try {
      const nodes = new GPUMultivariateLocalGearyPermutationTest({
        weights: chunked,
        values: view('float32', rows * 2),
        variableCount: 2,
        strategy: 'tiled',
        permutationCount: 3,
        rowTileSize,
        permutationTileSize,
        output: {
          statistic: view('float32', rows),
          pseudoPValues: view('float32', rows),
          classes: view('uint32', rows)
        }
      }).getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const {source} of sources) {
        expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
        expect(() => new WgslReflect(source)).not.toThrow();
      }
      return {
        sources: new Set(sources.map(item => item.source)),
        pipelines: getGPUCommandNodePipelineKeys(nodes)
      };
    } finally {
      device.destroy();
    }
  };
  const packed = record([769], 128, 3);
  const chunked = record([1, 255, 256, 257], 17, 1);
  expect(chunked.sources.size).toBe(packed.sources.size);
  expect(chunked.pipelines.size).toBe(packed.pipelines.size);
});

it('shared CSR output validation rejects aligned aliases between neighbor IDs and weights', () => {
  const {device, graph, rows, view} = createScene();
  try {
    const shared = view('uint32', rows * 2);
    const neighbors = graph.createDataView(shared.buffer, {format: 'uint32', length: rows});
    const weights = graph.createDataView(shared.buffer, {
      format: 'float32',
      length: rows,
      byteOffset: rows * 4
    });
    expect(
      () =>
        new GPULatticeWeights({
          width: rows,
          height: 1,
          criterion: 'rook',
          output: {
            offsets: view('uint32', rows + 1),
            neighbors,
            weights,
            status: {overflow: view('uint32', 1)}
          }
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('chunk transforms reject logical output overlap while allowing consecutive unaligned chunks', () => {
  const {device, graph, rows, view, chunked} = createScene([2, 3]);
  try {
    const shared = view('float32', rows);
    const first = graph.createDataView(shared.buffer, {format: 'float32', length: 2});
    const consecutive = graph.createDataView(shared.buffer, {
      format: 'float32',
      length: 3,
      byteOffset: 8
    });
    expect(
      () =>
        new GPUChunkedSpatialWeightsTransform({
          weights: chunked,
          operation: 'double',
          output: [first, consecutive]
        })
    ).not.toThrow();
    const overlapping = graph.createDataView(shared.buffer, {
      format: 'float32',
      length: 3,
      byteOffset: 4
    });
    expect(
      () =>
        new GPUChunkedSpatialWeightsTransform({
          weights: chunked,
          operation: 'double',
          output: [first, overlapping]
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('pick masks and region status ports reject output aliases', () => {
  const {device, graph, view, rows} = createScene();
  try {
    const shared = view('uint32', rows);
    // Status ports hold exactly one row, so alias the first row of the shared buffer.
    const sharedStatus = graph.createDataView(shared.buffer, {format: 'uint32', length: 1});
    expect(
      () =>
        new GPUPickRegionMask({
          picks: view('uint32', 6),
          output: {mask: shared, status: {overflow: sharedStatus}}
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPURegionMask({
          positions: view('float32x2', rows),
          region: {kind: 'rectangle', bounds: view('float32', 4)},
          output: {
            mask: view('uint32', rows),
            status: {
              overflow: view('uint32', 1),
              antimeridianUncertainCount: sharedStatus,
              poleUncertainCount: sharedStatus
            }
          }
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('region mask and counting kernels keep fixed pipelines and each unequal chunk dispatch size', () => {
  const record = (sizes: readonly number[]) => {
    const {device, graph, rows, view} = createScene(sizes);
    try {
      const maskNodes = new GPURegionMask({
        id: 'mask-review',
        positions: new GraphVectorView({
          id: 'position-chunks',
          name: 'positions',
          format: 'float32x2',
          length: rows,
          valueLength: rows * 2,
          stride: 2,
          byteStride: 8,
          rowByteLength: 8,
          data: sizes.map(size => view('float32x2', size))
        }),
        region: {kind: 'rectangle', bounds: view('float32', 4)},
        output: {mask: view('uint32', rows), status: {overflow: view('uint32', 1)}}
      }).getCommandNodes(graph);
      const regionNodes = new GPURegionStatistics({
        id: 'region-review',
        region: {kind: 'mask', mask: view('uint32', rows)},
        values: new GraphVectorView({
          id: 'value-chunks',
          name: 'values',
          format: 'float32',
          length: rows,
          valueLength: rows,
          stride: 1,
          byteStride: 4,
          rowByteLength: 4,
          data: sizes.map(size => view('float32', size))
        }),
        output: {summary: view('uint32', 8)}
      })
        .getCommandNodes(graph)
        .filter(node => /^region-review-count(?:-chunk-\d+)?$/.test(node.id));
      const nodes = [...maskNodes, ...regionNodes];
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const {source} of sources) {
        expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
        expect(() => new WgslReflect(source)).not.toThrow();
      }
      const countSizes = regionNodes.map(node => node.workload?.maximumInvocationCount);
      expect(countSizes).toEqual(sizes);
      return {
        sources: new Set(sources.map(item => item.source)),
        pipelines: getGPUCommandNodePipelineKeys(nodes)
      };
    } finally {
      device.destroy();
    }
  };
  const packed = record([769]);
  const chunked = record([1, 255, 256, 257]);
  expect(chunked.sources.size).toBe(packed.sources.size);
  expect(chunked.pipelines.size).toBe(packed.pipelines.size);
});

it('multivariate inference and rate standardization reject writable output aliases', () => {
  const {device, graph, rows, view, weights} = createScene();
  try {
    const statistic = view('float32', rows);
    expect(
      () =>
        new GPUMultivariateLocalGearyPermutationTest({
          weights,
          values: view('float32', rows),
          variableCount: 1,
          permutationCount: 3,
          output: {statistic, pseudoPValues: statistic, classes: view('uint32', rows)}
        })
    ).toThrow(/overlap/);
    const shared = view('float32', 16);
    const rates = graph.createDataView(shared.buffer, {format: 'float32', length: rows});
    expect(
      () =>
        new GPUAssuncaoReisStandardization({
          events: view('float32', rows),
          populations: view('float32', rows),
          output: {standardizedRates: rates, statistics: shared}
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('LOSH residual inference rejects aliases across descriptive and inferential count ports', () => {
  const {device, rows, view, weights} = createScene();
  try {
    const shared = view('uint32', 1);
    expect(
      () =>
        new GPULOSHResidualPermutationTest({
          weights,
          values: view('float32', rows),
          parameters: view('uint32', 4),
          maximumPermutations: 3,
          alternative: 'greater',
          output: {
            statistic: view('float32', rows),
            status: view('uint32', rows),
            residualCount: shared,
            inference: {
              exceedances: view('uint32', rows),
              pseudoPValues: view('float32', rows),
              status: {overflow: view('uint32', 1), invalidCount: shared}
            }
          }
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('contiguity snapping rejects float32 overflow and underflow before generating shader literals', () => {
  const {device, rows, view} = createScene();
  try {
    for (const snapTolerance of [1e300, 1e-300])
      expect(
        () =>
          new GPUContiguityWeights({
            criterion: 'queen',
            snapTolerance,
            polygons: {
              kind: 'polygons',
              positions: view('float32x2', rows),
              featureOffsets: view('uint32', rows + 1),
              polygonOffsets: view('uint32', rows + 1),
              ringOffsets: view('uint32', rows + 1)
            },
            output: {
              offsets: view('uint32', rows + 1),
              neighbors: view('uint32', rows),
              weights: view('float32', rows),
              status: {overflow: view('uint32', 1)}
            }
          })
      ).toThrow(/float32/);
  } finally {
    device.destroy();
  }
});

it('LOSH rejects a nonscalar residual count before graph construction', () => {
  const {device, graph, view, rows, weights} = createScene();
  try {
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(
      () =>
        new GPULocalSpatialHeteroscedasticity({
          weights,
          values: view('float32', rows),
          output: {
            statistic: view('float32', rows),
            status: view('uint32', rows),
            residualCount: view('uint32', 2)
          }
        })
    ).toThrow(/residualCount.length/);
    // Only the caller's four views were declared; construction adds no resources.
    expect(allocate).toHaveBeenCalledTimes(4);
  } finally {
    device.destroy();
  }
});

for (const kind of ['losh', 'geary', 'pick'] as const)
  it(`${kind} accepts disjoint packed writable ranges in separate dispatches`, () => {
    const {device, graph, rows, view, weights} = createScene();
    try {
      const packed = view('uint32', 128);
      const statistic = graph.createDataView(packed.buffer, {
        format: 'float32',
        length: rows,
        byteOffset: 4
      });
      const status = graph.createDataView(packed.buffer, {
        format: 'uint32',
        length: rows,
        byteOffset: 256
      });
      const count = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
      const inputs = {weights, values: view('float32', rows)};
      const contributor =
        kind === 'losh'
          ? new GPULocalSpatialHeteroscedasticity({
              ...inputs,
              output: {statistic, status, residualCount: count}
            })
          : kind === 'geary'
            ? new GPUMultivariateLocalGeary({
                ...inputs,
                variableCount: 1,
                output: {statistic, status, count}
              })
            : new GPUPickRegionMask({
                picks: view('uint32', 4),
                output: {
                  mask: status,
                  status: {overflow: count}
                }
              });
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
      if (kind !== 'pick') {
        expect(() =>
          kind === 'losh'
            ? new GPULocalSpatialHeteroscedasticity({
                ...inputs,
                output: {
                  statistic,
                  status,
                  residualCount: graph.createDataView(packed.buffer, {
                    format: 'uint32',
                    length: 1,
                    byteOffset: 4
                  })
                }
              })
            : new GPUMultivariateLocalGeary({
                ...inputs,
                variableCount: 1,
                output: {
                  statistic,
                  status,
                  count: graph.createDataView(packed.buffer, {
                    format: 'uint32',
                    length: 1,
                    byteOffset: 4
                  })
                }
              })
        ).toThrow(/logical.*overlap/);
      }
    } finally {
      device.destroy();
    }
  });

for (const emptyFirstBlock of [false, true])
  it(`Geary Chan merge initializes from its first nonempty block (empty prefix: ${emptyFirstBlock})`, () => {
    const {device, graph, view} = createScene();
    recordWGSLKernelNodes(graph);
    try {
      const nodes = getLocalVariabilityMomentNodes(graph, {
        id: 'large-offset',
        rows: emptyFirstBlock ? 4097 : 2,
        columns: 1,
        bindings: [{name: 'values', view: view('float32', 2), type: 'f32', access: 'read'}],
        declarations: `${GET_NAN_WGSL}\nfn included(row: u32) -> bool { return true; }`,
        variance: view('float32', 1),
        // Only the merge kernel (nodes[1]) runs below; the block kernel still binds `centered`.
        centered: view('float32', emptyFirstBlock ? 4097 : 2),
        count: view('uint32', 1)
      });
      const record = getWGSLKernelNodeRecord(graph, nodes[1].id)!;
      // For [2^67, 2^67 + 2^48], mean = 2^67 + 2^47 and sum of squared deviations = 2^95.
      const partials = Float32Array.from([
        ...(emptyFirstBlock ? [0, 0, 0] : []),
        2,
        2 ** 67 + 2 ** 47,
        2 ** 95
      ]);
      const variance = new Float32Array(1);
      const count = new Uint32Array(1);
      new WgslExec(new WgslParser().parse(record.source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: {0: partials, 1: variance, 2: count}},
        {constants: record.constants}
      );
      expect(count[0]).toBe(2);
      expect(variance[0]).toBe(2 ** 94);
    } finally {
      device.destroy();
    }
  });

it('lattice accepts disjoint packed writable stages and rejects candidate aliases and co-bound read/write storage', () => {
  const {device, graph, rows, view} = createScene();
  try {
    const packed = view('uint32', 256);
    const neighbors = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: rows * 2,
      byteOffset: 4
    });
    const weights = graph.createDataView(packed.buffer, {
      format: 'float32',
      length: rows * 2,
      byteOffset: 256
    });
    const overflow = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
    const candidateOverflow = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 768
    });
    const offsets = view('uint32', rows + 1);
    // Status ports hold one row: alias the first row of the `neighbors` range.
    const neighborsRow = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const props = {
      width: rows,
      height: 1,
      criterion: 'rook' as const,
      output: {offsets, neighbors, weights, status: {overflow, candidateOverflow}}
    };
    expect(
      () =>
        new GPULatticeWeights({
          ...props,
          output: {...props.output, status: {overflow, candidateOverflow: neighborsRow}}
        })
    ).toThrow(/neighbors and candidateOverflow logical byte ranges must not overlap/);
    expect(() => new GPULatticeWeights(props).getCommandNodes(graph)).not.toThrow();
    expect(
      () =>
        new GPULatticeWeights({
          ...props,
          output: {
            ...props.output,
            offsets: graph.createDataView(packed.buffer, {
              format: 'uint32',
              length: rows + 1,
              byteOffset: 512
            })
          }
        })
    ).toThrow(/share buffers/);
  } finally {
    device.destroy();
  }
});
