// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots';
import {GPULocalOutlierFactor} from '../../../src/gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor';
import {GPUSimilarLocations} from '../../../src/gpu-spatial-analysis/similar-locations/gpu-similar-locations';
import {GPUPettittTest} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test';
import {GPULocationChangePoints} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-colocation-quotient';
import {GPULocalCategoryDependence} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-local-category-dependence';
import {GPUNeighborhoodSummary} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUEmpiricalBayesRates,
  GPUSpatialEmpiricalBayesRates
} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';
import {
  GPUSegregation,
  getGPUSegregationLayout,
  getGPUSegregationParameterValues
} from '../../../src/gpu-spatial-analysis/segregation/index';
import {
  GPUSpatialEntropy,
  GPUCategoryPairCounts
} from '../../../src/gpu-spatial-analysis/spatial-entropy/index';
import {GPUGeographicalDetector} from '../../../src/gpu-spatial-analysis/geographical-detector/gpu-geographical-detector';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {GPUFuzzyContiguity} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32' | 'sint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `statistics-review-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 6),
    neighbors: view('uint32', 5),
    weights: view('float32', 5)
  };
  return {device, graph, view, weights};
}

for (const kind of [
  'rates',
  'spatial-rates',
  'segregation',
  'entropy',
  'category-pairs',
  'detector',
  'islands',
  'fuzzy'
] as const) {
  it(`${kind} preflights caller STORAGE usage before declaring any resource`, () => {
    const {device, graph, view, weights} = createScene();
    const buffer = device.createBuffer({byteLength: 40, usage: Buffer.COPY_DST});
    try {
      const invalidFloat = importGraphBuffer(graph, 'invalid-float', buffer, 'float32', 5);
      const invalidInteger = importGraphBuffer(graph, 'invalid-integer', buffer, 'uint32', 5);
      const invalidPositions = importGraphBuffer(
        graph,
        'invalid-positions',
        buffer,
        'float32x2',
        5
      );
      const contributor =
        kind === 'rates'
          ? new GPUEmpiricalBayesRates({
              events: invalidFloat,
              populations: view('float32', 5),
              output: {count: view('uint32', 1)}
            })
          : kind === 'spatial-rates'
            ? new GPUSpatialEmpiricalBayesRates({
                events: invalidFloat,
                populations: view('float32', 5),
                weights,
                output: {smoothedRates: view('float32', 5)}
              })
            : kind === 'segregation'
              ? new GPUSegregation({
                  groupCounts: importGraphBuffer(graph, 'invalid-groups', buffer, 'float32', 10),
                  groupCount: 2,
                  output: {indices: view('float32', getGPUSegregationLayout(2).stride)}
                })
              : kind === 'entropy'
                ? new GPUSpatialEntropy({
                    weights,
                    categories: invalidInteger,
                    categoryCount: 2,
                    output: {summary: view('float32', 8)}
                  })
                : kind === 'category-pairs'
                  ? new GPUCategoryPairCounts({
                      weights,
                      categories: invalidInteger,
                      categoryCount: 2,
                      output: {pairCounts: view('uint32', 4), categoryRowCounts: view('uint32', 2)}
                    })
                  : kind === 'detector'
                    ? new GPUGeographicalDetector({
                        values: invalidFloat,
                        labels: view('uint32', 5),
                        output: {summary: view('float32', 8)}
                      })
                    : kind === 'islands'
                      ? new GPUIslandAttachment({
                          weights,
                          positions: invalidPositions,
                          output: {
                            offsets: view('uint32', 6),
                            neighbors: view('uint32', 15),
                            weights: view('float32', 15),
                            attachment: view('uint32', 5),
                            added: view('uint32', 15),
                            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
                          }
                        })
                      : new GPUFuzzyContiguity({
                          polygons: {
                            kind: 'polygons',
                            positions: invalidPositions,
                            featureOffsets: view('uint32', 6),
                            polygonOffsets: view('uint32', 6),
                            ringOffsets: view('uint32', 6)
                          },
                          tolerance: 0.25,
                          output: {
                            offsets: view('uint32', 6),
                            neighbors: view('uint32', 15),
                            weights: view('float32', 15),
                            distances: view('float32', 15),
                            added: view('uint32', 15),
                            featureStatus: view('uint32', 5),
                            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
                          }
                        });
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const declareView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(declareView).not.toHaveBeenCalled();
    } finally {
      buffer.destroy();
      device.destroy();
    }
  });
}

it('rates, entropy and segregation reject writable aliases, including exact counts and status', () => {
  const {device, graph, view, weights} = createScene();
  try {
    const events = view('float32', 5);
    const populations = view('float32', 5);
    const shared = view('float32', 32);
    const floats = (length: number) =>
      graph.createDataView(shared.buffer, {format: 'float32', length});
    const integers = (length: number) =>
      graph.createDataView(shared.buffer, {format: 'uint32', length, byteOffset: 20});
    expect(
      () =>
        new GPUEmpiricalBayesRates({
          events,
          populations,
          output: {rawRates: floats(5), status: {invalidCount: integers(1)}}
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUSpatialEmpiricalBayesRates({
          events,
          populations,
          weights,
          output: {smoothedRates: floats(5), priorMeans: floats(5)}
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUSpatialEntropy({
          weights,
          categories: view('uint32', 5),
          categoryCount: 2,
          output: {summary: floats(8), counts: integers(2)}
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUCategoryPairCounts({
          weights,
          categories: view('uint32', 5),
          categoryCount: 2,
          output: {pairCounts: integers(4), categoryRowCounts: integers(2)}
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUSegregation({
          groupCount: 2,
          groupCounts: view('float32', 10),
          output: {indices: floats(getGPUSegregationLayout(2).stride), localEntropy: floats(5)}
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('entropy and geographical detector reject invalid capacities at construction', () => {
  const {device, view, weights} = createScene();
  try {
    for (const capacity of [NaN, Infinity, 0, 1.5]) {
      expect(
        () =>
          new GPUSpatialEntropy({
            weights,
            categories: view('uint32', 5),
            categoryCount: capacity,
            output: {summary: view('float32', 8)}
          })
      ).toThrow(/categoryCount/);
      expect(
        () =>
          new GPUGeographicalDetector({
            values: view('float32', 5),
            labels: view('uint32', 5),
            labelCapacity: capacity,
            output: {summary: view('float32', 8)}
          })
      ).toThrow(/labelCapacity/);
    }
  } finally {
    device.destroy();
  }
});

it('segregation runtime settings share shaders and reject values that round outside float32 domains', () => {
  const {device, graph, view} = createScene();
  try {
    expect(
      getGPUSegregationParameterValues({selfWeight: 2, atkinsonInequalityAversion: 0.25})
    ).toEqual(Float32Array.of(2, 0.25));
    for (const settings of [
      {selfWeight: 1e40},
      {atkinsonInequalityAversion: 1 - 1e-10},
      {atkinsonInequalityAversion: 1e-50}
    ]) {
      expect(() => getGPUSegregationParameterValues(settings)).toThrow(/float32/);
      expect(
        () =>
          new GPUSegregation({
            ...settings,
            groupCount: 2,
            groupCounts: view('float32', 10),
            output: {indices: view('float32', getGPUSegregationLayout(2).stride)}
          })
      ).toThrow(/float32/);
    }
    let buildIndex = 0;
    const build = (selfWeight: number, atkinsonInequalityAversion: number) =>
      new GPUSegregation({
        id: `runtime-segregation-${buildIndex++}`,
        groupCount: 2,
        groupCounts: view('float32', 10),
        selfWeight,
        atkinsonInequalityAversion,
        parameters: view('float32', 2),
        output: {indices: view('float32', getGPUSegregationLayout(2).stride)}
      }).getCommandNodes(graph);
    expect([...getGPUCommandNodePipelineKeys(build(1, 0.5))]).toEqual([
      ...getGPUCommandNodePipelineKeys(build(2, 0.25))
    ]);
    for (const {id, source} of getGPUCommandNodeKernelSources(build(1, 0.5))) {
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
      expect(() => new WgslReflect(source), id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('unequal spatial empirical-Bayes chunks share one source and pipeline and their own dispatch sizes', () => {
  const build = (sizes: number[]) => {
    const {device, graph, view} = createScene();
    let rowStart = 0;
    const chunks = sizes.map(rowCount => {
      const chunk = {
        rowStart,
        rowCount,
        weights: {
          offsets: view('uint32', rowCount + 1),
          neighbors: view('uint32', rowCount),
          weights: view('float32', rowCount)
        }
      };
      rowStart += rowCount;
      return chunk;
    });
    const nodes = new GPUSpatialEmpiricalBayesRates({
      events: view('float32', rowStart),
      populations: view('float32', rowStart),
      weights: {rowCount: rowStart, chunks},
      output: {smoothedRates: view('float32', rowStart)}
    }).getCommandNodes(graph);
    return {device, nodes};
  };
  const first = build([1000, 1048]);
  const second = build([1, 255, 256, 1536]);
  try {
    expect(getGPUCommandNodePipelineKeys(first.nodes).size).toBe(1);
    expect(getGPUCommandNodePipelineKeys(second.nodes).size).toBe(1);
    expect(
      new Set(getGPUCommandNodeKernelSources(second.nodes).map(({source}) => source)).size
    ).toBe(1);
    const dispatches = second.nodes.map(
      node => (node as {workload: {maximumWorkgroupCount: number}}).workload.maximumWorkgroupCount
    );
    expect(dispatches).toEqual([1, 1, 1, 6]);
    for (const {id, source} of getGPUCommandNodeKernelSources(second.nodes)) {
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
      expect(() => new WgslReflect(source), id).not.toThrow();
      expect(source).toContain('var count = 1u');
    }
  } finally {
    first.device.destroy();
    second.device.destroy();
  }
});

for (const kind of [
  'lof',
  'similar',
  'pettitt',
  'change',
  'colocation',
  'dependence',
  'summary'
] as const) {
  it(`${kind} rejects output aliases and preflights caller bindings before allocating scratch`, () => {
    const {device, graph, view, weights} = createScene();
    const buffer = device.createBuffer({byteLength: 40, usage: Buffer.COPY_DST});
    const shared = view('float32', 64);
    const floats = (length: number) =>
      graph.createDataView(shared.buffer, {format: 'float32', length});
    const integers = (length: number) =>
      graph.createDataView(shared.buffer, {format: 'uint32', length});
    const build = (invalid: boolean, overlapping: boolean) => {
      const input = (length: number) =>
        invalid
          ? importGraphBuffer(graph, `invalid-${kind}-${length}`, buffer, 'float32', length)
          : view('float32', length);
      const categories = invalid
        ? importGraphBuffer(graph, `invalid-categories-${kind}`, buffer, 'uint32', 5)
        : view('uint32', 5);
      const outputFloat = (length: number) =>
        overlapping ? floats(length) : view('float32', length);
      const outputInteger = (length: number) =>
        overlapping ? integers(length) : view('uint32', length);
      if (kind === 'lof')
        return new GPULocalOutlierFactor({
          neighbors: {...weights, distances: input(5)},
          parameters: view('float32', 4),
          output: {
            kDistance: outputFloat(5),
            localReachabilityDensity: outputFloat(5),
            lof: outputFloat(5)
          }
        });
      if (kind === 'similar')
        return new GPUSimilarLocations({
          values: input(10),
          columnCount: 2,
          referenceMask: view('uint32', 5),
          parameters: view('float32', 6),
          resultCapacity: 2,
          output: {
            ranks: outputInteger(5),
            distances: outputFloat(5),
            topIds: outputInteger(2),
            status: {count: outputInteger(1)}
          }
        });
      if (kind === 'pettitt')
        return new GPUPettittTest({
          values: input(10),
          cellCount: 5,
          sliceCount: 2,
          layout: 'cell-major',
          output: {
            statistic: outputInteger(5),
            changeSlice: outputInteger(5),
            pValues: outputFloat(5)
          }
        });
      if (kind === 'change')
        return new GPULocationChangePoints({
          values: input(10),
          cellCount: 5,
          sliceCount: 2,
          layout: 'cell-major',
          cost: 'mean',
          maximumChangePoints: 1,
          parameters: view('float32', 2),
          output: {
            changePointCount: outputInteger(5),
            changePoints: outputInteger(5),
            optimalCost: outputFloat(5)
          }
        });
      if (kind === 'colocation')
        return new GPUColocationQuotient({
          categories,
          weights,
          categoryCount: 2,
          output: {
            globalQuotients: outputFloat(4),
            localQuotients: outputFloat(10),
            categoryRowCounts: outputInteger(2)
          }
        });
      if (kind === 'dependence')
        return new GPULocalCategoryDependence({
          categories,
          weights,
          categoryCount: 2,
          output: {statistics: outputFloat(30), categoryRowCounts: outputInteger(2)}
        });
      return new GPUNeighborhoodSummary({
        categories,
        weights,
        output: {modes: outputInteger(5), entropy: outputFloat(5)}
      });
    };
    try {
      expect(() => build(false, true)).toThrow(/overlap/);
      const contributor = build(true, false);
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const declareView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
      expect(allocate).not.toHaveBeenCalled();
      expect(declareView).not.toHaveBeenCalled();
      allocate.mockRestore();
      declareView.mockRestore();
      for (const {id, source} of getGPUCommandNodeKernelSources(
        build(false, false).getCommandNodes(graph)
      )) {
        expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
        expect(() => new WgslReflect(source), id).not.toThrow();
      }
    } finally {
      buffer.destroy();
      device.destroy();
    }
  });
}

it('neighborhood focal weights reject finite JS numbers that overflow float32 before emitting WGSL', () => {
  const {device, view, weights} = createScene();
  try {
    expect(
      () =>
        new GPUNeighborhoodSummary({
          weights,
          categories: view('uint32', 5),
          includeFocal: true,
          focalWeight: 1e40,
          output: {modes: view('uint32', 5)}
        })
    ).toThrow(/focalWeight.*float32/);
  } finally {
    device.destroy();
  }
});

function createEmergingScene(sizes: number[] = [512, 512], invalidUsage = false) {
  const scene = createScene();
  const {device, graph, view} = scene;
  const cells = sizes.reduce((total, size) => total + size, 0);
  let rowStart = 0;
  const chunks = sizes.map(rowCount => {
    const chunk = {
      rowStart,
      rowCount,
      weights: {
        offsets: view('uint32', rowCount + 1),
        neighbors: view('uint32', rowCount),
        weights: view('float32', rowCount)
      }
    };
    rowStart += rowCount;
    return chunk;
  });
  const buffer = invalidUsage
    ? device.createBuffer({byteLength: cells * 3 * 4, usage: Buffer.COPY_DST})
    : undefined;
  const props = {
    values: buffer
      ? importGraphBuffer(graph, 'bad-cube', buffer, 'float32', cells * 3)
      : view('float32', cells * 3),
    weights: {rowCount: cells, chunks},
    sliceCount: 3,
    parameters: view('float32', 8),
    multipleTesting: 'none' as const,
    output: {
      giZScores: view('float32', cells * 3),
      trendZ: view('float32', cells),
      trendP: view('float32', cells),
      trendS: view('sint32', cells),
      category: view('uint32', cells),
      hotSliceCount: view('uint32', cells),
      coldSliceCount: view('uint32', cells),
      count: view('uint32', 1),
      globalStatistics: view('float32', 4)
    }
  };
  return {...scene, props, buffer};
}

it('emerging hot spots retains exact bin counts and shares Gi* sources across unequal chunks', () => {
  const first = createEmergingScene();
  const second = createEmergingScene([1, 255, 256, 512]);
  try {
    const before = new GPUEmergingHotSpots(first.props).getCommandNodes(first.graph);
    const after = new GPUEmergingHotSpots(second.props).getCommandNodes(second.graph);
    expect(getGPUCommandNodePipelineKeys(after).size).toBe(
      getGPUCommandNodePipelineKeys(before).size
    );
    const kernels = getGPUCommandNodeKernelSources(after);
    const giSources = kernels.filter(({id}) => id.includes('-gi-star'));
    expect(new Set(giSources.map(({source}) => source)).size).toBe(1);
    expect(
      after
        .filter(node => node.id.includes('-gi-star'))
        .map(
          node =>
            (node as {workload: {maximumWorkgroupCount: number}}).workload.maximumWorkgroupCount
        )
    ).toEqual([1, 3, 4, 7]);
    const mean = kernels.find(({id}) => id.endsWith('-mean'))!.source;
    expect(mean).toContain('var<workgroup> partialCounts: array<u32,');
    expect(mean).toContain('statistics[statisticsOffset + 4u] = f32(count >> 16u);');
    expect(mean).toContain('statistics[statisticsOffset + 5u] = f32(count & 65535u);');
    const meanCorrection = kernels.find(({id}) => id.endsWith('-mean-correction'))!.source;
    expect(meanCorrection).toContain(
      'let count = (u32(statistics[statisticsOffset + 4u]) << 16u) | u32(statistics[statisticsOffset + 5u]);'
    );
    expect(meanCorrection).toContain('divideBySpatialCount(getCompensated(partialSums[0]), count)');
    for (const {id, source} of kernels) {
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
      expect(() => new WgslReflect(source), id).not.toThrow();
    }
  } finally {
    first.device.destroy();
    second.device.destroy();
  }
});

it('emerging hot spots rejects output aliases, invalid float32 constants and caller usage before allocation', () => {
  const {device, graph, props, buffer} = createEmergingScene([5], true);
  try {
    expect(() => new GPUEmergingHotSpots({...props, selfWeight: 1e40})).toThrow(
      /selfWeight.*float32/
    );
    expect(() => new GPUEmergingHotSpots({...props, trendCorrectionCriticalZ: 1e40})).toThrow(
      /trendCorrectionCriticalZ/
    );
    expect(
      () =>
        new GPUEmergingHotSpots({...props, output: {...props.output, trendZ: props.output.trendP}})
    ).toThrow(/overlap/);
    const contributor = new GPUEmergingHotSpots(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const declareView = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
    expect(declareView).not.toHaveBeenCalled();
  } finally {
    buffer?.destroy();
    device.destroy();
  }
});
