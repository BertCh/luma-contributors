// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudNeighborhoods,
  decodeGPUPointCloudNeighborhoodsStatus
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-neighborhoods';
import {
  GPUPointCloudFeatures,
  decodeGPUPointCloudFeaturesStatus
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-features';
import {GPUPointCloudSMRF} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-smrf';
import {
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  GPUPointCloudStatisticalFilter,
  getGPUPointCloudStatisticalFilterParameterValues
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-statistical-filter';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';
import {expectPointCloudReadback} from './pointcloud-readback-assertions';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

it('review S2 rejects empty or zero large-distance readbacks', () => {
  const expected = [999999986991104, 999999986991104, 999999986991104];
  expect(() => expectPointCloudReadback([], expected, () => 67108864)).toThrow();
  expect(() => expectPointCloudReadback([0, 0, 0], expected, () => 67108864)).toThrow();
});

it('review S3 rejects a correct prefix with a missing suffix', () => {
  expect(() => expectPointCloudReadback([1], [1, 4, 9])).toThrow();
  expect(() => expectPointCloudReadback([1, 4, NaN], [1, 4, 9])).toThrow();
});
function declaration(rowCount = 3, capacity = 1) {
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `wave3-${sequence++}`, format, length);
  const counts = view('uint32', rowCount);
  const indices = view('uint32', rowCount * capacity);
  const squaredDistances = view('float32', rowCount * capacity);
  const positions = view('float32x3', rowCount);
  const filter = {
    id: 'lidar',
    counts,
    squaredDistances,
    capacity,
    output: {meanDistances: view('float32', rowCount), validity: view('uint32', rowCount)}
  };
  const features = {
    id: 'lidar',
    positions,
    counts,
    indices,
    capacity,
    output: {
      normals: view('float32x3', rowCount),
      eigenvalues: view('float32x3', rowCount),
      shapes: view('float32x3', rowCount),
      status: view('uint32', rowCount)
    }
  };
  return {
    graph,
    view,
    filter,
    features,
    neighborhoods: {positions, radius: 1, capacity, output: {counts, indices, squaredDistances}}
  };
}

it('OUT-07 initializes missing distances from the exact maximum finite word', () => {
  const {graph, neighborhoods} = declaration();
  const source = [
    ...getParsedKernelSources(
      new GPUPointCloudNeighborhoods(neighborhoods).getCommandNodes(graph)
    ).values()
  ].join('\n');
  expect(source).toContain('nearestDistances[slot] = bitcast<f32>(0x7f7fffffu)');
});

it('FLT-14 resolves rounded cutoff and retained-list ties, including cooperative merge', () => {
  const {graph, neighborhoods} = declaration();
  for (const strategy of ['serial', 'cooperative'] as const) {
    const source = [
      ...getParsedKernelSources(
        new GPUPointCloudNeighborhoods({...neighborhoods, strategy}).getCommandNodes(graph)
      ).values()
    ].join('\n');
    expect(source).toContain('comparePointCloudNorms');
    expect(source).toContain('pointCloudNeighborBefore');
    expect(source).toContain('sub_fp64');
  }
});

it('DGN-04 centers moments on a supported distance before restoring the threshold', () => {
  const {graph, filter} = declaration();
  const source = [
    ...getParsedKernelSources(
      new GPUPointCloudStatisticalFilter(filter).getCommandNodes(graph)
    ).values()
  ].join('\n');
  expect(source).toContain('readReference(0u)');
  expect(source).toContain('readMeanDistances(index) - readReference(0u)');
});

it('review B1 refines distance comparisons inside the arithmetic error bound', () => {
  for (const strategy of ['serial', 'cooperative'] as const) {
    const {graph, neighborhoods} = declaration();
    const source = [
      ...getParsedKernelSources(
        new GPUPointCloudNeighborhoods({...neighborhoods, radius: 53365, strategy}).getCommandNodes(
          graph
        )
      ).values()
    ].join('\n');
    expect(source).toContain('pointCloudNormKeysSeparated(distanceSquared,');
    expect(source).toContain('pointCloudNormKeysSeparated(firstDistance, secondDistance)');
    expect(source).toContain('if (!all((bitcast<vec3u>(difference)');
  }
});

it('review B2 emits floating literals for the extreme static radius', () => {
  const radius = new Float32Array(new Uint32Array([0x5f7fffff]).buffer)[0];
  for (const strategy of ['serial', 'cooperative'] as const) {
    const {graph, neighborhoods} = declaration();
    const source = [
      ...getParsedKernelSources(
        new GPUPointCloudNeighborhoods({...neighborhoods, radius, strategy}).getCommandNodes(graph)
      ).values()
    ].join('\n');
    expect(source).toContain('vec3f(18446742974197924000.0, 0.0, 0.0)');
    expect(source).not.toContain('vec3f(18446742974197924000,');
  }
});

it('review B3 centers each row before averaging mixed support counts', () => {
  const {graph, filter} = declaration(3, 3);
  const source = getParsedKernelSources(
    new GPUPointCloudStatisticalFilter({...filter, standardDeviations: 0}).getCommandNodes(graph)
  ).get('lidar-means');
  expect(source).toContain('let support = min(readCounts(index), 3u);');
  expect(source).toContain(
    'let distanceReference = sqrt_fp64(vec2f(readSquaredDistances(index * 3u), 0.0)).x;'
  );
  expect(source).toContain('for (var slot = 0u; slot < support; slot++)');
  expect(source).toContain(
    'distanceSum += sqrt_fp64(vec2f(readSquaredDistances(index * 3u + slot), 0.0)).x - distanceReference;'
  );
  expect(source).toContain(
    'writeMeanDistances(index, distanceReference + distanceSum / f32(support));'
  );
});

it('emits a WGSL floating-point literal for a large static statistical-filter multiplier', () => {
  const {graph, filter} = declaration();
  const source = getParsedKernelSources(
    new GPUPointCloudStatisticalFilter({...filter, standardDeviations: 1e20}).getCommandNodes(graph)
  ).get('lidar-threshold');
  expect(source).toContain(' + 100000002004087730000.0 * sqrt(');
});

it('review B4 distinguishes unsupported features, solve failure and degenerate rows', () => {
  expect(decodeGPUPointCloudFeaturesStatus(new Uint32Array([1]))).toEqual({
    degenerate: true,
    truncated: false,
    converged: false,
    notConverged: false,
    incompleteSupport: false,
    supportComplete: true,
    featureComplete: false
  });
  expect(decodeGPUPointCloudFeaturesStatus(new Uint32Array([4]))).toEqual({
    degenerate: false,
    truncated: false,
    converged: false,
    notConverged: true,
    incompleteSupport: false,
    supportComplete: true,
    featureComplete: false
  });
});

it('review N1 uses bound distances on static and runtime halo routes', () => {
  for (const runtime of [false, true]) {
    const {graph, neighborhoods, view} = declaration();
    const source = [
      ...getParsedKernelSources(
        new GPUPointCloudNeighborhoods({
          ...neighborhoods,
          radius: 0.25,
          supportBounds: {minimum: [9999998, -1, -1], maximum: [10000000, 1, 1]},
          ...(runtime ? {parameters: view('float32', 9)} : {}),
          output: {...neighborhoods.output, status: view('uint32', 3)}
        }).getCommandNodes(graph)
      ).values()
    ].join('\n');
    expect(source).toContain('pointCloudSupportContainsSphere(position,');
    expect(source).toContain('let margin = maximum - minimum;');
  }
});

it('review N2 rejects the retired SMRF mask spelling', () => {
  const {features, view} = declaration();
  const mask = view('uint32', 3);
  const props = {
    id: 'lidar',
    positions: features.positions,
    classifications: view('uint32', 3),
    grid: {origin: [0, 0] as const, cellSize: 1, width: 2, height: 2},
    window: 1,
    output: {
      groundSurface: view('float32', 4),
      groundMask: view('uint32', 3),
      outputClassifications: view('uint32', 3),
      status: view('uint32', 3)
    }
  };
  const retiredMask = {...props, mask, selection: mask};
  expect(() => new GPUPointCloudSMRF(retiredMask)).toThrow(/lidar selection.*removed.*use mask/);
  expect(() => new GPUPointCloudSMRF({...props, mask})).not.toThrow();
});

it('VAL-01 rejects wrong declared distance and feature index formats', () => {
  const {view, filter, features} = declaration();
  expect(
    () =>
      new GPUPointCloudStatisticalFilter({
        ...filter,
        squaredDistances: view('uint32', 3) as unknown as typeof filter.squaredDistances
      })
  ).toThrow(/lidar squaredDistances.*float32.*received uint32/);
  expect(
    () =>
      new GPUPointCloudFeatures({
        ...features,
        indices: view('float32', 3) as unknown as typeof features.indices
      })
  ).toThrow(/lidar indices.*uint32.*received float32/);
});

it('VAL-10 identifies missing counts before dereferencing it', () => {
  const {filter} = declaration();
  expect(
    () =>
      new GPUPointCloudStatisticalFilter({
        ...filter,
        counts: undefined as unknown as typeof filter.counts
      })
  ).toThrow(/lidar counts.*uint32.*received undefined/);
  try {
    new GPUPointCloudStatisticalFilter({
      ...filter,
      counts: undefined as unknown as typeof filter.counts
    });
    expect.fail('missing counts must fail');
  } catch (error) {
    expect(error).toMatchObject({
      id: 'lidar',
      code: 'MISSING_PROP',
      path: 'counts',
      received: undefined,
      expected: {viewType: 'data', formats: ['uint32']}
    });
  }
});

it('STA-04 validates optional status ports and preserves query chunk correspondence', () => {
  const {graph, view, features, filter} = declaration(4, 4);
  expect(
    () => new GPUPointCloudStatisticalFilter({...filter, neighborhoodStatus: view('uint32', 3)})
  ).toThrow(/neighborhoodStatus/);
  expect(
    () => new GPUPointCloudFeatures({...features, neighborhoodStatus: view('uint32', 3)})
  ).toThrow(/neighborhoodStatus/);
  const rows = <Format extends 'float32x3' | 'uint32'>(format: Format, lengths: number[]) =>
    createVectorView(
      'chunks',
      format,
      lengths.map(length => view(format, length))
    );
  const contributor = new GPUPointCloudFeatures({
    ...features,
    indices: rows('uint32', [4, 12]),
    counts: rows('uint32', [1, 3]),
    neighborhoodStatus: rows('uint32', [1, 3]),
    output: {
      normals: rows('float32x3', [1, 3]),
      eigenvalues: rows('float32x3', [1, 3]),
      shapes: rows('float32x3', [1, 3]),
      status: rows('uint32', [1, 3])
    }
  });
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  const support = [...sources.values()].join('\n');
  expect(support).toContain('readNeighborhoodStatus(index)');
  expect(support).toContain('flags |= 8u');
  expect(support).toContain('flags |= 2u');
  expect(
    () =>
      new GPUPointCloudFeatures({
        ...features,
        counts: rows('uint32', [1, 3]),
        neighborhoodStatus: rows('uint32', [2, 2]),
        indices: rows('uint32', [4, 12]),
        output: {
          normals: rows('float32x3', [1, 3]),
          eigenvalues: rows('float32x3', [1, 3]),
          shapes: rows('float32x3', [1, 3]),
          status: rows('uint32', [1, 3])
        }
      })
  ).toThrow(/chunk topology/);
});

it('STA-04 decodes distinct feature support and solve bits from words or readback bytes', () => {
  const buffer = new ArrayBuffer(12);
  new Uint32Array(buffer).set([0xffffffff, 8, 4]);
  expect(decodeGPUPointCloudFeaturesStatus(new Uint8Array(buffer, 4, 4))).toEqual({
    degenerate: false,
    truncated: false,
    converged: true,
    notConverged: false,
    incompleteSupport: true,
    supportComplete: false,
    featureComplete: false
  });
  expect(decodeGPUPointCloudFeaturesStatus(new Uint32Array(buffer, 8, 1))).toMatchObject({
    converged: false,
    incompleteSupport: false
  });
  expect(decodeGPUPointCloudNeighborhoodsStatus(new Uint32Array([4]))).toEqual({
    invalidInput: false,
    truncated: false,
    incompleteSupport: true,
    complete: false
  });
  for (const decode of [
    decodeGPUPointCloudFeaturesStatus,
    decodeGPUPointCloudNeighborhoodsStatus
  ]) {
    expect(() => decode(new Uint32Array())).toThrow(/at least 1/);
    expect(() => decode(new Uint8Array(3))).toThrow(/whole/);
  }
  const clearStatus = new Uint32Array([0]).buffer;
  expect(decodeGPUPointCloudFeaturesStatus(clearStatus)).toMatchObject({
    supportComplete: true,
    featureComplete: true
  });
  expect(decodeGPUPointCloudNeighborhoodsStatus(clearStatus)).toMatchObject({complete: true});
});

it('VAL-01 checks SMRF declared formats and preserves optional masks', () => {
  const {features, view} = declaration();
  const props = {
    id: 'lidar',
    positions: features.positions,
    classifications: view('uint32', 3),
    grid: {origin: [0, 0] as const, cellSize: 1, width: 2, height: 2},
    window: 1,
    output: {
      groundSurface: view('float32', 4),
      groundMask: view('uint32', 3),
      outputClassifications: view('uint32', 3),
      status: view('uint32', 3)
    }
  };
  expect(() => new GPUPointCloudSMRF(props)).not.toThrow();
  expect(
    () =>
      new GPUPointCloudSMRF({
        ...props,
        classifications: view('float32', 3) as unknown as typeof props.classifications
      })
  ).toThrow(/lidar classifications.*uint32.*received float32/);
});

it('generates statically valid WGSL for tied keys and complete-support routes', () => {
  for (const strategy of ['serial', 'cooperative'] as const) {
    const {graph, neighborhoods, features, filter, view} = declaration(4, 4);
    const neighborhoodStatus = view('uint32', 4);
    const contributors = [
      new GPUPointCloudNeighborhoods({...neighborhoods, strategy}),
      new GPUPointCloudFeatures({
        ...features,
        neighborhoodStatus,
        parameters: view('float32', 4),
        output: {...features.output, covariance: view('float32', 24)}
      }),
      new GPUPointCloudStatisticalFilter({
        ...filter,
        neighborhoodStatus,
        parameters: view('float32', 1),
        output: {...filter.output, thresholdComplete: view('uint32', 1)}
      })
    ];
    for (const contributor of contributors) {
      for (const {source, constants} of getGPUCommandNodeKernelSources(
        contributor.getCommandNodes(graph)
      )) {
        expect(auditWGSLModule(source, constants)).toEqual([]);
      }
    }
  }
});

it('VAL-14 rejects coerced multipliers and preserves original received values', () => {
  for (const received of [true, '2']) {
    try {
      getGPUPointCloudStatisticalFilterParameterValues({
        standardDeviations: received as unknown as number
      });
      expect.fail('expected numeric validation');
    } catch (error) {
      expect(error).toMatchObject({code: 'NUMBER_RANGE', path: 'standardDeviations', received});
    }
  }
});

it('DEF-11 infers exact nonempty neighbor capacity in all three producers', () => {
  const {features, filter, neighborhoods} = declaration(3, 2);
  expect(new GPUPointCloudFeatures({...features, capacity: undefined}).props.capacity).toBe(2);
  expect(new GPUPointCloudStatisticalFilter({...filter, capacity: undefined}).props.capacity).toBe(
    2
  );
  expect(
    new GPUPointCloudNeighborhoods({...neighborhoods, capacity: undefined}).props.capacity
  ).toBe(2);
});

it('DEF-11 refuses empty inference, indivisible slots and disagreeing outputs', () => {
  const empty = declaration(0);
  expect(() => new GPUPointCloudStatisticalFilter({...empty.filter, capacity: undefined})).toThrow(
    /capacity.*explicit/
  );
  expect(() => new GPUPointCloudFeatures({...empty.features, capacity: undefined})).toThrow(
    /capacity.*explicit/
  );
  expect(
    () => new GPUPointCloudNeighborhoods({...empty.neighborhoods, capacity: undefined})
  ).toThrow(/capacity.*explicit/);
  const {features, neighborhoods, view} = declaration(3, 2);
  expect(
    () => new GPUPointCloudFeatures({...features, capacity: undefined, indices: view('uint32', 7)})
  ).toThrow(/capacity/);
  expect(
    () =>
      new GPUPointCloudNeighborhoods({
        ...neighborhoods,
        capacity: undefined,
        output: {...neighborhoods.output, squaredDistances: view('float32', 9)}
      })
  ).toThrow(/squaredDistances/);
  const overLimit = declaration(3, 65);
  expect(() => new GPUPointCloudFeatures({...overLimit.features, capacity: undefined})).toThrow(
    /capacity.*64/
  );
  expect(
    () => new GPUPointCloudStatisticalFilter({...overLimit.filter, capacity: undefined})
  ).toThrow(/capacity.*64/);
  expect(
    () => new GPUPointCloudNeighborhoods({...overLimit.neighborhoods, capacity: undefined})
  ).toThrow(/capacity.*64/);
  expect(() => new GPUPointCloudFeatures(empty.features)).not.toThrow();
  expect(() => new GPUPointCloudStatisticalFilter(empty.filter)).not.toThrow();
  expect(() => new GPUPointCloudNeighborhoods(empty.neighborhoods)).not.toThrow();
});
