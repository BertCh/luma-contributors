// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {createSpatialJoinCandidateQueryRange} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-candidate-pages';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUSpatialJoinCandidates} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {getSpatialJoinCandidateNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-candidates';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {getReservedWGSLWords} from './wgsl-reserved-words';
import {getCandidateRankDeclarations} from './candidate-rank-test-utils';

function createFixture(polygon = false, candidateCapacity = 4) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const pointPositions = createTransientView(graph, 'points', 'float32x2', 1);
  const rightPositions = createTransientView(graph, 'right', 'float32x2', polygon ? 4 : 1);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const contributor = new GPUSpatialPredicateJoin({
    id: 'candidate-scratch',
    left: {kind: 'points', positions: pointPositions},
    right: polygon
      ? {
          kind: 'polygons',
          positions: rightPositions,
          featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
          polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
          ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
        }
      : {kind: 'points', positions: rightPositions},
    predicate: 'intersects',
    candidateCapacity,
    output: {
      pairs: {
        leftRows: createTransientView(graph, 'left-ids', 'uint32', 4),
        rightRows: createTransientView(graph, 'right-ids', 'uint32', 4),
        status: {
          count: scalar('count'),
          requiredCount: scalar('required'),
          overflow: scalar('overflow')
        }
      }
    }
  });
  return {device, graph, contributor};
}

for (const [polygon, capacity, expected] of [
  [false, 9, /candidate pairs/],
  [true, 4, /robust pair points/]
] as const) {
  it(`preflights ${expected.source} before graph mutation and permits retry`, () => {
    const {device, graph, contributor} = createFixture(polygon, capacity);
    try {
      const originalLimits = device.limits;
      Object.defineProperty(device, 'limits', {
        configurable: true,
        value: {...originalLimits, maxStorageBufferBindingSize: 64}
      });
      const declareScratch = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(graph)).toThrow(expected);
      expect(declareScratch).not.toHaveBeenCalled();
      Object.defineProperty(device, 'limits', {value: originalLimits});
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

it('checks candidate allocations against maxBufferSize before bounds and BVH declarations', () => {
  const {device, graph, contributor} = createFixture(false, 9);
  try {
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxBufferSize: 64}
    });
    const declareScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/candidate pairs.*maxBufferSize/);
    expect(declareScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects a finite JavaScript distance that would emit an infinite f32 WGSL literal', () => {
  const {device, contributor} = createFixture();
  try {
    expect(
      () =>
        new GPUSpatialPredicateJoin({...contributor.props, predicate: 'dwithin', distance: 1e100})
    ).toThrow(/candidate-scratch distance must be a finite number in/);
  } finally {
    device.destroy();
  }
});

it('predicate candidate offsets saturate monotonically before truncation and writing', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const counts = [2, 0xffffffff, 0, 2, 1];
  const rows = counts.length;
  const make = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  try {
    const nodes = getSpatialJoinCandidateNodes(graph, {
      id: 'candidate-saturation',
      leftCount: rows,
      rightCount: 1,
      leftMinima: createTransientView(graph, 'minima', 'float32x2', rows),
      leftMaxima: createTransientView(graph, 'maxima', 'float32x2', rows),
      tree: {
        nodeMinima: createTransientView(graph, 'tree-minima', 'float32x2', 1),
        nodeMaxima: createTransientView(graph, 'tree-maxima', 'float32x2', 1),
        leafIds: make('leaves', 1),
        internalNodeCount: 0
      },
      candidateCapacity: 4,
      candidatePairs: createTransientView(graph, 'pairs', 'uint32x2', 4),
      margin: 0,
      state: make('state', 4),
      candidateTruncated: make('truncated', rows)
    });
    const identifiers = nodes.map(node => node.id);
    const saturatedIndex = identifiers.indexOf('candidate-saturation-scan-candidates-saturate');
    expect(saturatedIndex).toBeGreaterThan(0);
    expect(saturatedIndex).toBeLessThan(
      identifiers.indexOf('candidate-saturation-write-candidates')
    );
    expect(identifiers).not.toContain('candidate-saturation-candidates-carry');
    const kernels = getGPUCommandNodeKernelSources(nodes);
    const saturate = kernels.find(kernel => kernel.id.endsWith('scan-candidates-saturate'))!;
    const truncated = kernels.find(kernel => kernel.id.endsWith('candidate-truncated'))!;
    const prefixes: bigint[] = [];
    let sum = 0n;
    for (const count of counts) {
      sum += BigInt(count);
      prefixes.push(sum);
    }
    const buffers: Record<string, Uint32Array> = {
      low: Uint32Array.from(prefixes.map(value => Number(value & 0xffffffffn))),
      high: Uint32Array.from(prefixes.map(value => Number(value >> 32n))),
      offsets: new Uint32Array(rows),
      total: new Uint32Array(4),
      counts: Uint32Array.from(counts),
      candidateTruncated: new Uint32Array(rows)
    };
    for (const kernel of [saturate, truncated]) {
      expect(getReservedWGSLWords(kernel.source)).toEqual([]);
      const bindings = Object.fromEntries(
        new WgslReflect(kernel.source).storage.map(binding => [
          binding.binding,
          buffers[binding.name]
        ])
      );
      new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {0: bindings},
        {constants: kernel.constants}
      );
    }
    // BigInt independently shows the prefix crossing 2^32 inside row 1. Rows 3 and 4
    // cannot wrap back into the first four retained slots.
    expect(Array.from(buffers['offsets'])).toEqual([0, 2, 0xffffffff, 0xffffffff, 0xffffffff]);
    expect(buffers['total'][0]).toBe(0xffffffff);
    expect(Array.from(buffers['candidateTruncated'])).toEqual([0, 1, 0, 1, 1]);
  } finally {
    device.destroy();
  }
});

it('predicate page pipelines and shader counts stay constant as the candidate envelope grows', () => {
  const sizes: number[][] = [];
  for (const capacity of [7, 19, 31]) {
    const {device, graph, contributor} = createFixture(true, capacity);
    try {
      const paged = new GPUSpatialPredicateJoin({...contributor.props, candidateTileCapacity: 3});
      const nodes = paged.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(paged.estimatedCommandNodeCount);
      const sources = getParsedKernelSources(nodes);
      for (const source of sources.values()) expect(getReservedWGSLWords(source)).toEqual([]);
      const writes = nodes.filter(node => /tile-\d+-write-candidates$/.test(node.id));
      expect(writes).toHaveLength(Math.ceil(capacity / 3));
      expect(getGPUCommandNodePipelineKeys(writes).size).toBe(1);
      sizes.push([new Set(sources.values()).size, getGPUCommandNodePipelineKeys(nodes).size]);
    } finally {
      device.destroy();
    }
  }
  expect(sizes[1]).toEqual(sizes[0]);
  expect(sizes[2]).toEqual(sizes[0]);
});

it('predicate pages declare capacity beyond one candidate binding', () => {
  const {device, graph, contributor} = createFixture(true, 20_000_000);
  try {
    const nodes = contributor.getCommandNodes(graph);
    expect(contributor.candidatePages.capacity).toBeLessThan(20_000_000);
    expect(contributor.candidatePages.pageCount).toBeGreaterThan(1);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(
      () => new GPUSpatialPredicateJoin({...contributor.props, candidateTileCapacity: NaN})
    ).toThrow(/candidateTileCapacity/);
  } finally {
    device.destroy();
  }
});

it('GPU page query ranges dispatch only intersecting query rows, including empty and saturated prefixes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const cases = [
    {
      prefix: Array.from({length: 1024}, (_, index) => index),
      total: 1024,
      first: 0,
      length: 513,
      chunkFirst: 0,
      chunkLength: 1024,
      expected: [0, 513, 3, 1, 1]
    },
    {
      prefix: [0, 0, 4, 4, 9],
      total: 12,
      first: 0,
      length: 3,
      chunkFirst: 0,
      chunkLength: 5,
      expected: [1, 1, 1, 1, 1]
    },
    {
      prefix: [0, 0, 4, 4, 9],
      total: 12,
      first: 3,
      length: 3,
      chunkFirst: 0,
      chunkLength: 5,
      expected: [1, 3, 1, 1, 1]
    },
    {
      prefix: [0, 0, 4, 4, 9],
      total: 12,
      first: 6,
      length: 3,
      chunkFirst: 0,
      chunkLength: 5,
      expected: [3, 1, 1, 1, 1]
    },
    {
      prefix: [0, 0, 4, 4, 9],
      total: 12,
      first: 9,
      length: 1,
      chunkFirst: 0,
      chunkLength: 5,
      expected: [4, 1, 1, 1, 1]
    },
    {
      prefix: [0, 0, 4, 4, 9],
      total: 12,
      first: 6,
      length: 3,
      chunkFirst: 0,
      chunkLength: 2,
      expected: [2, 0, 0, 1, 1]
    },
    {
      prefix: [0, 0xffffffff, 0xffffffff],
      total: 0xffffffff,
      first: 3,
      length: 3,
      chunkFirst: 0,
      chunkLength: 3,
      expected: [0, 1, 1, 1, 1]
    },
    {
      prefix: [0, 0, 0],
      total: 0,
      first: 0,
      length: 3,
      chunkFirst: 0,
      chunkLength: 3,
      expected: [3, 0, 0, 1, 1]
    }
  ];
  try {
    for (const [index, testCase] of cases.entries()) {
      const rows = testCase.prefix.length;
      const offsets = createTransientView(graph, `ranges-${index}`, 'uint32', rows + 5);
      const total = createTransientView(graph, `range-total-${index}`, 'uint32', 1);
      const {updateNode} = createSpatialJoinCandidateQueryRange(graph, {
        id: `range-${index}`,
        offsets,
        total,
        firstSlot: testCase.first,
        length: testCase.length,
        candidateCapacity: 1000,
        queryFirst: testCase.chunkFirst,
        queryLength: testCase.chunkLength,
        queryCount: rows,
        metadataRow: rows
      });
      const [kernel] = getGPUCommandNodeKernelSources([updateNode]);
      expect(getReservedWGSLWords(kernel.source)).toEqual([]);
      const parameters = new Uint32Array(64);
      parameters[1] = 1;
      parameters.set(
        [
          testCase.first,
          testCase.length,
          testCase.chunkFirst,
          testCase.chunkLength,
          rows,
          rows,
          65535,
          1000
        ],
        36
      );
      // Adapt only uniform reads for wgsl_reflect; the range search and dispatch packing are production WGSL.
      const executionSource = kernel.source.replace(
        /getChunkParameter\((\d+)u\)/g,
        (_match, word) => `${parameters[Number(word)]}u`
      );
      const values = new Uint32Array(rows + 5);
      values.set(testCase.prefix);
      new WgslExec(new WgslParser().parse(executionSource)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {0: {0: values, 1: new Uint32Array([testCase.total])}},
        {constants: kernel.constants}
      );
      expect(Array.from(values.slice(rows))).toEqual(testCase.expected);
      expect(Array.from(values.slice(0, rows))).toEqual(testCase.prefix);
    }
  } finally {
    device.destroy();
  }
});

it('page match totals saturate and never wrap the running output position', () => {
  const {device, graph, contributor} = createFixture(false, 9);
  try {
    const nodes = new GPUSpatialPredicateJoin({
      ...contributor.props,
      candidateTileCapacity: 3
    }).getCommandNodes(graph);
    const [kernel] = getGPUCommandNodeKernelSources(
      nodes.filter(node => /tile-0-match-total$/.test(node.id))
    );
    expect(getReservedWGSLWords(kernel.source)).toEqual([]);
    const values: Record<string, Uint32Array> = {
      state: new Uint32Array([9, 0, 0xfffffffe, 0, 0, 3]),
      flags: new Uint32Array([1, 1, 1, 0xfffffffe]),
      flagOffsets: new Uint32Array([0, 1, 2])
    };
    const bindings = Object.fromEntries(
      new WgslReflect(kernel.source).storage.map(binding => [binding.binding, values[binding.name]])
    );
    const execute = () =>
      new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {0: bindings},
        {constants: kernel.constants}
      );
    execute();
    expect(values['state'][2]).toBe(0xffffffff);
    execute();
    expect(values['state'][2]).toBe(0xffffffff);
  } finally {
    device.destroy();
  }
});

for (const mode of ['distance', 'relate', 'attribute', 'anti'] as const) {
  it(`page shaders audit the ${mode} route without tile-dependent pipelines`, () => {
    const {device, graph, contributor} = createFixture(true, 9);
    try {
      const props = contributor.props;
      const unmatched = {
        ids: createTransientView(graph, 'unmatched', 'uint32', 1),
        status: {
          count: createTransientView(graph, 'unmatched-count', 'uint32', 1),
          overflow: createTransientView(graph, 'unmatched-overflow', 'uint32', 1)
        }
      };
      const paged = new GPUSpatialPredicateJoin({
        ...props,
        candidateTileCapacity: 2,
        predicate: mode === 'distance' ? 'dwithin' : mode === 'relate' ? 'coveredBy' : 'intersects',
        engine: mode === 'distance' ? 'relate' : undefined,
        distance:
          mode === 'distance' ? createTransientView(graph, 'distance', 'float32', 1) : undefined,
        onAttribute:
          mode === 'attribute'
            ? {
                left: createTransientView(graph, 'left-key', 'uint32', 1),
                right: createTransientView(graph, 'right-key', 'uint32', 1)
              }
            : undefined,
        how: mode === 'anti' ? 'anti' : 'inner',
        output: mode === 'anti' ? {unmatched} : props.output
      });
      const nodes = paged.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(paged.estimatedCommandNodeCount);
      for (const source of getParsedKernelSources(nodes).values())
        expect(getReservedWGSLWords(source)).toEqual([]);
      expect(
        getGPUCommandNodePipelineKeys(
          nodes.filter(node => /tile-\d+-write-candidates$/.test(node.id))
        ).size
      ).toBe(1);
    } finally {
      device.destroy();
    }
  });
}

it('rejects nonrepresentable WGSL distance and sphere-radius literals before declaration', () => {
  const {device, contributor} = createFixture();
  try {
    const props = contributor.props;
    expect(
      () =>
        new GPUSpatialJoinCandidates({
          left: props.left,
          right: props.right,
          output: props.output.pairs!,
          distance: 1e100
        })
    ).toThrow(/distance must be a finite number/);
    for (const sphereRadius of [1e100, 1e-100]) {
      expect(
        () =>
          new GPUSpatialPredicateJoin({
            ...props,
            predicate: 'dwithin',
            distance: 0,
            spatialContext: {
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle',
              sphereRadius
            }
          })
      ).toThrow(/sphereRadius.*finite number other than 0/);
    }
  } finally {
    device.destroy();
  }
});

it('rejects writable fields aliasing persistent prepared storage at construction', () => {
  const {device, graph, contributor} = createFixture(true);
  const prepared = new GPUSpatialJoinPrepared({
    geometry: contributor.props.right,
    spatialSort: true
  });
  try {
    graph.add(prepared);
    expect(
      () =>
        new GPUSpatialPredicateJoin({
          ...contributor.props,
          prepared,
          candidateTileCapacity: 2,
          output: {
            ...contributor.props.output,
            pairs: {
              ...contributor.props.output.pairs!,
              status: {...contributor.props.output.pairs!.status, count: prepared.storage.count}
            }
          }
        })
    ).toThrow(/outputs must not overlap/);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('rank-zero selection skips additional BVH probes, while later ranks retain exact selection', () => {
  const count = 32;
  for (const legacy of [false, true]) {
    const declarations = getCandidateRankDeclarations(count, legacy);
    for (const rank of [0, 7, 31]) {
      const buffers = {
        leftMinima: new Float32Array(2),
        leftMaxima: new Float32Array(2),
        nodeMinima: new Float32Array((2 * count - 1) * 2),
        nodeMaxima: new Float32Array((2 * count - 1) * 2),
        leafIds: Uint32Array.from({length: count}, (_, index) => count - index - 1),
        probeWork: new Uint32Array(1),
        result: new Uint32Array(1)
      };
      const bindings = Object.entries(buffers).map(
        ([name, values], binding) =>
          `@group(0) @binding(${binding}) var<storage, read_write> ${name}: array<${values instanceof Float32Array ? 'f32' : 'u32'}>;\nconst ${name}Offset: u32 = 0u;`
      );
      const source = `${bindings.join('\n')}\n${declarations}
@compute @workgroup_size(1) fn main() { result[0] = findFirstPageRight(0u, ${rank}u); }`;
      expect(getReservedWGSLWords(source)).toEqual([]);
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {
        0: Object.fromEntries(Object.values(buffers).map((values, index) => [index, values]))
      });
      expect(buffers.result[0]).toBe(rank);
      expect(buffers.probeWork[0]).toBe(!legacy && rank === 0 ? 0 : 5 * count);
    }
  }
});

it('predicate joins accept aligned fields packed in one buffer and ignore unused planar status views', () => {
  const {device, graph, contributor} = createFixture();
  try {
    const storage = createTransientView(graph, 'packed-output', 'uint32', 256);
    const field = (byteOffset: number, length: number) =>
      graph.createDataView(storage.buffer, {format: 'uint32', byteOffset, length});
    const pairs = {
      leftRows: field(0, 4),
      rightRows: field(256, 4),
      status: {count: field(512, 1), overflow: field(768, 1)}
    };
    const join = new GPUSpatialPredicateJoin({
      ...contributor.props,
      output: {
        pairs,
        status: {
          // Both fields are untouched for planar coordinates, so this alias is valid.
          antimeridianUncertainCount: pairs.status.count,
          poleUncertainCount: pairs.status.count
        }
      }
    });
    expect(() => join.getCommandNodes(graph)).not.toThrow();
    // CSR offsets are scanned separately from the pair scatter, so an aligned prefix may overlap.
    const weighted = new GPUSpatialPredicateJoin({
      ...contributor.props,
      id: 'weighted-packed',
      output: {
        pairs,
        weights: {
          offsets: field(16, 2),
          neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
          weights: createTransientView(graph, 'weights', 'float32', 4)
        }
      }
    });
    expect(() => weighted.getCommandNodes(graph)).not.toThrow();
    // The weight fill reads rightRows while writing neighbors: those aligned ranges coexist.
    expect(
      () =>
        new GPUSpatialPredicateJoin({
          ...weighted.props,
          output: {
            ...weighted.props.output,
            weights: {...weighted.props.output.weights!, neighbors: field(272, 4)}
          }
        })
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(
      () =>
        new GPUSpatialPredicateJoin({
          ...weighted.props,
          output: {
            ...weighted.props.output,
            weights: {...weighted.props.output.weights!, neighbors: field(520, 4)}
          }
        })
    ).toThrow(/physical buffer.*read-only and writable/);
  } finally {
    device.destroy();
  }
});

it('predicate binding preflight rejects missing STORAGE usage without declaring scratch', () => {
  const {device, graph, contributor} = createFixture();
  try {
    const leftRows = graph.createDataView(
      graph.createTransientBuffer({id: 'non-storage', byteLength: 16, usage: Buffer.COPY_DST}),
      {format: 'uint32', length: 4}
    );
    const join = new GPUSpatialPredicateJoin({
      ...contributor.props,
      output: {
        pairs: {...contributor.props.output.pairs!, leftRows}
      }
    });
    const allocation = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => join.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(allocation).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('predicate joins preflight prepared BVH STORAGE usage before declaring scratch', () => {
  const {device, graph, contributor} = createFixture();
  const prepared = new GPUSpatialJoinPrepared({geometry: contributor.props.right});
  try {
    graph.add(prepared);
    const join = new GPUSpatialPredicateJoin({...contributor.props, prepared});
    Object.defineProperty(prepared.storage.nodeMinima.buffer, 'usage', {
      configurable: true,
      value: Buffer.COPY_DST
    });
    const allocation = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => join.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(allocation).not.toHaveBeenCalled();
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('predicate scalar publication audits every wired field and matches independent aggregate expectations', () => {
  const {device, graph, contributor} = createFixture(false, 8);
  const word = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const status = (prefix: string) => ({
    count: word(`${prefix}-count`),
    requiredCount: word(`${prefix}-required`),
    overflow: word(`${prefix}-overflow`),
    candidateOverflow: word(`${prefix}-candidate-overflow`),
    uncertainCount: word(`${prefix}-uncertain`)
  });
  try {
    const join = new GPUSpatialPredicateJoin({
      ...contributor.props,
      output: {
        pairs: {...contributor.props.output.pairs!, status: status('pairs')},
        weights: {
          offsets: word('weight-offsets', 2),
          neighbors: word('weight-neighbors', 4),
          weights: createTransientView(graph, 'weight-values', 'float32', 4),
          status: status('weights')
        },
        status: {
          overflow: word('overall-overflow'),
          candidateCount: word('candidate-count'),
          uncertainCount: word('uncertain-count'),
          candidateTruncatedCount: word('truncated-count')
        }
      }
    });
    const kernels = getGPUCommandNodeKernelSources(
      join.getCommandNodes(graph).filter(node => /-finalize(?:-\d+)?$/.test(node.id))
    );
    expect(kernels).toHaveLength(3);
    // Nine requested candidates, six retained matches, pair capacity four and two uncertain
    // candidates: these are aggregate inputs to the scalar publisher, independent of geometry.
    const buffers: Record<string, Uint32Array> = {
      state: new Uint32Array([9, 2, 6, 0, 1, 0]),
      bvhOverflow: new Uint32Array([0])
    };
    for (const kernel of kernels) {
      expect(getReservedWGSLWords(kernel.source)).toEqual([]);
      const bindings: Record<number, Uint32Array> = {};
      for (const binding of new WgslReflect(kernel.source).storage) {
        buffers[binding.name] ??= new Uint32Array(1);
        bindings[binding.binding] = buffers[binding.name];
      }
      // The interpreter does not implement WGSL phony pointer assignments. Remove only these
      // binding keepalives; parse/audit the unmodified shader and execute every aggregate write.
      new WgslParser().parse(kernel.source);
      const executable = kernel.source.replace(/^\s*_ = &\w+;$/gm, '');
      new WgslExec(new WgslParser().parse(executable)).dispatchWorkgroups(
        'main',
        1,
        {0: bindings},
        {constants: kernel.constants}
      );
    }
    expect(
      Object.fromEntries(
        Object.entries(buffers)
          .filter(([name]) => name !== 'state' && name !== 'bvhOverflow')
          .map(([name, values]) => [name, values[0]])
      )
    ).toEqual({
      overflow: 1,
      weightsCount: 4,
      weightsTotal: 6,
      weightsOverflow: 1,
      weightsCandidateOverflow: 1,
      weightsUncertainCount: 2,
      pairsOverflow: 1,
      pairsCandidateOverflow: 1,
      pairsCount: 4,
      pairsTotal: 6,
      candidateCount: 9,
      uncertainCount: 2,
      candidateTruncatedCount: 1
    });
  } finally {
    device.destroy();
  }
});
