// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {getSpatialJoinCandidateNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-candidates';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('predicate candidate scan preserves the retained prefix after synthetic uint32 carries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  // Inject large per-row counts into the real candidate pipeline without doing billions of probes.
  const counts = createInputBuffer(device, new Uint32Array([2, 0xffffffff, 0, 2, 1]));
  const minima = createInputBuffer(device, new Float32Array(10));
  const treeBounds = createInputBuffer(device, new Float32Array(2));
  const leaves = createInputBuffer(device, new Uint32Array([0]));
  const pairs = createOutputBuffer(device, 8);
  pairs.write(new Uint32Array(8).fill(0xffffffff));
  const state = createOutputBuffer(device, 4);
  const truncated = createOutputBuffer(device, 5);
  const offsets = createOutputBuffer(device, 5);
  const buffers = [counts, minima, treeBounds, leaves, pairs, state, truncated, offsets];
  const syntheticCounts = importGraphBuffer(graph, 'counts', counts, 'uint32');
  const boundView = importGraphBuffer(graph, 'bounds', minima, 'float32x2');
  const treeView = importGraphBuffer(graph, 'tree', treeBounds, 'float32x2');
  const outputOffsets = importGraphBuffer(graph, 'offsets', offsets, 'uint32');
  const nodes = getSpatialJoinCandidateNodes(graph, {
    id: 'candidate-carry',
    leftCount: 5,
    rightCount: 1,
    leftMinima: boundView,
    leftMaxima: boundView,
    tree: {
      nodeMinima: treeView,
      nodeMaxima: treeView,
      leafIds: importGraphBuffer(graph, 'leaves', leaves, 'uint32'),
      internalNodeCount: 0
    },
    candidateCapacity: 4,
    candidatePairs: importGraphBuffer(graph, 'pairs', pairs, 'uint32x2'),
    margin: 0,
    state: importGraphBuffer(graph, 'state', state, 'uint32'),
    candidateTruncated: importGraphBuffer(graph, 'truncated', truncated, 'uint32')
  });
  const countNode = nodes.find(node => node.id === 'candidate-carry-count-candidates')!;
  const saturateNode = nodes.find(node => node.id === 'candidate-carry-scan-candidates-saturate')!;
  const countResource = countNode.resources!.at(-1)!;
  if (!('buffer' in countResource)) throw new Error('Expected count storage');
  const countView = countResource.buffer as GraphDataView<'uint32'>;
  const offsetResource = saturateNode.resources![2];
  if (!('buffer' in offsetResource)) throw new Error('Expected offset storage');
  const scannedOffsets = offsetResource.buffer as GraphDataView<'uint32'>;
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'inject-counts',
      operation: 'CandidateCarryTest',
      invocationCount: 5,
      bindings: [
        {name: 'source', view: syntheticCounts, type: 'u32', access: 'read'},
        {name: 'destination', view: countView, type: 'u32', access: 'read_write'}
      ],
      body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
    })
  );
  for (const node of nodes) if (node !== countNode) graph.add(node);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'publish-offsets',
      operation: 'CandidateCarryTest',
      invocationCount: 5,
      bindings: [
        {name: 'source', view: scannedOffsets, type: 'u32', access: 'read'},
        {name: 'destination', view: outputOffsets, type: 'u32', access: 'read_write'}
      ],
      body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(offsets)).toEqual([0, 2, 0xffffffff, 0xffffffff, 0xffffffff]);
    expect((await readUint32(state))[0]).toBe(0xffffffff);
    expect(await readUint32(truncated)).toEqual([0, 1, 0, 1, 1]);
    expect(await readUint32(pairs)).toEqual([
      0, 0, 0xffffffff, 0xffffffff, 1, 0, 0xffffffff, 0xffffffff
    ]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('sorted-BVH candidate pages equal the Cartesian oracle across a split query', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const observed: number[][] = [];
  for (const firstSlot of [0, 3, 6, 9, 12]) {
    const length = Math.min(3, 13 - firstSlot);
    const fixture = createGeometryFixture(device, {
      inputs: {
        minima: {values: new Float32Array(8), format: 'float32x2'},
        maxima: {values: new Float32Array(8), format: 'float32x2'},
        treeMinima: {values: new Float32Array(14), format: 'float32x2'},
        treeMaxima: {values: new Float32Array(14), format: 'float32x2'},
        leaves: {values: new Uint32Array([3, 0, 2, 1]), format: 'uint32'},
        offsets: {values: new Uint32Array([0, 4, 8, 12]), format: 'uint32'},
        counts: {values: new Uint32Array([4, 4, 4, 4]), format: 'uint32'}
      },
      outputs: {pairs: {format: 'uint32', length: 6}, state: {format: 'uint32', length: 5}},
      create: ({inputs, outputs}) => ({
        getCommandNodes(graph) {
          return getSpatialJoinCandidateNodes(graph, {
            id: 'sorted-page',
            leftCount: 4,
            rightCount: 4,
            leftMinima: inputs['minima'] as GraphDataView<'float32x2'>,
            leftMaxima: inputs['maxima'] as GraphDataView<'float32x2'>,
            tree: {
              nodeMinima: inputs['treeMinima'] as GraphDataView<'float32x2'>,
              nodeMaxima: inputs['treeMaxima'] as GraphDataView<'float32x2'>,
              leafIds: inputs['leaves'] as GraphDataView<'uint32'>,
              internalNodeCount: 3,
              spatialSort: true
            },
            candidateCapacity: 13,
            margin: 0,
            candidatePairs: graph.createDataView(outputs['pairs'].buffer, {
              format: 'uint32x2',
              length: 3
            }),
            state: outputs['state'] as GraphDataView<'uint32'>,
            page: {
              firstSlot,
              length,
              initialize: false,
              counts: inputs['counts'] as GraphDataView<'uint32'>,
              offsets: inputs['offsets'] as GraphDataView<'uint32'>
            }
          });
        }
      })
    });
    try {
      const result = await fixture.run();
      for (let index = 0; index < length; index++)
        observed.push(result['pairs'].slice(index * 2, index * 2 + 2));
    } finally {
      fixture.destroy();
    }
  }
  // Every query intersects every leaf. Restore source-right order despite shuffled leaf IDs.
  expect(observed).toEqual(
    Array.from({length: 16}, (_, index) => [Math.floor(index / 4), index % 4]).slice(0, 13)
  );
});

it('point join evaluates more slab candidates than one core binding can hold', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pointCount = 2001;
  const featureCount = 4500;
  // 9,004,500 four-word slab candidates exceed the old 8,388,608-slot core binding ceiling.
  // Source inputs are small: every point is strictly inside every unit square.
  const positions = new Float32Array(featureCount * 8);
  for (let feature = 0; feature < featureCount; feature++)
    positions.set([-1, -1, 1, -1, 1, 1, -1, 1], feature * 8);
  const fixture = createGeometryFixture(device, {
    inputs: {
      points: {values: new Float32Array(pointCount * 2), format: 'float32x2'},
      vertices: {values: positions, format: 'float32x2'},
      features: {
        values: Uint32Array.from({length: featureCount + 1}, (_, index) => index),
        format: 'uint32'
      },
      polygons: {
        values: Uint32Array.from({length: featureCount + 1}, (_, index) => index),
        format: 'uint32'
      },
      rings: {
        values: Uint32Array.from({length: featureCount + 1}, (_, index) => 4 * index),
        format: 'uint32'
      }
    },
    outputs: {
      assignment: {format: 'uint32', length: pointCount},
      featureCounts: {format: 'uint32', length: featureCount},
      candidates: {format: 'uint32', length: 1},
      candidateOverflow: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      ids: {format: 'uint32', length: 3},
      count: {format: 'uint32', length: 1},
      required: {format: 'uint32', length: 1},
      matchOverflow: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) =>
      new GPUPointInPolygonJoin({
        points: {kind: 'points', positions: inputs['points'] as GraphDataView<'float32x2'>},
        polygons: {
          kind: 'polygons',
          positions: inputs['vertices'] as GraphDataView<'float32x2'>,
          featureOffsets: inputs['features'] as GraphDataView<'uint32'>,
          polygonOffsets: inputs['polygons'] as GraphDataView<'uint32'>,
          ringOffsets: inputs['rings'] as GraphDataView<'uint32'>
        },
        candidateCapacity: pointCount * featureCount,
        candidateTileCapacity: 8_388_608,
        edgeSlabs: true,
        spatialSort: false,
        querySort: false,
        output: {
          pointFeatureIds: outputs['assignment'] as GraphDataView<'uint32'>,
          featureCounts: outputs['featureCounts'] as GraphDataView<'uint32'>,
          matches: {
            ids: outputs['ids'] as GraphDataView<'uint32'>,
            status: {
              count: outputs['count'] as GraphDataView<'uint32'>,
              requiredCount: outputs['required'] as GraphDataView<'uint32'>,
              overflow: outputs['matchOverflow'] as GraphDataView<'uint32'>
            }
          },
          status: {
            overflow: outputs['overflow'] as GraphDataView<'uint32'>,
            candidateCount: outputs['candidates'] as GraphDataView<'uint32'>,
            candidateOverflow: outputs['candidateOverflow'] as GraphDataView<'uint32'>
          }
        }
      })
  });
  try {
    const result = await fixture.run();
    expect(result['assignment']).toEqual(new Array(pointCount).fill(0));
    expect(result['featureCounts']).toEqual([pointCount, ...new Array(featureCount - 1).fill(0)]);
    expect(result['candidates']).toEqual([pointCount * featureCount]);
    expect(result['candidateOverflow']).toEqual([0]);
    expect(result['ids']).toEqual([0, 1, 2]);
    expect(result['count']).toEqual([3]);
    expect(result['required']).toEqual([pointCount]);
    expect(result['overflow']).toEqual([1]);
    expect(result['matchOverflow']).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});
