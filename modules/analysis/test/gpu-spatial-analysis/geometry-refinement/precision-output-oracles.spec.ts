// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createTopologyOutput} from './topology-output';
import {createRefinementOracleHarness} from './refinement-output-oracle-utils';

it('precision publishes independently enumerated labeled rectangles and isolated optional uncertainty ports', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Shapely set_precision oracle: precision-output-oracles.oracle.py. Noding has no crossings:
  // each of the eight source edges yields exactly one [0,1] interval, followed by zero padding.
  // Every undirected edge has two orientations; CCW sides own rings 0 and 2, CW sides 1 and 3.
  // Each rectangle has one bounded face and one unbounded walk, hence four required face rings.
  const endpoints = [
    0, 0, 4, 0, 4, 0, 4, 2, 4, 2, 0, 2, 0, 2, 0, 0, 10, 4, 13, 4, 13, 4, 13, 7, 13, 7, 10, 7, 10, 7,
    10, 4
  ];
  for (const optional of ['absent', 'nodedSegments', 'directedEdges', 'status'] as const) {
    const harness = createRefinementOracleHarness(device);
    const {graph, view, read} = harness;
    const output = createTopologyOutput(view);
    if (optional === 'status') output.status.uncertainCount = view('uint32', 1);
    else if (optional !== 'absent') output[optional].status.uncertainCount = view('uint32', 1);
    graph.add(
      new GPUPolygonPrecisionReduction({
        polygons: {
          kind: 'polygons',
          positions: view(
            'float32x2',
            10,
            [
              0.2, 0.2, 4.2, 0.2, 4.2, 2.2, 0.2, 2.2, 0.2, 0.2, 10.2, 4.2, 13.2, 4.2, 13.2, 7.2,
              10.2, 7.2, 10.2, 4.2
            ]
          ),
          ringOffsets: view('uint32', 3, [0, 5, 10]),
          polygonOffsets: view('uint32', 3, [0, 1, 2]),
          featureOffsets: view('uint32', 3, [0, 1, 2]),
          sourceIds: view('uint32', 2, [71, 4093])
        },
        gridSize: 1,
        intersectionCapacity: 32,
        lattice: view('float32', 4),
        output
      })
    );
    const compiled = graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        submitGraph(device, compiled);
        expect(await read(output.nodedSegments.endpoints)).toEqual([
          ...endpoints,
          ...Array(96).fill(0)
        ]);
        for (const [name, prefix] of [
          ['sourceFeatureIds', [71, 71, 71, 71, 4093, 4093, 4093, 4093]],
          ['sourceRingIds', [0, 0, 0, 0, 1, 1, 1, 1]],
          ['sourceEdgeIds', [0, 1, 2, 3, 5, 6, 7, 8]]
        ] as const)
          expect(await read(output.nodedSegments[name])).toEqual([
            ...prefix,
            ...Array(24).fill(0xffffffff)
          ]);
        expect(await read(output.nodedSegments.sourceStartParameters)).toEqual(Array(32).fill(0));
        expect(await read(output.nodedSegments.sourceEndParameters)).toEqual([
          ...Array(8).fill(1),
          ...Array(24).fill(0)
        ]);
        const directed = endpoints.flatMap((_, index) =>
          index % 4
            ? []
            : [
                ...endpoints.slice(index, index + 4),
                endpoints[index + 2],
                endpoints[index + 3],
                endpoints[index],
                endpoints[index + 1]
              ]
        );
        expect(await read(output.directedEdges.endpoints)).toEqual([
          ...directed,
          ...Array(192).fill(0)
        ]);
        expect(await read(output.directedEdges.sourceSegmentIds)).toEqual([
          0,
          0,
          1,
          1,
          2,
          2,
          3,
          3,
          4,
          4,
          5,
          5,
          6,
          6,
          7,
          7,
          ...Array(48).fill(0xffffffff)
        ]);
        expect(await read(output.directedEdges.sourceFeatureIds)).toEqual([
          ...Array(8).fill(71),
          ...Array(8).fill(4093),
          ...Array(48).fill(0xffffffff)
        ]);
        expect(await read(output.directedEdges.twinIds)).toEqual([
          1,
          0,
          3,
          2,
          5,
          4,
          7,
          6,
          9,
          8,
          11,
          10,
          13,
          12,
          15,
          14,
          ...Array(48).fill(0xffffffff)
        ]);
        expect(await read(output.directedEdges.ringIds)).toEqual([
          0,
          1,
          0,
          1,
          0,
          1,
          0,
          1,
          2,
          3,
          2,
          3,
          2,
          3,
          2,
          3,
          ...Array(48).fill(0xffffffff)
        ]);
        expect(await read(output.directedEdges.flags)).toEqual(Array(64).fill(0));
        for (const [status, count] of [
          [output.nodedSegments.status, 8],
          [output.directedEdges.status, 16]
        ] as const) {
          expect(await read(status.count)).toEqual([count]);
          expect(await read(status.requiredCount)).toEqual([count]);
          expect(await read(status.overflow)).toEqual([0]);
          expect(await read(status.candidateOverflow)).toEqual([0]);
          if (status.uncertainCount) expect(await read(status.uncertainCount)).toEqual([0]);
        }
        expect(await read(output.status.count)).toEqual([2]);
        expect(await read(output.status.requiredRingCount)).toEqual([4]);
        expect(await read(output.status.overflow)).toEqual([0]);
        expect(await read(output.status.candidateOverflow)).toEqual([0]);
        if (output.status.uncertainCount)
          expect(await read(output.status.uncertainCount)).toEqual([0]);
        expect(await read(output.polygons.featureOffsets)).toEqual([0, 1, 2, 2, 2, 2, 2, 2, 2]);
        expect((await read(output.polygons.sourceIds)).slice(0, 2)).toEqual([71, 4093]);
        expect(await read(output.diagnostics.edgeClasses)).toEqual([
          ...Array(8).fill(1),
          ...Array(24).fill(0)
        ]);
        // These diagnostics count source segments, not rings: 8 edges belong to closed faces.
        for (const [name, count] of [
          ['closedRingCount', 8],
          ['openChainCount', 0],
          ['cutEdgeCount', 0],
          ['dangleCount', 0],
          ['unusedEdgeCount', 0]
        ] as const)
          expect(await read(output.diagnostics[name])).toEqual([count]);
      }
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it('precision distinguishes final segment exhaustion from upstream candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  // An ordinary square needs four segments. Only two fit; polygonization is incomplete.
  output.nodedSegments = {
    ...output.nodedSegments,
    endpoints: view('float32x4', 2),
    sourceFeatureIds: view('uint32', 2),
    sourceRingIds: view('uint32', 2),
    sourceEdgeIds: view('uint32', 2),
    sourceStartParameters: view('float32', 2),
    sourceEndParameters: view('float32', 2)
  };
  output.directedEdges = {
    ...output.directedEdges,
    endpoints: view('float32x4', 4),
    sourceSegmentIds: view('uint32', 4),
    sourceFeatureIds: view('uint32', 4),
    twinIds: view('uint32', 4),
    ringIds: view('uint32', 4),
    flags: view('uint32', 4)
  };
  output.diagnostics.edgeClasses = view('uint32', 2);
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 5, [0, 0, 4, 0, 4, 2, 0, 2, 0, 0]),
        ringOffsets: view('uint32', 2, [0, 5]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: 1,
      intersectionCapacity: 16,
      lattice: view('float32', 4),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([2]);
    expect(await read(output.nodedSegments.status.requiredCount)).toEqual([4]);
    expect(await read(output.nodedSegments.status.overflow)).toEqual([1]);
    expect(await read(output.nodedSegments.status.candidateOverflow)).toEqual([0]);
    expect(await read(output.directedEdges.status.count)).toEqual([4]);
    expect(await read(output.directedEdges.status.requiredCount)).toEqual([8]);
    expect(await read(output.directedEdges.status.overflow)).toEqual([1]);
    expect(await read(output.status.candidateOverflow)).toEqual([1]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('precision nodes a hand-intersected bowtie with nonzero source interval starts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  // The two diagonals cross at (2,2), exactly halfway along source edges 0 and 2.
  // Stable source-edge/parameter order therefore gives six segments with the intervals below.
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 5, [0, 0, 4, 4, 0, 4, 4, 0, 0, 0]),
        ringOffsets: view('uint32', 2, [0, 5]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1]),
        sourceIds: view('uint32', 1, [907])
      },
      gridSize: 1,
      intersectionCapacity: 32,
      lattice: view('float32', 4),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([6]);
    expect((await read(output.nodedSegments.endpoints)).slice(0, 24)).toEqual([
      0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 0, 4, 0, 4, 2, 2, 2, 2, 4, 0, 4, 0, 0, 0
    ]);
    expect((await read(output.nodedSegments.sourceEdgeIds)).slice(0, 6)).toEqual([
      0, 0, 1, 2, 2, 3
    ]);
    expect((await read(output.nodedSegments.sourceFeatureIds)).slice(0, 6)).toEqual([
      907, 907, 907, 907, 907, 907
    ]);
    expect((await read(output.nodedSegments.sourceStartParameters)).slice(0, 6)).toEqual([
      0, 0.5, 0, 0, 0.5, 0
    ]);
    expect((await read(output.nodedSegments.sourceEndParameters)).slice(0, 6)).toEqual([
      0.5, 1, 1, 0.5, 1, 1
    ]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('precision implicit closures preserve caller edge rows across rings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 8, [0, 0, 4, 0, 4, 2, 0, 2, 10, 4, 13, 4, 13, 7, 10, 7]),
        ringOffsets: view('uint32', 3, [0, 4, 8]),
        polygonOffsets: view('uint32', 3, [0, 1, 2]),
        featureOffsets: view('uint32', 3, [0, 1, 2])
      },
      gridSize: 1,
      intersectionCapacity: 16,
      lattice: view('float32', 4),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([8]);
    // Each rectangle contributes its four original outgoing edges, including implicit closure.
    expect((await read(output.nodedSegments.sourceEdgeIds)).slice(0, 8)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7
    ]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('precision recursively peels a collapsed chain into independently counted dangles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  // Six source edges traverse the three-unit chain out and back. No bounded face exists.
  // Removing the degree-one ends exposes the internal edges, which are peeled in turn.
  // Shapely polygonize_full (precision-output-oracles.oracle.py) reports three unique dangles
  // and no cuts. Both source occurrences of each geometric edge count: all six source rows
  // are unused|openChain|dangle = 16|2|8 = 26, and both directed sides carry dangling flag 2.
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 7, [0, 0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0, 0, 0]),
        ringOffsets: view('uint32', 2, [0, 7]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: 1,
      intersectionCapacity: 64,
      lattice: view('float32', 4),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([6]);
    expect(await read(output.diagnostics.edgeClasses)).toEqual([
      26,
      26,
      26,
      26,
      26,
      26,
      ...Array(26).fill(0)
    ]);
    expect(await read(output.directedEdges.flags)).toEqual([
      ...Array(12).fill(2),
      ...Array(52).fill(0)
    ]);
    expect(await read(output.directedEdges.ringIds)).toEqual(Array(64).fill(0xffffffff));
    for (const [name, count] of [
      ['closedRingCount', 0],
      ['openChainCount', 6],
      ['cutEdgeCount', 0],
      ['dangleCount', 6],
      ['unusedEdgeCount', 6]
    ] as const)
      expect(await read(output.diagnostics[name])).toEqual([count]);
    expect(await read(output.status.count)).toEqual([0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('precision distinguishes a true cut bridge from recursively peeled dangles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  // Two unit square loops joined by a bridge, traversed out and back. Every bridge endpoint
  // is anchored by a cycle, so peeling cannot remove it. GEOS reports two faces, one unique
  // cut, no dangles; source rows 4 and 9 both represent that cut (see the adjacent oracle.py).
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view(
          'float32x2',
          11,
          [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 3, 0, 4, 0, 4, 1, 3, 1, 3, 0, 1, 0]
        ),
        ringOffsets: view('uint32', 2, [0, 11]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: 1,
      intersectionCapacity: 64,
      lattice: view('float32', 4),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([10]);
    expect(await read(output.diagnostics.edgeClasses)).toEqual([
      1,
      1,
      1,
      1,
      22,
      1,
      1,
      1,
      1,
      22,
      ...Array(22).fill(0)
    ]);
    for (const [name, count] of [
      ['closedRingCount', 8],
      ['openChainCount', 2],
      ['cutEdgeCount', 2],
      ['dangleCount', 0],
      ['unusedEdgeCount', 2]
    ] as const)
      expect(await read(output.diagnostics[name])).toEqual([count]);
    expect(await read(output.status.count)).toEqual([2]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('precision propagates exhausted crossing candidates to every completeness level', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  // Two disjoint bowties have one non-adjacent proper crossing each. Capacity one cannot
  // retain both events, even though the final 32-segment destination is ample.
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view(
          'float32x2',
          10,
          [0, 0, 4, 4, 0, 4, 4, 0, 0, 0, 10, 0, 14, 4, 10, 4, 14, 0, 10, 0]
        ),
        ringOffsets: view('uint32', 3, [0, 5, 10]),
        polygonOffsets: view('uint32', 3, [0, 1, 2]),
        featureOffsets: view('uint32', 3, [0, 1, 2])
      },
      gridSize: 1,
      intersectionCapacity: 1,
      lattice: view('float32', 4),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.candidateOverflow)).toEqual([1]);
    expect(await read(output.directedEdges.status.candidateOverflow)).toEqual([1]);
    expect(await read(output.status.candidateOverflow)).toEqual([1]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
