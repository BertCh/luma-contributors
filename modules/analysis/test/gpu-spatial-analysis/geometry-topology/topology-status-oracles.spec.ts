// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULineNoding,
  GPUPolygonize,
  GPUMakeValid
} from '../../../src/gpu-spatial-analysis/geometry-topology/index';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {createTopologyOutput} from '../geometry-refinement/topology-output';
import {requireTestDevice} from '../../utils/require-test-device';

// Two diagonal segments cross exactly at (1,1), a representable lattice point.
// Each splits into two pieces: required=4; output capacity 2/8 writes 2/4.
// Eight event slots exceed the single intersection. Integer geometry has no
// uncertain predicate. Snapping to grid 1 preserves precisely the same pieces.
for (const snapped of [false, true])
  for (const capacity of [2, 8]) {
    it(`noding status has a complete poisoned crossing oracle (snapped ${snapped}, capacity ${capacity})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const status = {
        count: output('count', 'uint32', 1),
        requiredCount: output('required', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1),
        candidateOverflow: output('candidate', 'uint32', 1),
        uncertainCount: output('uncertain', 'uint32', 1)
      };
      const common = {
        lines: {
          kind: 'lines' as const,
          positions: input('positions', 'float32x2', Float32Array.of(0, 0, 2, 2, 0, 2, 2, 0)),
          lineOffsets: input('offsets', 'uint32', Uint32Array.of(0, 2, 4))
        },
        intersectionCapacity: 8,
        output: {
          endpoints: output('endpoints', 'float32x4', capacity),
          sourceFeatureIds: output('features', 'uint32', capacity),
          sourceRingIds: output('rings', 'uint32', capacity),
          sourceEdgeIds: output('edges', 'uint32', capacity),
          sourceStartParameters: output('starts', 'float32', capacity),
          sourceEndParameters: output('ends', 'float32', capacity),
          status
        }
      };
      graph.add(
        snapped
          ? new GPUSnapRoundedNoding({
              ...common,
              gridSize: 1,
              output: {
                ...common.output,
                lattice: output('lattice', 'float32', 4),
                latticeEndpoints: output('lattice-endpoints', 'float32x4', capacity)
              }
            })
          : new GPULineNoding(common)
      );
      const compiled = graph.compile();
      try {
        expect(
          await fixture.submitTwice(compiled, () =>
            fixture.readRows([
              status.count,
              status.requiredCount,
              status.overflow,
              status.candidateOverflow,
              status.uncertainCount
            ])
          )
        ).toEqual(capacity === 2 ? [2, 4, 1, 0, 0] : [4, 4, 0, 0, 0]);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }

// One isolated square has four noded segments and eight opposite half-edges.
// Its two directed face walks are the bounded square and the unbounded exterior;
// requiredRingCount=2 but only one polygon feature is published. All four source
// segments lie on a closed ring; none is open, dangling, cut, unused or uncertain.
// Make-valid keeps the same one valid square. A replay of all coincident vertices
// has no nonzero edge, no face and no ring, requiring every count to clear.
for (const repair of [false, true]) {
  it(`polygon topology status observes every stage and diagnostic word (repair ${repair})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    let serial = 0;
    const base = createTopologyOutput((format, length) =>
      output(`topology-${serial++}`, format, length)
    );
    const topology = {
      ...base,
      nodedSegments: {
        ...base.nodedSegments,
        status: {
          ...base.nodedSegments.status,
          uncertainCount: output('noded-uncertain', 'uint32', 1)
        }
      },
      directedEdges: {
        ...base.directedEdges,
        status: {
          ...base.directedEdges.status,
          uncertainCount: output('directed-uncertain', 'uint32', 1)
        }
      },
      status: {...base.status, uncertainCount: output('uncertain', 'uint32', 1)}
    };
    const positions = input(
      'positions',
      'float32x2',
      Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 0, 0)
    );
    const offsets = input('offsets', 'uint32', Uint32Array.of(0, 5));
    const common = {
      intersectionCapacity: 16,
      precision: {
        vertexTolerance: 1e-5,
        predicates: 'exact-or-uncertain' as const,
        storage: 'float32' as const
      },
      output: topology
    };
    graph.add(
      repair
        ? new GPUMakeValid({
            ...common,
            polygons: {
              kind: 'polygons',
              positions,
              ringOffsets: offsets,
              polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
              featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1))
            }
          })
        : new GPUPolygonize({...common, lines: {kind: 'lines', positions, lineOffsets: offsets}})
    );
    const compiled = graph.compile();
    const getStatusViews = (status: typeof topology.nodedSegments.status) => [
      status.count,
      status.requiredCount!,
      status.overflow,
      status.candidateOverflow!,
      status.uncertainCount
    ];
    try {
      for (const present of [true, false, true]) {
        fixture.writeInput(
          positions,
          Float32Array.from(present ? [0, 0, 2, 0, 2, 2, 0, 2, 0, 0] : Array(10).fill(0))
        );
        const actual = await fixture.submitTwice(compiled, async () => ({
          noded: await fixture.readRows(getStatusViews(topology.nodedSegments.status)),
          directed: await fixture.readRows(getStatusViews(topology.directedEdges.status)),
          final: await fixture.readRows([
            topology.status.count,
            topology.status.requiredRingCount!,
            topology.status.overflow,
            topology.status.candidateOverflow!,
            topology.status.uncertainCount
          ]),
          diagnostics: await fixture.readRows([
            topology.diagnostics.closedRingCount,
            topology.diagnostics.openChainCount,
            topology.diagnostics.cutEdgeCount,
            topology.diagnostics.dangleCount,
            topology.diagnostics.unusedEdgeCount
          ])
        }));
        expect(actual).toEqual(
          present
            ? {
                noded: [4, 4, 0, 0, 0],
                directed: [8, 8, 0, 0, 0],
                final: [1, 2, 0, 0, 0],
                diagnostics: [4, 0, 0, 0, 0]
              }
            : {
                noded: [0, 0, 0, 0, 0],
                directed: [0, 0, 0, 0, 0],
                final: [0, 0, 0, 0, 0],
                diagnostics: [0, 0, 0, 0, 0]
              }
        );
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
