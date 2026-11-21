// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {expect, it} from 'vitest';
import {GPUNetworkNoding} from '../../../src/gpu-network/network-noding/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

for (const route of ['candidate', 'aggregate'] as const) {
  it(`W5-02 exposes a dropped crossing with ample final storage through ${route}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const loss = output('loss', 'uint32');
    const count = output('piece-count', 'uint32');
    const finalOverflow = output('piece-overflow', 'uint32');
    const positions = output('piece-positions', 'float32x2', 8);
    try {
      graph.add(
        new GPUNetworkNoding({
          lines: {
            kind: 'lines',
            positions: input('positions', 'float32x2', [0, 1, 2, 1, 1, 0, 1, 2]),
            lineOffsets: input('lines', 'uint32', [0, 2, 4])
          },
          tolerance: input('tolerance', 'float32', [0]),
          intersectionCapacity: 8,
          leafCapacity: 2,
          spatialSort: false,
          output: {
            pieces: {
              geometry: {
                kind: 'lines',
                positions,
                lineOffsets: output('piece-offsets', 'uint32', 5)
              },
              sourceIds: output('piece-rows', 'uint32', 4),
              status: {
                count,
                overflow: finalOverflow,
                candidateOverflow: route === 'candidate' ? loss : undefined
              }
            },
            nodes: {
              positions: output('nodes', 'float32x2', 5),
              status: {count: output('node-count', 'uint32')}
            },
            edges: {
              fromNodes: output('from', 'uint32', 4),
              toNodes: output('to', 'uint32', 4),
              lengths: output('lengths', 'float32', 4)
            },
            topology: {
              offsets: output('offsets', 'uint32', 6),
              neighbors: output('neighbors', 'uint32', 8),
              weights: output('weights', 'float32', 8)
            },
            status: route === 'aggregate' ? {overflow: loss} : undefined
          }
        })
      );
      context.run();
      // Only segment slot 0 fits the right tree. Self pairs require left < right, so
      // the crossing with segment slot 2 is dropped. Both original lines fit unchanged.
      expect(await read(count)).toEqual([2]);
      expect(await read(finalOverflow)).toEqual([0]);
      expect(await read(loss)).toEqual([1]);
      expect((await read(positions)).slice(0, 8)).toEqual([0, 1, 2, 1, 1, 0, 1, 2]);
    } finally {
      context.destroy();
    }
  });
}

it('network noding reports candidate loss when three crossing pairs share one intersection slot', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withDiagnostic of [true, false]) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const count = output('piece-count', 'uint32');
    const nodeCount = output('node-count', 'uint32');
    const candidateOverflow = withDiagnostic ? output('candidate-overflow', 'uint32') : undefined;
    const overflow = output('whole-network-overflow', 'uint32');
    const vertexSourceEdges = output('source-edge-flags', 'uint32', 10);
    const positionsBuffer = device.createBuffer({
      data: Float32Array.of(0, 1, 2, 1, 1, 0, 1, 2, 0, 0, 2, 2),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    const positions = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 6);
    const pieceOverflow = output('piece-overflow', 'uint32');
    // Three segments meet properly at (1,1), so there are C(3,2)=3 intersecting pairs.
    // A capacity of one keeps exactly one pair, cutting two lines into four pieces while
    // the third stays whole: five pieces and seven unique endpoints, whichever pair wins.
    graph.add(
      new GPUNetworkNoding({
        lines: {
          kind: 'lines',
          positions,
          lineOffsets: input('lines', 'uint32', [0, 2, 4, 6])
        },
        tolerance: input('tolerance', 'float32', [0]),
        intersectionCapacity: 1,
        output: {
          status: {overflow},
          pieces: {
            vertexSourceEdges,
            geometry: {
              kind: 'lines',
              positions: output('piece-positions', 'float32x2', 10),
              lineOffsets: output('piece-offsets', 'uint32', 6)
            },
            sourceIds: output('piece-rows', 'uint32', 5),
            status: {count, overflow: pieceOverflow, candidateOverflow}
          },
          nodes: {positions: output('nodes', 'float32x2', 7), status: {count: nodeCount}},
          edges: {
            fromNodes: output('from', 'uint32', 5),
            toNodes: output('to', 'uint32', 5),
            lengths: output('lengths', 'float32', 5)
          },
          topology: {
            offsets: output('offsets', 'uint32', 8),
            neighbors: output('neighbors', 'uint32', 10),
            weights: output('weights', 'float32', 10)
          }
        }
      })
    );
    try {
      context.run();
      expect(await read(count)).toEqual([5]);
      expect(await read(nodeCount)).toEqual([7]);
      expect(await read(pieceOverflow)).toEqual([0]);
      if (candidateOverflow) expect(await read(candidateOverflow)).toEqual([1]);
      expect(await read(overflow)).toEqual([1]);
      // First piece starts on input edge zero. Pair loss sets the self-contact bit
      // on every provenance word even when the optional candidate flag is omitted.
      expect((await read(vertexSourceEdges))[0]).toBe(0x80000000);
      // Same declaration, now three disjoint unit segments: no candidates are lost and
      // all three final pieces fit. Every outer/candidate flag must recover on re-encoding.
      positionsBuffer.write(Float32Array.of(0, 0, 1, 0, 0, 2, 1, 2, 0, 4, 1, 4));
      context.run();
      expect(await read(count)).toEqual([3]);
      expect(await read(pieceOverflow)).toEqual([0]);
      if (candidateOverflow) expect(await read(candidateOverflow)).toEqual([0]);
      expect(await read(overflow)).toEqual([0]);
    } finally {
      context.destroy();
      positionsBuffer.destroy();
    }
  }
});

it('network noding writes independent crossing provenance and candidate diagnostics alone and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Horizontal (0,1)->(2,1) crosses vertical (1,0)->(1,2), producing four unit pieces.
  // Piece order follows input lines; each two-vertex piece starts on input edge 0 or 2.
  for (const selected of [
    undefined,
    'candidateOverflow',
    'vertexSourceEdges',
    'sourceFeatureIds'
  ] as const) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const pieces = {
      geometry: {
        kind: 'lines' as const,
        positions: output('piece-positions', 'float32x2', 8),
        lineOffsets: output('piece-offsets', 'uint32', 5)
      },
      sourceIds: output('piece-ids', 'uint32', 4),
      sourcePartRows: output('piece-rows', 'uint32', 4),
      status: {
        count: output('pieces', 'uint32'),
        overflow: output('piece-overflow', 'uint32'),
        candidateOverflow:
          selected === 'candidateOverflow' ? output('candidate', 'uint32') : undefined
      },
      vertexSourceEdges:
        selected === 'vertexSourceEdges' ? output('source-edges', 'uint32', 8) : undefined,
      sourceFeatureIds:
        selected === 'sourceFeatureIds' ? output('feature-ids', 'uint32', 4) : undefined
    };
    const nodes = {
      positions: output('nodes', 'float32x2', 5),
      status: {count: output('node-count', 'uint32')}
    };
    const edges = {
      fromNodes: output('from', 'uint32', 4),
      toNodes: output('to', 'uint32', 4),
      lengths: output('lengths', 'float32', 4)
    };
    const topology = {
      offsets: output('offsets', 'uint32', 6),
      neighbors: output('neighbors', 'uint32', 8),
      weights: output('weights', 'float32', 8)
    };
    graph.add(
      new GPUNetworkNoding({
        lines: {
          kind: 'lines',
          positions: input('positions', 'float32x2', [0, 1, 2, 1, 1, 0, 1, 2]),
          lineOffsets: input('lines', 'uint32', [0, 2, 4]),
          sourceIds: input('identities', 'uint32', [91, 17])
        },
        tolerance: input('tolerance', 'float32', [0]),
        intersectionCapacity: 8,
        output: {pieces, nodes, edges, topology}
      })
    );
    try {
      context.run();
      expect(await read(pieces.status.count)).toEqual([4]);
      expect(await read(pieces.status.overflow)).toEqual([0]);
      expect(await read(pieces.sourceIds)).toEqual([91, 91, 17, 17]);
      expect(await read(pieces.sourcePartRows)).toEqual([0, 0, 1, 1]);
      expect(await read(pieces.geometry.lineOffsets)).toEqual([0, 2, 4, 6, 8]);
      expect(await read(pieces.geometry.positions)).toEqual([
        0, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2
      ]);
      expect(await read(nodes.status.count)).toEqual([5]);
      expect(await read(edges.fromNodes)).toEqual([0, 2, 1, 2]);
      expect(await read(edges.toNodes)).toEqual([2, 4, 2, 3]);
      expect(await read(edges.lengths)).toEqual([1, 1, 1, 1]);
      expect(await read(topology.offsets)).toEqual([0, 1, 2, 6, 7, 8]);
      expect(await read(topology.neighbors)).toEqual([2, 2, 0, 4, 1, 3, 2, 2]);
      expect(await read(topology.weights)).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
      if (pieces.status.candidateOverflow)
        expect(await read(pieces.status.candidateOverflow)).toEqual([0]);
      if (pieces.sourceFeatureIds)
        expect(await read(pieces.sourceFeatureIds)).toEqual([91, 91, 17, 17]);
      // End-of-piece source-edge rows are explicitly unspecified by the API.
      if (pieces.vertexSourceEdges) {
        const sourceEdges = await read(pieces.vertexSourceEdges);
        expect([0, 2, 4, 6].map(index => sourceEdges[index])).toEqual([0, 0, 2, 2]);
      }
    } finally {
      context.destroy();
    }
  }
});
