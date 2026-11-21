// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkNoding} from '../../../src/gpu-network/network-noding/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createStatusOracleHarness} from '../network-analysis/network-status-oracles-harness';

for (const pieceCapacity of [3, 4])
  it(`writes retained and required noding counts independently (${pieceCapacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createStatusOracleHarness(device);
    const count = harness.output('uint32', 1);
    const required = harness.output('uint32', 1);
    const vertices = harness.output('uint32', 1);
    const requiredVertices = harness.output('uint32', 1);
    const nodeCount = harness.output('uint32', 1);
    const nodeRequired = harness.output('uint32', 1);
    const pieceOverflow = harness.output('uint32', 1);
    const overflow = harness.output('uint32', 1);
    const uncertain = harness.output('uint32', 1);
    // Integer horizontal and vertical segments cross at (1,1), strictly inside
    // both segments. Four pieces use eight vertices and five unique nodes.
    // Every determinant is an exact small integer, so uncertainty is zero.
    // Four piece slots fit; three retain only a prefix of the required geometry.
    await harness.run(
      new GPUNetworkNoding({
        lines: {
          kind: 'lines',
          positions: harness.input('float32x2', Float32Array.of(0, 1, 2, 1, 1, 0, 1, 2)),
          lineOffsets: harness.input('uint32', Uint32Array.of(0, 2, 4))
        },
        tolerance: harness.input('float32', Float32Array.of(0)),
        intersectionCapacity: 4,
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: harness.output('float32x2', 2 * pieceCapacity),
              lineOffsets: harness.output('uint32', pieceCapacity + 1)
            },
            sourceIds: harness.output('uint32', pieceCapacity),
            status: {
              count,
              requiredCount: required,
              vertexCount: vertices,
              requiredVertexCount: requiredVertices,
              overflow: pieceOverflow
            }
          },
          nodes: {
            positions: harness.output('float32x2', 5),
            status: {count: nodeCount, requiredCount: nodeRequired}
          },
          edges: {
            fromNodes: harness.output('uint32', pieceCapacity),
            toNodes: harness.output('uint32', pieceCapacity),
            lengths: harness.output('float32', pieceCapacity)
          },
          topology: {
            offsets: harness.output('uint32', 6),
            neighbors: harness.output('uint32', 2 * pieceCapacity),
            weights: harness.output('float32', 2 * pieceCapacity)
          },
          status: {overflow, uncertainCount: uncertain}
        }
      }),
      [
        read => {
          // Four pieces need eight vertices even when only three pieces/six
          // vertices can be retained. Copying retained counts into requirements fails.
          expect(
            [count, required, vertices, requiredVertices, pieceOverflow].flatMap(read)
          ).toEqual(pieceCapacity === 3 ? [3, 4, 6, 8, 1] : [4, 4, 8, 8, 0]);
          if (pieceCapacity === 4)
            expect(
              [
                count,
                required,
                vertices,
                requiredVertices,
                nodeCount,
                nodeRequired,
                pieceOverflow,
                overflow,
                uncertain
              ].flatMap(read)
            ).toEqual([4, 4, 8, 8, 5, 5, 0, 0, 0]);
        }
      ]
    );
  });

it('R6 one uncertifiable pair raises noding uncertainty and exact replay clears it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createStatusOracleHarness(device);
  // One pair whose orientation terms span about 2^226, outside the bounded
  // exact-predicate accumulator (the independent line-split uncertainty fixture).
  const positions = harness.input(
    'float32x2',
    Float32Array.of(1e-30, 1e-30, 2e-30, 2e-30, 1.5e-30, 1.5e-30, 1e38, 1e38)
  );
  const uncertainCount = harness.output('uint32', 1);
  await harness.run(
    new GPUNetworkNoding({
      lines: {
        kind: 'lines',
        positions,
        lineOffsets: harness.input('uint32', Uint32Array.of(0, 2, 4))
      },
      tolerance: harness.input('float32', Float32Array.of(0)),
      intersectionCapacity: 4,
      output: {
        pieces: {
          geometry: {
            kind: 'lines',
            positions: harness.output('float32x2', 8),
            lineOffsets: harness.output('uint32', 5)
          },
          sourceIds: harness.output('uint32', 4),
          status: {count: harness.output('uint32', 1), overflow: harness.output('uint32', 1)}
        },
        nodes: {
          positions: harness.output('float32x2', 8),
          status: {count: harness.output('uint32', 1)}
        },
        edges: {
          fromNodes: harness.output('uint32', 4),
          toNodes: harness.output('uint32', 4),
          lengths: harness.output('float32', 4)
        },
        topology: {
          offsets: harness.output('uint32', 9),
          neighbors: harness.output('uint32', 8),
          weights: harness.output('float32', 8)
        },
        status: {uncertainCount, overflow: harness.output('uint32', 1)}
      }
    }),
    [
      read => {
        expect(read(uncertainCount)).toEqual([1]);
        // Small integer determinants certify the exact crossing, with no uncertain pair.
        harness.rewrite(positions, Float32Array.of(0, 1, 2, 1, 1, 0, 1, 2));
      },
      read => expect(read(uncertainCount)).toEqual([0])
    ]
  );
});
