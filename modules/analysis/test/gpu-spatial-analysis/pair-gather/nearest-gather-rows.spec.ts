// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPairGather} from '../../../src/gpu-spatial-analysis/pair-gather/index';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

for (const sourceIds of [undefined, [1, 0], [100, 200]]) {
  it(`nearest join to gather uses rows with sourceIds=${sourceIds} and clears reused slots`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        queries: {values: Float32Array.of(0, 0, 10, 0, 100, 0), format: 'float32x2'},
        features: {values: Float32Array.of(0, 0, 10, 0), format: 'float32x2'},
        radius: {values: Float32Array.of(1), format: 'float32'},
        attributes: {values: Uint32Array.of(10, 20), format: 'uint32'},
        ...(sourceIds
          ? {sourceIds: {values: Uint32Array.from(sourceIds), format: 'uint32' as const}}
          : {})
      },
      outputs: {
        neighborIds: {format: 'uint32', length: 3},
        neighborRows: {format: 'uint32', length: 3},
        neighborCounts: {format: 'uint32', length: 3},
        neighborDistances: {format: 'float32', length: 3},
        joinOverflow: {format: 'uint32', length: 1},
        leftRows: {format: 'uint32', length: 3},
        rightRows: {format: 'uint32', length: 3},
        rowCount: {format: 'uint32', length: 1},
        attributes: {format: 'uint32', length: 3},
        overflow: {format: 'uint32', length: 1},
        candidateOverflow: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) => ({
        getCommandNodes(graph) {
          const neighborRows = outputs['neighborRows'] as GraphDataView<'uint32'>;
          const neighborCounts = outputs['neighborCounts'] as GraphDataView<'uint32'>;
          const joinOverflow = outputs['joinOverflow'] as GraphDataView<'uint32'>;
          const join = new GPUNearestFeatureJoin({
            spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'},
            queries: {kind: 'points', positions: inputs['queries'] as GraphDataView<'float32x2'>},
            features: {
              kind: 'points',
              positions: inputs['features'] as GraphDataView<'float32x2'>,
              ...(sourceIds ? {sourceIds: inputs['sourceIds'] as GraphDataView<'uint32'>} : {})
            },
            radius: inputs['radius'] as GraphDataView<'float32'>,
            output: {
              neighborIds: outputs['neighborIds'] as GraphDataView<'uint32'>,
              neighborRows,
              neighborCounts,
              neighborDistances: outputs['neighborDistances'] as GraphDataView<'float32'>,
              status: {overflow: joinOverflow}
            }
          });
          const neighbors = {rows: neighborRows, counts: neighborCounts, capacity: 1};
          const gatherProps = {
            neighbors,
            // Nearest's aggregate includes tie/BVH loss; preserve it as candidate loss here.
            candidateOverflowSources: [joinOverflow],
            rightColumns: [
              {
                source: inputs['attributes'] as GraphDataView<'uint32'>,
                output: outputs['attributes'] as GraphDataView<'uint32'>
              }
            ],
            output: {
              leftRows: outputs['leftRows'] as GraphDataView<'uint32'>,
              rightRows: outputs['rightRows'] as GraphDataView<'uint32'>,
              rowCount: outputs['rowCount'] as GraphDataView<'uint32'>,
              status: {
                overflow: outputs['overflow'] as GraphDataView<'uint32'>,
                candidateOverflow: outputs['candidateOverflow'] as GraphDataView<'uint32'>
              }
            }
          };
          const gather = new GPUPairGather(gatherProps);
          // Construction fixes the row binding and capacity even if the caller reuses its maps.
          neighbors.rows = outputs['neighborIds'] as GraphDataView<'uint32'>;
          neighbors.capacity = 0;
          gatherProps.neighbors = {
            rows: outputs['neighborIds'] as GraphDataView<'uint32'>,
            counts: neighborCounts,
            capacity: 1
          };
          return [...join.getCommandNodes(graph), ...gather.getCommandNodes(graph)];
        }
      })
    });
    try {
      const first = await fixture.run();
      expect(first['neighborIds']).toEqual([...(sourceIds ?? [0, 1]), 0xffffffff]);
      expect(first['neighborRows']).toEqual([0, 1, 0xffffffff]);
      expect(first['neighborCounts']).toEqual([1, 1, 0]);
      expect(first['neighborDistances']).toEqual([0, 0, -1]);
      expect(first['rowCount']).toEqual([2]);
      expect(first['leftRows']).toEqual([0, 1, 0xffffffff]);
      expect(first['rightRows']).toEqual([0, 1, 0xffffffff]);
      expect(first['attributes']).toEqual([10, 20, 0xffffffff]);
      expect(first['joinOverflow']).toEqual([0]);
      expect(first['overflow']).toEqual([0]);
      expect(first['candidateOverflow']).toEqual([0]);

      fixture.writeInput('queries', Float32Array.of(10, 0, 0, 0, 100, 0));
      const swapped = await fixture.run();
      expect(swapped['neighborRows']).toEqual([1, 0, 0xffffffff]);
      expect(swapped['neighborIds']).toEqual([
        (sourceIds ?? [0, 1])[1],
        (sourceIds ?? [0, 1])[0],
        0xffffffff
      ]);
      expect(swapped['attributes']).toEqual([20, 10, 0xffffffff]);

      fixture.writeInput('queries', Float32Array.of(100, 0, 100, 0, 100, 0));
      const empty = await fixture.run();
      expect(empty['neighborRows']).toEqual(Array(3).fill(0xffffffff));
      expect(empty['neighborIds']).toEqual(Array(3).fill(0xffffffff));
      expect(empty['neighborCounts']).toEqual([0, 0, 0]);
      expect(empty['neighborDistances']).toEqual([-1, -1, -1]);
      expect(empty['attributes']).toEqual(Array(3).fill(0xffffffff));
      expect(empty['rowCount']).toEqual([0]);
      expect(empty['overflow']).toEqual([0]);
      expect(empty['candidateOverflow']).toEqual([0]);
      expect(fixture.getCompileCount()).toBe(0);
    } finally {
      fixture.destroy();
    }
  });
}
