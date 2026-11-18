// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createScanNearestZoneNodes} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-nearest-index';
import {
  getSeparationWGSL,
  resolveInterpolationSeparation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/separation-wgsl';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {
  GPUExpectationBasedSpaceTimeScan,
  getGPUExpectationBasedScanParameterValues
} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import {getGPUSpatialScanStatisticParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {requireTestDevice} from '../../utils/require-test-device';

it('coincident centers can be displaced by the lowest zone row in either exact route', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const positions = rig.input(new Float32Array(6), 'float32x2', 3);
  const indexed = rig.output('uint32', 3);
  const exhaustive = rig.output('uint32', 3);
  const separationWGSL = getSeparationWGSL(resolveInterpolationSeparation('scan-ties', undefined));
  try {
    rig.run({
      getCommandNodes(graph) {
        return [
          ...createScanNearestZoneNodes(graph, {
            id: 'indexed-ties',
            operation: 'scan-ties',
            positions,
            neighbors: indexed.view,
            distances: rig.output('float32', 3).view,
            zones: 3,
            listLength: 1,
            geographic: false,
            separationWGSL,
            nearestZoneSearch: 'indexed'
          }),
          ...createScanNearestZoneNodes(graph, {
            id: 'exhaustive-ties',
            operation: 'scan-ties',
            positions,
            neighbors: exhaustive.view,
            distances: rig.output('float32', 3).view,
            zones: 3,
            listLength: 1,
            geographic: false,
            separationWGSL,
            nearestZoneSearch: 'full-scan'
          })
        ];
      }
    });
    expect(await indexed.readUint32()).toEqual([0, 0, 0]);
    expect(await exhaustive.readUint32()).toEqual([0, 0, 0]);
  } finally {
    rig.destroy();
  }
});

for (const geographic of [false, true])
  for (const listLength of [1, 2, 4, 7])
    it(`GPU coordinate index preserves exact keys and row ties: geographic ${geographic}, list ${listLength}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new GraphRig(device);
      const coordinates = geographic
        ? Float32Array.of(179.99, 0, -179.99, 0, 180, 0, -180, 0, 0, 90, 120, 90, 0, -90)
        : Float32Array.of(0, 0, 1, 0, -1, 0, 0, 1, 0, -1, -0, 0, 20, 0);
      const positions = rig.input(coordinates, 'float32x2', 7);
      const indexedRows = rig.output('uint32', 7 * listLength);
      const exhaustiveRows = rig.output('uint32', 7 * listLength);
      const indexedKeys = rig.output('float32', 7 * listLength);
      const exhaustiveKeys = rig.output('float32', 7 * listLength);
      const separation = resolveInterpolationSeparation(
        'scan-index-test',
        geographic ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} : undefined
      );
      try {
        rig.run({
          getCommandNodes(graph) {
            return [
              ...createScanNearestZoneNodes(graph, {
                id: 'indexed',
                operation: 'scan-index-test',
                positions,
                neighbors: indexedRows.view,
                distances: indexedKeys.view,
                zones: 7,
                listLength,
                separationWGSL: getSeparationWGSL(separation),
                geographic,
                nearestZoneSearch: 'indexed'
              }),
              ...createScanNearestZoneNodes(graph, {
                id: 'exhaustive',
                operation: 'scan-index-test',
                positions,
                neighbors: exhaustiveRows.view,
                distances: exhaustiveKeys.view,
                zones: 7,
                listLength,
                separationWGSL: getSeparationWGSL(separation),
                geographic,
                nearestZoneSearch: 'full-scan'
              })
            ];
          }
        });
        expect(await indexedRows.readUint32()).toEqual(await exhaustiveRows.readUint32());
        // Read the key bits to require exact ordering before any inverse-distance rounding.
        expect(await indexedKeys.readUint32()).toEqual(await exhaustiveKeys.readUint32());
        if (!geographic && listLength === 4)
          expect((await indexedRows.readUint32()).slice(0, 4)).toEqual([0, 5, 1, 2]);
        if (geographic && listLength === 2)
          expect((await indexedRows.readUint32()).slice(8, 10)).toEqual([4, 5]);
      } finally {
        rig.destroy();
      }
    });

for (const expectationBased of [false, true])
  it(`both complete scan results agree across routes and position updates: expectationBased ${expectationBased}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device);
    const positions = rig.output('float32', 8);
    positions.buffer.write(Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0));
    // Eight scalar coordinate words form four vector rows.
    const positionView = rig.graph.createDataView(positions.view.buffer, {
      format: 'float32x2',
      length: 4
    });
    const cases = rig.input(Uint32Array.of(10, 0, 0, 0), 'uint32');
    const baseline = rig.input(Float32Array.of(1, 1, 1, 1), 'float32');
    const results = (['indexed', 'full-scan'] as const).map(nearestZoneSearch => {
      const members = rig.output('uint32', expectationBased ? 2 : 4);
      const cluster = rig.output('float32', expectationBased ? 4 : 8);
      const statistics = rig.output('float32', 2);
      const summary = rig.output('uint32', expectationBased ? 5 : 4);
      const common = {
        id: nearestZoneSearch,
        positions: positionView,
        cases,
        baseline,
        timeBuckets: 1,
        nearestZoneSearch
      };
      const contributor = expectationBased
        ? new GPUExpectationBasedSpaceTimeScan({
            ...common,
            maximumNeighbors: 2,
            maximumWindowLength: 1,
            maximumReplicates: 1,
            parameters: rig.input(
              getGPUExpectationBasedScanParameterValues({seed: 1, replicateCount: 1}),
              'uint32'
            ),
            output: {
              clusterZones: members.view,
              clusterStatistics: cluster.view,
              statistics: statistics.view,
              summary: summary.view
            }
          })
        : new GPUSpatialScanStatistic({
            ...common,
            maximumWindowZones: 1,
            maximumClusters: 1,
            maximumPermutations: 1,
            parameters: rig.input(
              getGPUSpatialScanStatisticParameterValues({seed: 1, permutationCount: 1}),
              'uint32'
            ),
            output: {
              clusterIndices: members.view,
              clusterStatistics: cluster.view,
              statistics: statistics.view,
              summary: summary.view
            }
          });
      return {contributor, members, cluster, statistics, summary};
    });
    try {
      rig.run(...results.map(result => result.contributor));
      for (const coordinates of [
        [0, 0, 1, 0, 2, 0, 3, 0],
        [0, 3, 0, -1, 0, 2, 0, 0]
      ]) {
        positions.buffer.write(Float32Array.from(coordinates));
        rig.resubmit();
        const [indexed, exhaustive] = results;
        expect(await indexed.members.readUint32()).toEqual(await exhaustive.members.readUint32());
        expect(await indexed.summary.readUint32()).toEqual(await exhaustive.summary.readUint32());
        expect(await indexed.cluster.readFloat32()).toEqual(await exhaustive.cluster.readFloat32());
        expect(await indexed.statistics.readFloat32()).toEqual(
          await exhaustive.statistics.readFloat32()
        );
        expect((await indexed.summary.readUint32())[0]).toBe(1);
        const expected = expectationBased ? 10 * Math.log(10) - 9 : 10 * Math.log(4);
        const actual = (await indexed.statistics.readFloat32())[0];
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeCloseTo(expected, 5);
      }
    } finally {
      rig.destroy();
    }
  });
