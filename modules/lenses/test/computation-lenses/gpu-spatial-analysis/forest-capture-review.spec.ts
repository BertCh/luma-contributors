// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {createSubdividedContiguityFixture} from '../../../../analysis/test/gpu-spatial-analysis/spatial-weights/contiguity-loss.test-utils';
import {runExample as runForest} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-minimum-spanning-tree.example';
import {runExample as runSkater} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/skater-regions.example';
import {
  createSpatialRecordingGraph,
  uploadSpatialView,
  runSpatialScenario,
  finishSpatialForestRecord
} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {STATISTICS_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics';

for (const kind of ['forest', 'skater'] as const)
  for (const pairCapacity of [288, 1024])
    it(`R1 ${kind} capture respects candidate loss at pairCapacity ${pairCapacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = createSpatialRecordingGraph(
        device,
        `forest-capture-review-${kind}-${pairCapacity}`
      );
      const uploads: {destroy(): void}[] = [];
      const upload = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        name: string,
        values: Float32Array | Uint32Array,
        format: Format
      ) => {
        const resource = uploadSpatialView(graph, name, values, format);
        uploads.push(resource);
        return resource.view;
      };
      try {
        const polygons = createSubdividedContiguityFixture();
        const tracts = {
          kind: 'polygons' as const,
          positions: upload('positions', polygons.positions, 'float32x2'),
          ringOffsets: upload('rings', polygons.ringOffsets, 'uint32'),
          polygonOffsets: upload('polygons', polygons.polygonOffsets, 'uint32'),
          featureOffsets: upload('features', polygons.featureOffsets, 'uint32')
        };
        const execution = await (kind === 'forest' ? runForest : runSkater)(device, {
          graph,
          tracts,
          values: upload('values', Float32Array.of(1, 2, 4), 'float32'),
          parameters: upload('parameters', Uint32Array.of(1, 1), 'uint32'),
          pairCapacity
        });
        const views = {
          ...execution.views,
          ...(kind === 'forest'
            ? {treeSegmentOffsets: upload('treeSegmentOffsets', Uint32Array.of(0, 2, 4), 'uint32')}
            : {})
        };
        const capture = runSpatialScenario({
          id: `forest-capture-${kind}-${pairCapacity}`,
          graph,
          views,
          encodeOptions: {parameters: undefined},
          entry:
            STATISTICS_OPERATORS[
              kind === 'forest' ? 'GPUSpatialWeightsMinimumSpanningTree' : 'GPUSkaterRegions'
            ],
          choices:
            kind === 'forest'
              ? {criterion: 'queen', standardize: true}
              : {criterion: 'queen', regionCount: 1, minimumRegionSize: 1, maximumRegionCount: 8},
          datasetVersion: 'subdivided-three-tracts',
          seed: 1,
          axes: {
            keySpaces: [
              {id: 'edge', label: 'Forest edge', indexSpace: '1d'},
              {id: 'edgeVertex', label: 'Edge endpoint', indexSpace: '1d'},
              {id: 'edgeOffset', label: 'Segment offset', indexSpace: '1d'}
            ]
          },
          tags: {}
        });
        if (pairCapacity === 288) await expect(capture).rejects.toThrow(/weightsCandidateOverflow/);
        else {
          const record = await capture;
          const snapshots = record.captures!.snapshots;
          expect(
            Array.from(snapshots.find(snapshot => snapshot.name === 'weightsOverflow')!.values)
          ).toEqual([0]);
          expect(
            Array.from(
              snapshots.find(snapshot => snapshot.name === 'weightsCandidateOverflow')!.values
            )
          ).toEqual([0]);
          if (kind === 'forest') {
            const finished = finishSpatialForestRecord(record);
            expect(
              Array.from(
                finished.captures!.snapshots.find(snapshot => snapshot.name === 'edgeCount')!.values
              )
            ).toEqual([2]);
            expect(finished.axes.keySpaces?.find(space => space.id === 'edge')?.size).toBe(2);
          } else
            expect(
              Array.from(snapshots.find(snapshot => snapshot.name === 'regionCount')!.values)
            ).toEqual([1]);
        }
      } finally {
        uploads.forEach(resource => resource.destroy());
      }
    });
