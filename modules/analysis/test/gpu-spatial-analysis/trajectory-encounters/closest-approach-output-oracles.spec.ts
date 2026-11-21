// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryClosestApproach} from '../../../src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-closest-approach';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';

it('closest approach pins required pairs and aggregate candidate loss independently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const loss of ['none', 'entries', 'hits', 'pairs'] as const)
    for (const optional of ['required', 'aggregate', 'both', 'none'] as const) {
      // Candidate starvation has nondeterministic retained membership. Only execute/read
      // requiredCount where all three independently enumerated candidates are available.
      if (
        (loss === 'entries' || loss === 'hits') &&
        (optional === 'required' || optional === 'both')
      )
        continue;
      const harness = new GraphHarness(device, `closest-oracle-${loss}-${optional}`);
      // Three parallel unit segments at y=0,1,2 during [0,1], all within distance 3:
      // unordered qualifying track pairs are exactly {0,1}, {0,2}, {1,2}.
      const trajectory = importTrajectory(harness, {
        positions: [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, 1],
          [0, 2],
          [1, 2]
        ],
        times: Float32Array.of(0, 1, 0, 1, 0, 1),
        trackOffsets: [0, 2, 4, 6]
      });
      const ids = harness.output('ids', 'uint32', loss === 'pairs' ? 1 : 3);
      const count = harness.output('count', 'uint32', 1);
      const overflow = harness.output('overflow', 'uint32', 1);
      const entryOverflow = harness.output('entryOverflow', 'uint32', 1);
      const hitOverflow = harness.output('hitOverflow', 'uint32', 1);
      const required =
        optional === 'required' || optional === 'both'
          ? harness.output('required', 'uint32', 1)
          : undefined;
      const candidate =
        optional === 'aggregate' || optional === 'both'
          ? harness.output('candidate', 'uint32', 1)
          : undefined;
      for (const output of [ids, count, overflow, entryOverflow, hitOverflow, required, candidate])
        output?.buffer.write(new Uint32Array(output.buffer.byteLength / 4).fill(93));
      harness.graph.add(
        new GPUTrajectoryClosestApproach({
          ...trajectory,
          distance: harness.input('distance', Float32Array.of(3), 'float32', 1),
          cellSize: 10,
          timeCellSize: 10,
          entryCapacity: loss === 'entries' ? 1 : 128,
          hitCapacity: loss === 'hits' ? 1 : 128,
          output: {
            pairs: {
              ids: ids.view,
              status: {
                count: count.view,
                overflow: overflow.view,
                ...(required ? {requiredCount: required.view} : {})
              }
            },
            status: {
              entryOverflow: entryOverflow.view,
              hitOverflow: hitOverflow.view,
              ...(candidate ? {candidateOverflow: candidate.view} : {})
            }
          }
        })
      );
      const result = await harness.run(async () => ({
        count: await readUint32(count.buffer, 1),
        overflow: await readUint32(overflow.buffer, 1),
        required: required ? await readUint32(required.buffer, 1) : undefined,
        candidate: candidate ? await readUint32(candidate.buffer, 1) : undefined
      }));
      if (loss === 'none') expect(result.count).toEqual([3]);
      if (loss === 'pairs') expect(result.count).toEqual([1]);
      if (loss === 'none' || loss === 'pairs') {
        if (required) expect(result.required).toEqual([3]);
        expect(result.overflow).toEqual([Number(loss === 'pairs')]);
      }
      if (candidate)
        expect(result.candidate).toEqual([Number(loss === 'entries' || loss === 'hits')]);
    }
});
