// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPUSkaterRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {createSubdividedContiguityFixture} from '../spatial-weights/contiguity-loss.test-utils';

for (const criterion of ['queen', 'rook'] as const)
  it(`final2 ${criterion} subdivided adjacency withholds the forest until its candidate budget fits`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const pairCapacity of [288, 1024]) {
      const rig = new WeightsRig(device);
      try {
        const overflow = rig.output('uint32', 1);
        const candidate = rig.output('uint32', 1);
        const producer = new GPUContiguityWeights({
          criterion,
          pairCapacity,
          polygons: rig.polygons(createSubdividedContiguityFixture()),
          output: {
            ...rig.weightsOutput(3, 72).spatialWeights,
            status: {overflow: overflow.view, candidateOverflow: candidate.view}
          }
        });
        const labels = rig.output('uint32', 3);
        const count = rig.output('uint32', 1);
        const edgesOverflow = rig.output('uint32', 1);
        const edgeCandidate = rig.output('uint32', 1);
        rig.run(
          producer,
          new GPUSpatialWeightsMinimumSpanningTree({
            weights: producer.outputWeights,
            values: rig.input(Float32Array.of(1, 2, 4), 'float32', 3),
            standardize: false,
            output: {
              treeEdgeFlags: rig.output('uint32', 72).view,
              componentLabels: labels.view,
              edges: {
                ids: rig.output('uint32', 2).view,
                status: {
                  count: count.view,
                  overflow: edgesOverflow.view,
                  candidateOverflow: edgeCandidate.view
                }
              }
            }
          })
        );
        const incomplete = pairCapacity === 288;
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidate.buffer, 1)).toEqual([Number(incomplete)]);
        expect(await readUint32(edgesOverflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(edgeCandidate.buffer, 1)).toEqual([Number(incomplete)]);
        expect(await readUint32(count.buffer, 1)).toEqual([incomplete ? 0 : 2]);
        expect(await readUint32(labels.buffer, 3)).toEqual(
          incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
        );
      } finally {
        rig.destroy();
      }
    }
  });

it.each([
  'overflow',
  'candidateOverflow'
] as const)('final2 a reused forest suppresses %s and recovers the missing link', async flagName => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const candidate = rig.input(Uint32Array.of(0), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({
        offsets: [0, 1, 3, 4],
        neighbors: [1, 0, 2, 1],
        weights: [1, 1, 1, 1],
        distances: []
      }),
      status: {[flagName]: candidate}
    };
    const labels = rig.output('uint32', 3);
    const count = rig.output('uint32', 1);
    const finalLoss = rig.output('uint32', 1);
    const candidateLoss = rig.output('uint32', 1);
    const treeEdgeFlags = rig.output('uint32', 4);
    const values = rig.input(Float32Array.of(1, 2, 4), 'float32', 3);
    const baselineLabels = rig.output('uint32', 3);
    const baselineFlags = rig.output('uint32', 4);
    const baselineCount = rig.output('uint32', 1);
    rig.graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        id: 'unflagged-forest',
        weights: {...weights, status: undefined},
        standardize: false,
        values,
        output: {
          treeEdgeFlags: baselineFlags.view,
          componentLabels: baselineLabels.view,
          edges: {
            ids: rig.output('uint32', 2).view,
            status: {count: baselineCount.view, overflow: rig.output('uint32', 1).view}
          }
        }
      })
    );
    rig.graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        weights,
        standardize: false,
        values,
        output: {
          treeEdgeFlags: treeEdgeFlags.view,
          componentLabels: labels.view,
          edges: {
            ids: rig.output('uint32', 2).view,
            status: {
              count: count.view,
              overflow: finalLoss.view,
              candidateOverflow: candidateLoss.view
            }
          }
        }
      })
    );
    const regionLabels = rig.output('uint32', 3);
    const regionCount = rig.output('uint32', 2);
    const cutEdges = rig.output('uint32', 2);
    const cutGains = rig.output('float32', 2);
    regionCount.buffer.write(Uint32Array.of(0, 71));
    cutEdges.buffer.write(Uint32Array.of(0, 72));
    cutGains.buffer.write(Float32Array.of(0, 73));
    rig.graph.add(
      new GPUSkaterRegions({
        weights,
        values,
        treeEdgeFlags: treeEdgeFlags.view,
        componentLabels: labels.view,
        maximumRegionCount: 2,
        parameters: rig.input(Uint32Array.of(1, 1), 'uint32', 2),
        output: {
          labels: regionLabels.view,
          regionCount: regionCount.view,
          cutEdges: cutEdges.view,
          cutGains: cutGains.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const incomplete of [0, 1, 0]) {
        rig.bufferOf(candidate).write(Uint32Array.of(incomplete));
        rig
          .bufferOf(weights.offsets)
          .write(Uint32Array.from(incomplete ? [0, 1, 2, 2] : [0, 1, 3, 4]));
        rig
          .bufferOf(weights.neighbors)
          .write(Uint32Array.from(incomplete ? [1, 0, 0, 0] : [1, 0, 2, 1]));
        // Each submission must overwrite stale finite gains, including suppressed frames.
        cutGains.buffer.write(Float32Array.of(17));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(labels.buffer, 3)).toEqual(
          incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
        );
        expect(await readUint32(count.buffer, 1)).toEqual([incomplete ? 0 : 2]);
        expect(await readUint32(finalLoss.buffer, 1)).toEqual([0]);
        expect(await readUint32(candidateLoss.buffer, 1)).toEqual([incomplete]);
        expect(await readUint32(treeEdgeFlags.buffer, 4)).toEqual(
          incomplete ? [0, 0, 0, 0] : [1, 0, 1, 0]
        );
        expect(await readUint32(regionLabels.buffer, 3)).toEqual(
          incomplete ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
        );
        expect(await readUint32(regionCount.buffer, 1)).toEqual([incomplete ? 0 : 1]);
        expect(await readUint32(cutEdges.buffer, 1)).toEqual([0xffffffff]);
        const [gain] = await readFloat32(cutGains.buffer, 1);
        if (incomplete) expect(Number.isNaN(gain)).toBe(true);
        else expect(gain).toBe(0);
        if (!incomplete) {
          expect(await readUint32(labels.buffer, 3)).toEqual(
            await readUint32(baselineLabels.buffer, 3)
          );
          expect(await readUint32(treeEdgeFlags.buffer, 4)).toEqual(
            await readUint32(baselineFlags.buffer, 4)
          );
          expect(await readUint32(count.buffer, 1)).toEqual(
            await readUint32(baselineCount.buffer, 1)
          );
        }
        expect((await readUint32(regionCount.buffer, 2))[1]).toBe(71);
        expect((await readUint32(cutEdges.buffer, 2))[1]).toBe(72);
        expect((await readFloat32(cutGains.buffer, 2))[1]).toBe(73);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
