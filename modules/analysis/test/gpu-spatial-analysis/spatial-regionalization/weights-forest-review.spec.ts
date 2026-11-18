// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {GPUSkaterRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const flagName of ['overflow', 'candidateOverflow'] as const)
  it(`R2/R5 component-only forest status and SKATER canaries recover from ${flagName}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const flag = rig.input(Uint32Array.of(0), 'uint32', 1);
      const weights = {
        ...rig.uploadWeights({
          offsets: [0, 1, 3, 4],
          neighbors: [1, 0, 2, 1],
          weights: [1, 1, 1, 1],
          distances: []
        }),
        status: {[flagName]: flag}
      };
      const values = rig.input(Float32Array.of(1, 2, 4), 'float32', 3);
      const treeEdgeFlags = rig.output('uint32', 4);
      const componentLabels = rig.output('uint32', 3);
      const status = rig.output('uint32', 3);
      status.buffer.write(Uint32Array.of(88, 89, 97));
      rig.graph.add(
        new GPUSpatialWeightsMinimumSpanningTree({
          weights,
          values,
          standardize: false,
          output: {
            treeEdgeFlags: treeEdgeFlags.view,
            componentLabels: componentLabels.view,
            status: status.view
          }
        })
      );
      const labels = rig.output('uint32', 3);
      const regionCount = rig.output('uint32', 3);
      const cutEdges = rig.output('uint32', 4);
      const cutGains = rig.output('float32', 4);
      regionCount.buffer.write(Uint32Array.of(88, 98, 99));
      cutEdges.buffer.write(Uint32Array.of(88, 89, 98, 99));
      cutGains.buffer.write(Float32Array.of(88, 89, 98, 99));
      // Only the forest's packed status carries loss into this consumer; no weights loss port.
      rig.graph.add(
        new GPUSkaterRegions({
          weights: {...weights, status: undefined},
          values,
          treeStatus: status.view,
          treeEdgeFlags: treeEdgeFlags.view,
          componentLabels: componentLabels.view,
          maximumRegionCount: 3,
          parameters: rig.input(Uint32Array.of(2, 1), 'uint32', 2),
          output: {
            labels: labels.view,
            regionCount: regionCount.view,
            cutEdges: cutEdges.view,
            cutGains: cutGains.view
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const loss of [0, 9, 0]) {
          rig.bufferOf(flag).write(Uint32Array.of(loss));
          rig.bufferOf(weights.offsets).write(Uint32Array.from(loss ? [0, 1, 2, 2] : [0, 1, 3, 4]));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(status.buffer, 3)).toEqual([0, Number(loss !== 0), 97]);
          expect(await readUint32(componentLabels.buffer, 3)).toEqual(
            loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
          );
          expect(await readUint32(labels.buffer, 3)).toEqual(
            loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 2]
          );
          expect(await readUint32(regionCount.buffer, 3)).toEqual([loss ? 0 : 2, 98, 99]);
          expect(await readUint32(cutEdges.buffer, 4)).toEqual([
            loss ? 0xffffffff : 2,
            0xffffffff,
            98,
            99
          ]);
          const gains = await readFloat32(cutGains.buffer, 4);
          expect(gains.slice(2)).toEqual([98, 99]);
          if (loss) expect(gains.slice(0, 2).every(Number.isNaN)).toBe(true);
          else {
            expect(Number.isFinite(gains[0])).toBe(true);
            expect(gains[0]).toBeCloseTo(25 / 6, 5);
            expect(gains[1]).toBe(0);
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });

it('R3 REDCAP uses snapshotted loss and output ports on a lost chain and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const flag = rig.input(Uint32Array.of(9), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({
        offsets: [0, 1, 2, 2],
        neighbors: [1, 0, 2, 1],
        weights: [1, 1, 1, 1],
        distances: []
      }),
      status: {candidateOverflow: flag}
    };
    const labels = rig.output('uint32', 3);
    const count = rig.output('uint32', 1);
    const props = {
      weights,
      values: rig.input(Float32Array.of(1, 2, 4), 'float32', 3),
      parameters: rig.input(Uint32Array.of(1), 'uint32', 1),
      componentIterations: 3,
      standardize: false,
      output: {labels: labels.view, regionCount: count.view}
    };
    const contributor = new GPURedcapSingleLinkage(props);
    props.weights.status = {} as typeof props.weights.status;
    props.output = {
      labels: rig.output('uint32', 3).view,
      regionCount: rig.output('uint32', 1).view
    };
    rig.graph.add(contributor);
    const compiled = rig.graph.compile();
    try {
      for (const loss of [9, 0]) {
        rig.bufferOf(flag).write(Uint32Array.of(loss));
        rig.bufferOf(weights.offsets).write(Uint32Array.from(loss ? [0, 1, 2, 2] : [0, 1, 3, 4]));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(labels.buffer, 3)).toEqual(
          loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
        );
        expect(await readUint32(count.buffer, 1)).toEqual([loss ? 0 : 1]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

for (const flagName of ['overflow', 'candidateOverflow'] as const)
  it(`X4 REDCAP scalar tails survive ${flagName} loss and recovery`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const flag = rig.input(Uint32Array.of(0), 'uint32', 1);
      const weights = {
        ...rig.uploadWeights({
          offsets: [0, 1, 3, 4],
          neighbors: [1, 0, 2, 1],
          weights: [1, 1, 1, 1],
          distances: []
        }),
        status: {[flagName]: flag}
      };
      const labels = rig.output('uint32', 3);
      const regionCount = rig.output('uint32', 3);
      const converged = rig.output('uint32', 3);
      const iterationLimitReached = rig.output('uint32', 3);
      regionCount.buffer.write(Uint32Array.of(88, 98, 99));
      converged.buffer.write(Uint32Array.of(88, 88, 89));
      iterationLimitReached.buffer.write(Uint32Array.of(88, 90, 91));
      rig.graph.add(
        new GPURedcapSingleLinkage({
          weights,
          values: rig.input(Float32Array.of(1, 2, 4), 'float32', 3),
          parameters: rig.input(Uint32Array.of(1), 'uint32', 1),
          componentIterations: 3,
          standardize: false,
          output: {
            labels: labels.view,
            regionCount: regionCount.view,
            status: {converged: converged.view, iterationLimitReached: iterationLimitReached.view}
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const loss of [0, 9, 0]) {
          rig.bufferOf(flag).write(Uint32Array.of(loss));
          rig.bufferOf(weights.offsets).write(Uint32Array.from(loss ? [0, 1, 2, 2] : [0, 1, 3, 4]));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(labels.buffer, 3)).toEqual(
            loss ? [0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 0]
          );
          expect(await readUint32(regionCount.buffer, 3)).toEqual([loss ? 0 : 1, 98, 99]);
          expect(await readUint32(converged.buffer, 3)).toEqual([loss ? 0 : 1, 88, 89]);
          expect(await readUint32(iterationLimitReached.buffer, 3)).toEqual([0, 90, 91]);
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });

it('X5 REDCAP writes only the valid offset invalidCount scalar on repeated submissions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const backing = rig.output('uint32', 3);
    backing.buffer.write(Uint32Array.of(91, 92, 93));
    const invalidCount = rig.graph.createDataView(backing.view.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const values = rig.input(Float32Array.of(NaN, 2, NaN), 'float32', 3);
    const weights = rig.uploadWeights({
      offsets: [0, 1, 3, 4],
      neighbors: [1, 0, 2, 1],
      weights: [1, 1, 1, 1],
      distances: []
    });
    rig.graph.add(
      new GPURedcapSingleLinkage({
        weights,
        values,
        parameters: rig.input(Uint32Array.of(1), 'uint32', 1),
        componentIterations: 3,
        standardize: false,
        output: {labels: rig.output('uint32', 3).view, status: {invalidCount}}
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const invalid of [true, false]) {
        rig.bufferOf(values).write(Float32Array.from(invalid ? [NaN, 2, NaN] : [1, 2, 4]));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(backing.buffer, 3)).toEqual([91, invalid ? 2 : 0, 93]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('X7 reciprocal zero-weight forest links connect two rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = rig.uploadWeights({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [0, 0],
      distances: []
    });
    const flags = rig.output('uint32', 2);
    const labels = rig.output('uint32', 2);
    const ids = rig.output('uint32', 1);
    const count = rig.output('uint32', 1);
    const cost = rig.output('float32', 1);
    rig.graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        weights,
        values: rig.input(Float32Array.of(1, 4), 'float32', 2),
        standardize: false,
        output: {
          treeEdgeFlags: flags.view,
          componentLabels: labels.view,
          edges: {
            ids: ids.view,
            status: {count: count.view, overflow: rig.output('uint32', 1).view}
          },
          edgeDissimilarities: cost.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(flags.buffer, 2)).toEqual([1, 0]);
      expect(await readUint32(labels.buffer, 2)).toEqual([0, 0]);
      expect(await readUint32(count.buffer, 1)).toEqual([1]);
      expect(await readUint32(ids.buffer, 1)).toEqual([0]);
      expect(await readFloat32(cost.buffer, 1)).toEqual([9]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
