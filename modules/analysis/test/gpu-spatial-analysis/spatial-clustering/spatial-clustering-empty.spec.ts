// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialClustering} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-clustering';
import {getGPUSpatialClusteringParameterValues} from '../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes an empty clustering from a zero-byte input with packed separate-pass outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const positions = rig.input(new Float32Array(0), 'float32x2');
    const emptyRows = rig.graph.createDataView(positions.buffer, {format: 'uint32', length: 0});
    const packed = rig.output('uint32', 12);
    const sizes = rig.graph.createDataView(packed.buffer, {format: 'uint32', length: 4});
    const centroids = rig.graph.createDataView(packed.buffer, {
      format: 'float32x2',
      length: 4,
      byteOffset: 16
    });
    const clusterCount = rig.output('uint32', 1);
    const count = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const requiredCount = rig.output('uint32', 1);
    const drawInstanceCount = rig.output('uint32', 1);
    rig.run(
      new GPUSpatialClustering({
        positions,
        sourceIds: emptyRows,
        parameters: rig.input(
          getGPUSpatialClusteringParameterValues({
            bounds: [0, 0, 10, 10],
            epsilon: 1,
            minimumPoints: 1
          }),
          'float32'
        ),
        gridSize: [2, 2],
        output: {
          labels: emptyRows,
          rootRows: emptyRows,
          coreFlags: emptyRows,
          clusterCount,
          clusters: {
            // The empty route never writes ids, so aliasing this unused view is valid.
            ids: rig.graph.createDataView(packed.buffer, {format: 'uint32', length: 4}),
            status: {count, overflow, requiredCount}
          },
          clusterSizes: sizes,
          clusterCentroids: centroids,
          drawInstanceCount
        }
      })
    );
    expect(await rig.readUint(packed)).toEqual(new Array(12).fill(0));
    for (const field of [clusterCount, count, overflow, requiredCount, drawInstanceCount])
      expect(await rig.readUint(field)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});
