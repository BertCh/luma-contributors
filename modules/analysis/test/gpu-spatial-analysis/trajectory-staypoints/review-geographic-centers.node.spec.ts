// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUTrajectoryStaypoints} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {GPUTrajectoryLocations} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const positions = Float32Array.of(180 - 2 ** -16, 0, -180 - 2 ** -16, 0, 177, 0);

it('B1 CPU staypoint and location centers identify near-seam equivalent coordinates', () => {
  // Only two physical points: their longitude mean is 178.49999237060547,
  // halfway between f32 neighbors; nearest-even output is 178.5 degrees.
  const stays = computeTrajectoryStaypointsOnCPU(
    {
      positions,
      timestamps: Float32Array.of(0, 1, 2),
      trackOffsets: Uint32Array.of(0, 3),
      parameters: Float32Array.of(400000, 1, 0, 1)
    },
    {capacity: 1, spatialContext}
  );
  const locations = computeTrajectoryLocationsOnCPU(
    {
      centroids: positions,
      tracks: new Uint32Array(3),
      count: Uint32Array.of(3),
      parameters: Float32Array.of(400000, 0, 0, 0)
    },
    {spatialContext}
  );
  expect(Array.from(stays.centroids)).toEqual([178.5, 0]);
  expect(Array.from(locations.centers).slice(0, 2)).toEqual([178.5, 0]);
});

for (const producer of ['staypoints', 'locations'] as const) {
  it(`B1 actual ${producer} WGSL center kernel identifies near-seam equivalent coordinates`, () => {
    const context = createKernelAuditContext(`review-${producer}`);
    try {
      recordWGSLKernelNodes(context.graph);
      const points = context.view('float32x2', 3);
      const centers = context.view('float32x2', producer === 'staypoints' ? 1 : 3);
      const contributor =
        producer === 'staypoints'
          ? new GPUTrajectoryStaypoints({
              id: 'review',
              spatialContext,
              positions: points,
              timestamps: context.view('float32', 3),
              trackOffsets: context.view('uint32', 2),
              parameters: context.view('float32', 4),
              output: {
                ids: context.view('uint32', 1),
                centroids: centers,
                status: {
                  count: context.view('uint32', 1),
                  candidateOverflow: context.view('uint32', 1),
                  overflow: context.view('uint32', 1)
                }
              }
            })
          : new GPUTrajectoryLocations({
              id: 'review',
              spatialContext,
              centroids: points,
              tracks: context.view('uint32', 3),
              count: context.view('uint32', 1),
              parameters: context.view('float32', 4),
              output: {
                locationIds: context.view('uint32', 3),
                centers,
                status: {
                  count: context.view('uint32', 1),
                  iterationLimitReached: context.view('uint32', 1)
                }
              }
            });
      const nodes = contributor.getCommandNodes(context.graph);
      const identifier = producer === 'staypoints' ? 'review-centroids' : 'review-centers';
      const kernel = nodes.find(node => node.id === identifier);
      const record = getWGSLKernelNodeRecord(context.graph, identifier);
      if (!kernel || !record) throw new Error(`missing center kernel ${identifier}`);
      const memory = createCPUGraphMemory();
      const write = (name: string, values: Uint32Array) => {
        const binding = record.bindings.find(binding => binding.name === name);
        if (!binding) throw new Error(`missing center binding ${name}`);
        memory.write(binding.view, values);
      };
      memory.write(points, positions);
      if (producer === 'staypoints') {
        write('stayTotal', Uint32Array.of(1));
        write('startRows', Uint32Array.of(0));
        write('endRows', Uint32Array.of(3));
      } else {
        write('stayCount', Uint32Array.of(3));
        write('labels', new Uint32Array(3));
        write('changed', Uint32Array.of(0));
        write('groupEnds', Uint32Array.of(3, 3, 3));
        write('idMap', new Uint32Array(3));
        write('firstRows', new Uint32Array(3));
      }
      executeCommandNodesOnCPU([kernel], memory);
      expect(Array.from(memory.readFloat32(centers, 2))).toEqual([178.5, 0]);
    } finally {
      context.device.destroy();
    }
  });
}

it('N1 both CPU centers correct the endpoint after float32 rounding', () => {
  const points = Float32Array.of(179.99998474121094, 0, 180, 0);
  const stays = computeTrajectoryStaypointsOnCPU(
    {
      positions: points,
      timestamps: Float32Array.of(0, 1),
      trackOffsets: Uint32Array.of(0, 2),
      parameters: Float32Array.of(400000, 1, 0, 1)
    },
    {capacity: 1, spatialContext}
  );
  const locations = computeTrajectoryLocationsOnCPU(
    {
      centroids: points,
      tracks: new Uint32Array(2),
      count: Uint32Array.of(2),
      parameters: Float32Array.of(400000, 0, 0, 0)
    },
    {spatialContext}
  );
  // The unrounded mean 179.99999237060547 rounds to +180, which wraps to -180.
  expect(Array.from(stays.centroids)).toEqual([-180, 0]);
  expect(Array.from(locations.centers).slice(0, 2)).toEqual([-180, 0]);
});
