// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_BROWNIAN_BRIDGE_MAXIMUM_FIX_COUNT,
  GPU_BROWNIAN_BRIDGE_MAXIMUM_SUB_STEP_COUNT,
  GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS,
  GPUBrownianBridgeUtilization,
  GPUUtilizationVolumeIsopleths
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/index';
import {
  GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT,
  GPU_TRAJECTORY_RECURSIONS_MAXIMUM_TRACK_SEGMENTS,
  GPUTrajectoryRecursions
} from '../../../src/gpu-spatial-analysis/trajectory-recursion/index';
import type {KernelAuditCase} from './kernel-audit-context';

const FIX_COUNT = GPU_BROWNIAN_BRIDGE_MAXIMUM_FIX_COUNT;
const GRID_SIDE = 2048;

/** Kernel-audit cases of the Brownian bridge, its isopleths and the recursion analysis (MOB-07). */
export const MOB07_HOME_RANGE_RECURSION_AUDIT_CASES: readonly KernelAuditCase[] = [
  ...(['float32', 'uint32x2'] as const).map(
    (timestampFormat): KernelAuditCase => ({
      exportName: 'GPUBrownianBridgeUtilization',
      name: `maximum fixes, grid and sub steps, ${timestampFormat} times`,
      build: ({graph, view}) =>
        graph.add(
          new GPUBrownianBridgeUtilization({
            positions: view('float32x2', FIX_COUNT),
            timestamps: view(timestampFormat, FIX_COUNT),
            trackOffsets: view('uint32', 1025),
            motionVariances: view('float32', FIX_COUNT),
            locationErrors: view('float32', FIX_COUNT),
            gridSize: [GRID_SIDE, GRID_SIDE],
            parameters: view('float32', 4),
            subStepCount: GPU_BROWNIAN_BRIDGE_MAXIMUM_SUB_STEP_COUNT,
            output: {density: view('float32', GRID_SIDE * GRID_SIDE)}
          })
        )
    })
  ),
  {
    exportName: 'GPUUtilizationVolumeIsopleths',
    name: 'maximum binding-supported cell count with every output',
    build: ({graph, view, device}) => {
      const cellCount = Math.min(
        GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS,
        Math.floor(device.limits.maxStorageBufferBindingSize / 4),
        Math.floor(device.limits.maxBufferSize / 4)
      );
      graph.add(
        new GPUUtilizationVolumeIsopleths({
          density: view('float32', cellCount),
          parameters: view('float32', 4),
          output: {
            volume: view('float32', cellCount),
            inside: view('uint32', cellCount),
            totalMass: view('float32', 1)
          }
        })
      );
    }
  },
  ...(['float32', 'uint32x2'] as const).map(
    (timestampFormat): KernelAuditCase => ({
      exportName: 'GPUTrajectoryRecursions',
      name: `maximum radii and window, every output, ${timestampFormat} times`,
      build: ({graph, view}) => {
        const rowCount = 1 << 20;
        const outputRows = rowCount * GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT;
        graph.add(
          new GPUTrajectoryRecursions({
            positions: view('float32x2', rowCount),
            timestamps: view(timestampFormat, rowCount),
            trackOffsets: view('uint32', 4097),
            radii: view('float32', GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT),
            maximumTrackSegments: GPU_TRAJECTORY_RECURSIONS_MAXIMUM_TRACK_SEGMENTS,
            output: {
              revisits: view('uint32', outputRows),
              residenceTimes: view('float32', outputRows),
              firstPassageTimes: view('float32', outputRows),
              firstPassageCensored: view('uint32', outputRows),
              status: {candidateOverflow: view('uint32', 1)}
            }
          })
        );
      }
    })
  )
];
