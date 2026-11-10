// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_KERNEL_FLOW_SMOOTHING_MAXIMUM_ZONES,
  GPU_RADIATION_FLOW_MODEL_MAXIMUM_ZONES,
  GPUKernelFlowSmoothing,
  GPURadiationFlowModel
} from '../../../src/gpu-network/flow-aggregation/index';
import type {KernelAuditCase} from './kernel-audit-context';

const RADIATION_ZONES = GPU_RADIATION_FLOW_MODEL_MAXIMUM_ZONES;
const SMOOTHING_ZONES = GPU_KERNEL_FLOW_SMOOTHING_MAXIMUM_ZONES;

/** Kernel-audit cases of the radiation model and kernel flow smoothing (MOB-10). */
export const MOB10_FLOW_MODELS_AUDIT_CASES: readonly KernelAuditCase[] = [
  ...(['none', 'finite-size'] as const).map(
    (normalization): KernelAuditCase => ({
      exportName: 'GPURadiationFlowModel',
      name: `maximum zones, ${normalization} normalization, great circle`,
      build: ({graph, view}) =>
        graph.add(
          new GPURadiationFlowModel({
            spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
            positions: view('float32x2', RADIATION_ZONES),
            populations: view('float32', RADIATION_ZONES),
            outflows: view('float32', RADIATION_ZONES),
            normalization,
            output: {flows: view('float32', RADIATION_ZONES * RADIATION_ZONES)}
          })
        )
    })
  ),
  {
    exportName: 'GPUKernelFlowSmoothing',
    name: 'maximum zones, dense matrix, with radii',
    build: ({graph, view}) =>
      graph.add(
        new GPUKernelFlowSmoothing({
          positions: view('float32x2', SMOOTHING_ZONES),
          locationWeights: view('float32', SMOOTHING_ZONES),
          flows: {kind: 'matrix', matrix: view('float32', SMOOTHING_ZONES * SMOOTHING_ZONES)},
          parameters: view('float32', 4),
          output: {
            smoothed: view('float32', SMOOTHING_ZONES * SMOOTHING_ZONES),
            radii: view('float32', SMOOTHING_ZONES)
          }
        })
      )
  },
  {
    exportName: 'GPUKernelFlowSmoothing',
    name: 'maximum zones, aggregation pairs, planar',
    build: ({graph, view}) =>
      graph.add(
        new GPUKernelFlowSmoothing({
          positions: view('float32x2', SMOOTHING_ZONES),
          locationWeights: view('float32', SMOOTHING_ZONES),
          flows: {
            kind: 'pairs',
            originZoneIds: view('uint32', 65536),
            destinationZoneIds: view('uint32', 65536),
            weights: view('float32', 65536),
            validPairCount: view('uint32', 1)
          },
          parameters: view('float32', 4),
          output: {smoothed: view('float32', SMOOTHING_ZONES * SMOOTHING_ZONES)}
        })
      )
  }
];
