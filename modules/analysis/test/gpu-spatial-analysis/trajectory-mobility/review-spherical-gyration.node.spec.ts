// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUMobilityIndicators} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators';
import {computeMobilityIndicatorsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const spatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  sphereRadius: 6371000
} as const;
for (const cancelling of [false, true]) {
  const rowCount = cancelling ? 1800 : 1000;
  const positions = Float32Array.from({length: rowCount * 2}, (_, component) =>
    component % 2 === 1
      ? cancelling
        ? 0
        : 40.75
      : cancelling
        ? [17, 137, -103][Math.floor(component / 2 / 600)]
        : -74
  );
  const timestamps = Float32Array.from({length: rowCount}, (_, row) => row);
  const starts = Uint32Array.from({length: rowCount}, (_, row) => row);
  // Identical points have exact zero gyration. Equal masses at three equatorial
  // angles separated by 120 degrees sum to zero: 1 + exp(2*pi*i/3) + exp(4*pi*i/3)=0.
  it(`B2 CPU omitted spherical center handles ${cancelling ? 'grouped cancelling directions' : '1000 stationary stays'}`, () => {
    const result = computeMobilityIndicatorsOnCPU(
      {
        timestamps,
        staypoints: {
          tracks: new Uint32Array(rowCount),
          centroids: positions,
          startRows: starts,
          durations: new Float32Array(rowCount).fill(1),
          locationIds: new Uint32Array(rowCount),
          count: Uint32Array.of(rowCount)
        }
      },
      {trackCount: 1, spatialContext, includeRealEntropy: false}
    );
    expect(result.radiusOfGyration.length).toBe(1);
    if (cancelling) expect(Number.isNaN(result.radiusOfGyration[0])).toBe(true);
    else expect(result.radiusOfGyration[0]).toBe(0);
  });
  it(`B2 actual omitted-option WGSL gyration handles ${cancelling ? 'grouped cancelling directions' : '1000 stationary stays'}`, () => {
    const context = createKernelAuditContext('stable-gyration');
    try {
      recordWGSLKernelNodes(context.graph);
      const points = context.view('float32x2', rowCount);
      const radius = context.view('float32', 1);
      const nodes = new GPUMobilityIndicators({
        id: 'review',
        spatialContext,
        trackCount: 1,
        timestamps: context.view('float32', rowCount),
        staypoints: {
          tracks: context.view('uint32', rowCount),
          centroids: points,
          startRows: context.view('uint32', rowCount),
          durations: context.view('float32', rowCount),
          locationIds: context.view('uint32', rowCount),
          status: {count: context.view('uint32', 1)}
        },
        output: {radiusOfGyration: radius}
      }).getCommandNodes(context.graph);
      const kernel = nodes.find(node => node.id === 'review-radius-of-gyration');
      const record = getWGSLKernelNodeRecord(context.graph, 'review-radius-of-gyration');
      const offsets = record?.bindings.find(binding => binding.name === 'trackOffsets');
      if (!kernel || !record || !offsets)
        throw new Error('missing gyration kernel or track offsets');
      const memory = createCPUGraphMemory();
      memory.write(points, positions);
      memory.write(offsets.view, Uint32Array.of(0, rowCount));
      // The shared interpreter has no ldexp builtin. Retain the producer's body and
      // bindings, substituting exact power-of-two scaling for this builtin only.
      // All rounding-error corrections here are normal and far from overflow.
      const executable = createWGSLKernelNode(context.graph, {
        id: 'review-executable',
        operation: 'ReviewSphericalGyration',
        invocationCount: 1,
        bindings: record.bindings,
        body: record.body,
        declarations: `${record.declarations?.replace(/\bldexp\(/g, 'reviewLdexp(') ?? ''}
fn reviewLdexp(value: f32, power: i32) -> f32 {
  let first = power / 2i;
  return (value * exp2(f32(first))) * exp2(f32(power - first));
}`
      });
      executeCommandNodesOnCPU([executable], memory);
      const result = memory.readFloat32(radius);
      expect(result.length).toBe(1);
      if (cancelling) expect(Number.isNaN(result[0])).toBe(true);
      else expect(result[0]).toBe(0);
    } finally {
      context.device.destroy();
    }
  }, 60000);
}
