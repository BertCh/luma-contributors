// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A one-node GPU step that evaluates the variogram models GPUVariogramModelFit fitted, so the fitted
// curves are drawn from the fit's own output buffer: one row of [nugget, partial sill, range, residual]
// per model, evaluated at the distances in `distances`. Shapes match getVariogramModelShape.

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

const WORKGROUP_SIZE = 64;
const SHAPES = [
  'select(1.5 * q - 0.5 * q * q * q, 1.0, q >= 1.0)',
  '1.0 - exp(-3.0 * q)',
  '1.0 - exp(-3.0 * q * q)'
];

class FittedCurve implements GPUCommandNodeProducer {
  constructor(
    private readonly id: string,
    private readonly fitted: GraphDataView<'float32x4'>,
    private readonly distances: GraphDataView<'float32'>,
    private readonly curve: GraphDataView<'float32'>,
    private readonly modelIndex: number
  ) {}

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, fitted, distances, curve, modelIndex} = this;
    const dispatch = getBoundedDispatchLayout(
      id,
      curve.length,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const shaderSource = /* wgsl */ `
const fittedOffset: u32 = ${getViewElementOffset(fitted)}u;
const distancesOffset: u32 = ${getViewElementOffset(distances)}u;
const curveOffset: u32 = ${getViewElementOffset(curve)}u;
@group(0) @binding(0) var<storage, read> fitted: array<vec4f>;
@group(0) @binding(1) var<storage, read> distances: array<f32>;
@group(0) @binding(2) var<storage, read_write> curve: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatch, WORKGROUP_SIZE)}
  if (index >= ${curve.length}u) { return; }
  let model = fitted[fittedOffset + ${modelIndex}u];
  let distance = distances[distancesOffset + index];
  let q = distance / max(model.z, 1.0e-9);
  curve[curveOffset + index] = select(model.x + model.y * ${SHAPES[modelIndex]}, 0.0, distance <= 0.0);
}`;
    return [
      createGPUComputeCommandNode<Parameters>({
        id,
        resources: [
          {buffer: fitted, usage: 'storage-read'},
          {buffer: distances, usage: 'storage-read'},
          {buffer: curve, usage: 'storage-read-write'}
        ],
        workload: {
          operation: 'FittedCurve',
          variant: String(modelIndex),
          commandCount: 1,
          maximumWorkgroupCount: dispatch.x * dispatch.y * dispatch.z,
          maximumInvocationCount: dispatch.x * dispatch.y * dispatch.z * WORKGROUP_SIZE,
          readByteLength: curve.length * 8,
          writeByteLength: curve.length * 4
        },
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id,
            source: shaderSource,
            shaderLayout: {
              bindings: ['fitted', 'distances', 'curve'].map((name, location) => ({
                name,
                type: 'storage' as const,
                group: 0,
                location
              }))
            }
          });
          return {
            encode: ({computePass, getBuffer}) => {
              const bindings: Record<string, Binding> = {
                fitted: getViewBinding(fitted, getBuffer),
                distances: getViewBinding(distances, getBuffer),
                curve: getViewBinding(curve, getBuffer)
              };
              kernel.dispatch(computePass, {bindings, ...dispatch});
            },
            destroy: () => kernel.destroy()
          };
        }
      })
    ];
  }
}

/** Adds the curve of model row `modelIndex` (0 spherical, 1 exponential, 2 gaussian) at `distances`. */
export function addFittedCurve(
  builder: RunBuilder,
  id: string,
  fitted: RunBuffer<'float32x4'>,
  distances: RunBuffer<'float32'>,
  modelIndex: number
): RunBuffer<'float32'> {
  const curve = builder.allocate(id, 'float32', distances.length);
  builder.graph.add(
    new FittedCurve(`${id}-curve`, fitted.view, distances.view, curve.view, modelIndex)
  );
  return curve;
}
