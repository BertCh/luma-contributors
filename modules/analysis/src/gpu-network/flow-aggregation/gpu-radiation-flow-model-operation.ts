// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURadiationFlowModel, type GPURadiationFlowModelProps} from './gpu-radiation-flow-model';
import {resolveRadiationFlowModelNormalization} from './radiation-flow-model-normalization';

/** Shared graph-value properties of the GPU operation and its CPU twin. */
export type RadiationFlowModelOperationProps = Pick<
  GPURadiationFlowModelProps,
  'id' | 'spatialContext' | 'normalization'
> & {
  /** Compile-time number of zones, inferred from positions by the contributor. */
  zoneCount: number;
  /** float32x2 centroids. */
  positions: string;
  /** float32 populations and departing trip totals, one row per zone. */
  populations: string;
  outflows: string;
  output: {
    /** Dense origin-major float32 flows, zoneCount squared rows. */
    flows: string;
    /** Optional uint32 diagnostic, zoneCount rows; lowers to output.status.invalidInput. */
    invalidInput?: string;
  };
};

/** @internal Capture every declared field once, including inherited getters and context metadata. */
export function snapshotRadiationFlowModelOperationProps(
  originalProps: RadiationFlowModelOperationProps
): RadiationFlowModelOperationProps {
  const props = snapshotConstructionFields(originalProps, [
    'id',
    'zoneCount',
    'positions',
    'populations',
    'outflows',
    'normalization',
    'spatialContext',
    'output'
  ]);
  return Object.freeze({
    ...props,
    normalization: resolveRadiationFlowModelNormalization(
      props.id ?? 'radiation-flow-model',
      props.normalization
    ),
    spatialContext:
      props.spatialContext && snapshotConstructionSpatialContext(props.spatialContext),
    output: snapshotConstructionFields(props.output, ['flows', 'invalidInput'])
  });
}

/**
 * Operation form of GPURadiationFlowModel. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * All properties are compile-time; input graph contents are per-frame.
 */
export class GPURadiationFlowModelOperation extends GPUContributorOperation {
  constructor(originalProps: RadiationFlowModelOperationProps) {
    const props = snapshotRadiationFlowModelOperationProps(originalProps);
    const id = props.id ?? 'radiation-flow-model';
    super({
      id,
      inputs: {
        positions: props.positions,
        populations: props.populations,
        outflows: props.outflows
      },
      outputs: {
        flows: {value: props.output.flows, format: 'float32', length: props.zoneCount ** 2},
        ...(props.output.invalidInput !== undefined
          ? {
              invalidInput: {
                value: props.output.invalidInput,
                format: 'uint32' as const,
                length: props.zoneCount
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPURadiationFlowModel({
          id,
          spatialContext: props.spatialContext,
          normalization: props.normalization,
          positions: getView<'float32x2'>('positions'),
          populations: getView<'float32'>('populations'),
          outflows: getView<'float32'>('outflows'),
          output: {
            flows: getView<'float32'>('flows'),
            ...(props.output.invalidInput !== undefined
              ? {
                  status: {invalidInput: getView<'uint32'>('invalidInput')}
                }
              : {})
          }
        })
    });
  }
}
