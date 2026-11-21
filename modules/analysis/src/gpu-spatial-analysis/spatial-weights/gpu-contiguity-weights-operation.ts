// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotContiguityOperationProps} from './weights-construction-snapshot';
import {snapshotPolygonReviewProps} from './weights-review-snapshot';
import {validateContiguityOperationOutput} from './contiguity-operation-output';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUContiguityWeights, type GPUContiguityCriterion} from './gpu-contiguity-weights';

/** Packed polygon and CSR graph values shared by the two operations. */
export type ContiguityWeightsOperationProps = {
  id?: string;
  criterion: GPUContiguityCriterion;
  polygons: {
    positions: string;
    ringOffsets: string;
    polygonOffsets: string;
    featureOffsets: string;
  };
  /** featureOffsets length minus one, explicitly carried for the output shape. */
  featureCount: number;
  /** Neighbor and weight output length. The used count is offsets[featureCount]. */
  capacity: number;
  snapTolerance?: number;
  coordinateExtent?: number;
  pairCapacity?: number;
  output: {
    offsets: string;
    neighbors: string;
    weights: string;
    /** Both causes need public destinations; pair capacity is bounded even when omitted. */
    status: {overflow: string; requiredCount?: string; candidateOverflow: string};
  };
};

/** @internal Shared scalar validation precedes named output-port validation in both backends. */
export function validateContiguityWeightsOperationScalars(
  id: string,
  props: ContiguityWeightsOperationProps
): void {
  validateCapacity(id, 'featureCount', props.featureCount, 1, 0xfffffffe);
  validateCapacity(id, 'capacity', props.capacity);
  if (props.pairCapacity !== undefined) validateCapacity(id, 'pairCapacity', props.pairCapacity);
  validateRequirement(
    id,
    props.criterion === 'queen' || props.criterion === 'rook',
    "criterion must be 'queen' or 'rook'"
  );
  validateRequirement(id, (props.snapTolerance ?? 0) >= 0, 'snapTolerance must be >= 0');
  validateFiniteNumber(id, 'snapTolerance', props.snapTolerance ?? 0, 0);
  if (props.coordinateExtent !== undefined)
    validateFiniteNumber(id, 'coordinateExtent', props.coordinateExtent, 0);
}

/**
 * Operation form; all props are compile-time, polygon contents are per-frame.
 * The optional distances port is omitted because the contributor never writes it.
 */
export class GPUContiguityWeightsOperation extends GPUContributorOperation {
  constructor(props: ContiguityWeightsOperationProps) {
    props = snapshotPolygonReviewProps(props);
    validateContiguityOperationOutput(props.id ?? 'contiguity-weights', props.output);
    props = snapshotContiguityOperationProps(props);
    const id = props.id ?? 'contiguity-weights';
    validateContiguityWeightsOperationScalars(id, props);
    validateRequirement(
      id,
      Boolean(props.output.status.candidateOverflow),
      'output.status.candidateOverflow is required for bounded contiguity candidates'
    );
    super({
      id,
      inputs: {...props.polygons},
      outputs: {
        offsets: {value: props.output.offsets, format: 'uint32', length: props.featureCount + 1},
        neighbors: {value: props.output.neighbors, format: 'uint32', length: props.capacity},
        weights: {value: props.output.weights, format: 'float32', length: props.capacity},
        overflow: {value: props.output.status.overflow, format: 'uint32', length: 1},
        ...(props.output.status.requiredCount
          ? {
              requiredCount: {
                value: props.output.status.requiredCount,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {}),
        ...(props.output.status.candidateOverflow
          ? {
              candidateOverflow: {
                value: props.output.status.candidateOverflow,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUContiguityWeights({
          id,
          criterion: props.criterion,
          snapTolerance: props.snapTolerance,
          coordinateExtent: props.coordinateExtent,
          pairCapacity: props.pairCapacity,
          polygons: {
            kind: 'polygons',
            positions: getView<'float32x2'>('positions'),
            ringOffsets: getView<'uint32'>('ringOffsets'),
            polygonOffsets: getView<'uint32'>('polygonOffsets'),
            featureOffsets: getView<'uint32'>('featureOffsets')
          },
          output: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            status: {
              overflow: getView<'uint32'>('overflow'),
              requiredCount: props.output.status.requiredCount
                ? getView<'uint32'>('requiredCount')
                : undefined,
              candidateOverflow: props.output.status.candidateOverflow
                ? getView<'uint32'>('candidateOverflow')
                : undefined
            }
          }
        })
    });
  }
}
