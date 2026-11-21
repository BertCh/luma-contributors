// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUMantelTest, type GPUMantelTransform} from './gpu-mantel-test';
import {GPU_SPACE_TIME_SUMMARY_LENGTH} from './space-time-parameters';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {snapshotSpaceTimeTemporalContext} from './space-time-construction';
import {resolveGPUSpaceTimeTemporalContext} from './space-time-bands';

/** Shared graph-value properties; parameters retain getGPUSpaceTimeParameterValues layout. */
export type MantelTestOperationProps = {
  id?: string;
  rowCount: number;
  /** Weights are shape-checked; either supplied loss word makes inference unavailable. */
  pairs: {
    offsets: string;
    neighbors: string;
    weights: string;
    distances: string;
    status?: {overflow?: string; candidateOverflow?: string};
  };
  times: string;
  /** Metadata of the live float32-relative times, captured at construction. */
  temporalContext?: GPUTemporalContext;
  parameters: string;
  maximumPermutations: number;
  stopAfterExceedances?: number;
  distanceTransform?: GPUMantelTransform;
  timeTransform?: GPUMantelTransform;
  output: {
    statistics: string;
    summary: string;
    permutationCount?: string;
    usablePermutationCount?: string;
  };
};

/** Capture the schema and optional port presence once for both operation forms. @internal */
export function snapshotMantelOperationProps(
  props: MantelTestOperationProps
): MantelTestOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'rowCount',
    'pairs',
    'times',
    'temporalContext',
    'parameters',
    'maximumPermutations',
    'stopAfterExceedances',
    'distanceTransform',
    'timeTransform',
    'output'
  ]);
  const id = captured.id ?? 'mantel-test';
  validateRequirement(
    id,
    Boolean(captured.pairs && typeof captured.pairs === 'object'),
    `pairs must be a port map; received ${String(captured.pairs)}`
  );
  validateRequirement(
    id,
    Boolean(captured.output && typeof captured.output === 'object'),
    `output must be a port map; received ${String(captured.output)}`
  );
  const pairs = snapshotConstructionFields(captured.pairs, [
    'offsets',
    'neighbors',
    'weights',
    'distances',
    'status'
  ]);
  return Object.freeze({
    ...captured,
    temporalContext:
      captured.temporalContext && snapshotSpaceTimeTemporalContext(captured.temporalContext),
    pairs: Object.freeze({
      ...pairs,
      status:
        pairs.status && snapshotConstructionFields(pairs.status, ['overflow', 'candidateOverflow'])
    }),
    output: snapshotConstructionFields(captured.output, [
      'statistics',
      'summary',
      'permutationCount',
      'usablePermutationCount'
    ]),
    distanceTransform:
      captured.distanceTransform &&
      snapshotConstructionFields(captured.distanceTransform, ['constant', 'power']),
    timeTransform:
      captured.timeTransform &&
      snapshotConstructionFields(captured.timeTransform, ['constant', 'power'])
  });
}

/** Shared host checks before either operation declares outputs. @internal */
export function validateMantelOperationProps(id: string, props: MantelTestOperationProps): void {
  validateCapacity(id, 'rowCount', props.rowCount, 2);
  validateCapacity(id, 'maximumPermutations', props.maximumPermutations, 1, 0xfffffffe);
  if (props.stopAfterExceedances !== undefined)
    validateCapacity(
      id,
      'stopAfterExceedances',
      props.stopAfterExceedances,
      1,
      props.maximumPermutations
    );
  validateRequirement(
    id,
    props.stopAfterExceedances === undefined || props.output.permutationCount !== undefined,
    'output.permutationCount is required when stopAfterExceedances is supplied'
  );
  for (const [name, transform] of [
    ['distanceTransform', props.distanceTransform],
    ['timeTransform', props.timeTransform]
  ] as const) {
    if (transform) {
      validateFiniteNumber(id, `${name}.constant`, transform.constant);
      validateFiniteNumber(id, `${name}.power`, transform.power);
    }
  }
  resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
}

/**
 * Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Optional pair loss ports are read on every execution, including CPU execution.
 */
export class GPUMantelTestOperation extends GPUContributorOperation {
  constructor(props: MantelTestOperationProps) {
    props = snapshotMantelOperationProps(props);
    validateMantelOperationProps(props.id ?? 'mantel-test', props);
    super({
      id: props.id ?? 'mantel-test',
      inputs: {
        offsets: props.pairs.offsets,
        neighbors: props.pairs.neighbors,
        weights: props.pairs.weights,
        distances: props.pairs.distances,
        ...(props.pairs.status?.overflow !== undefined
          ? {pairsOverflow: props.pairs.status.overflow}
          : {}),
        ...(props.pairs.status?.candidateOverflow !== undefined
          ? {pairsCandidateOverflow: props.pairs.status.candidateOverflow}
          : {}),
        times: props.times,
        parameters: props.parameters
      },
      outputs: {
        statistics: {
          value: props.output.statistics,
          format: 'float32',
          length: props.maximumPermutations + 1
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_SPACE_TIME_SUMMARY_LENGTH
        },
        ...(props.output.permutationCount === undefined
          ? {}
          : {
              permutationCount: {
                value: props.output.permutationCount,
                format: 'uint32' as const,
                length: 1
              }
            }),
        ...(props.output.usablePermutationCount === undefined
          ? {}
          : {
              usablePermutationCount: {
                value: props.output.usablePermutationCount,
                format: 'uint32' as const,
                length: 1
              }
            })
      },
      createContributor: getView => {
        const id = props.id ?? 'mantel-test';
        const times = getView<'float32'>('times');
        const offsets = getView<'uint32'>('offsets');
        validateViewLength(id, 'times (rowCount)', times, props.rowCount);
        validateViewLength(id, 'pairs.offsets (rowCount + 1)', offsets, props.rowCount + 1);
        return new GPUMantelTest({
          ...props,
          pairs: {
            offsets,
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            distances: getView<'float32'>('distances'),
            status: {
              overflow:
                props.pairs.status?.overflow !== undefined
                  ? getView<'uint32'>('pairsOverflow')
                  : undefined,
              candidateOverflow:
                props.pairs.status?.candidateOverflow !== undefined
                  ? getView<'uint32'>('pairsCandidateOverflow')
                  : undefined
            }
          },
          times,
          parameters: getView<'uint32'>('parameters'),
          output: {
            statistics: getView<'float32'>('statistics'),
            summary: getView<'float32'>('summary'),
            permutationCount:
              props.output.permutationCount === undefined
                ? undefined
                : getView<'uint32'>('permutationCount'),
            usablePermutationCount:
              props.output.usablePermutationCount === undefined
                ? undefined
                : getView<'uint32'>('usablePermutationCount')
          }
        });
      }
    });
  }
}
