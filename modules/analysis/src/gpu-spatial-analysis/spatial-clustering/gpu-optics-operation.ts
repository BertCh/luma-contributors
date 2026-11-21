// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUOPTICS, OPTICS_NAMED_STATUS_FIELDS, type GPUOPTICSProps} from './gpu-optics';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

/** OPTICS graph values; each per-row output has the explicit rowCount inferred from positions. */
export type OPTICSOperationProps = Omit<GPUOPTICSProps, 'positions' | 'times' | 'output'> & {
  positions: string;
  times?: string;
  rowCount: number;
  output: {
    ordering: string;
    coreDistances: string;
    reachability: string;
    predecessors: string;
    labels: string;
    clusterIntervals?: string;
    clusterSummary?: string;
    ports?: {
      status: {
        candidateOverflow?: string;
        reason?: string;
        iterationCount?: string;
        converged?: string;
        iterationLimitReached?: string;
        invalidCount?: string;
      };
    };
  };
};

/** @internal Retains the operation's construction fields across deferred execution/lowering. */
export function snapshotOPTICSOperationProps(props: OPTICSOperationProps): OPTICSOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'positions',
    'times',
    'rowCount',
    'output',
    'spatialContext',
    'temporalContext',
    'maximumTimeDistance',
    'minimumSamples',
    'maximumDistance',
    'extractionDistance',
    'extractionMethod',
    'xi',
    'minimumClusterSize',
    'predecessorCorrection',
    'frontierBatchSize',
    'maximumIterations',
    'neighborIndex',
    'orderingMethod',
    'neighborCapacity'
  ]);
  validateRequirement(
    snapshot.id ?? 'optics',
    !('status' in snapshot.output),
    'output.status was removed; use output.ports.status for iterationCount, converged and invalidCount'
  );
  const output = snapshotConstructionFields(snapshot.output, [
    'ordering',
    'coreDistances',
    'reachability',
    'predecessors',
    'labels',
    'clusterIntervals',
    'clusterSummary',
    'ports'
  ]);
  const ports = output.ports && snapshotConstructionFields(output.ports, ['status']);
  const temporal =
    snapshot.temporalContext &&
    snapshotConstructionFields(snapshot.temporalContext, [
      'unit',
      'epoch',
      'encoding',
      'timeZone',
      'axis'
    ]);
  const axis =
    temporal &&
    snapshotConstructionFields(temporal.axis, ['kind', 'step'] as (keyof typeof temporal.axis)[]);
  return Object.freeze({
    ...snapshot,
    ...(snapshot.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
      : {}),
    ...(temporal
      ? {
          temporalContext: Object.freeze({
            ...temporal,
            axis: Object.freeze({
              ...axis,
              ...(axis && 'step' in axis && typeof axis.step === 'object'
                ? {
                    step: snapshotConstructionFields(axis.step, ['calendarUnit', 'interval'])
                  }
                : {})
            }) as typeof temporal.axis
          })
        }
      : {}),
    output: Object.freeze({
      ...output,
      ...(ports
        ? {
            ports: Object.freeze({
              status: snapshotConstructionFields(ports.status, [
                'candidateOverflow',
                ...OPTICS_NAMED_STATUS_FIELDS
              ])
            })
          }
        : {})
    })
  });
}

/** OPTICS contributor operation. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUOPTICSOperation extends GPUContributorOperation {
  constructor(props: OPTICSOperationProps) {
    props = snapshotOPTICSOperationProps(props);
    const id = props.id ?? 'optics';
    const {output} = props;
    const length = props.rowCount;
    super({
      id,
      inputs: {positions: props.positions, ...(props.times ? {times: props.times} : {})},
      outputs: {
        ordering: {value: output.ordering, format: 'uint32', length},
        coreDistances: {value: output.coreDistances, format: 'float32', length},
        reachability: {value: output.reachability, format: 'float32', length},
        predecessors: {value: output.predecessors, format: 'uint32', length},
        labels: {value: output.labels, format: 'uint32', length},
        ...(output.clusterIntervals
          ? {
              clusterIntervals: {
                value: output.clusterIntervals,
                format: 'uint32x2' as const,
                length
              }
            }
          : {}),
        ...(output.clusterSummary
          ? {clusterSummary: {value: output.clusterSummary, format: 'uint32' as const, length: 2}}
          : {}),
        ...Object.fromEntries(
          OPTICS_NAMED_STATUS_FIELDS.filter(name => output.ports?.status[name]).map(name => [
            name,
            {value: output.ports!.status[name]!, format: 'uint32' as const, length: 1}
          ])
        ),
        ...(output.ports?.status.candidateOverflow
          ? {
              candidateOverflow: {
                value: output.ports.status.candidateOverflow,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUOPTICS({
          ...props,
          id,
          positions: getView<'float32x2'>('positions'),
          times: props.times ? getView<'float32'>('times') : undefined,
          output: {
            ordering: getView<'uint32'>('ordering'),
            coreDistances: getView<'float32'>('coreDistances'),
            reachability: getView<'float32'>('reachability'),
            predecessors: getView<'uint32'>('predecessors'),
            labels: getView<'uint32'>('labels'),
            clusterIntervals: output.clusterIntervals
              ? getView<'uint32x2'>('clusterIntervals')
              : undefined,
            clusterSummary: output.clusterSummary ? getView<'uint32'>('clusterSummary') : undefined,
            ports: {
              status: {
                ...Object.fromEntries(
                  OPTICS_NAMED_STATUS_FIELDS.filter(name => output.ports?.status[name]).map(
                    name => [name, getView<'uint32'>(name)]
                  )
                ),
                candidateOverflow: output.ports?.status.candidateOverflow
                  ? getView<'uint32'>('candidateOverflow')
                  : undefined
              }
            }
          }
        })
    });
  }
}
