// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUOPTICS, type GPUOPTICSProps} from './gpu-optics';

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
    status: string;
    clusterIntervals?: string;
    clusterSummary?: string;
    ports?: {status: {candidateOverflow?: string}};
  };
};

/** OPTICS contributor operation. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUOPTICSOperation extends GPUContributorOperation {
  constructor(props: OPTICSOperationProps) {
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
        status: {value: output.status, format: 'uint32', length: 3},
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
            status: getView<'uint32'>('status'),
            clusterIntervals: output.clusterIntervals
              ? getView<'uint32x2'>('clusterIntervals')
              : undefined,
            clusterSummary: output.clusterSummary ? getView<'uint32'>('clusterSummary') : undefined,
            ports: {
              status: {
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
