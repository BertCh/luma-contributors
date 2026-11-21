// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateCellReadWriteDispatch} from '../cell-indexing/cell-dispatch-validation';
import {
  validateRequirement,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** A cell prefix together with the evidence that qualifies its completeness. */
export type GPUCountedCellSource = {
  cells: GraphDataView<'uint32x2'>;
  status: {
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
    /** Optional producer completeness, including non-capacity loss; zero means incomplete. */
    complete?: GraphDataView<'uint32'>;
  };
};

/** Bare cell views remain supported; bounded producers should pass their whole source status. */
export type GPUCellSourceInput =
  | {
      cells: GraphDataView<'uint32x2'>;
      count?: GraphDataView<'uint32'>;
      source?: GPUCountedCellSource;
    }
  | {
      source: GPUCountedCellSource;
      cells?: GraphDataView<'uint32x2'>;
      count?: GraphDataView<'uint32'>;
    };

type ResolvedCellSource<Props extends GPUCellSourceInput> = Omit<
  Props,
  'cells' | 'count' | 'source'
> & {
  cells: GraphDataView<'uint32x2'>;
  count?: GraphDataView<'uint32'>;
  source?: GPUCountedCellSource;
};

function snapshotSourceMetadata<Metadata extends object>(
  metadata: Metadata,
  fields: readonly (keyof Metadata)[]
): Metadata {
  return snapshotConstructionFields(metadata, [
    ...new Set([...(Reflect.ownKeys(metadata) as (keyof Metadata)[]), ...fields])
  ]);
}

/** Resolve fields as replacements; capture metadata without copying live GPU resources. @internal */
export function resolveCountedCellSource<
  Props extends GPUCellSourceInput & {source: GPUCountedCellSource}
>(
  id: string,
  props: Props
): Omit<Props, 'cells' | 'count' | 'source'> & {
  cells: GraphDataView<'uint32x2'>;
  count: GraphDataView<'uint32'>;
  source: GPUCountedCellSource;
};
export function resolveCountedCellSource<Props extends GPUCellSourceInput>(
  id: string,
  props: Props
): ResolvedCellSource<Props>;
export function resolveCountedCellSource<Props extends GPUCellSourceInput>(
  id: string,
  props: Props
): ResolvedCellSource<Props> {
  const source = props.source;
  if (!source) return props as ResolvedCellSource<Props>;
  const fields = snapshotSourceMetadata(source, ['cells', 'status']);
  validateRequirement(
    id,
    Boolean(fields.status),
    'source.status must supply count and overflow; received missing status'
  );
  const status = snapshotSourceMetadata(fields.status, [
    'count',
    'overflow',
    'candidateOverflow',
    'complete'
  ]);
  validateRequirement(
    id,
    !props.cells || props.cells === fields.cells,
    'cells must equal source.cells when both are supplied; pass source alone instead'
  );
  validateRequirement(
    id,
    !props.count || props.count === status.count,
    'count must equal source.status.count when both are supplied; pass source alone instead'
  );
  validateRequirement(
    id,
    Boolean(status.count && status.overflow),
    'source.status must include count and overflow; pass the whole producer status'
  );
  if (status.complete) {
    validateSinglePackedView(status.complete, ['uint32'], `${id} source.status.complete`);
    validateRequirement(
      id,
      status.complete.length >= 1,
      `source.status.complete must contain at least one row; received ${status.complete.length}`
    );
  }
  const snapshot: ResolvedCellSource<Props> = {
    ...props,
    cells: fields.cells,
    count: status.count,
    source: Object.freeze({...fields, status: Object.freeze(status)})
  };
  Object.freeze(snapshot);
  return snapshot;
}

/** Validate the producer-loss reader against its separately published destination. @internal */
export function validateCountedCellSourceLossBindings(
  id: string,
  source: GPUCountedCellSource | undefined,
  output: {count: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>}
): void {
  if (!source) return;
  validateCellReadWriteDispatch(
    id,
    {
      sourceOverflow: source.status.overflow,
      sourceCandidateOverflow: source.status.candidateOverflow,
      sourceComplete: source.status.complete
    },
    output.candidateOverflow ? {candidateOverflow: output.candidateOverflow} : {count: output.count}
  );
}

/** Keep inherited loss separate from local sizing; suppress the prefix if no loss port exists. @internal */
export function getCountedCellSourceLossNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  source: GPUCountedCellSource | undefined,
  output: {count: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters>[] {
  if (!source && !output.candidateOverflow) return [];
  if (source?.status.complete) validateGraphViewsBelongToGraph(id, graph, [source.status.complete]);
  const bindings: WGSLKernelBinding[] = source
    ? [
        {name: 'sourceOverflow', view: source.status.overflow, type: 'u32', access: 'read'},
        ...(source.status.candidateOverflow
          ? [
              {
                name: 'sourceCandidate',
                view: source.status.candidateOverflow,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(source.status.complete
          ? [
              {
                name: 'sourceComplete',
                view: source.status.complete,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ]
    : [];
  const loss = source
    ? `sourceOverflow[sourceOverflowOffset] != 0u${source.status.candidateOverflow ? ' || sourceCandidate[sourceCandidateOffset] != 0u' : ''}${source.status.complete ? ' || sourceComplete[sourceCompleteOffset] == 0u' : ''}`
    : 'false';
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-source-loss`,
      operation,
      variant: 'source-loss',
      invocationCount: 1,
      bindings: [
        ...bindings,
        output.candidateOverflow
          ? {name: 'candidate', view: output.candidateOverflow, type: 'u32', access: 'read_write'}
          : {name: 'count', view: output.count, type: 'u32', access: 'read_write'}
      ],
      body: output.candidateOverflow
        ? `candidate[candidateOffset] = select(0u, 1u, ${loss});`
        : `if (${loss}) { count[countOffset] = 0u; }`
    })
  ];
}
