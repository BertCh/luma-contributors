// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateGraphPorts,
  validateViewLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_QUIET_NAN_SEEDED_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getInterpolationNeighborhoodLossViews,
  type InterpolationNeighborhoodWeights
} from './interpolation-neighborhood-loss';

/** Optional live loss certificate of the producer of a Vecchia CSR. */
export type VecchiaNeighborhoodStatus = InterpolationNeighborhoodWeights['status'];

/** These routes consume negative loss flags, not an independently positive certificate. */
function validateVecchiaCompleteness(id: string, status?: {complete?: unknown}): void {
  validateRequirement(
    id,
    status?.complete === undefined,
    'neighborhoodStatus.complete requires a completeness-aware consumer; a candidateOverflow view alone does not certify equivalence; supply only overflow/candidateOverflow loss flags on Vecchia routes'
  );
}

/** @internal Capture new metadata around the established construction snapshots. */
export function snapshotVecchiaLossProps<
  Props extends {id?: string; neighborhoodStatus?: object; output: object}
>(props: Props, snapshot: (props: Props) => Props): Props {
  const status = props.neighborhoodStatus as
    | {overflow?: unknown; candidateOverflow?: unknown; complete?: unknown}
    | undefined;
  const neighborhoodStatus =
    status && snapshotConstructionFields(status, ['overflow', 'candidateOverflow', 'complete']);
  const output = props.output as {neighborhoodStatus?: unknown};
  const neighborhoodOutput = output.neighborhoodStatus;
  const captured = snapshot(props);
  const id =
    captured.id ??
    ('times' in captured
      ? 'vecchia-space-time-log-likelihood'
      : 'totals' in captured.output
        ? 'vecchia-fisher-scoring'
        : 'vecchia-log-likelihood');
  validateVecchiaCompleteness(id, neighborhoodStatus);
  // Operation forms validate graph-value names. Packed contributors validate the captured
  // views here, before the shared loss collector can filter out malformed falsy inputs.
  if (typeof (captured as {positions?: unknown}).positions !== 'string')
    validateGraphPorts(
      id,
      {neighborhoodStatus, output: {neighborhoodStatus: neighborhoodOutput}},
      {
        'neighborhoodStatus.overflow': {required: false, viewType: 'data', formats: ['uint32']},
        'neighborhoodStatus.candidateOverflow': {
          required: false,
          viewType: 'data',
          formats: ['uint32']
        },
        'output.neighborhoodStatus': {required: false, viewType: 'data', formats: ['uint32']}
      }
    );
  return Object.freeze({
    ...captured,
    neighborhoodStatus,
    output: Object.freeze({
      ...captured.output,
      ...(neighborhoodOutput === undefined ? {} : {neighborhoodStatus: neighborhoodOutput})
    })
  }) as Props;
}

/** @internal Explicit fixed formats must be checked before snapshots or length dereferences. */
export function validateVecchiaPorts(
  id: string,
  props: Record<string, unknown>,
  fisher = false,
  spaceTime = false
): void {
  const formats = {
    positions: 'float32x2',
    observations: 'float32x4',
    neighborhoodOffsets: 'uint32',
    neighborhoods: 'uint32',
    parameters: 'float32',
    ...(spaceTime ? {times: 'float32' as const} : {}),
    ...(!fisher
      ? ({
          'output.conditionals': 'float32x4',
          'output.conditionalStatus': 'uint32',
          'output.logLikelihood': 'float32'
        } as const)
      : ({'output.totals': 'float32'} as const)),
    'output.siteTerms': 'float32',
    'output.siteStatus': 'uint32',
    'output.status': 'uint32'
  } as const;
  validateGraphPorts(
    id,
    props,
    Object.fromEntries(
      Object.entries(formats).map(([name, format]) => [
        name,
        {required: true, viewType: 'data' as const, formats: [format]}
      ])
    )
  );
}

/** @internal Capture the direct evaluator's declared fields before applying a default ID. */
export function snapshotCPUVecchiaInput<
  Props extends {
    id?: string;
    positions: Float32Array;
    observations: Float32Array;
    neighborhoodOffsets: Uint32Array;
    neighborhoods: Uint32Array;
    parameters: Float32Array;
    neighborhoodStatus?: object;
    times?: Float32Array;
  }
>(input: Props): Props {
  return snapshotConstructionFields(input, [
    'id',
    'positions',
    'observations',
    'neighborhoodOffsets',
    'neighborhoods',
    'parameters',
    'neighborhoodStatus',
    'times'
  ]);
}

/** @internal Suppress totals after accumulation, publishing loss separately from solve counts. */
export function createVecchiaNeighborhoodStatusNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  status: VecchiaNeighborhoodStatus,
  totals: GraphDataView<'float32'>,
  evidence?: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const losses = getInterpolationNeighborhoodLossViews(id, {neighborhoodStatus: status});
  if (evidence) validateViewLength(id, 'output.neighborhoodStatus', evidence, 3);
  if (!losses.length && !evidence) return [];
  const named = {overflow: status?.overflow, candidateOverflow: status?.candidateOverflow};
  validateGraphChunkBindingRanges(graph, id, {
    ...Object.fromEntries(Object.entries(named).filter(([, view]) => view)),
    ...(evidence ? {neighborhoodStatus: evidence} : {})
  });
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-neighborhood-status`,
      operation: 'VecchiaNeighborhoodStatus',
      invocationCount: Math.max(totals.length, evidence ? 3 : 0),
      workgroupSize: 32,
      bindings: [
        ...Object.entries(named).flatMap(([name, view]) =>
          view ? [{name, view, type: 'u32' as const, access: 'read' as const}] : []
        ),
        {name: 'totals', view: totals, type: 'f32', access: 'read_write'},
        ...(evidence
          ? [
              {
                name: 'evidence',
                view: evidence,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      declarations: GET_QUIET_NAN_SEEDED_WGSL,
      body: `let overflowFlag = ${status?.overflow ? 'u32(overflow[overflowOffset] != 0u)' : '0u'};
let candidateFlag = ${status?.candidateOverflow ? 'u32(candidateOverflow[candidateOverflowOffset] != 0u)' : '0u'};
let complete = u32((overflowFlag | candidateFlag) == 0u);
if (complete == 0u && index < ${totals.length}u) { totals[totalsOffset + index] = getQuietNaN(index); }
${evidence ? 'if (index == 0u) { evidence[evidenceOffset] = complete; }\nif (index == 1u) { evidence[evidenceOffset + 1u] = overflowFlag; }\nif (index == 2u) { evidence[evidenceOffset + 2u] = candidateFlag; }' : ''}`
    })
  ];
}

/** @internal CPU loss arrays have the same scalar uint32 contract as GPU ports. */
export function getCPUVecchiaNeighborhoodStatus(
  id: string,
  status?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array; complete?: Uint32Array}
): Uint32Array {
  status =
    status && snapshotConstructionFields(status, ['overflow', 'candidateOverflow', 'complete']);
  validateVecchiaCompleteness(id, status);
  for (const [name, values] of Object.entries(status ?? {}))
    validateRequirement(
      id,
      values === undefined || values instanceof Uint32Array,
      `neighborhoodStatus.${name} must be Uint32Array; received ${Object.prototype.toString.call(values)}`
    );
  for (const [name, values] of Object.entries(status ?? {}))
    validateRequirement(
      id,
      values === undefined || values.length >= 1,
      `neighborhoodStatus.${name} must contain at least 1 word; received ${values?.length}`
    );
  const overflow = Number((status?.overflow?.[0] ?? 0) !== 0);
  const candidateOverflow = Number((status?.candidateOverflow?.[0] ?? 0) !== 0);
  return Uint32Array.of(Number(!(overflow || candidateOverflow)), overflow, candidateOverflow);
}
