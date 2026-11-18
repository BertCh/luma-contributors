// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';

type VisibilityViews = Readonly<Record<string, GraphDataView | undefined>>;

/** Checks aligned ranges within CSR dispatches and logical separation of final values before declaration. @internal */
export function validateVisibilityGraphCSRBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: VisibilityViews,
  output: unknown,
  network = false
): void {
  const supplied: Record<string, GraphDataView> = {};
  const collect = (value: unknown, prefix = '') => {
    if (!value || typeof value !== 'object') return;
    if ('buffer' in value && 'format' in value && 'byteOffset' in value)
      supplied[prefix] = value as GraphDataView;
    else
      for (const [name, child] of Object.entries(value))
        collect(child, prefix ? `${prefix}.${name}` : name);
  };
  collect(output);
  const fields: Record<string, GraphDataView | undefined> = {};
  const include = (names: readonly string[]) => {
    for (const name of names) fields[name] = supplied[name];
  };
  include([
    'offsets',
    'neighbors',
    'weights',
    'degree',
    'gridVisits',
    'viewpointClassifications',
    'status.count',
    'status.requiredCount',
    'status.overflow',
    'status.candidateOverflow',
    'status.invalidCount'
  ]);
  if (network)
    include([
      'betweenness',
      'closeness',
      'reachedCounts',
      'metrics',
      'integration',
      'extendedIntegration',
      'normalizedChoice',
      'status.approximation',
      'status.converged',
      'status.iterationLimitReached',
      'status.zeroWeightCycle'
    ]);
  validateGraphChunkBindingRanges(graph, id, {
    ...inputs,
    ...Object.fromEntries(Object.entries(fields).map(([name, view]) => [`output.${name}`, view]))
  });
  const physicalBuffer = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const overlaps = (first: GraphDataView, second: GraphDataView, aligned: boolean) => {
    if (physicalBuffer(first) !== physicalBuffer(second)) return false;
    const range = (view: GraphDataView) =>
      aligned
        ? getViewBindingRange(view)
        : {
            offset: view.byteOffset,
            size: view.length === 0 ? 0 : (view.length - 1) * view.byteStride + view.rowByteLength
          };
    const firstRange = range(first);
    const secondRange = range(second);
    return (
      firstRange.size > 0 &&
      secondRange.size > 0 &&
      firstRange.offset < secondRange.offset + secondRange.size &&
      secondRange.offset < firstRange.offset + firstRange.size
    );
  };
  const dispatch = (writes: readonly string[], reads: readonly string[] = []) => {
    const writable = Object.fromEntries(writes.map(name => [name, fields[name]]));
    for (const name of writes) {
      const view = fields[name];
      if (!view) continue;
      for (const otherName of reads) {
        const other = fields[otherName] ?? inputs[otherName];
        if (!other || physicalBuffer(view) !== physicalBuffer(other)) continue;
        if (fields[otherName]) {
          // Readbacks of other output fields are promoted to writable storage in this dispatch.
          // Validate every promoted binding, including its alignment prefix.
          writable[otherName] = other;
        } else {
          validateRequirement(
            id,
            false,
            `${name} and input ${otherName} must not share a physical buffer in one dispatch`
          );
        }
      }
    }
    validateDisjointWritableViews(id, writable, {strictAlignedBindings: true});
  };
  dispatch(['neighbors', 'weights']);
  dispatch(['weights'], ['neighbors']);
  dispatch(['neighbors'], ['segments']);
  dispatch(['offsets'], ['status.candidateOverflow', 'status.invalidCount']);
  dispatch(
    ['degree', 'status.count', 'status.requiredCount', 'status.overflow'],
    ['offsets', 'status.candidateOverflow']
  );
  if (network) {
    dispatch(['closeness', 'reachedCounts']);
    dispatch(['status.converged', 'status.iterationLimitReached', 'status.zeroWeightCycle']);
    dispatch(['normalizedChoice'], ['betweenness']);
    dispatch(['metrics', 'integration', 'extendedIntegration'], ['degree']);
    dispatch(
      ['betweenness', 'closeness', 'reachedCounts', 'status.converged'],
      ['status.overflow', 'status.candidateOverflow', 'status.invalidCount']
    );
    dispatch(
      ['metrics', 'integration', 'extendedIntegration', 'normalizedChoice'],
      ['status.overflow', 'status.candidateOverflow', 'status.invalidCount']
    );
  }
  const writable = Object.entries(fields).filter((entry): entry is [string, GraphDataView] =>
    Boolean(entry[1])
  );
  for (const [index, [name, view]] of writable.entries()) {
    for (const [otherName, other] of writable.slice(index + 1))
      validateRequirement(
        id,
        !overlaps(view, other, false),
        `${name} and ${otherName} logical byte ranges must not overlap`
      );
    for (const [otherName, other] of Object.entries(inputs))
      if (other)
        validateRequirement(
          id,
          !overlaps(view, other, false),
          `${name} and ${otherName} logical byte ranges must not overlap`
        );
  }
}
