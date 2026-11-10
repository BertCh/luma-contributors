// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Dispatch numbers of two recorded kernels, derived from a view model. Nothing here is typed in:
// invocations and workgroups come from the record, the workgroup size is implied by them.

import type {LensViewModel} from '../../../../lens/types';
import type {WorkgroupGrid} from '../../scenes/workgroup-scene';

export type RecordedDispatch = {
  label: string;
  invocations: number;
  workgroups: number;
  /** Power-of-two workgroup size implied by the record, or null when none reproduces it. */
  workgroupSize: number | null;
  slots: number;
  guarded: number;
};

export function getRecordedDispatch(
  model: LensViewModel | undefined,
  label: string
): RecordedDispatch | null {
  const node = model?.nodes.find(candidate => candidate.label === label);
  if (!node?.invocations || !node.workgroups) return null;
  const size = 2 ** Math.ceil(Math.log2(node.invocations / node.workgroups));
  const isConsistent = Math.ceil(node.invocations / size) === node.workgroups;
  return {
    label,
    invocations: node.invocations,
    workgroups: node.workgroups,
    workgroupSize: isConsistent ? size : null,
    slots: isConsistent ? node.workgroups * size : node.workgroups,
    guarded: isConsistent ? node.workgroups * size - node.invocations : 0
  };
}

/** A grid of workgroups with the recorded dispatch's workgroup count: the divisor of it nearest a wide frame. */
export function getRecordedGrid(dispatch: RecordedDispatch): WorkgroupGrid | null {
  if (!dispatch.workgroupSize) return null;
  const laneSide = Math.sqrt(dispatch.workgroupSize);
  if (!Number.isInteger(laneSide)) return null;
  const target = Math.sqrt(dispatch.workgroups * 3);
  let columns = 1;
  for (let divisor = 1; divisor <= dispatch.workgroups; divisor++) {
    if (
      dispatch.workgroups % divisor === 0 &&
      Math.abs(divisor - target) < Math.abs(columns - target)
    ) {
      columns = divisor;
    }
  }
  return {
    columns,
    rows: dispatch.workgroups / columns,
    laneSide,
    invocationCount: dispatch.invocations,
    concurrency: 7
  };
}
