// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';

type SpatialOutputDispatch =
  | readonly (GraphDataView | undefined)[]
  | {read: readonly (GraphDataView | undefined)[]; write: readonly (GraphDataView | undefined)[]};

function getPhysicalBuffer(view: GraphDataView) {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** Logical outputs survive across passes; aligned prefixes matter only within one dispatch. */
export function validateSpatialOutputViews(
  id: string,
  outputs: Readonly<Record<string, GraphDataView | undefined>>,
  dispatches: readonly SpatialOutputDispatch[]
): void {
  const fields = Object.entries(outputs).filter((entry): entry is [string, GraphDataView] =>
    Boolean(entry[1]?.length)
  );
  for (let firstIndex = 0; firstIndex < fields.length; firstIndex++) {
    const [firstName, first] = fields[firstIndex];
    const firstBuffer = getPhysicalBuffer(first);
    for (const [secondName, second] of fields.slice(firstIndex + 1)) {
      const secondBuffer = getPhysicalBuffer(second);
      const firstEnd =
        first.byteOffset + (first.length - 1) * first.byteStride + first.rowByteLength;
      const secondEnd =
        second.byteOffset + (second.length - 1) * second.byteStride + second.rowByteLength;
      validateRequirement(
        id,
        firstBuffer !== secondBuffer ||
          first.byteOffset >= secondEnd ||
          second.byteOffset >= firstEnd,
        `${firstName} and ${secondName} must not overlap writable byte ranges`
      );
    }
  }
  for (const dispatch of dispatches) {
    const views = 'write' in dispatch ? dispatch.write : dispatch;
    validateDisjointWritableViews(
      id,
      Object.fromEntries(views.map((view, index) => [`binding[${index}]`, view]))
    );
    if ('read' in dispatch) {
      for (const input of dispatch.read) {
        if (!input?.length) continue;
        for (const output of dispatch.write) {
          if (!output?.length) continue;
          // WebGPU usage scopes apply to the whole physical buffer, even for disjoint bindings.
          validateRequirement(
            id,
            getPhysicalBuffer(input) !== getPhysicalBuffer(output),
            'read-only storage and writable storage bindings must use different physical buffers'
          );
        }
      }
    }
  }
}
