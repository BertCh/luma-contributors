// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {doGraphDataViewsOverlap, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Packed statistics ports written in separate dispatches need logical separation; only co-bound
 * writable ports need aligned separation. Include only ports actually written by this route.
 * Caller inputs remain subject to the contributor's read/write buffer separation check.
 * @internal
 */
export function validateSpatialStatisticsWritableDispatches(
  id: string,
  fields: Readonly<Record<string, GraphDataView | undefined>>,
  dispatches: readonly (readonly string[])[]
): void {
  const entries = Object.entries(fields).filter((entry): entry is [string, GraphDataView] =>
    Boolean(entry[1])
  );
  for (const [index, [name, view]] of entries.entries()) {
    for (const [otherName, other] of entries.slice(index + 1)) {
      const physical = view.buffer.defaultBuffer && getCoreDefaultBuffer(view.buffer.defaultBuffer);
      const otherPhysical =
        other.buffer.defaultBuffer && getCoreDefaultBuffer(other.buffer.defaultBuffer);
      validateRequirement(
        id,
        !physical || physical !== otherPhysical || view.buffer === other.buffer,
        `${name} and ${otherName} share a physical buffer: import it once and use offset views`
      );
      validateRequirement(
        id,
        !doGraphDataViewsOverlap(view, other),
        `${name} and ${otherName} logical byte ranges must not overlap`
      );
    }
  }
  for (const names of dispatches)
    validateDisjointWritableViews(id, Object.fromEntries(names.map(name => [name, fields[name]])));
}
