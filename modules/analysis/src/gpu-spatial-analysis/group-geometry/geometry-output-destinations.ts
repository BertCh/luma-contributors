// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';

/** Newly added diagnostic words cannot overwrite any existing geometry output. @internal */
export function validateGeometryAuxiliaryOutput(
  id: string,
  name: string,
  view: GraphDataView | undefined,
  others: Readonly<Record<string, GraphDataView | undefined>>
): void {
  if (!view) return;
  for (const [otherName, other] of Object.entries(others)) {
    validateDisjointWritableViews(
      id,
      {[name]: view, [otherName]: other},
      {separateDispatches: true}
    );
  }
}
