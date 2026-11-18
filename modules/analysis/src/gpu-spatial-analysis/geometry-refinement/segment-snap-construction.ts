// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

type SegmentSnapConstruction = {
  spatialContext?: GPUSpatialContext;
  output: {
    positions: unknown;
    sourceEdges: unknown;
    counts: unknown;
    validity?: unknown;
    issueCode?: unknown;
    candidateOverflow?: unknown;
    status?: {candidateOverflow?: unknown};
  };
};

/** Capture validated policy and port maps without cloning caller-owned graph views. @internal */
export function snapshotSegmentSnapConstruction<Props extends SegmentSnapConstruction>(
  props: Props,
  fields: readonly (keyof Props)[]
): Props {
  const snapshot = snapshotConstructionFields(props, fields);
  const output = snapshotConstructionFields(snapshot.output, [
    'positions',
    'sourceEdges',
    'counts',
    'validity',
    'issueCode',
    'candidateOverflow',
    'status'
  ]);
  return Object.freeze({
    ...snapshot,
    ...(snapshot.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(snapshot.spatialContext)}
      : {}),
    output: Object.freeze({
      ...output,
      ...(output.status
        ? {status: snapshotConstructionFields(output.status, ['candidateOverflow'])}
        : {})
    })
  }) as Props;
}
