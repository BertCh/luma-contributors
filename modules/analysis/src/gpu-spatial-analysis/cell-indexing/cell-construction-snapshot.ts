// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

/** Explicit configuration fields; graph views are leaves and keep their live identity. @internal */
export type CellConstructionFields = readonly (
  | string
  | readonly [string, CellConstructionFields]
)[];

/** Capture validated cell configuration, including nested options and output selections. @internal */
export function snapshotCellConstruction<Props extends object>(
  props: Props,
  fields: CellConstructionFields
): Props {
  const captured = snapshotConstructionFields(
    props as Record<string, unknown>,
    fields.map(field => (typeof field === 'string' ? field : field[0]))
  );
  const result = {...captured};
  for (const field of fields) {
    if (typeof field !== 'string' && captured[field[0]] !== undefined) {
      result[field[0]] = snapshotCellConstruction(captured[field[0]] as object, field[1]);
    }
  }
  if (captured['spatialContext'])
    result['spatialContext'] = snapshotConstructionSpatialContext(
      captured['spatialContext'] as GPUSpatialContext
    );
  return Object.freeze(result) as Props;
}
