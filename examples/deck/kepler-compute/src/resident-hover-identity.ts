// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DataRow} from '@kepler.gl/utils';

/** A native DataRow carrying identity only; no source container is retained or read. */
export class ResidentHoverIdentity extends DataRow {
  constructor(readonly sourceRowIndex: number) {
    super(null, sourceRowIndex);
  }
}

/** Kepler passes picking information fifth even to native classes with shorter signatures. */
export function getResidentHoverIdentity(
  parameters: readonly unknown[]
): ResidentHoverIdentity | null {
  const information = parameters[4];
  if (
    !information ||
    typeof information !== 'object' ||
    !('index' in information) ||
    typeof information.index !== 'number' ||
    information.index < 0
  )
    return null;
  const layer = 'layer' in information ? information.layer : undefined;
  const properties =
    layer && typeof layer === 'object' && 'props' in layer ? layer.props : undefined;
  const resolveSourceRow =
    properties && typeof properties === 'object' && 'nativeSourceRowIndex' in properties
      ? properties.nativeSourceRowIndex
      : undefined;
  const sourceRowIndex =
    typeof resolveSourceRow === 'function'
      ? resolveSourceRow(information.index)
      : information.index;
  return new ResidentHoverIdentity(sourceRowIndex);
}
