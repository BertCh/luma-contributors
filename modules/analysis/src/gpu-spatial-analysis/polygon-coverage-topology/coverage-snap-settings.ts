// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateGeometryFloat32Setting} from '../geometry-edit/float32-setting-validation';

/** Validates the grid spacing before selecting the coverage kernel. @internal */
export function resolveCoverageSnapGridSize(id: string, props: {snapGridSize?: number}): number {
  const {snapGridSize} = props;
  if (snapGridSize !== undefined) {
    validateGeometryFloat32Setting(id, 'snapGridSize', snapGridSize, true, true);
  }
  return snapGridSize ?? 0;
}
