// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateOption} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Shared accepted set for direct contributors, named operations and the CPU reference. @internal */
export function validateSphericalOverlayOperation(id: string, operation: string): void {
  validateOption(id, 'operation', operation, [
    'intersection',
    'union',
    'difference',
    'symmetric-difference'
  ]);
}
