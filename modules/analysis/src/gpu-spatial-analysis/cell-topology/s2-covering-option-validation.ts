// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Validate shared string options before either execution or deferred contributor lowering. @internal */
export function validateS2CoveringOptions(
  id: string,
  options: {mode?: string; refinement?: string},
  refinements: readonly string[]
): void {
  validateRequirement(
    id,
    options.mode === undefined || options.mode === 'exterior' || options.mode === 'interior',
    `mode must be 'exterior' or 'interior'; received ${String(options.mode)}`
  );
  validateRequirement(
    id,
    options.refinement === undefined || refinements.includes(options.refinement),
    `refinement must be ${refinements.map(value => `'${value}'`).join(' or ')}; received ${String(options.refinement)}`
  );
}
