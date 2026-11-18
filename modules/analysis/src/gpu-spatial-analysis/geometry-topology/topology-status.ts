// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import type {GPUPolygonizeOutput} from './topology-types';

/** Checks final feature and ring cardinality scalars without conflating their storage. */
export function validatePolygonizeStatus(id: string, status: GPUPolygonizeOutput['status']): void {
  validateRequirement(
    id,
    Boolean(status.requiredRingCount && status.candidateOverflow),
    'output.status requires requiredRingCount and candidateOverflow'
  );
  for (const [name, view] of Object.entries(status)) {
    if (!view) continue;
    validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
    validateCapacity(id, `output.status.${name}.length`, view.length, 1);
  }
  validateOutlineOutputPasses(`${id} output.status`, status, []);
}
