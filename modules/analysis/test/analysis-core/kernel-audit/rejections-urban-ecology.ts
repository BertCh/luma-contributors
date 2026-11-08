// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the urban ecology cases. */
export const URBAN_ECOLOGY_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'addSparseResistanceConnectivityRecipe',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 1048576,
    rejects:
      /^Error: sparse-resistance-connectivity vertexCount must be an integer in \[2, 1048576\]$/
  },
  {
    exportName: 'addSparseResistanceConnectivityRecipe',
    name: 'edges above the maximum',
    kind: 'binding',
    limit: 2097152,
    rejects: /^Error: sparse-resistance-connectivity edges must have between 1 and 2097152 rows$/
  }
];
