// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {POINT_NEIGHBOR_GRID_UNIFICATION_AUDIT_CASES} from './kernel-audit/cases-point-neighbor-grid-unification';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('point-neighbor-grid-unification', POINT_NEIGHBOR_GRID_UNIFICATION_AUDIT_CASES);
