// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {K_SHORTEST_PATHS_AUDIT_CASES} from './kernel-audit/cases-k-shortest-paths';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('k-shortest-paths', K_SHORTEST_PATHS_AUDIT_CASES);
