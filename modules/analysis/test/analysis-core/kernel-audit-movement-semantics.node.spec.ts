// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MOVEMENT_SEMANTICS_AUDIT_CASES} from './kernel-audit/cases-movement-semantics';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('movement-semantics', MOVEMENT_SEMANTICS_AUDIT_CASES);
