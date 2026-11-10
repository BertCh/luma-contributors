// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MOB06_LAGRANGIAN_DIAGNOSTICS_AUDIT_CASES} from './kernel-audit/cases-mob06-lagrangian-diagnostics';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('mob06-lagrangian-diagnostics', MOB06_LAGRANGIAN_DIAGNOSTICS_AUDIT_CASES);
