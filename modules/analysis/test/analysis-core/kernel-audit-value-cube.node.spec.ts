// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {VALUE_CUBE_AUDIT_CASES} from './kernel-audit/cases-value-cube';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('value-cube', VALUE_CUBE_AUDIT_CASES);
