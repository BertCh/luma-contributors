// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MOB07_HOME_RANGE_RECURSION_AUDIT_CASES} from './kernel-audit/cases-mob07-home-range-recursion';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('mob07-home-range-recursion', MOB07_HOME_RANGE_RECURSION_AUDIT_CASES);
