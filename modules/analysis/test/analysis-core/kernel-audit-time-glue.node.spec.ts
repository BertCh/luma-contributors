// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {TIME_GLUE_AUDIT_CASES} from './kernel-audit/cases-time-glue';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('time-glue', TIME_GLUE_AUDIT_CASES);
