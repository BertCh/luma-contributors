// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SPACE_TIME_LOCAL_STATISTICS_AUDIT_CASES} from './kernel-audit/cases-space-time-local-statistics';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('space-time-local-statistics', SPACE_TIME_LOCAL_STATISTICS_AUDIT_CASES);
