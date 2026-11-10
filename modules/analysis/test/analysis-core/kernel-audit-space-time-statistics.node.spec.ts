// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SPACE_TIME_STATISTICS_AUDIT_CASES} from './kernel-audit/cases-space-time-statistics';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('space-time-statistics', SPACE_TIME_STATISTICS_AUDIT_CASES);
