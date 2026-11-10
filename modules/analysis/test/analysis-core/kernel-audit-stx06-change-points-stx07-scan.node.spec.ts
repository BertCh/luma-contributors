// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {STX06_CHANGE_POINTS_STX07_SCAN_AUDIT_CASES} from './kernel-audit/cases-stx06-change-points-stx07-scan';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('stx06-change-points-stx07-scan', STX06_CHANGE_POINTS_STX07_SCAN_AUDIT_CASES);
