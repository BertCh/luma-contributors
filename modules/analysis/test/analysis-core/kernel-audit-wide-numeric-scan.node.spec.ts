// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WIDE_NUMERIC_SCAN_CASES} from './kernel-audit/cases-wide-numeric-scan';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('wide-numeric-scan', WIDE_NUMERIC_SCAN_CASES);
