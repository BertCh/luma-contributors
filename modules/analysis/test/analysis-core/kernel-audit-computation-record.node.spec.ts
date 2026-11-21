// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COMPUTATION_RECORD_AUDIT_CASES} from './kernel-audit/cases-computation-record';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('computation-record', COMPUTATION_RECORD_AUDIT_CASES);
