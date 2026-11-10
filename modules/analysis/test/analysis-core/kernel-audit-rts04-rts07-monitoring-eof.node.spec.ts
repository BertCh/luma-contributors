// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RTS04_RTS07_MONITORING_EOF_AUDIT_CASES} from './kernel-audit/cases-rts04-rts07-monitoring-eof';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('rts04-rts07-monitoring-eof', RTS04_RTS07_MONITORING_EOF_AUDIT_CASES);
