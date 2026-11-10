// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RTS_TIME_SERIES_AUDIT_CASES} from './kernel-audit/cases-rts-time-series';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('rts-time-series', RTS_TIME_SERIES_AUDIT_CASES);
