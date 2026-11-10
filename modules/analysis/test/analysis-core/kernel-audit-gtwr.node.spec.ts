// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GTWR_AUDIT_CASES} from './kernel-audit/cases-gtwr';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';
registerKernelAudit('gtwr', GTWR_AUDIT_CASES);
