// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RTS05_COMPOSITING_AUDIT_CASES} from './kernel-audit/cases-rts05-compositing';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('rts05-compositing', RTS05_COMPOSITING_AUDIT_CASES);
