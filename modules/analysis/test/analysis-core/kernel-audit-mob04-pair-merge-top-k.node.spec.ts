// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MOB04_PAIR_MERGE_TOP_K_AUDIT_CASES} from './kernel-audit/cases-mob04-pair-merge-top-k';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('mob04-pair-merge-top-k', MOB04_PAIR_MERGE_TOP_K_AUDIT_CASES);
