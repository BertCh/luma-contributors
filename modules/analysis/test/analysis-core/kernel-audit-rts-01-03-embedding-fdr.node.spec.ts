// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RTS_01_03_EMBEDDING_FDR_AUDIT_CASES} from './kernel-audit/cases-rts-01-03-embedding-fdr';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('rts-01-03-embedding-fdr', RTS_01_03_EMBEDDING_FDR_AUDIT_CASES);
