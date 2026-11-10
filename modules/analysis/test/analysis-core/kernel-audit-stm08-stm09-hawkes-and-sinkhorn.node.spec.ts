// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {STM08_STM09_HAWKES_AND_SINKHORN_AUDIT_CASES} from './kernel-audit/cases-stm08-stm09-hawkes-and-sinkhorn';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('stm08-stm09-hawkes-and-sinkhorn', STM08_STM09_HAWKES_AND_SINKHORN_AUDIT_CASES);
