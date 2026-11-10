// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MOB05_MOB09_AUDIT_CASES} from './kernel-audit/cases-mob05-mob09-edge-volumes-time-geography';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('mob05-mob09-edge-volumes-time-geography', MOB05_MOB09_AUDIT_CASES);
