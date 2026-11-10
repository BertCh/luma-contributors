// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {STX08_SECOND_GROUP_AUDIT_CASES} from './kernel-audit/cases-stx08-colocation-licd-adaptive-density';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('stx08-colocation-licd-adaptive-density', STX08_SECOND_GROUP_AUDIT_CASES);
