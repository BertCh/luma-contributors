// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MOB10_FLOW_MODELS_AUDIT_CASES} from './kernel-audit/cases-mob10-flow-models';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('mob10-flow-models', MOB10_FLOW_MODELS_AUDIT_CASES);
