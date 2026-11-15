// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {FOREST_INFERENCE_AUDIT_CASES} from './kernel-audit/cases-forest-inference';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('forest-inference', FOREST_INFERENCE_AUDIT_CASES);
