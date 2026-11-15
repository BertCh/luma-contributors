// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COREGIONALIZATION_FIT_AUDIT_CASES} from './kernel-audit/cases-coregionalization-fit';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('coregionalization-fit', COREGIONALIZATION_FIT_AUDIT_CASES);
