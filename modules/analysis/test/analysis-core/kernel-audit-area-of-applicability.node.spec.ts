// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {AREA_OF_APPLICABILITY_AUDIT_CASES} from './kernel-audit/cases-area-of-applicability';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';
registerKernelAudit('area-of-applicability', AREA_OF_APPLICABILITY_AUDIT_CASES);
