// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {STX08_SPATIAL_METHODS_AUDIT_CASES} from './kernel-audit/cases-stx08-spatial-methods';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('stx08-spatial-methods', STX08_SPATIAL_METHODS_AUDIT_CASES);
