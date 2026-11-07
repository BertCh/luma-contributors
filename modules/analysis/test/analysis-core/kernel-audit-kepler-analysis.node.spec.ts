// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {QUANTILE_LISA_AUDIT_CASES} from './kernel-audit/cases-quantile-lisa';
import {EXACT_NATURAL_BREAKS_AUDIT_CASES} from './kernel-audit/cases-exact-natural-breaks';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';
import {DORLING_CARTOGRAM_AUDIT_CASES} from './kernel-audit/cases-dorling-cartogram';
import {VARIABLE_STANDARDIZATION_AUDIT_CASES} from './kernel-audit/cases-variable-standardization';
import {RATE_MAPPING_KERNEL_AUDIT_CASES} from './kernel-audit/cases-rate-mapping';

registerKernelAudit('kepler-analysis', [
  ...QUANTILE_LISA_AUDIT_CASES,
  ...EXACT_NATURAL_BREAKS_AUDIT_CASES,
  ...DORLING_CARTOGRAM_AUDIT_CASES,
  ...VARIABLE_STANDARDIZATION_AUDIT_CASES,
  ...RATE_MAPPING_KERNEL_AUDIT_CASES
]);
