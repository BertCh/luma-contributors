// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {STX04_STX05_SPACE_TIME_DENSITY_AND_K_AUDIT_CASES} from './kernel-audit/cases-stx04-stx05-space-time-density-and-k';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit(
  'stx04-stx05-space-time-density-and-k',
  STX04_STX05_SPACE_TIME_DENSITY_AND_K_AUDIT_CASES
);
