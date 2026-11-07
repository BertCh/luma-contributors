// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COVERAGE_GLOBAL_CASES} from './kernel-audit/cases-coverage-global';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('global-coordinates', COVERAGE_GLOBAL_CASES);
