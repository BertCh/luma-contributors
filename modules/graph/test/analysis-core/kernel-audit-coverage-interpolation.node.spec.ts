// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COVERAGE_INTERPOLATION_KERNEL_CASES} from './kernel-audit/cases-coverage-interpolation';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('coverage-interpolation', COVERAGE_INTERPOLATION_KERNEL_CASES);
