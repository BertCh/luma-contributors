// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COVERAGE_POINT_PROCESS_CASES} from './kernel-audit/cases-coverage-point-process';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('coverage-point-process', COVERAGE_POINT_PROCESS_CASES);
