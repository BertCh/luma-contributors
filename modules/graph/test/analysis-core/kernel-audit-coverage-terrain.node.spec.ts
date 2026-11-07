// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COVERAGE_TERRAIN_CASES} from './kernel-audit/cases-coverage-terrain';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('coverage-terrain', COVERAGE_TERRAIN_CASES);
