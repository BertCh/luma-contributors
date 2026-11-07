// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COVERAGE_URBAN_ECOLOGY_CASES} from './kernel-audit/cases-coverage-urban-ecology';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('urban-ecology', COVERAGE_URBAN_ECOLOGY_CASES);
