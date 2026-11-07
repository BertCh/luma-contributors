// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COVERAGE_RASTER_KERNEL_CASES} from './kernel-audit/cases-coverage-raster';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('coverage-raster', COVERAGE_RASTER_KERNEL_CASES);
