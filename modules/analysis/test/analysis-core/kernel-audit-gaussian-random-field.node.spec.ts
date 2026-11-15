// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GAUSSIAN_RANDOM_FIELD_KERNEL_CASES} from './kernel-audit/cases-gaussian-random-field';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';
registerKernelAudit('gaussian-random-field', GAUSSIAN_RANDOM_FIELD_KERNEL_CASES);
