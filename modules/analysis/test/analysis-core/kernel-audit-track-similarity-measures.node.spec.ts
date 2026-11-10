// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {TRACK_SIMILARITY_MEASURE_AUDIT_CASES} from './kernel-audit/cases-track-similarity-measures';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('track-similarity-measures', TRACK_SIMILARITY_MEASURE_AUDIT_CASES);
