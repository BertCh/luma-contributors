// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {CONFORMAL_PREDICTION_INTERVALS_AUDIT_CASES} from './kernel-audit/cases-conformal-prediction-intervals';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';
registerKernelAudit('conformal-prediction-intervals', CONFORMAL_PREDICTION_INTERVALS_AUDIT_CASES);
