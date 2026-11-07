// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {vi} from 'vitest';
import {NETWORK_CASES} from './kernel-audit/cases-network';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

// 164 cases build and audit WGSL for every network contributor (about 55 s in total on a loaded machine,
// 0.3 s per case on average); a single case that starts while other workers saturate the CPU can exceed the
// default 60 s budget although no case regressed, so give each one explicit headroom.
vi.setConfig({testTimeout: 120_000});

registerKernelAudit('network', NETWORK_CASES);
