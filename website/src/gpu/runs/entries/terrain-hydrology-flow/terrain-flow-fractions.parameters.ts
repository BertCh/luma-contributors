// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// routing is the operation's compile-time prop; flowExponent is the MFD exponent packed into the
// shared flow parameters (0 selects the published exponent).
export const parameters: GpuSceneParameter[] = [
  {
    id: 'routing',
    label: 'routing',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['d-infinity', 'mfd-freeman', 'mfd-quinn']
  },
  {id: 'flowExponent', label: 'flowExponent', min: 0.5, max: 6, step: 0.1, default: 1.1}
];
