// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// exponent is GPUHuffTradeAreas' per-frame attractiveness exponent; beta and costLimit shape the
// decay weights it reads.
export const parameters: GpuSceneParameter[] = [
  {id: 'exponent', label: 'exponent', min: 0.5, max: 4, step: 0.25, default: 1},
  {id: 'beta', label: 'beta', min: 0.5, max: 8, step: 0.5, default: 3},
  {id: 'costLimit', label: 'costLimit', min: 0.3, max: 2, step: 0.1, default: 1.5}
];
