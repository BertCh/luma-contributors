// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// distance fills the operation's per-row distances buffer; bearings are the fan of directions.
export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 200, max: 9000, step: 200, default: 4000, unit: ' km'},
  {
    id: 'bearingSpan',
    label: 'Data: bearing span',
    min: 30,
    max: 360,
    step: 15,
    default: 360,
    unit: '°'
  }
];
