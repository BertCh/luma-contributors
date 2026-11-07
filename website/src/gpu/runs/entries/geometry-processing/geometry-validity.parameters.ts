// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// orientation is a constructor prop of GPUGeometryValidity; defects damages that many polygons.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'orientation',
    label: 'orientation',
    min: 0,
    max: 2,
    step: 1,
    default: 0,
    options: ['counter-clockwise-shell', 'clockwise-shell', 'ignore']
  },
  {id: 'defects', label: 'Data: defects', min: 0, max: 12, step: 1, default: 6}
];
