// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0.002, max: 0.1, step: 0.0005, default: 0.04}
];
