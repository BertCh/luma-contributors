// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.04, max: 0.4, step: 0.01, default: 0.2},
  {id: 'streetWidth', label: 'Data: streetWidth', min: 0.08, max: 0.4, step: 0.01, default: 0.2}
];
