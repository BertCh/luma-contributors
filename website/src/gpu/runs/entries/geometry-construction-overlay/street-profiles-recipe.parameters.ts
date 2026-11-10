// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.04, max: 0.5, step: 0.01, default: 0.4},
  {id: 'streetWidth', label: 'Data: street widening', min: 0, max: 0.2, step: 0.01, default: 0}
];
