// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `strength` varies the spatial error of the generated panel; the estimates follow it.
export const parameters: GpuSceneParameter[] = [
  {id: 'strength', label: 'Data: spatial error lambda', min: 0, max: 0.7, step: 0.05, default: 0.5}
];
