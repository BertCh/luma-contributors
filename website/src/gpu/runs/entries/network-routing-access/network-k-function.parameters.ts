// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumDistance and the simulation count are GPUNetworkKFunction's per-frame views; clustering
// moves the observed events.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.2, max: 1.2, step: 0.05, default: 0.8},
  {id: 'activeSimulations', label: 'activeSimulations', min: 1, max: 19, step: 1, default: 19},
  {id: 'clustering', label: 'Data: clustering', min: 0, max: 1, step: 0.05, default: 0.7}
];
