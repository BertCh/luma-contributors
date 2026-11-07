// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumDistance is the per-frame packed setting and seedCount the per-frame `seedCount` view.
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumDistance', label: 'maximumDistance', min: 0.05, max: 0.8, step: 0.05, default: 0.4},
  {id: 'seedCount', label: 'seedCount', min: 1, max: 12, step: 1, default: 8}
];
