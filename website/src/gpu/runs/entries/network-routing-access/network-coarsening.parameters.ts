// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// cellSize relabels the nodes each frame (the `labels` view); closures thin the network.
export const parameters: GpuSceneParameter[] = [
  {id: 'cellSize', label: 'labels (cell size)', min: 0.2, max: 0.6, step: 0.02, default: 0.3},
  {id: 'closures', label: 'Data: closures', min: 0, max: 0.4, step: 0.05, default: 0.1}
];
