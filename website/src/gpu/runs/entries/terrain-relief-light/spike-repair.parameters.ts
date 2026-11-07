// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Bound to GPUTerrainSpikeRepair({jump, tolerance}); the spike count changes the input tile.
export const parameters: GpuSceneParameter[] = [
  {id: 'jump', label: 'jump', min: 50, max: 300, step: 10, default: 200, unit: ' m'},
  {id: 'tolerance', label: 'tolerance', min: 5, max: 100, step: 5, default: 40, unit: ' m'},
  {id: 'spikeCount', label: 'Data: spikes', min: 0, max: 12, step: 1, default: 8}
];
