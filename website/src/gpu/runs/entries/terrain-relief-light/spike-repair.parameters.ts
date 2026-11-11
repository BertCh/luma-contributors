// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The only slider changes the input tile; GPUTerrainSpikeRepair's jump (200 m) and tolerance (40 m) are fixed.
export const parameters: GpuSceneParameter[] = [
  {id: 'spikeCount', label: 'Data: spikes', min: 0, max: 12, step: 1, default: 8}
];
