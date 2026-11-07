// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUMakeValid has no per-frame prop; the slider damages that many input polygons.
export const parameters: GpuSceneParameter[] = [
  {id: 'defects', label: 'Data: bow-ties', min: 0, max: 12, step: 1, default: 6}
];
