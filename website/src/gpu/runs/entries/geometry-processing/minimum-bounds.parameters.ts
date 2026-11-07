// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// maximumHullVertices caps the hull size; a polygon whose hull is larger loses its shapes (NaN rows).
export const parameters: GpuSceneParameter[] = [
  {id: 'maximumHullVertices', label: 'maximumHullVertices', min: 3, max: 12, step: 1, default: 12},
  {id: 'stretch', label: 'Data: stretch', min: 0, max: 1, step: 0.05, default: 0.7}
];
