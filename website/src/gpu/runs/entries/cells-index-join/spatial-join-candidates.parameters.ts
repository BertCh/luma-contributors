// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `distance` is GPUSpatialJoinCandidates' per-frame view: it grows every zone's bounds before the
// bounding-box test. `size` scales the zones (a per-frame rewrite of their vertices).
export const parameters: GpuSceneParameter[] = [
  {id: 'distance', label: 'distance', min: 0, max: 0.12, step: 0.005, default: 0.03},
  {id: 'size', label: 'Data: size', min: 0.6, max: 1.3, step: 0.02, default: 1}
];
