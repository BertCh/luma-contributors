// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// `from` and `to` choose which entry of the categoryCount x categoryCount quotient table is drawn.
// `neighborCount` is how many nearest events the weights give each event.
const ERAS = ['early third', 'middle third', 'late third'];

export const parameters: GpuSceneParameter[] = [
  {id: 'from', label: 'from category', min: 0, max: 2, step: 1, default: 0, options: ERAS},
  {id: 'to', label: 'to category', min: 0, max: 2, step: 1, default: 0, options: ERAS},
  {id: 'neighborCount', label: 'Weights: neighbors', min: 3, max: 10, step: 1, default: 6}
];
