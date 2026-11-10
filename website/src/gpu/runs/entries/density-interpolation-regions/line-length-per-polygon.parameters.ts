// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The lines are the operation's input; `lineCount` varies how much line there is to measure.
export const parameters: GpuSceneParameter[] = [
  {id: 'lineCount', label: 'Data: lineCount', min: 3, max: 40, step: 1, default: 16},
  {id: 'tractRows', label: 'Data: tract window size', min: 4, max: 12, step: 1, default: 7}
];
