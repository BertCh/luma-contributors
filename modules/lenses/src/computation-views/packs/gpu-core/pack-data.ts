// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewPackData} from '../../registry/view-registration';
import {GPU_CORE_FORMULAS} from './formulas';

/** Pure coverage entry: importing this file performs no registration and loads no GPU runtime. */
export function loadGPUCorePackData(): ViewPackData {
  return {
    id: 'gpu-core',
    formulas: GPU_CORE_FORMULAS,
    analysisStages: ['input', 'operator', 'output']
  };
}
