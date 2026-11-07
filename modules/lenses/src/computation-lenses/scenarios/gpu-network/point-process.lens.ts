// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-network.network-kernel-density',
  element: 'GPUNetworkKernelDensity',
  title: 'GPUNetworkKernelDensity: explicit small point-process example',
  rungs: DEFAULT_LENS_RUNGS,
  dataset: {
    id: 'point-process-illustrative/1',
    source:
      'Repository-authored small planar or network frame; supplied intensity and supplied null curves are explicit illustrative inputs.',
    licence: 'MIT'
  },
  async load() {
    const {runNetworkPointProcess, describeNetworkPointProcess} = await import(
      './point-process-examples'
    );
    return {
      run: device => runNetworkPointProcess(device, 'GPUNetworkKernelDensity'),
      describeStructure: device => describeNetworkPointProcess(device, 'GPUNetworkKernelDensity')
    };
  }
});
defineLensScenario({
  id: 'gpu-network.network-inhomogeneous-k',
  element: 'GPUNetworkInhomogeneousK',
  title: 'GPUNetworkInhomogeneousK: explicit small point-process example',
  rungs: DEFAULT_LENS_RUNGS,
  dataset: {
    id: 'point-process-illustrative/1',
    source:
      'Repository-authored small planar or network frame; supplied intensity and supplied null curves are explicit illustrative inputs.',
    licence: 'MIT'
  },
  async load() {
    const {runNetworkPointProcess, describeNetworkPointProcess} = await import(
      './point-process-examples'
    );
    return {
      run: device => runNetworkPointProcess(device, 'GPUNetworkInhomogeneousK'),
      describeStructure: device => describeNetworkPointProcess(device, 'GPUNetworkInhomogeneousK')
    };
  }
});
