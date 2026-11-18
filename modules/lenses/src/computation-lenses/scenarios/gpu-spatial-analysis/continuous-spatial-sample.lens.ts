// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-spatial-analysis.continuous-spatial-sample',
  element: 'GPUContinuousSpatialSample',
  title: 'GPUContinuousSpatialSample: primary and replacement area designs',
  rungs: DEFAULT_LENS_RUNGS,
  dataset: {
    id: 'continuous-frame/1',
    source:
      'Repository-authored unit square represented by two triangles; two independently shifted lattice waves.',
    licence: 'MIT'
  },
  async load() {
    const [{runS2Lens}, {runExample}] = await Promise.all([
      import('./s2-lens-run'),
      import('./continuous-spatial-sample.example')
    ]);
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.continuous-spatial-sample',
          element: 'GPUContinuousSpatialSample',
          datasets: ['continuous-frame/1'],
          seed: 43,
          choices: {spacing: 0.5, seed: 43, waves: 2},
          keySpaces: [{id: 'entry', label: 'Continuous frame sample row', indexSpace: '1d'}],
          build: async (graph, uploads) => {
            const frame = uploads.view(
              'frame',
              new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]),
              'float32x2'
            );
            const positions = uploads.view('positions', new Float32Array(16), 'float32x2');
            const designWeights = uploads.view('designWeights', new Float32Array(8), 'float32');
            const provenance = uploads.view('provenance', new Uint32Array(32), 'uint32x4');
            const status = uploads.view('status', new Uint32Array(8), 'uint32x4');
            await runExample(graph, {
              frame: {type: 'area', positions: frame, bounds: [0, 0, 1, 1]},
              spacing: 0.5,
              seed: 43,
              waves: 2,
              capacity: 4,
              output: {positions, designWeights, provenance, status}
            });
            return {
              frame: {view: frame, role: 'input', keySpace: 'entry'},
              positions: {view: positions, role: 'output', keySpace: 'entry'},
              designWeights: {view: designWeights, role: 'output', keySpace: 'entry'},
              provenance: {view: provenance, role: 'output', keySpace: 'entry'},
              status: {view: status, role: 'output', keySpace: 'entry'}
            };
          }
        })
    };
  }
});
