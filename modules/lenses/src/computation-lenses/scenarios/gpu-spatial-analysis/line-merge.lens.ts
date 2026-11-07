// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';

const SCENARIO_ID = 'gpu-spatial-analysis.line-merge';

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPULineMerge',
  title: 'Merge Wicker Park streets through degree-two endpoints',
  rungs: DEFAULT_LENS_RUNGS,
  dataset: {
    id: 's3-chicago-roads',
    source: 'showcase public/data/chicago-roads (OpenStreetMap)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./line-merge.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);

    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineMerge',
          datasets: [data.id],
          keySpaces: [
            {id: 'road', label: 'road', size: data.lineCount, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {directed: false},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const output = await runExample(graph, {roads});
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              chainOffsets: {view: output.chainOffsets, role: 'intermediate'},
              lineChains: {
                view: output.lineChains,
                role: 'intermediate',
                keySpace: 'road',
                meaning: 'merged chain index for each source road (zero-based)'
              },
              count: {view: output.status.count, role: 'intermediate'},
              mergedRoads: {
                view: output.mergedRoads,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'chainOffsets'})
              }
            };
          }
        })
    };
  }
});
