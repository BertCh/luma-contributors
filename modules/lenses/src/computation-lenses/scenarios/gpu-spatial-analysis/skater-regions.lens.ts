// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-spatial-analysis.skater-regions',
  element: 'GPUSkaterRegions',
  title: 'SKATER: cutting the poverty spanning tree of north-side Chicago tracts',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 's1-chicago-tracts/1',
    source: 'Census cb_2022_17_tract_500k; CDC/ATSDR SVI 2022 (ACS 2018-2022)',
    licence:
      'US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain; CDC/ATSDR SVI 2022, public domain'
  },
  async load() {
    const [{runExample}, support, utilities, spatial, {STATISTICS_OPERATORS}] = await Promise.all([
      import('./skater-regions.example'),
      import('./s1-regions-support'),
      import('./spatial-scenario-utils'),
      import('@luma-contributors/analysis/gpu-spatial-analysis'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await support.loadRegionTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.skater-regions',
          tracts,
          entry: STATISTICS_OPERATORS['GPUSkaterRegions'],
          columns: {values: 'poverty150Pct'},
          choices: {
            criterion: 'queen',
            regionCount: 5,
            minimumRegionSize: 5,
            maximumRegionCount: 8
          },
          async execute({graph, polygons, columns, upload}) {
            const parameters = upload(
              'parameters',
              spatial.getGPUSkaterRegionsParameterValues({regionCount: 5, minimumRegionSize: 5}),
              'uint32'
            );
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values'],
              parameters
            });
            return {
              ...execution,
              tags: {
                values: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Population below 150% poverty line (%)'
                },
                labels: {
                  role: 'output',
                  polygons: true,
                  meaning: 'SKATER region (nominal; label is the row nearest the tree root)'
                },
                regionCount: {
                  role: 'intermediate',
                  keySpace: 'status',
                  meaning: 'Regions after the cuts'
                },
                cutEdges: {
                  role: 'intermediate',
                  keySpace: 'cut',
                  meaning: 'Edge slot cut at each step; 0xffffffff means no cut'
                },
                cutGains: {
                  role: 'intermediate',
                  keySpace: 'cut',
                  meaning: 'Within-region sum of squares removed by each cut'
                }
              },
              keySpaces: [
                {id: 'cut', label: 'Greedy cut step', size: 7, indexSpace: '1d'},
                {id: 'status', label: 'Scalar and status words', indexSpace: '1d'}
              ]
            };
          }
        })
    };
  }
});
