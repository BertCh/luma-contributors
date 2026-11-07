// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma-contributors/graph/computation-record';

defineLensScenario({
  id: 'gpu-spatial-analysis.minimum-clearance',
  element: 'GPUMinimumClearance',
  title: 'Minimum clearance of Chicago North Side parks',
  rungs: FORMULA_LENS_RUNGS,
  dataset: S2_DATASETS.parks,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, parks] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./minimum-clearance.example'),
        import('../../data/s2-chicago-north-parks.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.minimum-clearance',
          element: 'GPUMinimumClearance',
          datasets: ['s2-chicago-north-parks'],
          retain: ['clearances', 'vertexIds'],
          keySpaces,
          choices: {outputs: 'clearances, vertexIds'},
          async build(graph, uploads) {
            const parkGeometry = uploadS2JoinPolygons(uploads, frame, 'parks', parks.polygons);
            keySpaces.push(...parkGeometry.keySpaces);
            const result = await runExample(graph, {
              parks: parkGeometry.geometry,
              spatialContext: frame.spatialContext
            });
            const captures: Record<string, S2CaptureSpec> = {...parkGeometry.captures};
            captures['resultParks'] = {
              view: parkGeometry.display,
              role: 'output',
              keySpace: 'parks',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'parksDisplayOffsets',
                valuesCapture: 'clearances',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['clearances'] = {
              view: result.clearances,
              role: 'output',
              keySpace: 'parks',
              meaning: 'metres; Infinity where a park has no clearance'
            };
            captures['vertexIds'] = {
              view: result.vertexIds,
              role: 'output',
              keySpace: 'parks',
              meaning: 'row of the fragile vertex in the uploaded vertices; 0xffffffff when none'
            };
            return captures;
          }
        });
      }
    };
  }
});
