// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EMBEDDED_DATASET_LOADERS} from '../../embedded-dataset-loaders.js';
import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma-contributors/analysis/computation-record';

defineLensScenario({
  id: 'gpu-spatial-analysis.pair-gather',
  element: 'GPUPairGather',
  title: 'Pair Gather, Chicago North Side',
  rungs: DEFAULT_LENS_RUNGS,
  dataset: S2_DATASETS.parks,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, parks, tracts] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./pair-gather.example'),
        EMBEDDED_DATASET_LOADERS['s2-chicago-north-parks.json']().then(module => module.default),
        EMBEDDED_DATASET_LOADERS['s2-chicago-north-tracts.json']().then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const capacity = parks.name.length * tracts.geoid.length;
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'tracts', label: 'tract', size: tracts.geoid.length, indexSpace: '1d'},
          {
            id: 'pair',
            label: 'pair capacity; read count for active rows',
            size: capacity,
            indexSpace: '1d'
          },
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.pair-gather',
          element: 'GPUPairGather',
          datasets: ['s2-chicago-north-parks', 's2-chicago-north-tracts'],
          keySpaces,
          relations: [
            {
              id: 'pair-parks',
              from: 'pair',
              to: 'parks',
              cardinality: 'N:1',
              kind: 'buffer',
              timeDependent: false,
              ref: {layout: 'index', resource: 'leftRows', capture: 'leftRows'}
            },
            {
              id: 'pair-tracts',
              from: 'pair',
              to: 'tracts',
              cardinality: 'N:1',
              kind: 'buffer',
              timeDependent: false,
              ref: {layout: 'index', resource: 'rightRows', capture: 'rightRows'}
            }
          ],
          choices: {how: 'inner'},
          async build(graph, uploads) {
            const parkGeometry = uploadS2JoinPolygons(uploads, frame, 'parks', parks.polygons);
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(
              ...parkGeometry.keySpaces,
              ...tractGeometry.keySpaces.filter(space => space.id !== 'tracts')
            );
            const pairs = {
              leftRows: uploads.view(
                'leftRows',
                new Uint32Array(capacity).fill(0xffffffff),
                'uint32'
              ),
              rightRows: uploads.view(
                'rightRows',
                new Uint32Array(capacity).fill(0xffffffff),
                'uint32'
              ),
              status: {
                count: uploads.view('pairCount', new Uint32Array(1), 'uint32'),
                overflow: uploads.view('pairOverflow', new Uint32Array(1), 'uint32'),
                candidateOverflow: uploads.view(
                  'pairCandidateOverflow',
                  new Uint32Array(1),
                  'uint32'
                )
              }
            };
            const parkAreas = uploads.view('parkAreas', new Float32Array(parks.areaKm2), 'float32');
            const result = await runExample(graph, {
              parks: parkGeometry.geometry,
              tracts: tractGeometry.geometry,
              pairs,
              spatialContext: frame.spatialContext,
              parkAreas
            });
            const captures: Record<string, S2CaptureSpec> = {
              ...parkGeometry.captures,
              ...tractGeometry.captures,
              leftRows: {view: pairs.leftRows, role: 'output', keySpace: 'pair'},
              rightRows: {view: pairs.rightRows, role: 'output', keySpace: 'pair'},
              pairCount: {view: pairs.status.count, role: 'output', keySpace: 'status'},
              pairOverflow: {view: pairs.status.overflow, role: 'output', keySpace: 'status'},
              pairCandidateOverflow: {
                view: pairs.status.candidateOverflow,
                role: 'output',
                keySpace: 'status'
              },
              joinedTracts: {
                view: tractGeometry.display,
                role: 'output',
                keySpace: 'tracts-display-vertex',
                geometry: {
                  kind: 'polygons',
                  featureKeySpace: 'tracts',
                  offsetsCapture: 'tractsDisplayOffsets',
                  valuesCapture: 'tractMatches',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              }
            };
            captures['parkAreas'] = {
              view: parkAreas,
              role: 'input',
              keySpace: 'parks',
              meaning: 'Full source park area in km², copied without clipping or apportionment'
            };
            captures['tractMatches'] = {
              view: result.tractMatches,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['parkRows'] = {view: result.parkRows, role: 'output', keySpace: 'pair'};
            captures['tractRows'] = {view: result.tractRows, role: 'output', keySpace: 'pair'};
            captures['joinedAreas'] = {view: result.joinedAreas, role: 'output', keySpace: 'pair'};
            captures['rowCount'] = {view: result.rowCount, role: 'output', keySpace: 'status'};
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            captures['candidateOverflow'] = {
              view: result.candidateOverflow,
              role: 'output',
              keySpace: 'status'
            };
            return captures;
          }
        });
      }
    };
  }
});
