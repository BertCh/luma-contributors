// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';
import {checkCellPyramidCompleteness, checkDrawnPrefix} from './cell-display-check';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-level-selection';
const ZOOMS = [18, 17, 16];
/** Index into ZOOMS: level 1 is zoom 17. */
const ACTIVE_LEVEL = 1;

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPUCellLevelSelection',
  title: 'Selecting the zoom-17 level of a place-count pyramid',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 's3-chicago-places',
    source: 'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1)',
    licence: 'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)'
  },
  load: async () => {
    const [
      {runS3Lens, geographicGeometry, geographicMetadata, createFixedOffsets},
      {runExample},
      {
        addCellBoundaries,
        addQuadbinPyramid,
        countQuadbinTiles,
        CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
        CELL_DISPLAY_TABLE_HEADROOM
      },
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-level-selection.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    const occupied = ZOOMS.map(zoom => countQuadbinTiles(data.positionsLonLat, zoom));
    const capacities = occupied.map(count => count + CELL_DISPLAY_TABLE_HEADROOM);
    const activeCapacity = capacities[ACTIVE_LEVEL] as number;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellLevelSelection',
          datasets: [data.id],
          keySpaces: [
            {id: 'cell', label: 'zoom-17 cell', size: activeCapacity, indexSpace: '1d'},
            {id: 'place', label: 'place', size: data.pointCount, indexSpace: '1d'},
            {id: 'level', label: 'pyramid level', size: ZOOMS.length, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'zoom-17 boundary vertex',
              size: activeCapacity * CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {activeLevel: ACTIVE_LEVEL},
          metadata: geographicMetadata(),
          observations: snapshots => {
            checkCellPyramidCompleteness(snapshots, ZOOMS.length);
            // Draw exactly the rows the GPU wrote.
            checkDrawnPrefix(snapshots, [
              {capture: 'selectedCount', drawn: occupied[ACTIVE_LEVEL] as number}
            ]);
            return {};
          },
          async build(graph, uploads) {
            const places = uploads.view('places', data.positionsLonLat, 'float32x2');
            // The pyramid the selection reads (setup): GPUCellPyramid of the places at zooms 18, 17, 16.
            const pyramid = addQuadbinPyramid(graph, places, ZOOMS, capacities);
            const activeLevel = uploads.view('activeLevel', Uint32Array.of(ACTIVE_LEVEL), 'uint32');
            const output = await runExample(graph, {
              levelCounts: pyramid.levelCounts,
              activeLevel
            });
            const active = pyramid.levels[ACTIVE_LEVEL];
            if (!active) throw new Error('active pyramid level missing');
            // Boundaries of the selected level from GPUCellGeometry, for display only.
            const display = addCellBoundaries(graph, active.cells, {
              family: 'quadbin',
              vertexCount: CELL_DISPLAY_QUADBIN_VERTEX_COUNT
            });
            const boundaryOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(
                occupied[ACTIVE_LEVEL] as number,
                CELL_DISPLAY_QUADBIN_VERTEX_COUNT
              ),
              'uint32'
            );
            return {
              ...Object.fromEntries(
                pyramid.levels.flatMap((table, level) => [
                  [
                    `level${level}Overflow`,
                    {
                      view: table.status.overflow,
                      role: 'intermediate' as const,
                      meaning: `level ${level} destination capacity loss`
                    }
                  ],
                  [
                    `level${level}CandidateOverflow`,
                    {
                      view: table.status.candidateOverflow!,
                      role: 'intermediate' as const,
                      meaning: `level ${level} inherited source loss`
                    }
                  ]
                ])
              ),
              places: {
                view: places,
                role: 'input',
                keySpace: 'place',
                geometry: geographicGeometry('points')
              },
              levelCounts: {
                view: pyramid.levelCounts,
                role: 'intermediate',
                keySpace: 'level',
                meaning: 'occupied cells per pyramid level, finest (zoom 18) first'
              },
              activeLevel: {
                view: activeLevel,
                role: 'intermediate',
                meaning: 'active level index (0 = zoom 18, 1 = zoom 17, 2 = zoom 16)'
              },
              selectedCount: {
                view: output.count,
                role: 'intermediate',
                meaning: 'occupied cells of the selected level'
              },
              selectedFirstRow: {
                view: output.firstRow,
                role: 'intermediate',
                meaning: 'first row of the selected level in its buffer (0: one buffer per level)'
              },
              level1Cells: {
                view: active.cells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'zoom-17 Quadbin cell key as (low, high) uint32 words'
              },
              level1Counts: {
                view: active.counts,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'places per zoom-17 cell (count)'
              },
              boundaryOffsets: {view: boundaryOffsets, role: 'intermediate'},
              boundaries: {
                view: display.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundaryOffsets',
                  valuesCapture: 'level1Counts'
                }),
                meaning:
                  'the selected zoom-17 level valued by place count; boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
