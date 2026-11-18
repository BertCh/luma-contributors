// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EMBEDDED_DATASET_LOADERS} from '../../embedded-dataset-loaders.js';
import {lngLatToWorld, worldToLngLat} from '@math.gl/geospatial';
import {defineLensScenario} from '../../define-lens-scenarios';
import {S2_DATASETS} from './s2-datasets';

/** Width of the Web Mercator world in `@math.gl/geospatial` world units. */
const WORLD_SIZE = 512;

/** Quadbin resolution of the analysis cells, about 610 m by 450 m at this latitude. */
const RESOLUTION = 16;

type Ring = {start: number; end: number};

/** Whether `(x, y)` lies inside any ring of `vertices` (even-odd, unclosed rings). */
function isInsideRings(
  vertices: readonly number[],
  rings: readonly Ring[],
  x: number,
  y: number
): boolean {
  let inside = false;
  for (const {start, end} of rings) {
    for (let index = start, previous = end - 1; index < end; previous = index++) {
      const [ax, ay, bx, by] = [
        vertices[2 * index],
        vertices[2 * index + 1],
        vertices[2 * previous],
        vertices[2 * previous + 1]
      ];
      if (ay > y !== by > y && x < ax + ((y - ay) / (by - ay)) * (bx - ax)) inside = !inside;
    }
  }
  return inside;
}

/** Longitude and latitude of the centre of Web Mercator tile `(x, y)` at zoom `z`. */
function getTileCenter(x: number, y: number, z: number): [number, number] {
  const tiles = 2 ** z;
  // World space is 512 units wide with y up, so tile rows count down from the top edge.
  const [longitude, latitude] = worldToLngLat([
    ((x + 0.5) / tiles) * WORLD_SIZE,
    (1 - (y + 0.5) / tiles) * WORLD_SIZE
  ]);
  return [longitude, latitude];
}

/** Tile column and row containing a longitude and latitude at zoom `z`. */
function getTile(lng: number, lat: number, z: number): [number, number] {
  const tiles = 2 ** z;
  const [worldX, worldY] = lngLatToWorld([lng, lat]);
  return [Math.floor((worldX / WORLD_SIZE) * tiles), Math.floor((1 - worldY / WORLD_SIZE) * tiles)];
}

defineLensScenario({
  id: 'gpu-spatial-analysis.hot-spot-analysis-recipe',
  element: 'addHotSpotAnalysisRecipe',
  title: 'Observation hot spots on Quadbin cells, Chicago North Side',
  rungs: ['result', 'kernel', 'graph'],
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens, closeS2Rings}, {runExample}, nature, areas] = await Promise.all([
      import('./s2-lens-run'),
      import('./hot-spot-analysis-recipe.example'),
      EMBEDDED_DATASET_LOADERS['s2-chicago-north-nature.json']().then(module => module.default),
      EMBEDDED_DATASET_LOADERS['s2-chicago-north-areas.json']().then(module => module.default)
    ]);
    return {
      run: device => {
        const {vertices, featureOffsets, ringOffsets} = areas.polygons;
        // Count the tiles whose centre lies in a community area, to size the study table, and the
        // tiles in each polygon's bounding box, to size the cover's candidate list.
        const rings = ringOffsets.slice(0, -1).map((start, ring) => ({
          start,
          end: ringOffsets[ring + 1]
        }));
        let cellCount = 0;
        let candidateCount = 0;
        for (const ring of rings) {
          const lngs = Array.from(
            {length: ring.end - ring.start},
            (_, i) => vertices[2 * (ring.start + i)]
          );
          const lats = Array.from(
            {length: ring.end - ring.start},
            (_, i) => vertices[2 * (ring.start + i) + 1]
          );
          const [x0, y1] = getTile(Math.min(...lngs), Math.min(...lats), RESOLUTION);
          const [x1, y0] = getTile(Math.max(...lngs), Math.max(...lats), RESOLUTION);
          candidateCount += (x1 - x0 + 1) * (y1 - y0 + 1);
          for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
              const [lng, lat] = getTileCenter(x, y, RESOLUTION);
              if (isInsideRings(vertices, [ring], lng, lat)) cellCount++;
            }
          }
        }
        const cellCapacity = cellCount + 2;
        const closed = closeS2Rings(Float32Array.from(vertices), ringOffsets);
        const observationCount = nature.positions.length / 2;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.hot-spot-analysis-recipe',
          element: 'addHotSpotAnalysisRecipe',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-areas'],
          keySpaces: [
            {id: 'cell', label: 'study cell', size: cellCapacity, indexSpace: '1d'},
            {id: 'observation', label: 'observation', size: observationCount, indexSpace: '1d'},
            {
              id: 'area',
              label: 'community area',
              size: featureOffsets.length - 1,
              indexSpace: '1d'
            },
            {
              id: 'area-offset',
              label: 'ring offset',
              size: closed.ringOffsets.length,
              indexSpace: '1d'
            },
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {
            resolution: RESOLUTION,
            neighborRings: 1,
            minimumNeighbors: 8,
            multipleTesting: 'benjamini-hochberg',
            significanceLevel: 0.05
          },
          async build(graph, uploads) {
            const observations = uploads.view(
              'observations',
              Float32Array.from(nature.positions),
              'float32x2'
            );
            const studyArea = {
              positions: uploads.view(
                'studyAreaVertices',
                Float32Array.from(vertices),
                'float32x2'
              ),
              featureOffsets: uploads.view(
                'studyAreaFeatureOffsets',
                Uint32Array.from(featureOffsets, (_, index) => index),
                'uint32'
              ),
              polygonOffsets: uploads.view(
                'studyAreaPolygonOffsets',
                Uint32Array.from(featureOffsets),
                'uint32'
              ),
              ringOffsets: uploads.view(
                'studyAreaRingOffsets',
                Uint32Array.from(ringOffsets),
                'uint32'
              )
            };
            const display = uploads.view('studyAreaDisplay', closed.positions, 'float32x2');
            const displayOffsets = uploads.view(
              'studyAreaDisplayOffsets',
              closed.ringOffsets,
              'uint32'
            );
            const result = await runExample(graph, {
              observations,
              studyArea,
              cellCapacity,
              candidateCapacity: candidateCount + 256
            });
            uploads.own(result.parameters);
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'geographic'}
              },
              studyAreaDisplayOffsets: {
                view: displayOffsets,
                role: 'input',
                keySpace: 'area-offset'
              },
              studyArea: {
                view: display,
                role: 'context',
                keySpace: 'area',
                geometry: {
                  kind: 'polygons',
                  offsetsCapture: 'studyAreaDisplayOffsets',
                  coordinateModel: 'geographic'
                }
              },
              centers: {view: result.centers, role: 'output', keySpace: 'cell'},
              resultCells: {
                view: result.centers,
                role: 'output',
                keySpace: 'cell',
                geometry: {kind: 'points', valuesCapture: 'zScores', coordinateModel: 'geographic'}
              },
              values: {view: result.values, role: 'output', keySpace: 'cell'},
              zScores: {view: result.zScores, role: 'output', keySpace: 'cell'},
              pValues: {view: result.pValues, role: 'output', keySpace: 'cell'},
              bins: {view: result.bins, role: 'output', keySpace: 'cell'},
              neighborCounts: {view: result.neighborCounts, role: 'output', keySpace: 'cell'},
              pointsOutsideStudyArea: {
                view: result.pointsOutsideStudyArea,
                role: 'output',
                keySpace: 'status'
              }
            };
          }
        });
      }
    };
  }
});
