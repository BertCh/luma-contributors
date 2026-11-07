// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';
import {assert} from '@luma.gl/core';

const SCENARIO_ID = 'gpu-spatial-analysis.shape-descriptors';

defineLensScenario({
  id: SCENARIO_ID,
  element: 'GPUShapeDescriptors',
  title: 'Compactness of Chicago parks',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 's3-chicago-parks',
    source: 'OpenStreetMap via Overpass (showcase chicago-parks, 2026-10-07)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./shape-descriptors.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
      ]);
    const {getGPUShapeDescriptorsParameterValues, GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD} =
      await import('@luma-contributors/analysis/gpu-spatial-analysis');
    // The park dataset has exactly one ring per park, so the drawn paths are the parks.
    assert(data.ringOffsets.length - 1 === data.featureCount);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUShapeDescriptors',
          datasets: [data.id],
          keySpaces: [
            {id: 'park', label: 'park', size: data.featureCount, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {
            sliverThreshold: GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD,
            convexityMethod: 'auto'
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPUShapeDescriptorsParameterValues({
                sliverThreshold: GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD
              }),
              'float32'
            );
            const output = await runExample(graph, {parks, parameters});
            const column = (
              view: typeof output.areas,
              meaning: string
            ): {
              view: typeof output.areas;
              role: 'intermediate';
              keySpace: string;
              meaning: string;
            } => ({
              view,
              role: 'intermediate',
              keySpace: 'park',
              meaning
            });
            return {
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate'},
              areas: column(output.areas, 'park area in m²'),
              perimeters: column(output.perimeters, 'park perimeter in m'),
              polsbyPopper: column(
                output.polsbyPopper,
                'Polsby-Popper compactness 4πA/P², 1 for a circle (ratio)'
              ),
              elongation: column(
                output.elongation,
                'elongation 1 - √(λ2/λ1) from second moments, 0 for a circle (ratio)'
              ),
              orientation: column(
                output.orientation,
                'major axis angle in radians, counter-clockwise from +x'
              ),
              convexity: column(output.convexity, 'area over convex hull area (ratio)'),
              sliver: {
                view: output.sliver,
                role: 'intermediate',
                keySpace: 'park',
                meaning: '1 when Polsby-Popper is below the 0.05 threshold, else 0'
              },
              parksByCompactness: {
                view: parks.positions,
                role: 'output',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {
                  offsetsCapture: 'ringOffsets',
                  valuesCapture: 'polsbyPopper'
                }),
                meaning: 'park outlines coloured by Polsby-Popper compactness'
              }
            };
          }
        })
    };
  }
});
