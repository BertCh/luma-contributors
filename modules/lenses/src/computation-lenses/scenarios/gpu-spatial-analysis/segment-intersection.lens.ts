// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EMBEDDED_DATASET_LOADERS} from '../../embedded-dataset-loaders.js';
import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';

const PAIR_CAPACITY = 8192;

defineLensScenario({
  id: 'gpu-spatial-analysis.segment-intersection',
  element: 'GPUSegmentIntersection',
  title: 'Street segment intersections, Lincoln Park and Lake View',
  rungs: DEFAULT_LENS_RUNGS,
  dataset: S2_DATASETS.roads,
  presentation: {
    primaryOutputCaptureId: 'gpu-spatial-analysis-segment-intersection-snapshots-intersections-0'
  },
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, projectS2Positions}, {runExample}, roads] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./segment-intersection.example'),
        EMBEDDED_DATASET_LOADERS['s2-chicago-north-roads.json']().then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, roads.lines.vertices).positions;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.segment-intersection',
          element: 'GPUSegmentIntersection',
          datasets: ['s2-chicago-north-roads'],
          keySpaces: [
            {id: 'pair', label: 'segment pair', size: PAIR_CAPACITY, indexSpace: '1d'},
            {id: 'vertex', label: 'road vertex', size: projected.length / 2, indexSpace: '1d'},
            {
              id: 'road-offset',
              label: 'road offset',
              size: roads.lines.offsets.length,
              indexSpace: '1d'
            },
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {mode: 'self', sameFeatureOnly: false, pairCapacity: PAIR_CAPACITY},
          async build(graph, uploads) {
            const positions = uploads.view('positions', projected, 'float32x2');
            const lineOffsets = uploads.view(
              'lineOffsets',
              Uint32Array.from(roads.lines.offsets),
              'uint32'
            );
            const points = uploads.view(
              'points',
              new Float32Array(PAIR_CAPACITY * 2).fill(Number.NaN),
              'float32x2'
            );
            const result = await runExample(graph, {
              positions,
              lineOffsets,
              points,
              spatialContext: frame.spatialContext
            });
            const captures: Record<string, S2CaptureSpec> = {
              roads: {
                view: positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: {
                  kind: 'lines',
                  offsetsCapture: 'lineOffsets',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              },
              lineOffsets: {view: lineOffsets, role: 'input', keySpace: 'road-offset'},
              intersections: {
                view: points,
                role: 'output',
                keySpace: 'pair',
                geometry: {
                  kind: 'points',
                  countCapture: 'count',
                  valuesCapture: 'kinds',
                  coordinateModel: 'projected',
                  unit: 'm'
                },
                meaning:
                  'intersection point per pair; rows past the pair count stay NaN (uploaded as NaN)'
              },
              kinds: {
                view: result.kinds,
                role: 'output',
                keySpace: 'pair',
                meaning:
                  '1 proper, 2 touch, 3 collinear touch, 4 overlap, 5 uncertain; rows past the pair count are unspecified'
              },
              leftRows: {view: result.leftRows, role: 'output', keySpace: 'pair'},
              rightRows: {view: result.rightRows, role: 'output', keySpace: 'pair'}
            };
            for (const name of ['count', 'overflow', 'requiredCount', 'uncertainCount'] as const) {
              captures[name] = {view: result[name], role: 'output', keySpace: 'status'};
            }
            return captures;
          }
        });
      }
    };
  }
});
