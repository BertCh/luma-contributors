// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUWardRegions merges neighboring tracts bottom-up, always the pair that raises the within-
// region variance least, until `targetRegionCount` contiguous regions remain.

import {
  getGPUWardRegionsParameterValues,
  GPU_WARD_STATUS_LAYOUT,
  GPUWardRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './ward-regions.parameters';
import {uploadTractMesh} from './support/geometry';
import {createRegionScene, drawLabels} from './support/regions';

// The merge sequence the pointer can stop at, from many regions to few; the default is the third.
const TARGETS = [12, 9, 6, 4, 2] as const;

export default defineOperationRun({
  parameters,
  caption: 'Contiguous regions by Ward agglomeration.',
  createInputs: ({world}) => createRegionScene(makeRandom(7), world, 6, 2),
  build(builder, scene, current) {
    const {tracts, attributes, columnCount} = scene;
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUWardRegions({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        values: builder.upload('attributes', 'float32', attributes).view,
        columnCount,
        standardize: current.standardize === 1,
        targetRegionCount: current.targetRegionCount,
        maximumIterations: tracts.count - current.targetRegionCount,
        mergeSchedule: current.mergeSchedule === 0 ? 'sequential' : 'mutual-nearest',
        parameters: builder.upload(
          'parameters',
          'uint32',
          getGPUWardRegionsParameterValues({seed: 1})
        ).view,
        output: {
          labels: labels.view,
          objective: builder.allocate('objective', 'float32', 1).view,
          status: builder.allocate('status', 'uint32', GPU_WARD_STATUS_LAYOUT.length + 1).view
        }
      })
    );
    return {outputs: {labels, ...uploadTractMesh(builder, tracts)}};
  },
  buildCache: 5,
  rebuildOn: ['targetRegionCount', 'mergeSchedule', 'standardize'],
  // x walks the merge sequence: the regions are rebuilt once the pointer rests on a stop.
  interaction: {
    kind: 'time',
    hint: 'walk the merges',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, world}) {
      const index = Math.min(TARGETS.length - 1, Math.floor(pointer.unitX * TARGETS.length));
      const target = TARGETS[index];
      const glyphs: RunGlyph[] = TARGETS.map((_, stop) => ({
        shape: 'dot',
        at: [((stop + 0.5) / TARGETS.length) * world.width, 0.03]
      }));
      glyphs.push({shape: 'crosshair', at: [((index + 0.5) / TARGETS.length) * world.width, 0.03]});
      return {
        settle: {targetRegionCount: target},
        glyphs,
        readout: `targetRegionCount ${target} · ${inputs.tracts.count - target} merges`
      };
    }
  },
  draw: ({inputs, outputs}) => drawLabels(inputs, outputs, outputs.labels)
});
