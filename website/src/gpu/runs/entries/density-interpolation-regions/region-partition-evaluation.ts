// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURegionPartitionEvaluation scores a partition (here the Ward regions on the map). Bars show
// the within-region sum of squares of each region (indexed by label); the summary also holds the
// between and total sums of squares, the pseudo-F and the share of adjacencies that cross a border.

import {
  getGPUWardRegionsParameterValues,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPURegionPartitionEvaluation,
  GPUWardRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './region-partition-evaluation.parameters';
import {createTractGlyph, findMiddleTract, findTractAt, uploadTractMesh} from './support/geometry';
import {
  allocateRegionOutline,
  createRegionScene,
  drawLabels,
  drawRegionOutline,
  writeRegionOutline
} from './support/regions';

export default defineOperationRun({
  parameters,
  caption:
    'Within-region variance of each region of a partition (bars); hover a region to outline and score it.',
  createInputs: ({world}) => {
    const scene = createRegionScene(makeRandom(7), world, 6, 2);
    return {...scene, focus: {tract: findMiddleTract(scene.tracts, world)}};
  },
  build(builder, scene, current) {
    const {tracts, attributes, columnCount} = scene;
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const values = builder.upload('attributes', 'float32', attributes);
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUWardRegions({
        weights,
        values: values.view,
        columnCount,
        standardize: current.standardize === 1,
        targetRegionCount: current.regionCount,
        maximumIterations: tracts.count - current.regionCount,
        mergeSchedule: 'mutual-nearest',
        parameters: builder.upload(
          'ward-parameters',
          'uint32',
          getGPUWardRegionsParameterValues({seed: 1})
        ).view,
        output: {
          labels: labels.view,
          objective: builder.allocate('ward-objective', 'float32', 1).view,
          status: builder.allocate('ward-status', 'uint32', 5).view
        }
      })
    );
    const within = builder.allocate('region-within', 'float32', tracts.count);
    const summary = builder.allocate(
      'summary',
      'float32',
      GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
    );
    builder.graph.add(
      new GPURegionPartitionEvaluation({
        values: values.view,
        columnCount,
        labels: labels.view,
        weights,
        output: {
          summary: summary.view,
          regionSizes: builder.allocate('region-sizes', 'uint32', tracts.count).view,
          regionWithinSumOfSquares: within.view
        }
      })
    );
    return {
      outputs: {
        labels,
        within,
        summary,
        highlight: allocateRegionOutline(builder, scene),
        ...uploadTractMesh(builder, tracts)
      }
    };
  },
  // Only the regions that exist get a bar: the within-region sums, compacted after each run.
  async afterRun({outputs, inputs, upload}) {
    const [labelBytes, withinBytes] = await Promise.all([
      outputs.labels.buffer.readAsync(),
      outputs.within.buffer.readAsync()
    ]);
    const labels = new Uint32Array(labelBytes.buffer, labelBytes.byteOffset, inputs.tracts.count);
    const within = new Float32Array(
      withinBytes.buffer,
      withinBytes.byteOffset,
      inputs.tracts.count
    );
    const present = [...new Set(labels)]
      .filter(label => label < inputs.tracts.count)
      .sort((a, b) => a - b);
    const bars = new Float32Array(Math.max(1, present.length));
    present.forEach((label, row) => (bars[row] = within[label]));
    return {bars: upload('within-bars', 'float32', bars)};
  },
  // The hovered tract picks its region: the region is outlined on the map and scored.
  interaction: {
    kind: 'probe',
    hint: 'hover a region',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.tracts.centers[inputs.focus.tract * 2],
        inputs.tracts.centers[inputs.focus.tract * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      const hovered = findTractAt(inputs.tracts, pointer.x, pointer.y);
      if (hovered >= 0) inputs.focus.tract = hovered;
      const labels = peek(outputs.labels);
      if (!labels) return {};
      const label = labels[inputs.focus.tract];
      const tractCount = writeRegionOutline(inputs, labels, label, outputs.highlight);
      const within = peek(outputs.within)?.[label];
      const pseudoF = peek(outputs.summary)?.[GPU_REGION_PARTITION_EVALUATION_LAYOUT.pseudoF];
      return {
        focus: label,
        glyphs: [createTractGlyph(inputs.tracts, inputs.focus.tract)],
        readout: `region ${label} · ${tractCount} tracts · within ${within?.toFixed(1) ?? '...'} · pseudo-F ${pseudoF?.toFixed(1) ?? '...'}`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    ...drawLabels(inputs, outputs, outputs.labels),
    drawRegionOutline(inputs, outputs.highlight),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.bars,
      range: 'auto',
      rectangle: [0.03, 0.03, 0.3, 0.14],
      ramp: 'sequential'
    }
  ]
});
