// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUShapeDescriptors over polygons of varying roundness. Fill flags the slivers (compactness
// below `sliverThreshold`); the strip below places each polygon at its Polsby-Popper compactness (1 is a circle).

import {
  getGPUShapeDescriptorsParameterValues,
  GPUShapeDescriptors
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {clamp, findParameter, snapToParameter} from '../../interaction';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {addKernel} from './shared/shape-segments';
import {parameters} from './shape-descriptors.parameters';

const OPTIONS = {irregularity: 0.12, vertexCount: 16, rows: 2, columns: 5, originY: 0.2};
const THRESHOLD = findParameter(parameters, 'sliverThreshold');
const STRIP_Y = 0.09;
/** The compactness strip spans the parameter range across 84% of the picture width. */
const getStripX = (score: number, width: number) =>
  width *
  (0.08 +
    (0.84 * (clamp(score, THRESHOLD.min, THRESHOLD.max) - THRESHOLD.min)) /
      (THRESHOLD.max - THRESHOLD.min));
const getScoreAt = (x: number, width: number) =>
  THRESHOLD.min + ((x / width - 0.08) / 0.84) * (THRESHOLD.max - THRESHOLD.min);

export default defineOperationRun({
  parameters,
  caption: 'Slivers flagged by compactness; the strip ranks every polygon against the threshold.',
  createInputs: ({seed, world}) => ({
    seed,
    world: {width: world.width, height: 0.8},
    width: world.width,
    set: createPolygonSet(
      makeRandom(seed),
      {width: world.width, height: 0.8},
      {...OPTIONS, stretch: 0.8}
    )
  }),
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const polsbyPopper = builder.allocate('polsby-popper', 'float32', inputs.set.count);
    const sliver = builder.allocate('sliver', 'uint32', inputs.set.count);
    builder.graph.add(
      new GPUShapeDescriptors({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        parameters: parameterBuffer.view,
        output: {polsbyPopper: polsbyPopper.view, sliver: sliver.view}
      })
    );
    // One tick per polygon on the compactness strip, and the strip's baseline.
    const ticks = builder.allocate('ticks', 'float32x2', inputs.set.count);
    addKernel(builder, {
      id: 'compactness-ticks',
      invocationCount: inputs.set.count,
      bindings: [
        {name: 'scores', buffer: polsbyPopper, access: 'read'},
        {name: 'ticks', buffer: ticks, access: 'read_write'}
      ],
      body: `let score = clamp(scores[scoresOffset + index], ${THRESHOLD.min}, ${THRESHOLD.max});
  ticks[ticksOffset + index * 2u] = ${inputs.width} * (0.08 + 0.84 * (score - ${THRESHOLD.min}) / ${THRESHOLD.max - THRESHOLD.min});
  ticks[ticksOffset + index * 2u + 1u] = ${STRIP_Y};`
    });
    const baseline = builder.upload(
      'baseline',
      'float32x4',
      Float32Array.of(inputs.width * 0.08, STRIP_Y, inputs.width * 0.92, STRIP_Y)
    );
    return {
      outputs: {...polygons, polsbyPopper, sliver, ticks, baseline},
      update: values => {
        parameterBuffer.write(
          getGPUShapeDescriptorsParameterValues({sliverThreshold: values.sliverThreshold})
        );
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            ...OPTIONS,
            stretch: values.stretch
          })
        );
      }
    };
  },
  // The pointer's x is the sliverThreshold on the compactness strip below the polygons.
  interaction: {
    kind: 'level',
    hint: 'slide the threshold',
    rest: ({inputs, toUnit}) => toUnit([getStripX(THRESHOLD.default, inputs.width), 0.5]),
    resolve(pointer, {inputs, outputs, peek}) {
      const sliverThreshold = snapToParameter(THRESHOLD, getScoreAt(pointer.x, inputs.width));
      const flags = peek(outputs.sliver);
      let slivers = 0;
      if (flags) for (const flag of flags) slivers += flag;
      const x = getStripX(sliverThreshold, inputs.width);
      return {
        values: {sliverThreshold},
        glyphs: [{shape: 'line', from: [x, STRIP_Y - 0.05], to: [x, STRIP_Y + 0.05]}],
        readout: `sliverThreshold ${sliverThreshold.toFixed(2)} · ${flags ? slivers : '...'} slivers`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.sliver,
      categories: [
        [0.45, 0.55, 0.72],
        [1, 0.55, 0.25]
      ],
      triangleCount: inputs.set.fillTriangleCount
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', context: true, segments: outputs.baseline, radius: 1},
    {
      layer: 'dots',
      positions: outputs.ticks,
      values: outputs.sliver,
      categories: [
        [0.45, 0.55, 0.72],
        [1, 0.55, 0.25]
      ],
      radius: 5
    }
  ]
});
