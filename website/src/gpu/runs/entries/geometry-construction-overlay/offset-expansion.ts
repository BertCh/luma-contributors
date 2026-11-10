// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOffsetExpansion turns run offsets (owner k owns elements [offsets[k], offsets[k + 1])) into a
// per-element owner column and a local index within the run. The top bars are the owner of every
// element (one color per owner, height = owner id); the bottom bars are the local index, a
// sawtooth that restarts with every owner.

import vesselLegs from '../../../data/sf-bay-ais.json';
import {GPUOffsetExpansion} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getChartPosition} from '../../pick';
import {parameters} from './offset-expansion.parameters';

const TOP_RECTANGLE = [0.07, 0.56, 0.95, 0.92] as const;
const BOTTOM_RECTANGLE = [0.07, 0.08, 0.95, 0.44] as const;

function getBarWorldX(
  world: {width: number},
  rectangle: readonly [number, number, number, number],
  fraction: number
): number {
  return (rectangle[0] + fraction * (rectangle[2] - rectangle[0])) * world.width;
}

export default defineOperationRun({
  parameters,
  caption:
    'Each bar is an AIS fix. The GPU reconstructs its vessel index and its position within the track.',
  createInputs: ({world}) => ({fixCounts: vesselLegs.map(leg => leg.length), world}),
  build(builder, {fixCounts, world}, values) {
    const rowCount = values.rowCount;
    const offsets = [0];
    for (let owner = 0; owner < rowCount; owner++) {
      offsets.push(offsets[owner] + Math.min(fixCounts[owner], values.maximumFixes));
    }
    const elementCount = offsets[rowCount];
    const offsetBuffer = builder.upload('offsets', 'uint32', Uint32Array.from(offsets));
    const owners = builder.allocate('owners', 'uint32', elementCount);
    const localIndex = builder.allocate('local-index', 'uint32', elementCount);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUOffsetExpansion({
        offsets: offsetBuffer.view,
        output: {
          owners: owners.view,
          localIndex: localIndex.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    // A tick at every run boundary, across both charts, shows where the offsets cut the elements.
    const ticks = new Float32Array(rowCount * 4);
    for (let owner = 1; owner <= rowCount; owner++) {
      const x = getBarWorldX(world, TOP_RECTANGLE, offsets[owner] / elementCount);
      ticks.set([x, 0.08 * world.height, x, 0.92 * world.height], (owner - 1) * 4);
    }
    return {
      outputs: {
        owners,
        localIndex,
        rowCount,
        elementCount,
        offsets,
        ticks: builder.upload('ticks', 'float32x4', ticks),
        longestRun: Math.max(...offsets.slice(1).map((offset, owner) => offset - offsets[owner]))
      }
    };
  },
  rebuildOn: ['rowCount', 'maximumFixes'],
  // The pointer picks one element (bar): its owner's whole run is boxed in both charts, and the
  // readout gives the owner and local index the GPU computed for it.
  interaction: {
    kind: 'probe',
    hint: 'hover a fix',
    resolve(pointer, {inputs, outputs, peek}) {
      const hit = getChartPosition(
        TOP_RECTANGLE,
        inputs.world,
        pointer.x,
        0.74 * inputs.world.height
      );
      const element = Math.min(outputs.elementCount - 1, Math.floor(hit.u * outputs.elementCount));
      const owner = peek(outputs.owners)?.[element];
      const localIndex = peek(outputs.localIndex)?.[element];
      if (owner === undefined || localIndex === undefined) return {};
      const runStart = outputs.offsets[owner];
      const runEnd = outputs.offsets[owner + 1];
      const [left, right] = [runStart, runEnd].map(offset =>
        getBarWorldX(inputs.world, TOP_RECTANGLE, offset / outputs.elementCount)
      );
      const x = getBarWorldX(inputs.world, TOP_RECTANGLE, (element + 0.5) / outputs.elementCount);
      const height = inputs.world.height;
      return {
        focus: element,
        glyphs: [
          {shape: 'box', minimum: [left, 0.56 * height], maximum: [right, 0.92 * height]},
          {shape: 'box', minimum: [left, 0.08 * height], maximum: [right, 0.44 * height]},
          {shape: 'line', from: [x, 0.08 * height], to: [x, 0.92 * height], dashed: true}
        ],
        readout: `fix ${element} -> vessel ${owner}, local index ${localIndex}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.owners,
      range: [-1, outputs.rowCount],
      rectangle: TOP_RECTANGLE
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.localIndex,
      range: [-1, outputs.longestRun],
      ramp: 'sequential',
      rectangle: BOTTOM_RECTANGLE
    },
    {layer: 'lines', context: true, segments: outputs.ticks, radius: 1, alpha: 0.35}
  ]
});
