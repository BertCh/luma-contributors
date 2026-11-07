// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOffsetExpansion turns run offsets (owner k owns elements [offsets[k], offsets[k + 1])) into a
// per-element owner column and a local index within the run. The top bars are the owner of every
// element (one color per owner, height = owner id); the bottom bars are the local index, a
// sawtooth that restarts with every owner.

import {GPUOffsetExpansion} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './offset-expansion.parameters';

function getHash(index: number): number {
  const value = Math.sin(index * 91.7 + 3.3) * 43758.5453;
  return value - Math.floor(value);
}

export default defineOperationRun({
  parameters,
  caption: 'Run offsets expanded into an owner column and local indices.',
  createInputs: () => ({}),
  build(builder, _inputs, values) {
    const rowCount = values.rowCount;
    const offsets = [0];
    for (let owner = 0; owner < rowCount; owner++) {
      offsets.push(offsets[owner] + 1 + Math.floor(getHash(owner) * (2 + 6 * values.spread)));
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
    return {
      outputs: {
        owners,
        localIndex,
        rowCount,
        longestRun: Math.max(...offsets.slice(1).map((offset, owner) => offset - offsets[owner]))
      }
    };
  },
  rebuildOn: ['rowCount', 'spread'],
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.owners,
      range: [-1, outputs.rowCount],
      rectangle: [0.07, 0.56, 0.95, 0.92]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.localIndex,
      range: [-1, outputs.longestRun],
      ramp: 'sequential',
      rectangle: [0.07, 0.08, 0.95, 0.44]
    }
  ]
});
