// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURedcapSingleLinkage: regionalization of Chicago tracts by REDCAP single linkage. It builds a
// minimum spanning forest over rook contiguity weighted by the distance between attribute rows
// (poverty, unemployment and 65+ shares), then cuts the longest edges until `regionCount` contiguous
// regions remain. Sweeping the pointer right cuts more edges. Bars: the tract count of each region.

import {
  getGPURedcapSingleLinkageParameterValues,
  GPURedcapSingleLinkage,
  GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {bindAxes} from '../../interaction';
import {readUints} from './_lib/readback';
import {
  CHART_RECTANGLE,
  createTractMap,
  drawTractOutline,
  drawTractValues,
  uploadTractMap
} from './_lib/tract-map';
import {parameters} from './redcap-single-linkage.parameters';

const COLUMN_COUNT = 3;
const BAR_CAPACITY = 30;

export default defineOperationRun({
  parameters,
  caption:
    'Tracts merged by poverty, unemployment and 65+ shares. The largest region is grey; amber links were cut. Single linkage peels outliers first. Bars: region sizes (log).',
  createInputs: ({random, world}) => {
    const tracts = createTractMap(random, world);
    const table = new Float32Array(tracts.count * COLUMN_COUNT);
    for (let tract = 0; tract < tracts.count; tract++) {
      table[tract * COLUMN_COUNT] = tracts.povertyShare[tract];
      table[tract * COLUMN_COUNT + 1] = tracts.unemploymentShare[tract];
      table[tract * COLUMN_COUNT + 2] = tracts.age65[tract] / tracts.population[tract];
    }
    return {...tracts, table, largest: 0};
  },
  build(builder, tracts, values) {
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH
    );
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    const regionCount = builder.allocate('region-count', 'uint32', 1);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPURedcapSingleLinkage({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        values: builder.upload('attributes', 'float32', tracts.table).view,
        columnCount: COLUMN_COUNT,
        standardize: values.standardize === 1,
        parameters: parameterBuffer.view,
        output: {
          labels: labels.view,
          regionCount: regionCount.view,
          status: {converged: converged.view}
        }
      })
    );
    return {
      outputs: {labels, regionCount, map: uploadTractMap(builder, tracts)},
      update: current =>
        parameterBuffer.write(
          getGPURedcapSingleLinkageParameterValues({regionCount: current.regionCount})
        )
    };
  },
  rebuildOn: ['standardize'],
  buildCache: 2,
  async afterRun({inputs, outputs, upload}) {
    const labels = await readUints(outputs.labels);
    const sizes = new Map<number, number>();
    for (const label of labels) sizes.set(label, (sizes.get(label) ?? 0) + 1);
    const ranked = [...sizes.entries()].sort((first, second) => second[1] - first[1]);
    const rank = new Map(ranked.map(([label], index) => [label, index]));
    // The largest region stays neutral (unreached color); the split-off regions get the colors.
    const display = Uint32Array.from(labels, label =>
      rank.get(label) === 0 ? 0xffffffff : (rank.get(label) as number) - 1
    );
    // Borders: contiguity links between tracts of different regions (where the tree was cut).
    const borders: number[] = [];
    const {offsets, neighbors} = inputs.contiguity;
    for (let tract = 0; tract < inputs.count; tract++) {
      for (let slot = offsets[tract]; slot < offsets[tract + 1]; slot++) {
        const other = neighbors[slot];
        if (other > tract && labels[other] !== labels[tract]) {
          borders.push(
            inputs.centers[tract * 2],
            inputs.centers[tract * 2 + 1],
            inputs.centers[other * 2],
            inputs.centers[other * 2 + 1]
          );
        }
      }
    }
    inputs.largest = ranked[0]?.[1] ?? 0;
    const bars = new Float32Array(BAR_CAPACITY);
    ranked.slice(0, BAR_CAPACITY).forEach(([, size], index) => {
      bars[index] = size;
    });
    return {
      display: upload('display', 'uint32', display),
      borders: upload(
        'borders',
        'float32x4',
        borders.length ? Float32Array.from(borders) : new Float32Array(4)
      ),
      borderCount: borders.length / 4,
      bars: upload('region-sizes', 'float32', bars)
    };
  },
  interaction: bindAxes<{largest: number}, {regionCount: {format: string; buffer: unknown}}>(
    parameters,
    {
      kind: 'level',
      hint: 'cut the tree',
      x: 'regionCount',
      decorate: (current, _pointer, state) => {
        const resulting = state.peek(state.outputs.regionCount)?.[0];
        return {
          readout: `regionCount ${current.regionCount} · ${resulting ?? '…'} regions · largest ${state.inputs.largest} tracts`
        };
      }
    }
  ),
  draw: ({inputs, outputs, extra}) => [
    drawTractValues(outputs.map, inputs, extra.display),
    drawTractOutline(outputs.map, inputs),
    {
      layer: 'lines',
      segments: extra.borders,
      count: extra.borderCount,
      color: [1, 0.72, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.bars,
      ramp: 'sequential',
      logarithmic: true,
      range: [0, 7.5],
      rectangle: CHART_RECTANGLE
    }
  ]
});
