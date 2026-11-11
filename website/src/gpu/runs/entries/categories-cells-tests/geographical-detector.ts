// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeographicalDetector: how much of the variance of a tract variable a stratification explains.
// The pointer places a cross that cuts the Chicago tracts into four strata (labels 0 to 3: south-west,
// south-east, north-west, north-east). Left bars: the mean of the response in each stratum, in the stratum's map color. Right bar:
// the q statistic, 1 - SSW / SST. Strata at or above `labelCapacity` are left out and drawn muted.

import {
  getGeographicalDetectorPValue,
  GPU_GEOGRAPHICAL_DETECTOR_LAYOUT,
  GPUGeographicalDetector
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {readFloats} from './_lib/readback';
import {
  createTractMap,
  drawTractOutline,
  drawTractValues,
  MAP_SHARE,
  uploadTractMap
} from './_lib/tract-map';
import {allocateBarMesh, getBarPositions} from './_lib/grid-panel';
import {getQuantileClasses} from './_lib/tract-map';
import {parameters} from './geographical-detector.parameters';

const LAYOUT = GPU_GEOGRAPHICAL_DETECTOR_LAYOUT;
const NONE = 0xffffffff;
const STRATUM_COUNT = 5;
const MEANS_RECTANGLE = [0.64, 0.2, 0.86, 0.84] as const;
const Q_RECTANGLE = [0.9, 0.2, 0.97, 0.84] as const;

export default defineOperationRun({
  parameters,
  caption: 'Chicago tract variables stratified by a cross you place; the bar on the right is q.',
  createInputs: ({random, world}) => {
    const tracts = createTractMap(random, world);
    return {...tracts, world, cut: [(world.width * MAP_SHARE) / 2, 0.5] as [number, number]};
  },
  build(builder, tracts, values) {
    const responses = builder.allocate('responses', 'float32', tracts.count);
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    const summary = builder.allocate('summary', 'float32', LAYOUT.length);
    builder.graph.add(
      new GPUGeographicalDetector({
        values: responses.view,
        labels: labels.view,
        labelCapacity: values.labelCapacity,
        output: {summary: summary.view}
      })
    );
    const strata = builder.allocate('strata', 'uint32', tracts.count);
    const bars = allocateBarMesh(builder, 'means', STRATUM_COUNT);
    return {
      outputs: {summary, strata, bars, map: uploadTractMap(builder, tracts)},
      update: current => {
        const column = [
          tracts.povertyShare,
          tracts.unemploymentShare,
          Float32Array.from(tracts.age65, (count, tract) => count / tracts.population[tract])
        ][current.variable];
        const strataLabels =
          current.stratifier === 0
            ? Uint32Array.from(
                {length: tracts.count},
                (_, tract) =>
                  (tracts.centers[tract * 2] > tracts.cut[0] ? 1 : 0) +
                  (tracts.centers[tract * 2 + 1] > tracts.cut[1] ? 2 : 0)
              )
            : getQuantileClasses(
                current.stratifier === 1
                  ? Float32Array.from(
                      tracts.age65,
                      (count, tract) => count / tracts.population[tract]
                    )
                  : tracts.unemploymentShare,
                STRATUM_COUNT
              );
        responses.write(column);
        labels.write(strataLabels);
        strata.write(
          Uint32Array.from(strataLabels, label => (label < current.labelCapacity ? label : NONE))
        );
        const sums = new Float32Array(STRATUM_COUNT);
        const counts = new Float32Array(STRATUM_COUNT);
        strataLabels.forEach((label, tract) => {
          if (label >= current.labelCapacity) return;
          sums[label] += column[tract];
          counts[label]++;
        });
        // Bar heights: the stratum mean as a share of 0.6 (the plot height).
        bars.positions.write(
          getBarPositions(
            STRATUM_COUNT,
            sums.map((sum, label) => (counts[label] ? sum / counts[label] / 0.6 : 0)),
            MEANS_RECTANGLE,
            tracts.world
          )
        );
      }
    };
  },
  rebuildOn: ['labelCapacity'],
  buildCache: 3,
  async afterRun({outputs, upload}) {
    const summary = await readFloats(outputs.summary);
    const q = Number.isFinite(summary[LAYOUT.q]) ? summary[LAYOUT.q] : 0;
    return {q: upload('q', 'float32', Float32Array.of(q)), summary};
  },
  interaction: {
    kind: 'place',
    hint: 'place the cross',
    rest: [MAP_SHARE / 2, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const x = Math.min(pointer.x, world.width * MAP_SHARE);
      inputs.cut = [x, pointer.y];
      const summary = peek(outputs.summary);
      const p = summary ? getGeographicalDetectorPValue(summary) : Number.NaN;
      return {
        rerun: values.stratifier === 0,
        glyphs:
          values.stratifier !== 0
            ? []
            : [
                {shape: 'line', from: [x, 0], to: [x, world.height], dashed: true},
                {
                  shape: 'line',
                  from: [0, pointer.y],
                  to: [world.width * MAP_SHARE, pointer.y],
                  dashed: true
                },
                {shape: 'crosshair', at: [x, pointer.y], size: 7}
              ],
        readout: summary
          ? `${values.stratifier + values.variable === 3 && values.stratifier * values.variable === 2 ? 'circular (same variable) · ' : ''}${['cross', 'age 65+ quintiles', 'unemployment quintiles'][values.stratifier]} · q ${summary[LAYOUT.q].toFixed(2)} · F ${summary[LAYOUT.fStatistic].toFixed(1)} · p ${
              Number.isFinite(p) ? p.toPrecision(2) : '–'
            }`
          : 'q …'
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    drawTractValues(outputs.map, inputs, outputs.strata),
    drawTractOutline(outputs.map, inputs),
    {
      layer: 'polygons',
      positions: outputs.bars.positions,
      indices: outputs.bars.indices,
      owners: outputs.bars.owners,
      values: outputs.bars.classes,
      triangleCount: STRATUM_COUNT * 2
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.q,
      ramp: 'sequential',
      range: [0, 1],
      rectangle: Q_RECTANGLE
    }
  ]
});
