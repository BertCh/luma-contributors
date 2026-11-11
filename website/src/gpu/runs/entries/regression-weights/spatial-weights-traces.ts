// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsTraces of rook-contiguous tract weights. Top right: tr((W/s)^q) for q = 1..maximumPower
// as a line (exact, or a stochastic estimate whose noise shrinks with `probeCount`, over the exact
// line). Bottom right: the power sums of the same operation as bars. Left: the tracts a walk of q
// steps along W reaches from the middle tract; the pointer along the charts sets q.

import {GPUSpatialWeightsTraces} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights, type CsrWeights} from '../../inputs';
import {getChartWorldX} from '../../pick';
import {getChartRectangle} from '../../text';
import {parameters} from './spatial-weights-traces.parameters';

const MAP_SHARE = 0.44;
const TRACE_RECTANGLE = [0.52, 0.56, 0.96, 0.92] as const;
const SUM_RECTANGLE = [0.52, 0.08, 0.96, 0.46] as const;
const REST_POWER = 3;

/** The rows a walk of exactly `power` steps from `start` can end on (the pattern of row `start` of W^power). */
function getReachedRows(csr: CsrWeights, start: number, power: number): Float32Array {
  let frontier = new Set([start]);
  for (let step = 0; step < power; step++) {
    const next = new Set<number>();
    for (const row of frontier) {
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        next.add(csr.neighbors[slot]);
      }
    }
    frontier = next;
  }
  // NaN draws transparent: only reached tracts are filled.
  const reached = new Float32Array(csr.rowCount).fill(Number.NaN);
  for (const row of frontier) reached[row] = 1;
  return reached;
}

/** The tract nearest the middle of the map. */
function getMiddleTract(centers: Float32Array, world: {width: number; height: number}): number {
  let nearest = 0;
  let nearestDistance = Infinity;
  for (let tract = 0; tract < centers.length / 2; tract++) {
    const distance = Math.hypot(
      centers[tract * 2] - (world.width * MAP_SHARE) / 2,
      centers[tract * 2 + 1] - world.height / 2
    );
    if (distance < nearestDistance) {
      nearest = tract;
      nearestDistance = distance;
    }
  }
  return nearest;
}

export default defineOperationRun({
  parameters,
  caption:
    'Power traces of W/s (top right), power sums (bottom right) and the tracts a q-step walk reaches (left).',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, {width: world.width * MAP_SHARE, height: world.height}, 10);
    return {tracts, middle: getMiddleTract(tracts.centers, world), world};
  },
  build(builder, {tracts, middle}, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const addTraces = (id: string, mode: 'exact' | 'stochastic') => {
      const powerTraces = builder.allocate(`${id}-traces`, 'float32', values.maximumPower);
      const powerSums = builder.allocate(`${id}-sums`, 'float32', values.maximumPower);
      builder.graph.add(
        new GPUSpatialWeightsTraces({
          id: `traces-${id}`,
          weights,
          maximumPower: values.maximumPower,
          mode,
          probeCount: mode === 'exact' ? undefined : values.probeCount,
          output: {
            powerTraces: powerTraces.view,
            powerSums: powerSums.view,
            powerTraceStandardErrors: builder.allocate(
              `${id}-errors`,
              'float32',
              values.maximumPower
            ).view,
            infinityNorm: builder.allocate(`${id}-norm`, 'float32', 1).view
          }
        })
      );
      return {powerTraces, powerSums};
    };
    const main = addTraces('main', values.mode === 0 ? 'exact' : 'stochastic');
    // The exact traces drawn behind a stochastic estimate show how far the estimate is off.
    const exact = values.mode === 0 ? main : addTraces('exact', 'exact');
    const reached = builder.allocate('reached', 'float32', tracts.count);
    reached.write(
      getReachedRows(tracts.contiguity, middle, Math.min(REST_POWER, values.maximumPower))
    );
    return {
      outputs: {
        ...main,
        exactTraces: exact.powerTraces,
        reached,
        vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
        indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
        owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
        outline: builder.upload('outline', 'float32x4', tracts.outline)
      }
    };
  },
  rebuildOn: ['mode', 'maximumPower', 'probeCount'],
  // The pointer along the charts picks the power q; the map shows where a q-step walk from the middle tract can end.
  interaction: {
    kind: 'level',
    hint: 'slide along the powers',
    rest: [0.05 + 0.9 * (2.5 / 20), 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      // The whole canvas width sets q, so the map side is as live as the charts.
      const u = Math.min(1, Math.max(0, (pointer.unitX - 0.05) / 0.9));
      const power = Math.min(
        values.maximumPower,
        Math.max(1, Math.floor(u * values.maximumPower) + 1)
      );
      outputs.reached.write(getReachedRows(inputs.tracts.contiguity, inputs.middle, power));
      const traces = peek(outputs.powerTraces);
      const sums = peek(outputs.powerSums);
      const ruleX = getChartWorldX(TRACE_RECTANGLE, world, (power - 0.5) / values.maximumPower);
      return {
        glyphs: [
          {
            shape: 'line',
            from: [ruleX, SUM_RECTANGLE[1] * world.height],
            to: [ruleX, TRACE_RECTANGLE[3] * world.height]
          }
        ],
        readout: `q ${power} · trace ${traces ? traces[power - 1].toPrecision(2) : '-'} · sum ${sums ? sums[power - 1].toPrecision(2) : '-'}`
      };
    }
  },
  draw: ({inputs, outputs, world}) => {
    const traceRectangle = getChartRectangle(world, TRACE_RECTANGLE);
    const sumRectangle = getChartRectangle(world, SUM_RECTANGLE);
    const titles = {
      labels: [
        {
          text: 'traces of W/s, powers 1 to maximumPower',
          at: [traceRectangle[0], traceRectangle[3]] as const,
          anchor: 'bottom-left' as const,
          offset: [0, 3] as const,
          role: 'label' as const,
          priority: 4
        },
        {
          text: 'power sums of W/s',
          at: [sumRectangle[0], sumRectangle[3]] as const,
          anchor: 'bottom-left' as const,
          offset: [0, 3] as const,
          role: 'label' as const,
          priority: 4
        },
        {
          text: 'tracts a q-step walk reaches',
          at: [inputs.world.width * MAP_SHARE * 0.5, inputs.world.height] as const,
          anchor: 'center' as const,
          offset: [0, -14] as const,
          role: 'label' as const,
          priority: 4
        }
      ]
    };
    return [
      {
        layer: 'polygons',
        positions: outputs.vertices,
        indices: outputs.indices,
        owners: outputs.owners,
        values: outputs.reached,
        triangleCount: inputs.tracts.triangleCount,
        ramp: 'sequential',
        range: [0, 1.4]
      },
      {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
      {
        layer: 'chart',
        kind: 'line',
        context: true,
        values: outputs.exactTraces,
        rectangle: TRACE_RECTANGLE,
        radius: 1.4
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.powerTraces,
        rectangle: TRACE_RECTANGLE,
        radius: 2.2
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.powerSums,
        color: [0.55, 0.65, 0.95, 0.9],
        rectangle: SUM_RECTANGLE
      },
      {layer: 'text', ...titles}
    ];
  }
});
