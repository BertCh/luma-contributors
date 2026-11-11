// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointPatternEnvelope bounds 99 simulated complete-spatial-randomness curves of K(r) - pi r^2
// (grey) and tests the observed curve (gold) against them. The observed pattern leaves the
// envelope where it is clustered; `rank` widens or narrows the pointwise envelope. The shaded band
// is the GPU's lower and upper curves filled between, orange dots mark the radii where the observed
// curve leaves it, and the inset shows the observed epicenters the gold curve was measured on.

import {GPUPointPatternEnvelope} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {createClusteredPoints} from '../../inputs';
import {getChartPosition, getChartWorldX} from '../../pick';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {parameters} from './point-pattern-envelope.parameters';
import {computeKDeviation} from './support/ripley';

const RADII = Float32Array.from({length: 20}, (_, index) => 0.01 + index * 0.01);
const STEP = 0.01;
const SIMULATION_COUNT = 99;
const POINT_COUNT = 60;
// The ranks the pointer can settle on; the default is the second.
const RANKS = [1, 3, 5, 7, 10] as const;
const RECTANGLE = [0.07, 0.44, 0.95, 0.92] as const;
// The observed pattern, drawn at this fraction of its size in the lower left.
const INSET_SCALE = 0.3;
const INSET_ORIGIN = [0.07, 0.05] as const;
const BAND_SEGMENT_COUNT = 300;
const RANGE = [-0.04, 0.4] as const;
const STATISTICS = ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Japanese earthquake epicenters against 99 simulated patterns of complete spatial randomness.',
  createInputs: ({world}) => {
    // The 99 simulated curves do not depend on any slider: compute them once.
    const random = makeRandom(5);
    const simulations = new Float32Array(SIMULATION_COUNT * RADII.length);
    for (let replicate = 0; replicate < SIMULATION_COUNT; replicate++) {
      const points = Float32Array.from({length: POINT_COUNT * 2}, (_, index) =>
        index % 2 === 0 ? random() * world.width : random()
      );
      simulations.set(computeKDeviation(points, RADII, world.width), replicate * RADII.length);
    }
    return {world, simulations};
  },
  build(builder, {world, simulations}, values) {
    const random = makeRandom(6);
    const area = world.width;
    const uniform = () => [random() * world.width, random()];
    const observedPoints = createClusteredPoints(random, world, POINT_COUNT, 1, 0).positions;
    for (let point = 0; point < POINT_COUNT; point++) {
      if (random() > values.clustering) observedPoints.set(uniform(), point * 2);
    }
    const observed = builder.upload(
      'observed',
      'float32',
      computeKDeviation(observedPoints, RADII, area)
    );
    const lower = builder.allocate('lower', 'float32', RADII.length);
    const upper = builder.allocate('upper', 'float32', RADII.length);
    builder.graph.add(
      new GPUPointPatternEnvelope({
        observed: observed.view,
        simulations: builder.upload('simulations', 'float32', simulations).view,
        reference: builder.upload('reference', 'float32', new Float32Array(RADII.length)).view,
        integrationWeights: builder.upload(
          'weights',
          'float32',
          new Float32Array(RADII.length).fill(0.01)
        ).view,
        simulationCount: SIMULATION_COUNT,
        rank: values.rank,
        statistic: STATISTICS[values.statistic],
        nullModel: 'complete spatial randomness',
        fittedParameters: 'none',
        output: {
          lower: lower.view,
          upper: upper.view,
          statistics: builder.allocate('statistics', 'float32', 4).view,
          status: builder.allocate('status', 'uint32', 1).view,
          pValueInterval:
            values.statistic === 2 ? builder.allocate('p-interval', 'float32', 2).view : undefined
        }
      })
    );
    const inset = Float32Array.from(observedPoints, (value, index) =>
      index % 2 === 0
        ? INSET_ORIGIN[0] * world.width + value * INSET_SCALE
        : INSET_ORIGIN[1] + value * INSET_SCALE
    );
    return {
      outputs: {
        observed,
        lower,
        upper,
        pattern: builder.upload('pattern', 'float32x2', inset),
        band: builder.allocate('band', 'float32x4', BAND_SEGMENT_COUNT),
        outside: builder.allocate('outside', 'float32x2', RADII.length),
        zero: builder.upload('zero', 'float32', new Float32Array(RADII.length))
      }
    };
  },
  buildCache: 5,
  rebuildOn: ['rank', 'statistic', 'clustering'],
  // x reads the curves at one radius r (the rule); y chooses the rank of the envelope, which is
  // rebuilt once the pointer rests on one of five ranks.
  interaction: {
    kind: 'level',
    hint: 'read r, pick the rank',
    rest: [0.5, 1 - 1.5 / RANKS.length],
    resolve(pointer, {outputs, values, world, peek}) {
      // Sample i sits at (i + 0.5) / count across the plot, as the line chart places it.
      const index = Math.min(
        RADII.length - 1,
        Math.max(
          0,
          Math.floor(getChartPosition(RECTANGLE, world, pointer.x, pointer.y).u * RADII.length)
        )
      );
      const rank =
        RANKS[Math.min(RANKS.length - 1, Math.floor((1 - pointer.unitY) * RANKS.length))];
      const observed = peek(outputs.observed)?.[index];
      const lower = peek(outputs.lower)?.[index];
      const upper = peek(outputs.upper)?.[index];
      const verdict =
        observed === undefined || lower === undefined || upper === undefined
          ? ''
          : observed > upper
            ? ' above envelope'
            : observed < lower
              ? ' below envelope'
              : ' inside envelope';
      // Fill between the GPU's lower and upper curves, and mark the radii outside them.
      const lowerCurve = peek(outputs.lower);
      const upperCurve = peek(outputs.upper);
      const observedCurve = peek(outputs.observed);
      let bandCount = 0;
      let outsideCount = 0;
      if (lowerCurve && upperCurve && observedCurve) {
        const toY = (value: number) =>
          (RECTANGLE[1] +
            ((Math.min(RANGE[1], Math.max(RANGE[0], value)) - RANGE[0]) / (RANGE[1] - RANGE[0])) *
              (RECTANGLE[3] - RECTANGLE[1])) *
          world.height;
        const band = new Float32Array(BAND_SEGMENT_COUNT * 4);
        for (let segment = 0; segment < BAND_SEGMENT_COUNT; segment++) {
          const position = (segment / (BAND_SEGMENT_COUNT - 1)) * (RADII.length - 1);
          const left = Math.floor(position);
          const right = Math.min(RADII.length - 1, left + 1);
          const share = position - left;
          const x = getChartWorldX(RECTANGLE, world, (position + 0.5) / RADII.length);
          band.set(
            [
              x,
              toY(lowerCurve[left] * (1 - share) + lowerCurve[right] * share),
              x,
              toY(upperCurve[left] * (1 - share) + upperCurve[right] * share)
            ],
            segment * 4
          );
        }
        bandCount = BAND_SEGMENT_COUNT;
        const outside = new Float32Array(RADII.length * 2);
        for (let radius = 0; radius < RADII.length; radius++) {
          if (
            observedCurve[radius] <= upperCurve[radius] &&
            observedCurve[radius] >= lowerCurve[radius]
          ) {
            continue;
          }
          outside.set(
            [
              getChartWorldX(RECTANGLE, world, (radius + 0.5) / RADII.length),
              toY(observedCurve[radius])
            ],
            outsideCount++ * 2
          );
        }
        outputs.band.write(band);
        outputs.outside.write(outside);
      }
      return {
        settle: {rank},
        focus: [bandCount, outsideCount],
        glyphs: [
          {shape: 'curtain', x: getChartWorldX(RECTANGLE, world, (index + 0.5) / RADII.length)}
        ],
        readout: `r ${RADII[index].toFixed(2)} · observed ${observed?.toFixed(3) ?? '...'}${verdict} · rank ${rank === values.rank ? rank : `${values.rank} → ${rank}`}`
      };
    }
  },
  draw: ({outputs, interaction, theme, world}) => {
    const [bandCount, outsideCount] = Array.isArray(interaction.focus) ? interaction.focus : [0, 0];
    const axes = createChartAxes({
      rectangle: getChartRectangle(world, RECTANGLE),
      x: {
        range: [RADII[0] - STEP / 2, RADII[RADII.length - 1] + STEP / 2],
        ticks: [0.05, 0.1, 0.15, 0.2],
        title: 'r'
      },
      y: {range: RANGE, ticks: [0, 0.1, 0.2, 0.3, 0.4], title: 'K(r) − CSR'}
    });
    const legendX = 0.62 * world.width;
    return [
      {layer: 'text', context: true, ...axes.grid},
      {
        layer: 'dots',
        context: true,
        positions: outputs.pattern,
        color: [...theme.ink, 0.8],
        radius: 4.5
      },
      ...(bandCount > 0
        ? [
            {
              layer: 'lines' as const,
              segments: outputs.band,
              count: bandCount,
              radius: 2.2,
              color: [0.6, 0.75, 0.9, 0.4] as const
            }
          ]
        : []),
      {
        layer: 'chart',
        kind: 'line',
        context: true,
        values: outputs.zero,
        range: RANGE,
        rectangle: RECTANGLE,
        color: [0.5, 0.5, 0.5, 0.5]
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.lower,
        range: RANGE,
        rectangle: RECTANGLE,
        radius: 3.4,
        color: [0.6, 0.75, 0.9, 1]
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.upper,
        range: RANGE,
        rectangle: RECTANGLE,
        radius: 3.4,
        color: [0.6, 0.75, 0.9, 1]
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.observed,
        range: RANGE,
        rectangle: RECTANGLE,
        radius: 3.4,
        color: [1, 0.85, 0.3, 1]
      },
      ...(outsideCount > 0
        ? [
            {
              layer: 'dots' as const,
              positions: outputs.outside,
              count: outsideCount,
              color: [...theme.categories[1], 1] as const,
              radius: 5
            }
          ]
        : []),
      {
        layer: 'text',
        ...mergeText(axes, {
          labels: [
            {
              text: 'observed pattern',
              at: [INSET_ORIGIN[0] * world.width, INSET_ORIGIN[1] + INSET_SCALE],
              anchor: 'bottom-left',
              offset: [0, 3],
              role: 'label'
            },
            {text: 'observed K', at: [legendX, 0.26], anchor: 'left', swatch: [1, 0.85, 0.3]},
            {
              text: 'envelope of 99 random patterns',
              at: [legendX, 0.18],
              anchor: 'left',
              swatch: [0.6, 0.75, 0.9]
            },
            {text: 'observed outside it', at: [legendX, 0.1], anchor: 'left', swatch: 1}
          ]
        })
      }
    ];
  }
});
