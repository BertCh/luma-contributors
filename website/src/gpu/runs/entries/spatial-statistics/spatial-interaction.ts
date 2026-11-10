// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialInteraction: fits a Poisson gravity model of flows between 24 origins and 24
// destinations, choosing the distance-decay coefficient from a ladder of candidates. Left: the 24
// tracts. Right: each dot is one origin-destination pair, observed flow (x) against the flow the
// fitted model predicts (y); a good fit hugs the diagonal and the constraint decides which margins
// are matched exactly. Hovering a tract draws its flows on the map and rings its pairs in the plot.

import {GPUSpatialInteraction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunLayer} from '../../define-run';
import {COMMUTE_FLOWS, getCommutePositions} from '../../../data/commute-flows';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {addLogCounts} from '../regression-weights/shared/weights';
import {whenShown} from './shared/point-field';
import {parameters} from './spatial-interaction.parameters';

const CONSTRAINTS = ['unconstrained', 'origin', 'destination', 'doubly'] as const;
const COEFFICIENTS = [-3, -2.5, -2, -1.5, -1, -0.5, 0];
const SIDE = 24;
const MAP_SHARE = 0.4;
const PLOT = [0.5, 0.1, 0.94, 0.92] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Chicago workplace commutes across selected tract pairs, observed against fitted flows on log(1 + jobs) axes.',
  createInputs: ({world}) => {
    // Retain only published links between the first 24 tract IDs in this high-volume extract.
    const links = COMMUTE_FLOWS.filter(
      ([origin, destination]) => origin < SIDE && destination < SIDE
    );
    const origin = Uint32Array.from(links, link => link[0]);
    const destination = Uint32Array.from(links, link => link[1]);
    const mapPositions = getCommutePositions({
      width: world.width * MAP_SHARE,
      height: world.height
    });
    const logDistances = Float32Array.from(links, ([from, to]) =>
      Math.log(
        Math.max(
          0.1,
          Math.hypot(
            mapPositions[from * 2] - mapPositions[to * 2],
            mapPositions[from * 2 + 1] - mapPositions[to * 2 + 1]
          ) * 10
        )
      )
    );
    const flows = Float32Array.from(links, link => link[2]);
    const tractPositions = mapPositions.slice(0, SIDE * 2);
    // The log1p axes start at the smallest flow, so the small flows are not squeezed into a corner.
    const logFlows = flows.map(flow => Math.log1p(flow));
    const axisMinimum = Math.min(...logFlows) * 0.8;
    const axisMaximum = Math.max(...logFlows) * 1.08;
    return {
      origin,
      destination,
      logDistances,
      flows,
      tractPositions,
      index: createPointIndex(tractPositions),
      focus: {row: -1},
      axis: [axisMinimum, axisMaximum] as const,
      diagonal: Float32Array.of(
        PLOT[0] * world.width,
        PLOT[1] * world.height,
        PLOT[2] * world.width,
        PLOT[3] * world.height
      )
    };
  },
  build(builder, inputs, values) {
    const flows = builder.upload('flows', 'float32', inputs.flows);
    const predictions = builder.allocate('predictions', 'float32', flows.length);
    const summary = builder.allocate('summary', 'float32', 4);
    const status = builder.allocate('status', 'uint32', 1);
    const candidates = builder.allocate('candidates', 'float32', COEFFICIENTS.length * 4);
    builder.graph.add(
      new GPUSpatialInteraction({
        origins: builder.upload('origins', 'uint32', inputs.origin).view,
        destinations: builder.upload('destinations', 'uint32', inputs.destination).view,
        originCount: SIDE,
        destinationCount: SIDE,
        flows: flows.view,
        covariate: builder.upload('covariate', 'float32', inputs.logDistances).view,
        constraint: CONSTRAINTS[values.constraint],
        coefficients: COEFFICIENTS.map(value => value * values.decayScale),
        output: {
          predictions: predictions.view,
          summary: summary.view,
          status: status.view,
          candidates: candidates.view
        }
      })
    );
    return {
      outputs: {
        flows: addLogCounts(builder, 'observed-flows', flows),
        predictions: addLogCounts(builder, 'fitted-flows', predictions),
        rawPredictions: predictions,
        // Render-only: plot positions of the hovered tract's pairs.
        pairs: builder.allocate('pairs', 'float32x2', flows.length),
        tracts: builder.upload('tracts', 'float32x2', inputs.tractPositions),
        diagonal: builder.upload('diagonal', 'float32x4', inputs.diagonal)
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.2, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      const predictions = peek(outputs.rawPredictions);
      if (row < 0 || !predictions) return {focus: 0};
      const glyphs: RunGlyph[] = [];
      const plotted = new Float32Array(inputs.flows.length * 2);
      let pairCount = 0;
      let observedTotal = 0;
      let fittedTotal = 0;
      const [axisMinimum, axisMaximum] = inputs.axis;
      const getPlot = (value: number) =>
        Math.min(1, Math.max(0, (Math.log1p(value) - axisMinimum) / (axisMaximum - axisMinimum)));
      const from = [inputs.tractPositions[row * 2], inputs.tractPositions[row * 2 + 1]] as const;
      for (let pair = 0; pair < inputs.flows.length; pair++) {
        if (inputs.origin[pair] !== row) continue;
        const destination = inputs.destination[pair];
        glyphs.push({
          shape: 'line',
          from,
          to: [inputs.tractPositions[destination * 2], inputs.tractPositions[destination * 2 + 1]]
        });
        plotted[pairCount * 2] =
          (PLOT[0] + getPlot(inputs.flows[pair]) * (PLOT[2] - PLOT[0])) * world.width;
        plotted[pairCount * 2 + 1] =
          (PLOT[1] + getPlot(predictions[pair]) * (PLOT[3] - PLOT[1])) * world.height;
        pairCount++;
        observedTotal += inputs.flows[pair];
        fittedTotal += predictions[pair];
      }
      outputs.pairs.write(plotted);
      glyphs.push({shape: 'crosshair', at: from, size: 12});
      return {
        focus: pairCount,
        glyphs,
        readout: `tract ${row} · ${pairCount} flows · observed ${Math.round(observedTotal)} · fitted ${Math.round(fittedTotal)} jobs`
      };
    }
  },
  draw: ({inputs, outputs, interaction, theme}): readonly RunLayer[] => {
    const shownCount = typeof interaction.focus === 'number' ? interaction.focus : 0;
    return [
      {layer: 'dots', positions: outputs.tracts, color: [...theme.ink, 0.6], radius: 4},
      {layer: 'lines', context: true, segments: outputs.diagonal, radius: 1},
      {
        layer: 'chart',
        kind: 'scatter',
        values: outputs.predictions,
        xValues: outputs.flows,
        xRange: inputs.axis,
        range: inputs.axis,
        color: [0.4, 0.68, 1, 0.85],
        radius: 5.5,
        rectangle: PLOT
      },
      ...whenShown(shownCount, {
        layer: 'dots',
        positions: outputs.pairs,
        count: shownCount,
        color: [...theme.categories[1], 1],
        radius: 8
      })
    ];
  }
});
