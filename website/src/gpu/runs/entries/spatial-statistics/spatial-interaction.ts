// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialInteraction: fits a Poisson gravity model of flows between 24 origins and 24
// destinations, choosing the distance-decay coefficient from a ladder of candidates. Each dot is
// one origin-destination pair: observed flow (x) against the flow the fitted model predicts (y).
// A good fit hugs the diagonal; the constraint decides which margins are matched exactly.

import {GPUSpatialInteraction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './spatial-interaction.parameters';

const CONSTRAINTS = ['unconstrained', 'origin', 'destination', 'doubly'] as const;
const COEFFICIENTS = [-3, -2.5, -2, -1.5, -1, -0.5, 0];
const SIDE = 24;
const PAIRS = SIDE * SIDE;
const MAXIMUM_FLOW = 120;
const PLOT = [0.12, 0.1, 0.88, 0.92] as const;

export default defineOperationRun({
  parameters,
  caption: 'Observed against fitted flows of a calibrated gravity model.',
  createInputs: ({random, world}) => {
    const place = () => [random() * world.width, random()];
    const origins = Array.from({length: SIDE}, place);
    const destinations = Array.from({length: SIDE}, place);
    const logDistances = new Float32Array(PAIRS);
    const origin = new Uint32Array(PAIRS);
    const destination = new Uint32Array(PAIRS);
    for (let pair = 0; pair < PAIRS; pair++) {
      origin[pair] = Math.floor(pair / SIDE);
      destination[pair] = pair % SIDE;
      const [ox, oy] = origins[origin[pair]];
      const [dx, dy] = destinations[destination[pair]];
      logDistances[pair] = Math.log(Math.max(Math.hypot(ox - dx, oy - dy), 0.4));
    }
    const sizes = Float32Array.from({length: SIDE * 2}, () => 0.4 + random());
    const gaussian = Float32Array.from({length: PAIRS}, () => random() + random() + random() - 1.5);
    return {
      origin,
      destination,
      logDistances,
      sizes,
      gaussian,
      diagonal: Float32Array.of(
        PLOT[0] * world.width,
        PLOT[1] * world.height,
        PLOT[2] * world.width,
        PLOT[3] * world.height
      )
    };
  },
  build(builder, inputs, values) {
    const flows = builder.allocate('flows', 'float32', PAIRS);
    const predictions = builder.allocate('predictions', 'float32', PAIRS);
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
        coefficients: COEFFICIENTS,
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
        flows,
        predictions,
        diagonal: builder.upload('diagonal', 'float32x4', inputs.diagonal)
      },
      update: current => {
        // Gravity flows O_i D_j d^-decay with multiplicative noise, scaled so the largest is 120.
        const raw = Float32Array.from({length: PAIRS}, (_, pair) => {
          const size =
            inputs.sizes[inputs.origin[pair]] * inputs.sizes[SIDE + inputs.destination[pair]];
          return (
            size *
            Math.exp(
              -current.distanceDecay * inputs.logDistances[pair] +
                current.noise * inputs.gaussian[pair]
            )
          );
        });
        const scale = MAXIMUM_FLOW / Math.max(...raw);
        flows.write(Float32Array.from(raw, value => Math.round(value * scale)));
      }
    };
  },
  rebuildOn: ['constraint'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.diagonal, radius: 1},
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.predictions,
      xValues: outputs.flows,
      xRange: [0, MAXIMUM_FLOW],
      range: [0, MAXIMUM_FLOW],
      color: [0.4, 0.68, 1, 0.85],
      radius: 5,
      rectangle: PLOT
    }
  ]
});
