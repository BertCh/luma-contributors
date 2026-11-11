// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBrownianBridgeUtilization for one San Francisco Bay vessel track (53 AIS fixes): where the
// vessel probably was between the fixes. Brighter cells are more probable; the line is the path.

import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {
  addBrownianBridge,
  createHomeRangeInputs,
  getReferenceDensity,
  type HomeRangeInputs
} from './_lib/home-range';
import {formatDistance} from './_lib/units';
import {parameters} from './trajectory-brownian-bridge-utilization.parameters';

type Outputs = {density: RunBuffer; segments: RunBuffer; positions: RunBuffer};

export default defineOperationRun({
  parameters,
  caption:
    'Utilization density of one vessel track. Bridges widen with motionVariances (how erratic the movement is) and locationErrors (how far a fix may be from the truth). Distances and speeds are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => {
    const inputs = createHomeRangeInputs(world);
    // The ramp tops out at a quarter of the default density's peak, so the long stays do not hide the
    // faint bridges between fixes.
    const reference = getReferenceDensity(inputs, parameters[0].default, parameters[1].default);
    return {
      ...inputs,
      top: reference.reduce((maximum, value) => Math.max(maximum, value), 0) * 0.25
    };
  },
  build(builder, inputs) {
    const bridge = addBrownianBridge(builder, inputs);
    const outputs: Outputs = {
      density: bridge.density,
      segments: bridge.segments,
      positions: builder.upload('fix-positions', 'float32x2', inputs.positions)
    };
    return {
      outputs,
      update: current => bridge.write(current.motionVariance, current.locationError)
    };
  },
  // A pad: x is the motion variance (log scale), y the location error.
  interaction: bindAxes<HomeRangeInputs & {top: number}, Outputs>(parameters, {
    kind: 'pad',
    hint: 'variance across, error up',
    x: {parameter: 'motionVariance', scale: 'logarithmic'},
    y: 'locationError',
    decorate: (values, _pointer, {inputs}) => ({
      readout: `motion variance ${Math.round((values.motionVariance * inputs.metersPerUnit ** 2) / 144)} m²/s · location error ${formatDistance(values.locationError, inputs.metersPerUnit)}`
    })
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.density,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, inputs.top] as const,
      smoothing: true
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.segments,
      count: inputs.segmentCount,
      radius: 1.5
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3}
  ]
});
