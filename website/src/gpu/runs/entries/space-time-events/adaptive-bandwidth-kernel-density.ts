// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAdaptiveBandwidthKernelDensity of earthquake epicenters around Japan: every event gets its
// own bandwidth from a fixed-bandwidth pilot density (Abramson's square-root rule at sensitivity
// 0.5), narrow inside the dense aftershock zones and wide among scattered events. Hovering an event
// draws its bandwidth (solid ring, the operation's pointBandwidths) against the global h (dashed).

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  getGPUAdaptiveBandwidthKernelDensityParameterValues,
  GPUAdaptiveBandwidthKernelDensity,
  GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getEarthquakes} from '../../../data/point-observations';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import {formatValue} from './_lib/events';
import {parameters} from './adaptive-bandwidth-kernel-density.parameters';

const EVENT_COUNT = 1400;
const ROWS = 60;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const events = getEarthquakes(random, world, EVENT_COUNT, 0.35);
  const columns = Math.round(ROWS * world.width);
  return {
    ...events,
    width: world.width,
    columns,
    index: createPointIndex(events.positions),
    focus: {row: -1},
    grid: makeGPURasterMetadata({
      width: columns,
      height: ROWS,
      cellSize: [world.width / columns, 1 / ROWS],
      origin: [0, 1]
    })
  };
}

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {positions: RunBuffer; density: RunBuffer; pointBandwidths: RunBuffer};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'USGS ComCat M4+ earthquakes around Japan, 2020–2024: adaptive kernel density; each event’s bandwidth narrows where events are dense. 35% of the sample is drawn from sparse cells, which adds to the contrast.',
  createInputs,
  build(builder, events, values) {
    const positions = builder.upload('positions', 'float32x2', events.positions);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH
    );
    const density = builder.allocate('density', 'float32', events.columns * ROWS);
    const pointBandwidths = builder.allocate('point-bandwidths', 'float32', events.count);
    builder.graph.add(
      new GPUAdaptiveBandwidthKernelDensity({
        positions: positions.view,
        parameters: parameterBuffer.view,
        grid: events.grid,
        indexGridSize: [40, 24],
        indexBounds: [0, 0, events.width, 1],
        bandwidthFactorLimit: values.bandwidthFactorLimit,
        output: {values: density.view, pointBandwidths: pointBandwidths.view}
      })
    );
    return {
      outputs: {positions, density, pointBandwidths},
      update: current =>
        parameterBuffer.write(
          getGPUAdaptiveBandwidthKernelDensityParameterValues(
            {bandwidth: current.bandwidth, sensitivity: current.sensitivity},
            events.grid
          )
        )
    };
  },
  rebuildOn: ['bandwidthFactorLimit'],
  buildCache: 5,
  interaction: {
    kind: 'probe',
    hint: 'hover an event',
    rest: [0.3, 0.4],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const at = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const bandwidth = peek(outputs.pointBandwidths)?.[row];
      return {
        focus: row,
        glyphs: [
          {shape: 'crosshair', at, size: 10},
          {shape: 'ring', at, radius: values.bandwidth, dashed: true},
          ...(bandwidth && Number.isFinite(bandwidth)
            ? [{shape: 'ring' as const, at, radius: bandwidth}]
            : [])
        ],
        readout: `bandwidth ${formatValue(bandwidth, 3)} = h × ${formatValue(
          bandwidth === undefined ? undefined : bandwidth / values.bandwidth
        )} · M${inputs.magnitude[row].toFixed(1)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.density,
      columns: inputs.columns,
      rows: ROWS,
      range: [6, 15],
      logarithmic: true,
      smoothing: true
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 1.6, alpha: 0.6}
  ]
});
