// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeKriging: Helene log rainfall residuals at Carolina stations on two days (day 0 and
// day 1) predicted over the map at one query time. The separable covariance is exponential in space
// and in time, simple kriging around the planar trend, 16 nearest neighbors in the space-time
// metric. Left: the predicted raster (a quad per cell) and the stations of the nearer day. Right:
// how strongly a station day still speaks at that time lag, over (query time, temporal range), and
// below it the latent kriging variance (uncertainty) of the same prediction. The pointer anywhere on
// the canvas is the query time (x) and the temporal range (y). Far from both days the prediction fades to the
// trend (zero residual).

import {
  getGPUSpaceTimeKrigingParameterValues,
  GPUSpaceTimeKriging,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH,
  type GPUSpaceTimeKrigingProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuilder} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {createRainScene} from './_lib/rain';
import {
  createGridMesh,
  createPanelPad,
  getFrameGlyph,
  getPanelPoint,
  type PanelAxis
} from './_lib/surface';
import {parameters} from './space-time-kriging.parameters';

const STATION_COUNT = 300;
const NEIGHBOR_COUNT = 16;
const COLUMNS = 44;
const ROWS = 40;
const PANEL_COLUMNS = 32;
const PANEL_ROWS = 20;
const SECONDS_PER_DAY = 86400;
/** Lowest latent variance drawn: the spatial nugget share keeps it above this even at a station. */
const VARIANCE_FLOOR = 0.1;
/** Variance of the measurement error of one station day, in squared log units. */
const OBSERVATION_ERROR_VARIANCE = 0.05;
/** Neighbor metric: map units per day, the same scale for both the distance and the time lag. */
const SEARCH_ANISOTROPY = 0.5 / SECONDS_PER_DAY;
const TEMPORAL_CONTEXT: GPUSpaceTimeKrigingProps['temporalContext'] = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};
const TIME_AXIS: PanelAxis = {parameter: parameters[0]};
const RANGE_AXIS: PanelAxis = {parameter: parameters[1], logarithmic: true};
const TEMPORAL_SHARE = 0.95;

function getCovariance(values: Record<string, number>) {
  return {
    type: 'separable' as const,
    sill: 1,
    spatial: {model: 'exponential' as const, nugget: 0.1, sill: 0.9, range: values.spatialRange},
    temporal: {
      model: 'exponential' as const,
      nugget: 1 - TEMPORAL_SHARE,
      sill: TEMPORAL_SHARE,
      range: values.temporalRange * SECONDS_PER_DAY
    }
  };
}

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const rainScene = createRainScene(random, world, STATION_COUNT);
  // The right side holds the pad panel (upper) and a small variance map (lower).
  const padPanel = {...rainScene.panel, y0: 0.5};
  const varianceRectangle = {
    x0: rainScene.panel.x0,
    y0: 0.08,
    x1: rainScene.panel.x0 + 0.36 * rainScene.mapWidth,
    y1: 0.44
  };
  const scene = {...rainScene, panel: padPanel};
  // The pointer sets queryTime and temporalRange from anywhere on the canvas.
  const padRectangle = {x0: 0, y0: 0, x1: world.width, y1: 1};
  const mapRectangle = {x0: 0, y0: 0, x1: scene.mapWidth, y1: 1};
  const queries = new Float32Array(COLUMNS * ROWS * 2);
  for (let cell = 0; cell < COLUMNS * ROWS; cell++) {
    queries[cell * 2] = ((cell % COLUMNS) + 0.5) * (scene.mapWidth / COLUMNS);
    queries[cell * 2 + 1] = (Math.floor(cell / COLUMNS) + 0.5) / ROWS;
  }
  const observations = new Float32Array(STATION_COUNT * 2 * 4);
  const positions = new Float32Array(STATION_COUNT * 4);
  const times = new Float32Array(STATION_COUNT * 2);
  const magnitudes: number[] = [];
  for (let row = 0; row < STATION_COUNT; row++) {
    for (let day = 0; day < 2; day++) {
      const value = day === 0 ? scene.previousResiduals[row] : scene.residuals[row];
      const sample = day * STATION_COUNT + row;
      observations.set(
        [value, OBSERVATION_ERROR_VARIANCE, 0, Number.isFinite(value) ? 1 : 0],
        sample * 4
      );
      positions.set([scene.positions[row * 2], scene.positions[row * 2 + 1]], sample * 2);
      times[sample] = day * SECONDS_PER_DAY;
      if (Number.isFinite(value)) magnitudes.push(Math.abs(value));
    }
  }
  magnitudes.sort((first, second) => first - second);
  const panelMesh = createGridMesh(scene.panel, PANEL_COLUMNS, PANEL_ROWS);
  const varianceMesh = createGridMesh(varianceRectangle, COLUMNS, ROWS);
  // Correlation of a time lag to the nearest observation day, over (query time, temporal range).
  const correlation = new Float32Array(PANEL_COLUMNS * PANEL_ROWS);
  for (let cell = 0; cell < correlation.length; cell++) {
    const queryTime =
      TIME_AXIS.parameter.min +
      (((cell % PANEL_COLUMNS) + 0.5) / PANEL_COLUMNS) *
        (TIME_AXIS.parameter.max - TIME_AXIS.parameter.min);
    const fraction = (Math.floor(cell / PANEL_COLUMNS) + 0.5) / PANEL_ROWS;
    const {min, max} = RANGE_AXIS.parameter;
    const range = min * Math.pow(max / min, fraction);
    const lag = Math.min(Math.abs(queryTime), Math.abs(queryTime - 1));
    correlation[cell] = lag === 0 ? 1 : TEMPORAL_SHARE * Math.exp((-3 * lag) / range);
  }
  return {
    ...scene,
    mapMesh: createGridMesh(mapRectangle, COLUMNS, ROWS),
    varianceMesh,
    padRectangle,
    varianceRectangle,
    panelMesh,
    correlation,
    queries,
    observations,
    samplePositions: positions,
    times,
    amplitude: magnitudes[Math.floor(magnitudes.length * 0.7)] ?? 1
  };
}

type Inputs = ReturnType<typeof createInputs>;

function build(builder: RunBuilder, inputs: Inputs) {
  const queryCount = COLUMNS * ROWS;
  const queryTimes = builder.allocate('query-times', 'float32', queryCount);
  const parameterBuffer = builder.allocate(
    'parameters',
    'float32',
    GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
  );
  const predictions = builder.allocate('predictions', 'float32x4', queryCount);
  builder.graph.add(
    new GPUSpaceTimeKriging({
      temporalContext: TEMPORAL_CONTEXT,
      positions: builder.upload('sample-positions', 'float32x2', inputs.samplePositions).view,
      times: builder.upload('sample-times', 'float32', inputs.times).view,
      observations: builder.upload('observations', 'float32x4', inputs.observations).view,
      queries: builder.upload('queries', 'float32x2', inputs.queries).view,
      queryTimes: queryTimes.view,
      mode: 'simple',
      parameters: parameterBuffer.view,
      neighborCount: NEIGHBOR_COUNT,
      neighborSearch: 'metric',
      output: {
        values: predictions.view,
        status: builder.allocate('status', 'uint32', queryCount).view
      }
    })
  );
  return {
    outputs: {
      predictions,
      stationPositions: builder.upload('station-positions', 'float32x2', inputs.positions),
      previousValues: builder.upload('previous-residuals', 'float32', inputs.previousResiduals),
      currentValues: builder.upload('residuals', 'float32', inputs.residuals),
      mapPositions: builder.upload('map-positions', 'float32x2', inputs.mapMesh.positions),
      mapIndices: builder.upload('map-indices', 'uint32', inputs.mapMesh.indices),
      mapOwners: builder.upload('map-owners', 'uint32', inputs.mapMesh.owners),
      panelPositions: builder.upload('panel-positions', 'float32x2', inputs.panelMesh.positions),
      panelIndices: builder.upload('panel-indices', 'uint32', inputs.panelMesh.indices),
      variancePositions: builder.upload(
        'variance-positions',
        'float32x2',
        inputs.varianceMesh.positions
      ),
      varianceIndices: builder.upload('variance-indices', 'uint32', inputs.varianceMesh.indices),
      varianceOwners: builder.upload('variance-owners', 'uint32', inputs.varianceMesh.owners),
      panelOwners: builder.upload('panel-owners', 'uint32', inputs.panelMesh.owners),
      correlation: builder.upload('correlation', 'float32', inputs.correlation)
    },
    update: (values: Record<string, number>) => {
      queryTimes.write(new Float32Array(queryCount).fill(values.queryTime * SECONDS_PER_DAY));
      parameterBuffer.write(
        getGPUSpaceTimeKrigingParameterValues({
          covariance: getCovariance(values),
          searchAnisotropy: SEARCH_ANISOTROPY,
          mean: 0
        })
      );
    }
  };
}

type Outputs = ReturnType<typeof build>['outputs'];

export default defineOperationRun({
  parameters,
  caption:
    'Space-time kriging of Helene log rainfall residuals (Carolinas, two days) at a query time; right: temporal correlation over (queryTime, temporalRange).',
  createInputs,
  build,
  async afterRun({outputs, upload}) {
    // The raster layer reads one float per cell: the prediction channel of each query row.
    const bytes = await outputs.predictions.buffer.readAsync();
    const rows = new Float32Array(bytes.buffer, bytes.byteOffset, COLUMNS * ROWS * 4);
    return {
      prediction: upload(
        'prediction',
        'float32',
        Float32Array.from({length: COLUMNS * ROWS}, (_, cell) => rows[cell * 4])
      ),
      variance: upload(
        'variance',
        'float32',
        Float32Array.from({length: COLUMNS * ROWS}, (_, cell) => rows[cell * 4 + 1])
      )
    };
  },
  interaction: createPanelPad<Inputs, Outputs>({
    hint: 'query time and temporal range',
    rectangle: ({inputs}) => inputs.padRectangle,
    x: TIME_AXIS,
    y: RANGE_AXIS,
    decorate(values, {inputs}) {
      const {panel} = inputs;
      const dayLine = (day: number): RunGlyph => {
        const at = getPanelPoint(panel, TIME_AXIS, RANGE_AXIS, day, RANGE_AXIS.parameter.min);
        return {shape: 'line', from: at, to: [at[0], panel.y1], dashed: true};
      };
      const nearest = values.queryTime < 0.5 ? 0 : 1;
      return {
        glyphs: [
          getFrameGlyph(panel),
          getFrameGlyph(inputs.varianceRectangle),
          dayLine(0),
          dayLine(1),
          {
            shape: 'crosshair',
            at: getPanelPoint(panel, TIME_AXIS, RANGE_AXIS, values.queryTime, values.temporalRange)
          }
        ],
        spotlight: null,
        readout: `queryTime ${formatNumber(values.queryTime, 2)} d · range ${formatNumber(values.temporalRange, 1)} d · day ${nearest}`
      };
    }
  }),
  draw: ({inputs, outputs, values, extra}) => [
    {
      layer: 'polygons',
      context: true,
      positions: outputs.panelPositions,
      indices: outputs.panelIndices,
      owners: outputs.panelOwners,
      values: outputs.correlation,
      triangleCount: inputs.panelMesh.triangleCount,
      ramp: 'sequential',
      range: [0, 1]
    },
    ...(extra
      ? [
          {
            layer: 'polygons' as const,
            positions: outputs.mapPositions,
            indices: outputs.mapIndices,
            owners: outputs.mapOwners,
            values: extra.prediction,
            triangleCount: inputs.mapMesh.triangleCount,
            ramp: 'diverging' as const,
            range: [-inputs.amplitude, inputs.amplitude] as const
          },
          {
            layer: 'polygons' as const,
            positions: outputs.variancePositions,
            indices: outputs.varianceIndices,
            owners: outputs.varianceOwners,
            values: extra.variance,
            triangleCount: inputs.varianceMesh.triangleCount,
            ramp: 'sequential' as const,
            // Fixed, so the reader sees uncertainty grow as the query time leaves both days.
            range: [VARIANCE_FLOOR, 1] as const
          }
        ]
      : []),
    {
      layer: 'dots',
      context: true,
      positions: outputs.stationPositions,
      values: values.queryTime < 0.5 ? outputs.previousValues : outputs.currentValues,
      ramp: 'diverging',
      range: [-inputs.amplitude, inputs.amplitude],
      radius: 4.5
    }
  ]
});
