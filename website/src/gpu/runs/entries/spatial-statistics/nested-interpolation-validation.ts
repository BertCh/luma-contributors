// GPUNestedInterpolationValidation: for every outer fold, picks the interpolation settings (a
// kriging model or Gaussian RBF bandwidths) by an inner cross-validation on the training blocks
// only, then predicts the held-out outer block. Left: the stations colored by outer fold. Right:
// each held-out station's observed response (x) against its nested prediction (y); points on the
// diagonal were predicted well. Hovering a station picks its fold and rings that fold's points.

import {
  GPUNestedInterpolationValidation,
  GPUSpatialCrossValidation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunLayer} from '../../define-run';
import {getRainStations} from '../../../data/point-observations';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './nested-interpolation-validation.parameters';

const POINT_COUNT = 240;
const MAP_SHARE = 0.46;
const PLOT = [0.55, 0.12, 0.96, 0.9] as const;
const EPSILONS = [2, 6, 18];

export default defineOperationRun({
  parameters,
  caption: 'Held-out Carolina rainfall against nested predictions, on a log precipitation scale.',
  createInputs: ({random, world}) => {
    const mapWidth = world.width * MAP_SHARE;
    const points = getRainStations(
      random,
      {width: mapWidth, height: world.height},
      POINT_COUNT,
      true
    );
    const signal = Float32Array.from(points.rain, rain => Math.log1p(rain) / Math.log1p(343.4));
    const outerBlocks = new Uint32Array(POINT_COUNT);
    const innerBlocks = new Uint32Array(POINT_COUNT);
    for (let row = 0; row < POINT_COUNT; row++) {
      const column = Math.min(Math.floor((points.positions[row * 2] / mapWidth) * 5), 4);
      const line = Math.min(Math.floor(points.positions[row * 2 + 1] * 3), 2);
      outerBlocks[row] = line * 5 + column;
      innerBlocks[row] =
        Math.min(Math.floor((points.positions[row * 2] / mapWidth) * 10), 9) +
        10 * Math.min(Math.floor(points.positions[row * 2 + 1] * 6), 5);
    }
    return {
      positions: points.positions,
      index: createPointIndex(points.positions),
      focus: {row: -1},
      signal,
      outerBlocks,
      innerBlocks,
      noise: Float32Array.from({length: POINT_COUNT}, () => random() - 0.5),
      diagonal: Float32Array.of(
        PLOT[0] * world.width,
        PLOT[1] * world.height,
        PLOT[2] * world.width,
        PLOT[3] * world.height
      )
    };
  },
  build(builder, inputs, values) {
    const foldCount = values.foldCount;
    const positions = builder.upload('positions', 'float32x2', inputs.positions);
    const response = builder.allocate('response', 'float32', POINT_COUNT);
    const folds = builder.allocate('folds', 'uint32', POINT_COUNT);
    const training = builder.allocate('training', 'uint32', foldCount * POINT_COUNT);
    const testing = builder.allocate('testing', 'uint32', foldCount * POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', foldCount * 3);
    const foldStatus = builder.allocate('fold-status', 'uint32', foldCount);
    const candidates = [
      {mode: 'rbf', kernel: 'gaussian', epsilon: 2, smoothing: 0.1},
      {mode: 'rbf', kernel: 'gaussian', epsilon: 6, smoothing: 0.1},
      {mode: 'rbf', kernel: 'gaussian', epsilon: 18, smoothing: 0.1}
    ] as const;
    const selected = builder.allocate('selected', 'uint32', foldCount);
    const scores = builder.allocate('scores', 'float32', foldCount * candidates.length * 3);
    const predictions = builder.allocate('predictions', 'float32', foldCount * POINT_COUNT);
    const metrics = builder.allocate('metrics', 'float32', foldCount * 6);
    const status = builder.allocate('status', 'uint32', foldCount);
    // Render-only: the held-out points in the plot, and those of the hovered fold, as world positions.
    const scatter = builder.allocate('scatter', 'float32x2', foldCount * POINT_COUNT);
    const scatterFolds = builder.allocate('scatter-folds', 'uint32', foldCount * POINT_COUNT);
    const highlight = builder.allocate('highlight', 'float32x2', foldCount * POINT_COUNT);
    builder.graph.add(
      new GPUSpatialCrossValidation({
        id: 'outer-folds',
        positions: positions.view,
        blocks: builder.upload('outer-blocks', 'uint32', inputs.outerBlocks).view,
        foldCount,
        seed: 5,
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: foldStatus.view
        }
      })
    );
    builder.graph.add(
      new GPUNestedInterpolationValidation({
        positions: positions.view,
        response: response.view,
        training: training.view,
        testing: testing.view,
        foldCount,
        innerBlocks: builder.upload('inner-blocks', 'uint32', inputs.innerBlocks).view,
        innerFoldCount: 3,
        seed: 2,
        neighborCount: values.neighborCount,
        candidates,
        output: {
          selected: selected.view,
          scores: scores.view,
          predictions: predictions.view,
          metrics: metrics.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        folds,
        testing,
        predictions,
        response,
        selected,
        metrics,
        scatter,
        scatterFolds,
        highlight,
        foldCount,
        positions,
        diagonal: builder.upload('diagonal', 'float32x4', inputs.diagonal)
      },
      update: current =>
        response.write(
          Float32Array.from(
            inputs.signal,
            (value, row) => value + current.noise * inputs.noise[row]
          )
        )
    };
  },
  rebuildOn: ['foldCount', 'neighborCount'],
  interaction: {
    kind: 'probe',
    hint: 'hover a station',
    rest: [0.25, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      const folds = peek(outputs.folds);
      const testing = peek(outputs.testing);
      const predictions = peek(outputs.predictions);
      const response = peek(outputs.response);
      if (!folds || !testing || !predictions || !response) return {focus: [0, 0]};
      const hoveredFold = row >= 0 ? folds[row] : -1;
      let count = 0;
      let highlightCount = 0;
      const points = new Float32Array(outputs.scatter.length * 2);
      const foldIds = new Uint32Array(outputs.scatter.length);
      const marked = new Float32Array(outputs.scatter.length * 2);
      for (let fold = 0; fold < outputs.foldCount; fold++) {
        for (let station = 0; station < POINT_COUNT; station++) {
          if (testing[fold * POINT_COUNT + station] !== 1) continue;
          const predicted = predictions[fold * POINT_COUNT + station];
          if (!Number.isFinite(predicted)) continue;
          const x =
            (PLOT[0] + Math.min(1, Math.max(0, response[station])) * (PLOT[2] - PLOT[0])) *
            world.width;
          const y =
            (PLOT[1] + Math.min(1, Math.max(0, predicted)) * (PLOT[3] - PLOT[1])) * world.height;
          points.set([x, y], count * 2);
          foldIds[count] = fold;
          count++;
          if (fold === hoveredFold) marked.set([x, y], highlightCount++ * 2);
        }
      }
      outputs.scatter.write(points);
      outputs.scatterFolds.write(foldIds);
      outputs.highlight.write(marked);
      if (row < 0) return {focus: [count, 0]};
      const selected = peek(outputs.selected)?.[hoveredFold];
      const metrics = peek(outputs.metrics);
      const meanSquaredError = metrics?.[hoveredFold * 6 + 2];
      return {
        focus: [count, highlightCount],
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
            size: 12
          }
        ],
        readout: `outer fold ${hoveredFold + 1}/${outputs.foldCount} · epsilon ${
          selected === undefined ? '…' : EPSILONS[selected]
        } · RMSE ${meanSquaredError === undefined ? '…' : Math.sqrt(meanSquaredError).toFixed(2)}`
      };
    }
  },
  draw: ({outputs, interaction, theme}): readonly RunLayer[] => {
    const [count, highlightCount] = Array.isArray(interaction.focus) ? interaction.focus : [0, 0];
    return [
      {layer: 'lines', context: true, segments: outputs.diagonal, radius: 1},
      {layer: 'dots', positions: outputs.positions, values: outputs.folds, radius: 4},
      ...(highlightCount > 0
        ? [
            {
              layer: 'dots' as const,
              positions: outputs.highlight,
              count: highlightCount,
              color: [...theme.ink, 0.9] as const,
              radius: 7
            }
          ]
        : []),
      ...(count > 0
        ? [
            {
              layer: 'dots' as const,
              positions: outputs.scatter,
              values: outputs.scatterFolds,
              count,
              radius: 3.5
            }
          ]
        : [])
    ];
  }
});
