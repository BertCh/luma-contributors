// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudICP registering a rotated and shifted copy of a terrain cloud onto the original.
// Gray dots are the target, blue dots the misaligned source, orange dots the source after the
// transform the GPU solved, each blue dot joined to its orange one by a line (a subset of the
// points is drawn so the three clouds can be told apart); the curve is the RMSE per iteration
// from the history output.

import {GPUPointCloudICP} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createChartAxes, createLegend, getChartRectangle, mergeText} from '../../text';
import {addFloatCopy} from './_lib/convert';
import {createTerrainCloud} from './_lib/helpers';
import {addTransformedFootprint} from './_lib/transform-points';
import {parameters} from './pointcloud-icp.parameters';
import {getPointCloudICPSupportText} from './_lib/pointcloud-display';
import {addPointCloudDisplayCopy} from './_lib/pointcloud-display-run';

const TERRAIN_COUNT = 9000;
const POINT_COUNT = 1500;
// Dots drawn per cloud, and every this many of those get a line from before to after.
const DRAWN_COUNT = 400;
const LINK_STRIDE = 4;
const RMSE_RECTANGLE = [0.66, 0.07, 0.97, 0.3] as const;
const COLOR_TARGET = [0.62, 0.66, 0.76] as const;
const COLOR_SOURCE = [0.3, 0.7, 1] as const;
const COLOR_ALIGNED = [1, 0.6, 0.2] as const;
// The misalignment's shift is along this unit direction; the `shift` parameter is its length.
const SHIFT_DIRECTION = [0.832, -0.555] as const;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const cloud = createTerrainCloud(random, world.width, TERRAIN_COUNT);
  const target = new Float32Array(POINT_COUNT * 3);
  for (let index = 0; index < POINT_COUNT; index++) {
    target.set(cloud.positions.subarray(index * 6 * 3, index * 6 * 3 + 3), index * 3);
  }
  // The first RMSE of the history is at most this: the root mean square offset of the source.
  return {width: world.width, target, rmseTop: 0.05};
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = Record<'target' | 'initial' | 'aligned' | 'rmse' | 'status', RunBuffer>;

type Extra = {links: RunBuffer; iterations: number; status: number};

export default defineOperationRun<Inputs, Outputs, Extra>({
  parameters,
  caption:
    'The King County scan (gray), a rotated and shifted copy of it (blue) and that copy after ICP (orange); the curve is the RMSE per iteration. Closest-point matching on this nearly flat terrain approaches the scan gradually, so a large rotation may not be fully undone within the iterations.',
  createInputs,
  build(builder, inputs, values) {
    const target = builder.upload('target', 'float32x3', inputs.target);
    // The source is the target turned about the cloud center and shifted: ICP has to undo it.
    const source = builder.upload('source', 'float32x3', new Float32Array(POINT_COUNT * 3));
    const transform = builder.allocate('transform', 'float32', 16);
    const history = builder.allocate('history', 'float32x3', values.iterations);
    const status = builder.allocate('status', 'uint32', 1);
    builder.graph.add(
      new GPUPointCloudICP({
        source: source.view,
        target: target.view,
        maximumDistance: values.maximumDistance,
        iterations: values.iterations,
        output: {
          transform: transform.view,
          history: history.view,
          inlierMask: builder.allocate('inlier-mask', 'uint32', POINT_COUNT).view,
          status: status.view
        }
      })
    );
    const identity = identityTransform(builder);
    const aligned = addTransformedFootprint(builder, 'aligned-footprint', source, transform);
    return {
      outputs: {
        target: addTransformedFootprint(builder, 'target-footprint', target, identity),
        initial: addTransformedFootprint(builder, 'initial-footprint', source, identity),
        aligned: addPointCloudDisplayCopy(
          builder,
          'converged-footprint',
          aligned,
          status,
          POINT_COUNT,
          {components: 2, statusIndex: 0}
        ),
        status,
        rmse: addFloatCopy(builder, 'rmse', history, values.iterations, {componentStart: 1})
      },
      update: current => {
        const angle = (current.rotation * Math.PI) / 180;
        const centerX = inputs.width / 2;
        let sumOfSquares = 0;
        const moved = Float32Array.from({length: POINT_COUNT * 3}, (_, index) => {
          const row = Math.floor(index / 3);
          const x = inputs.target[row * 3] - centerX;
          const y = inputs.target[row * 3 + 1] - 0.5;
          if (index % 3 === 0) {
            return (
              centerX +
              x * Math.cos(angle) -
              y * Math.sin(angle) +
              current.shift * SHIFT_DIRECTION[0]
            );
          }
          if (index % 3 === 1) {
            return (
              0.5 + x * Math.sin(angle) + y * Math.cos(angle) + current.shift * SHIFT_DIRECTION[1]
            );
          }
          return inputs.target[index];
        });
        for (let row = 0; row < POINT_COUNT; row++) {
          sumOfSquares +=
            (moved[row * 3] - inputs.target[row * 3]) ** 2 +
            (moved[row * 3 + 1] - inputs.target[row * 3 + 1]) ** 2;
        }
        inputs.rmseTop = Math.max(Math.sqrt(sumOfSquares / POINT_COUNT), 0.01);
        source.write(moved);
      }
    };
  },
  rebuildOn: ['iterations', 'maximumDistance'],
  // The aligned points are read back to draw a line from every source point to where ICP put it.
  async afterRun({outputs, upload, values}) {
    const statusBytes = await outputs.status.buffer.readAsync();
    const status = new Uint32Array(
      statusBytes.buffer.slice(
        statusBytes.byteOffset,
        statusBytes.byteOffset + statusBytes.byteLength
      )
    )[0];
    const bytes = await outputs.aligned.buffer.readAsync();
    const aligned = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    const initialBytes = await outputs.initial.buffer.readAsync();
    const initial = new Float32Array(
      initialBytes.buffer.slice(
        initialBytes.byteOffset,
        initialBytes.byteOffset + initialBytes.byteLength
      )
    );
    const links: number[] = [];
    for (let row = 0; row < DRAWN_COUNT; row += LINK_STRIDE) {
      links.push(initial[row * 2], initial[row * 2 + 1], aligned[row * 2], aligned[row * 2 + 1]);
    }
    return {
      links: upload('links', 'float32x4', Float32Array.from(links)),
      iterations: values.iterations,
      status
    };
  },
  // The pointer sets the misalignment ICP has to undo: x is the rotation of the source about the
  // cloud center, y its shift. An arc at the center marks the rotation.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'rotate across, shift up',
    x: 'rotation',
    y: 'shift',
    decorate: (values, _pointer, {inputs, outputs, peek}) => {
      const centerX = inputs.width / 2;
      const arc: number[] = [];
      const sweep = (values.rotation * Math.PI) / 180;
      for (let step = 0; step <= 12; step++) {
        const angle = (step / 12) * sweep;
        arc.push(centerX + Math.cos(angle) * 0.12, 0.5 + Math.sin(angle) * 0.12);
      }
      const history = peek(outputs.rmse);
      const status = peek(outputs.status)?.[0];
      return {
        glyphs: [
          {shape: 'crosshair', at: [centerX, 0.5], size: 6},
          {shape: 'polyline', points: arc}
        ],
        readout: `${getPointCloudICPSupportText(status)}${status === 0 ? '' : ' (alignment hidden)'} · rotation ${values.rotation}° · shift ${values.shift.toFixed(3)}${
          history
            ? ` · RMSE ${history[0].toFixed(3)} -> ${history[history.length - 1].toFixed(3)}`
            : ''
        }`
      };
    }
  }),
  draw: ({inputs, outputs, extra, theme, world}) => {
    const rectangle = getChartRectangle(world, RMSE_RECTANGLE);
    const axes = createChartAxes({
      rectangle,
      x: {range: [1, Math.max(2, extra?.iterations ?? 2)], ticks: 3, title: 'iteration'},
      y: {range: [0, inputs.rmseTop], ticks: 2, title: 'RMSE'}
    });
    return [
      {
        layer: 'dots',
        context: true,
        positions: outputs.target,
        count: DRAWN_COUNT,
        color: [...COLOR_TARGET, 0.9],
        radius: 5.2
      },
      {
        layer: 'dots',
        context: true,
        positions: outputs.initial,
        count: DRAWN_COUNT,
        color: [...COLOR_SOURCE, 0.95],
        radius: 2.6
      },
      ...(extra
        ? [
            {
              layer: 'lines' as const,
              segments: extra.links,
              color: [...theme.ink, 0.6] as const,
              radius: 0.9
            }
          ]
        : []),
      {
        layer: 'dots',
        positions: outputs.aligned,
        count: DRAWN_COUNT,
        color: [...COLOR_ALIGNED, 1],
        radius: 2.6
      },
      {layer: 'text', ...axes.grid},
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.rmse,
        range: [0, inputs.rmseTop],
        rectangle: RMSE_RECTANGLE,
        radius: 2.4,
        color: [0.3, 0.75, 1, 1]
      },
      {
        layer: 'text',
        ...mergeText(
          axes,
          createLegend({
            corner: 'top-left',
            entries: [
              {label: 'target', swatch: COLOR_TARGET},
              {label: 'source before ICP', swatch: COLOR_SOURCE},
              {
                label: `source after ICP: ${getPointCloudICPSupportText(extra?.status)}${extra?.status === 0 ? '' : ' (hidden)'}`,
                swatch: COLOR_ALIGNED
              }
            ]
          })
        )
      }
    ];
  }
});

function identityTransform(builder: Parameters<typeof addTransformedFootprint>[0]) {
  return builder.upload(
    'identity',
    'float32',
    Float32Array.from({length: 16}, (_, index) => (index % 5 === 0 ? 1 : 0))
  );
}

// Bound to GPUPointCloudICP's `iterations` and `maximumDistance` (compile-time), and the rotation
// and shift that misalign the source cloud.
