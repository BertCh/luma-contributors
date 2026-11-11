// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers shared by the space-time-models runs: a quad mesh for a panel of cells placed anywhere in
// the picture (the raster layer always fills the canvas), a parameter axis on that panel, a sweep
// that encodes the run's compiled graph once per grid point and copies one output after each, and
// a pad gesture whose two axes are the panel's axes.

import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import type {RunBuffer} from '../../../define-run';
import {
  clamp,
  snapToParameter,
  type Point,
  type RunGlyph,
  type RunInteraction,
  type RunInteractionResult,
  type RunInteractionState
} from '../../../interaction';
import type {GpuSceneParameter} from '../../../../types';

export type Rectangle = {x0: number; y0: number; x1: number; y1: number};

export type GridMesh = {
  columns: number;
  rows: number;
  rectangle: Rectangle;
  positions: Float32Array;
  indices: Uint32Array;
  owners: Uint32Array;
  triangleCount: number;
};

/** Quads for `columns x rows` cells over `rectangle`; cell `row * columns + column`, row 0 at the bottom. */
export function createGridMesh(rectangle: Rectangle, columns: number, rows: number): GridMesh {
  const positions = new Float32Array(columns * rows * 8);
  const indices = new Uint32Array(columns * rows * 6);
  const owners = new Uint32Array(columns * rows * 2);
  const cellWidth = (rectangle.x1 - rectangle.x0) / columns;
  const cellHeight = (rectangle.y1 - rectangle.y0) / rows;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      const x0 = rectangle.x0 + column * cellWidth;
      const y0 = rectangle.y0 + row * cellHeight;
      positions.set(
        [x0, y0, x0 + cellWidth, y0, x0 + cellWidth, y0 + cellHeight, x0, y0 + cellHeight],
        cell * 8
      );
      const base = cell * 4;
      indices.set([base, base + 1, base + 2, base, base + 2, base + 3], cell * 6);
      owners.set([cell, cell], cell * 2);
    }
  }
  return {
    columns,
    rows,
    rectangle,
    positions,
    indices,
    owners,
    triangleCount: columns * rows * 2
  };
}

/** Closed outline of a rectangle, as a glyph. */
export function getFrameGlyph(rectangle: Rectangle, dashed = false): RunGlyph {
  return {
    shape: 'box',
    minimum: [rectangle.x0, rectangle.y0],
    maximum: [rectangle.x1, rectangle.y1],
    dashed
  };
}

/** A parameter laid along one side of a panel, linear or logarithmic in value. */
export type PanelAxis = {parameter: GpuSceneParameter; logarithmic?: boolean};

/** Value at `fraction` (0..1) of the axis, unsnapped. */
export function getAxisValue(axis: PanelAxis, fraction: number): number {
  const {min, max} = axis.parameter;
  const amount = clamp(fraction, 0, 1);
  return axis.logarithmic ? min * Math.pow(max / min, amount) : min + (max - min) * amount;
}

/** Fraction (0..1) at which `value` sits on the axis. */
export function getAxisFraction(axis: PanelAxis, value: number): number {
  const {min, max} = axis.parameter;
  return clamp(
    axis.logarithmic ? Math.log(value / min) / Math.log(max / min) : (value - min) / (max - min),
    0,
    1
  );
}

/** Values at the centers of `count` cells along the axis. */
export function getAxisSamples(axis: PanelAxis, count: number): number[] {
  return Array.from({length: count}, (_, index) => getAxisValue(axis, (index + 0.5) / count));
}

/** World point of an (x, y) value pair on the panel. */
export function getPanelPoint(
  rectangle: Rectangle,
  xAxis: PanelAxis,
  yAxis: PanelAxis,
  xValue: number,
  yValue: number
): Point {
  return [
    rectangle.x0 + getAxisFraction(xAxis, xValue) * (rectangle.x1 - rectangle.x0),
    rectangle.y0 + getAxisFraction(yAxis, yValue) * (rectangle.y1 - rectangle.y0)
  ];
}

/**
 * Pad over a panel: the pointer's position inside `rectangle` (clamped to it) sets the x and y
 * parameters. `rest` is the panel point of the defaults, so the poster matches the sliders.
 */
export function createPanelPad<Inputs, Outputs>(options: {
  hint: string;
  rectangle: (state: RunInteractionState<Inputs, Outputs>) => Rectangle;
  x: PanelAxis;
  y: PanelAxis;
  decorate: (
    values: Record<string, number>,
    state: RunInteractionState<Inputs, Outputs>
  ) => RunInteractionResult;
}): RunInteraction<Inputs, Outputs> {
  const {x, y} = options;
  return {
    kind: 'pad',
    hint: options.hint,
    rest: state =>
      state.toUnit(
        getPanelPoint(options.rectangle(state), x, y, x.parameter.default, y.parameter.default)
      ),
    resolve(pointer, state) {
      const rectangle = options.rectangle(state);
      const values = {
        [x.parameter.id]: snapToParameter(
          x.parameter,
          getAxisValue(x, (pointer.x - rectangle.x0) / (rectangle.x1 - rectangle.x0))
        ),
        [y.parameter.id]: snapToParameter(
          y.parameter,
          getAxisValue(y, (pointer.y - rectangle.y0) / (rectangle.y1 - rectangle.y0))
        )
      };
      const decorated = options.decorate({...state.values, ...values}, state);
      return {...decorated, values: {...decorated.values, ...values}};
    }
  };
}

/**
 * Encodes the compiled graph once per point and copies `length` rows of `source` after each into
 * `destination` (at least `pointCount * length` rows), then reads it back. `prepare(point)` writes
 * the inputs of that point. Writes and submissions are queued in order, so each copy sees its own
 * point's result.
 */
export async function sweepCompiledGraph(options: {
  device: Device;
  compiled: CompiledGPUCommandGraph<void>;
  pointCount: number;
  prepare: (point: number) => void;
  source: RunBuffer;
  destination: RunBuffer;
  length: number;
}): Promise<Float32Array> {
  const {device, compiled, pointCount, prepare, source, destination, length} = options;
  for (let point = 0; point < pointCount; point++) {
    prepare(point);
    const encoder = device.createCommandEncoder({id: 'surface-sweep'});
    compiled.encode(encoder, {parameters: undefined});
    encoder.copyBufferToBuffer({
      sourceBuffer: source.buffer,
      destinationBuffer: destination.buffer,
      destinationOffset: point * length * 4,
      size: length * 4
    });
    device.submit(encoder.finish());
  }
  const bytes = await destination.buffer.readAsync();
  return new Float32Array(bytes.buffer, bytes.byteOffset, pointCount * length).slice();
}

/** Index and value of the largest finite entry, or -1. */
export function findMaximum(values: ArrayLike<number>): {
  index: number;
  value: number;
} {
  let index = -1;
  let value = -Infinity;
  for (let entry = 0; entry < values.length; entry++) {
    if (Number.isFinite(values[entry]) && values[entry] > value) {
      value = values[entry];
      index = entry;
    }
  }
  return {index, value};
}
