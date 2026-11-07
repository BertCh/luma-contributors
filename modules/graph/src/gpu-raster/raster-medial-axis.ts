// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma-contributors/graph/gpu-core';
import {validateGPUStorageBindingByteLength} from '../utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../utils/wgsl-kernel-nodes';
import {GPUDistanceField} from './distance-field/gpu-distance-field';
import {getDistanceFieldBandLayout} from './distance-field/distance-field-kernels';
import type {GPURasterMedialAxisProps} from './gpu-raster-skeleton';

/**
 * Exact continuous-disk containment on integer pixel centers, without a pairwise search.
 *
 * Let background b be nearest to p. If the disk at z contains the disk at p, then
 * |z-p| + r(p) <= r(z) <= |z-b| <= |z-p| + |p-b|. All inequalities are equalities:
 * b, p, z are collinear, in that order. The first lattice point q after p on this ray lies
 * between p and z. Since distance to background is 1-Lipschitz, r(q) = |q-b| too. Conversely,
 * that equality certifies containment by q. Thus a single primitive lattice step suffices,
 * for any chosen nearest background b, even when several seeds tie.
 *
 * This is containment of continuous Euclidean disks centered on pixels, not containment of
 * their sets of lattice points, a discrete-ball skeleton, or topology-preserving thinning.
 * @internal
 */
export function getRasterMedialAxisNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPURasterMedialAxisProps
): GPUCommandNode<Parameters>[] {
  const {grid, output} = props;
  const count = grid.width * grid.height;
  const bands = getDistanceFieldBandLayout(grid);
  const largestScratchLength = Math.max(
    count,
    8,
    4 * bands.columnBandCount * grid.width,
    2 * bands.rowBandCount * grid.height
  );
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'distance-transform scratch',
    largestScratchLength * 4,
    'reduce the raster dimensions'
  );
  const seeds = createTransientView(graph, `${id}-background`, 'uint32', count);
  const nearestCells = createTransientView(graph, `${id}-nearest-background`, 'uint32', count);
  const parameters = createTransientView(graph, `${id}-distance-parameters`, 'float32', 8);
  // Preserve the existing capacity contract: only the first width * height output rows are used.
  const distances = graph.createDataView(output.radius.buffer, {...output.radius, length: count});
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-background`,
      operation: 'GPURasterMedialAxis',
      variant: 'background',
      invocationCount: count,
      bindings: [
        {name: 'input', view: props.input, type: 'u32', access: 'read'},
        {name: 'seeds', view: seeds, type: 'u32', access: 'read_write'}
      ],
      body: 'seeds[seedsOffset + index] = u32(input[inputOffset + index] == 0u);'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-distance-parameters`,
      operation: 'GPURasterMedialAxis',
      variant: 'pixel-metric',
      invocationCount: 8,
      bindings: [{name: 'settings', view: parameters, type: 'f32', access: 'read_write'}],
      body: `var value = select(0.0, 1.0, index >= 2u && index <= 5u);
  if (index == 6u) { value = bitcast<f32>(0x7f800000u | (index & 0u)); }
  settings[settingsOffset + index] = value;`
    }),
    ...new GPUDistanceField({
      id: `${id}-distance`,
      grid,
      parameters,
      seedMask: seeds,
      mode: 'exact',
      output: {distances, nearestCells}
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-maximal-disks`,
      operation: 'GPURasterMedialAxis',
      variant: 'primitive-step',
      invocationCount: count,
      bindings: [
        {name: 'nearest', view: nearestCells, type: 'u32', access: 'read'},
        {name: 'radius', view: output.radius, type: 'f32', access: 'read_write'},
        {name: 'axis', view: output.medialAxis, type: 'u32', access: 'read_write'}
      ],
      declarations: `const WIDTH: u32 = ${grid.width}u;
const HEIGHT: u32 = ${grid.height}u;
fn squaredLength(value: vec2<i32>) -> u32 {
  let absolute = vec2<u32>(abs(value));
  return absolute.x * absolute.x + absolute.y * absolute.y;
}
// Four nearest virtual exterior centers suffice; farther exterior centers cannot win.
fn getBackgroundOffset(cell: u32) -> vec2<i32> {
  let column = i32(cell % WIDTH);
  let row = i32(cell / WIDTH);
  var offset = vec2<i32>(column + 1, 0);
  if (i32(WIDTH) - column < offset.x) { offset = vec2<i32>(column - i32(WIDTH), 0); }
  if (row + 1 < abs(offset.x)) { offset = vec2<i32>(0, row + 1); }
  if (i32(HEIGHT) - row < abs(offset.x) + abs(offset.y)) {
    offset = vec2<i32>(0, row - i32(HEIGHT));
  }
  let background = nearest[nearestOffset + cell];
  if (background != 0xffffffffu) {
    let interior = vec2<i32>(column - i32(background % WIDTH), row - i32(background / WIDTH));
    if (squaredLength(interior) < squaredLength(offset)) { offset = interior; }
  }
  return offset;
}
fn greatestCommonDivisor(first: u32, second: u32) -> u32 {
  var dividend = first;
  var divisor = second;
  loop {
    if (divisor == 0u) { return dividend; }
    let remainder = dividend % divisor;
    dividend = divisor;
    divisor = remainder;
  }
}`,
      body: `let column = index % WIDTH;
  let row = index / WIDTH;
  let exteriorRadius = min(min(column + 1u, WIDTH - column), min(row + 1u, HEIGHT - row));
  radius[radiusOffset + index] = min(radius[radiusOffset + index], f32(exteriorRadius));
  let offset = getBackgroundOffset(index);
  let squaredRadius = squaredLength(offset);
  var maximal = squaredRadius > 0u;
  if (maximal) {
    let divisor = greatestCommonDivisor(u32(abs(offset.x)), u32(abs(offset.y)));
    let step = offset / i32(divisor);
    let candidate = vec2<i32>(i32(column), i32(row)) + step;
    if (all(candidate >= vec2<i32>(0)) && candidate.x < i32(WIDTH) && candidate.y < i32(HEIGHT)) {
      let candidateCell = u32(candidate.y) * WIDTH + u32(candidate.x);
      // q and its background b are within [−1, dimension], so the two squares sum to < 2^32.
      maximal = squaredLength(getBackgroundOffset(candidateCell)) != squaredLength(offset + step);
    }
  }
  axis[axisOffset + index] = u32(maximal);`
    })
  ];
}
