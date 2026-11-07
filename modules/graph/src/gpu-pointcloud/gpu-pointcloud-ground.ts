// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {createTransientView} from '../gpu-core/graph-data-view-utils';
import {getSpatialOperationNodes} from '../utils/gpu-spatial-operation';
import {getPointCloudCellSource, validatePointCloudGrid} from './gpu-pointcloud-surfaces';
import type {PointCloudRasterGrid} from './gpu-pointcloud-surfaces';

export type GPUPointCloudGroundProps = {
  id?: string;
  positions: GraphDataView<'float32x3'>;
  classifications: GraphDataView<'uint32'>;
  grid: PointCloudRasterGrid;
  /** Minimum elevation per occupied cell, before classification. */
  minimumSurface: GraphDataView<'float32'>;
  coverage: GraphDataView<'uint32'>;
  /** Strictly increasing square-window radii in cells, each 1–32. */
  radii: readonly number[];
  initialThreshold: number;
  slope: number;
  maximumThreshold: number;
  groundMask: GraphDataView<'uint32'>;
  /** Ground becomes LAS class 2; rejected existing class 2 becomes class 1; other classes survive. */
  outputClassifications: GraphDataView<'uint32'>;
  /** Bit 1 outside/unsupported, bit 2 final morphology window has incomplete support. */
  status: GraphDataView<'uint32'>;
};

/** Bounded progressive morphological filter; a square-window PMF, not SMRF equivalence. */
export class GPUPointCloudGround {
  readonly id: string;
  constructor(readonly props: GPUPointCloudGroundProps) {
    this.id = props.id ?? 'gpu-pointcloud-ground';
    validatePointCloudGrid(props.grid);
    assert(props.radii.length > 0 && props.radii.length <= 32);
    assert(
      props.radii.every(
        (radius, index) =>
          Number.isInteger(radius) &&
          radius >= 1 &&
          radius <= 32 &&
          (!index || radius > props.radii[index - 1])
      )
    );
    assert(
      [props.initialThreshold, props.slope, props.maximumThreshold].every(
        value => value >= 0 && Number.isFinite(Math.fround(value))
      )
    );
    assert(props.maximumThreshold >= props.initialThreshold);
    assert(
      props.minimumSurface.length === props.grid.width * props.grid.height &&
        props.coverage.length === props.minimumSurface.length
    );
    for (const view of [
      props.classifications,
      props.groundMask,
      props.outputClassifications,
      props.status
    ])
      assert(view.length === props.positions.length);
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    const {
      positions,
      classifications,
      grid,
      minimumSurface,
      coverage,
      radii,
      groundMask,
      outputClassifications,
      status
    } = this.props;
    const erosion = createTransientView(
      graph,
      `${this.id}-erosion`,
      'float32',
      minimumSurface.length
    );
    const opening = createTransientView(
      graph,
      `${this.id}-opening`,
      'float32',
      minimumSurface.length
    );
    const erosionSupport = createTransientView(
      graph,
      `${this.id}-erosion-support`,
      'uint32',
      minimumSurface.length
    );
    const support = createTransientView(
      graph,
      `${this.id}-support`,
      'uint32',
      minimumSurface.length
    );
    const masks = [0, 1].map(value =>
      createTransientView(graph, `${this.id}-mask-${value}`, 'uint32', positions.length)
    );
    const nodes = getSpatialOperationNodes(graph, {
      id: `${this.id}-initialize`,
      length: positions.length,
      inputs: {},
      outputs: {mask: masks[0]},
      body: 'writeMask(index, 1u);'
    });
    for (let iteration = 0; iteration < radii.length; iteration++) {
      const radius = radii[iteration];
      for (const operation of ['erode', 'dilate'] as const) {
        nodes.push(
          ...getSpatialOperationNodes(graph, {
            id: `${this.id}-${operation}-${iteration}`,
            length: minimumSurface.length,
            inputs: {
              surface: operation === 'erode' ? minimumSurface : erosion,
              coverage,
              ...(operation === 'dilate' ? {partial: erosionSupport} : {})
            },
            outputs:
              operation === 'erode'
                ? {result: erosion, support: erosionSupport}
                : {result: opening, support},
            body: `let column = i32(index % ${grid.width}u); let row = i32(index / ${grid.width}u);
var value = ${operation === 'erode' ? '' : '-'}3.402823e38; var count = 0u; var complete = true;
for (var vertical = -${radius}; vertical <= ${radius}; vertical++) {
  for (var horizontal = -${radius}; horizontal <= ${radius}; horizontal++) {
    let otherColumn = column + horizontal; let otherRow = row + vertical;
    if (otherColumn < 0 || otherColumn >= ${grid.width} || otherRow < 0 || otherRow >= ${grid.height}) { continue; }
    let other = u32(otherRow) * ${grid.width}u + u32(otherColumn);
    if (readCoverage(other) == 0u) { continue; }
    ${operation === 'dilate' ? 'complete = complete && readPartial(other) == 0u;' : ''}
    value = ${operation === 'erode' ? 'min' : 'max'}(value, readSurface(other)); count++;
  }
}
writeResult(index, select(0.0, value, count > 0u));
writeSupport(index, select(2u, 0u, count == ${(2 * radius + 1) ** 2}u && complete));`
          })
        );
      }
      const threshold = Math.min(
        this.props.maximumThreshold,
        this.props.initialThreshold +
          this.props.slope * grid.cellSize * (radius - (radii[iteration - 1] ?? 0))
      );
      const final = iteration === radii.length - 1;
      nodes.push(
        ...getSpatialOperationNodes(graph, {
          id: `${this.id}-classify-${iteration}`,
          length: positions.length,
          inputs: {
            positions,
            opening,
            coverage,
            previous: masks[iteration % 2],
            ...(final ? {support} : {})
          },
          outputs: final ? {groundMask, status} : {next: masks[(iteration + 1) % 2]},
          body: `let position = readPositions(index);
${getPointCloudCellSource(grid)}
let valid = inside && all(abs(position) <= vec3f(3.402823e38)) && readCoverage(cellIndex) != 0u;
let ground = valid && readPrevious(index) != 0u && position.z - readOpening(cellIndex) <= ${Math.fround(threshold)};
${
  final
    ? `writeGroundMask(index, u32(ground));
writeStatus(index, select(1u, readSupport(cellIndex), valid));`
    : 'writeNext(index, u32(ground));'
}`
        })
      );
    }
    nodes.push(
      ...getSpatialOperationNodes(graph, {
        id: `${this.id}-classes`,
        length: positions.length,
        inputs: {classifications, groundMask},
        outputs: {outputClassifications},
        body: `let original = readClassifications(index);
writeOutputClassifications(index, select(select(original, 1u, original == 2u), 2u, readGroundMask(index) != 0u));`
      })
    );
    return nodes;
  }
}
