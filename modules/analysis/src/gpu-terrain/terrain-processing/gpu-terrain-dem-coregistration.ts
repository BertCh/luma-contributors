// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  checkTerrainViews,
  checkTerrainGraph,
  terrainBinding,
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

export type GPUTerrainDEMCoregistrationProps = {
  id?: string;
  grid: GPURasterMetadata;
  reference: GraphDataView<'float32'>;
  moving: GraphDataView<'float32'>;
  stableMask: GraphDataView<'uint32'>;
  /** Known variance of the raw difference per cell, in squared vertical units. */
  differenceVariance: GraphDataView<'float32'>;
  /** Exponential covariance range in planar affine units; zero means independent measurement errors. */
  correlationLength?: number;
  output: {
    /** [moving-minus-reference median, residual NMAD, stable count, estimated bias standard error]. */
    fit: GraphDataView<'float32'>;
    /** Two floats per cell: corrected moving-minus-reference difference and its standard deviation. */
    differences: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    /** [integrated change, estimated standard deviation, valid area]. */
    volume: GraphDataView<'float32'>;
  };
};

/** Robust vertical-shift model, matching xDEM's median VerticalShift objective. No horizontal registration is implied. */
export class GPUTerrainDEMCoregistration implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainDEMCoregistrationProps) {
    this.id = props.id ?? 'terrain-dem-coregistration';
    const count = validateRasterGrid(this.id, props.grid);
    for (const view of [
      props.reference,
      props.moving,
      props.stableMask,
      props.differenceVariance,
      props.output.validity
    ])
      assert(view.length === count);
    assert(
      props.output.fit.length === 4 &&
        props.output.differences.length === count * 2 &&
        props.output.volume.length === 3
    );
    assert(Number.isFinite(props.correlationLength ?? 0) && (props.correlationLength ?? 0) >= 0);
    assert(
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      ) > 0
    );
    checkTerrainViews(
      this.id,
      [props.reference, props.moving, props.stableMask, props.differenceVariance],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.reference.length;
    checkTerrainGraph(id, graph, [
      props.reference,
      props.moving,
      props.stableMask,
      props.differenceVariance,
      ...Object.values(props.output)
    ]);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', count);
    const cells = createTransientView(graph, `${id}-cells`, 'uint32', count);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', count);
    const sortedCells = createTransientView(graph, `${id}-sorted-cells`, 'uint32', count);
    const covarianceRows = createTransientView(graph, `${id}-covariance`, 'float32', count * 2);
    const standardDeviations = createTransientView(
      graph,
      `${id}-standard-deviations`,
      'float32',
      count
    );
    const inputs = [
      terrainBinding('reference', props.reference),
      terrainBinding('moving', props.moving),
      terrainBinding('stable', props.stableMask)
    ];
    const affine = props.grid.affine.map(getWGSLFloatLiteral);
    const correlation = props.correlationLength ?? 0;
    const declarations = `${TERRAIN_FINITE_WGSL}
fn residual(cell: u32) -> f32 { return moving[movingOffset + cell] - reference[referenceOffset + cell]; }
fn usable(cell: u32) -> bool { return stable[stableOffset + cell] != 0u && finite(residual(cell)); }
fn correlation(first: u32, second: u32) -> f32 {
 ${
   correlation === 0
     ? 'return select(0.0, 1.0, first == second);'
     : `let delta = vec2<f32>(f32(first % ${props.grid.width}u) - f32(second % ${props.grid.width}u), f32(first / ${props.grid.width}u) - f32(second / ${props.grid.width}u));
 return exp(-length(vec2<f32>(${affine[0]} * delta.x + ${affine[1]} * delta.y, ${affine[3]} * delta.x + ${affine[4]} * delta.y)) / ${getWGSLFloatLiteral(correlation)});`
 }
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      // Materialize square roots across a dispatch boundary: fast-math can otherwise rewrite
      // sqrt(a) * sqrt(b) as sqrt(a * b), overflowing/underflowing finite covariance inputs.
      createWGSLKernelNode(graph, {
        id: `${id}-standard-deviations`,
        operation: 'GPUTerrainDEMCoregistration',
        invocationCount: count,
        bindings: [
          terrainBinding('variance', props.differenceVariance),
          terrainBinding('deviations', standardDeviations, 'read_write')
        ],
        body: 'deviations[deviationsOffset + index] = sqrt(variance[varianceOffset + index]);'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-covariance-rows`,
        operation: 'GPUTerrainDEMCoregistration',
        invocationCount: count,
        declarations,
        bindings: [
          ...inputs,
          terrainBinding('variance', props.differenceVariance),
          terrainBinding('deviations', standardDeviations),
          terrainBinding('covariance', covarianceRows, 'read_write')
        ],
        body: `var stableCovariance = 0.0; var measurementCovariance = 0.0;
let firstVariance = variance[varianceOffset + index];
let valid = finite(residual(index)) && finite(firstVariance) && firstVariance >= 0.0;
${
  correlation === 0
    ? `stableCovariance = f32(usable(index));
measurementCovariance = select(0.0, firstVariance, valid);`
    : `for (var second = 0u; second < ${count}u; second++) {
 let secondVariance = variance[varianceOffset + second];
 let weight = correlation(index, second);
 if (usable(index) && usable(second)) { stableCovariance += weight; }
 if (valid && finite(residual(second)) && finite(secondVariance) && secondVariance >= 0.0) { measurementCovariance += deviations[deviationsOffset + index] * deviations[deviationsOffset + second] * weight; }
}`
}
covariance[covarianceOffset + index * 2u] = stableCovariance;
covariance[covarianceOffset + index * 2u + 1u] = measurementCovariance;`
      })
    ];
    for (const absoluteDeviation of [false, true]) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${absoluteDeviation ? 'deviation' : 'bias'}-rank`,
          operation: 'GPUTerrainDEMCoregistration',
          invocationCount: count,
          declarations,
          bindings: [
            ...inputs,
            ...(absoluteDeviation ? [terrainBinding('fit', props.output.fit)] : []),
            terrainBinding('keys', keys, 'read_write'),
            terrainBinding('cells', cells, 'read_write')
          ],
          body: `let value = ${absoluteDeviation ? 'abs(residual(index) - fit[fitOffset])' : 'residual(index)'};
let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
keys[keysOffset + index] = select(0xffffffffu, select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u), usable(index));
cells[cellsOffset + index] = index;`
        }),
        ...new GPUSort({
          id: `${id}-${absoluteDeviation ? 'deviation' : 'bias'}-sort`,
          keys,
          values: cells,
          outputKeys: sortedKeys,
          outputValues: sortedCells
        }).getCommandNodes(graph),
        createWGSLKernelNode(graph, {
          id: `${id}-${absoluteDeviation ? 'deviation' : 'bias'}-reduce`,
          operation: 'GPUTerrainDEMCoregistration',
          invocationCount: 1,
          declarations,
          bindings: [
            ...inputs,
            terrainBinding('order', sortedCells),
            terrainBinding('covarianceRows', covarianceRows),
            terrainBinding('fit', props.output.fit, 'read_write')
          ],
          body: `var sampleCount = 0u; var covariance = 0.0;
for (var cell = 0u; cell < ${count}u; cell++) { sampleCount += u32(usable(cell)); covariance += covarianceRows[covarianceRowsOffset + cell * 2u]; }
let samples = f32(sampleCount);
var median = missing();
if (sampleCount > 0u) {
 let lower = order[orderOffset + (sampleCount - 1u) / 2u]; let upper = order[orderOffset + sampleCount / 2u];
 median = ${absoluteDeviation ? '(abs(residual(lower) - fit[fitOffset]) + abs(residual(upper) - fit[fitOffset])) * 0.5' : '(residual(lower) + residual(upper)) * 0.5'};
}
${
  absoluteDeviation
    ? `let deviation = median * 1.4826022;
fit[fitOffset + 1u] = deviation;
fit[fitOffset + 3u] = select(missing(), 1.2533141 * deviation * sqrt(covariance) / max(samples, 1.0), samples >= 2.0);`
    : 'fit[fitOffset] = median; fit[fitOffset + 1u] = missing(); fit[fitOffset + 2u] = samples; fit[fitOffset + 3u] = missing();'
}`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-apply`,
        operation: 'GPUTerrainDEMCoregistration',
        invocationCount: count,
        declarations,
        bindings: [
          ...inputs,
          terrainBinding('variance', props.differenceVariance),
          terrainBinding('fit', props.output.fit),
          terrainBinding('differences', props.output.differences, 'read_write'),
          terrainBinding('validity', props.output.validity, 'read_write')
        ],
        body: `let difference = residual(index) - fit[fitOffset];
let cellVariance = variance[varianceOffset + index];
let biasError = fit[fitOffset + 3u];
let valid = finite(difference) && finite(cellVariance) && cellVariance >= 0.0 && finite(biasError);
validity[validityOffset + index] = u32(valid);
differences[differencesOffset + index * 2u] = select(missing(), difference, valid);
differences[differencesOffset + index * 2u + 1u] = select(missing(), sqrt(max(cellVariance, 0.0) + biasError * biasError), valid);`
      })
    );
    const area = getWGSLFloatLiteral(
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      )
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-volume`,
        operation: 'GPUTerrainDEMCoregistration',
        invocationCount: 1,
        declarations:
          declarations.slice(0, declarations.indexOf('fn residual')) +
          declarations.slice(declarations.indexOf('fn correlation')),
        bindings: [
          terrainBinding('covarianceRows', covarianceRows),
          terrainBinding('fit', props.output.fit),
          terrainBinding('differences', props.output.differences),
          terrainBinding('validity', props.output.validity),
          terrainBinding('volume', props.output.volume, 'read_write')
        ],
        body: `var change = 0.0; var samples = 0u; var covariance = 0.0;
for (var first = 0u; first < ${count}u; first++) {
 if (validity[validityOffset + first] == 0u) { continue; }
 change += differences[differencesOffset + first * 2u]; samples++;
 covariance += covarianceRows[covarianceRowsOffset + first * 2u + 1u];
}
let biasError = fit[fitOffset + 3u];
volume[volumeOffset] = select(missing(), change * ${area}, samples > 0u);
volume[volumeOffset + 1u] = select(missing(), sqrt(covariance + f32(samples) * f32(samples) * biasError * biasError) * ${area}, samples > 0u);
volume[volumeOffset + 2u] = f32(samples) * ${area};`
      })
    );
    return nodes;
  }
}
