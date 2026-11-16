// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateRasterGrid,
  validateRasterGridAxisAligned
} from '../../raster-analysis/raster-band-utils';
import {
  validateFiniteNumber,
  validateMaximumCount,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  createTerrainValidityMask,
  getTerrainGroundMeasureWGSL,
  terrainBinding,
  validateTerrainValidityView,
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

/** Maximum covariance terms summed by one invocation, independent of raster size. */
export const TERRAIN_COREGISTRATION_COVARIANCE_TILE_SIZE = 256;
/**
 * Largest correlated grid whose 4 * cells * ceil(cells / 256) byte partial plane fits a core
 * 128 MiB storage binding: 92,672 cells use 362 tiles; the next cell needs 363. This also keeps
 * partial indices below uint32. Enforced at construction even on devices with larger bindings.
 * @internal
 */
const TERRAIN_COREGISTRATION_MAXIMUM_CORRELATED_CELL_COUNT = 92672;

export type GPUTerrainDEMCoregistrationProps = {
  id?: string;
  /**
   * Grid. Without `cellSizeMode` the horizontal units are the affine's own (geographic degrees
   * are not ground meters): `correlationLength` is in those units and `output.volume` is in
   * vertical units times squared affine units. With nonzero `correlationLength`, at most
   * 92,672 cells (`width * height`) are accepted at construction.
   */
  grid: GPURasterMetadata;
  /**
   * Ground model for `correlationLength` distances and the volume. Default `'uniform'` (affine
   * units). `'web-mercator'` and `'geographic'` give ground meters (volume in vertical units
   * times square meters) and need an axis-aligned grid; their volume uncertainty scales the
   * correlated term by the mean valid cell area, an approximation of the per-cell weights.
   * Compile-time.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  reference: GraphDataView<'float32'>;
  moving: GraphDataView<'float32'>;
  /** Optional per-cell validity of both DEMs (uint32, nonzero valid); a zero word counts as NaN. */
  validity?: GraphDataView<'uint32'>;
  stableMask: GraphDataView<'uint32'>;
  /** Known variance of the raw difference per cell, in squared vertical units. */
  differenceVariance: GraphDataView<'float32'>;
  /**
   * E-folding length `L` of the exponential error correlation `exp(-d / L)`, in the horizontal
   * units above; zero means independent measurement errors. The correlation is 0.37 at `L` and
   * 0.05 at `3 L`. It is not the practical (effective) range of an exponential variogram, the
   * distance where the variogram reaches 95 % of its sill, which is `3 L`: when a tool reports that
   * range (scikit-gstat, which xDEM builds on, documents its exponential model by the effective
   * range; check the convention of yours), pass a third of it. Passing the practical range itself
   * gives correlation 0.72 instead of 0.37 at a third of the range and overstates the volume
   * uncertainty.
   */
  correlationLength?: number;
  output: {
    /** [moving-minus-reference median, residual NMAD, stable count, estimated bias standard error]. */
    fit: GraphDataView<'float32'>;
    /** Two floats per cell: corrected moving-minus-reference difference and its standard deviation. */
    differences: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    /** [integrated change, estimated standard deviation, valid area], in vertical units times the area units above. */
    volume: GraphDataView<'float32'>;
  };
};

/**
 * Robust vertical-shift model, matching xDEM's median VerticalShift objective. No horizontal registration is implied.
 * The stable count, covariance sum, volume sums and counts are `GPUReduction` sums and the median
 * comes from a `GPUSort`. With `correlationLength > 0`, every covariance row is split into
 * 256-cell tiles and joined with a fixed-order segmented reduction. The full dense model retains
 * quadratic arithmetic; no invocation walks an entire raster row of covariance terms. Scratch
 * uses two float32 planes of cellCount * ceil(cellCount / 256) entries, checked against the device
 * binding limit before graph nodes are created. Nonzero correlation is capped at 92,672 grid
 * cells at construction so both planes fit the core 128 MiB binding limit and uint32 indices.
 * Independent errors keep the linear path and are not subject to this correlated-grid cap.
 */
export class GPUTerrainDEMCoregistration implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainDEMCoregistrationProps) {
    this.id = props.id ?? 'terrain-dem-coregistration';
    const count = validateRasterGrid(this.id, props.grid);
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    validateViewLength(this.id, 'reference', props.reference, count);
    validateViewLength(this.id, 'moving', props.moving, count);
    validateViewLength(this.id, 'stableMask', props.stableMask, count);
    validateViewLength(this.id, 'differenceVariance', props.differenceVariance, count);
    validateViewLength(this.id, 'output.validity', props.output.validity, count);
    validateTerrainValidityView(this.id, props.validity, count);
    validateViewLength(this.id, 'output.fit', props.output.fit, 4);
    validateViewLength(this.id, 'output.differences', props.output.differences, count * 2);
    validateViewLength(this.id, 'output.volume', props.output.volume, 3);
    validateFiniteNumber(this.id, 'correlationLength', props.correlationLength ?? 0, 0);
    if ((props.correlationLength ?? 0) > 0) {
      validateMaximumCount(
        this.id,
        'grid cell count with correlationLength > 0',
        count,
        TERRAIN_COREGISTRATION_MAXIMUM_CORRELATED_CELL_COUNT
      );
    }
    validateRequirement(
      this.id,
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      ) > 0,
      'grid affine must have a nonzero determinant'
    );
    checkTerrainViews(
      this.id,
      [
        props.reference,
        props.moving,
        props.stableMask,
        props.differenceVariance,
        ...(props.validity ? [props.validity] : [])
      ],
      Object.values(props.output),
      [
        {
          reads: [...(!props.validity ? [props.reference, props.moving] : []), props.stableMask],
          writes: [props.output.fit]
        },
        {
          reads: [
            ...(!props.validity ? [props.reference, props.moving] : []),
            props.stableMask,
            props.differenceVariance,
            props.output.fit
          ],
          writes: [props.output.differences, props.output.validity]
        },
        {reads: [props.output.fit], writes: [props.output.volume]}
      ]
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
      ...(props.validity ? [props.validity] : []),
      ...Object.values(props.output)
    ]);
    const correlation = props.correlationLength ?? 0;
    const covarianceTileCount =
      correlation === 0 ? 1 : Math.ceil(count / TERRAIN_COREGISTRATION_COVARIANCE_TILE_SIZE);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'covariance partial plane',
      count * covarianceTileCount * 4,
      'reduce grid size or use independent errors (correlationLength = 0)'
    );
    const operation = 'GPUTerrainDEMCoregistration';
    const maskedReference = createTerrainValidityMask(graph, {
      id: `${id}-reference`,
      operation,
      values: props.reference,
      validity: props.validity
    });
    const maskedMoving = createTerrainValidityMask(graph, {
      id: `${id}-moving`,
      operation,
      values: props.moving,
      validity: props.validity
    });
    const isUniform = (props.cellSizeMode ?? 'uniform') === 'uniform';
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', count);
    const cells = createTransientView(graph, `${id}-cells`, 'uint32', count);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', count);
    const sortedCells = createTransientView(graph, `${id}-sorted-cells`, 'uint32', count);
    const usableFlags = createTransientView(graph, `${id}-usable`, 'uint32', count);
    const stableCovarianceRows = createTransientView(
      graph,
      `${id}-stable-covariance`,
      'float32',
      count
    );
    const measurementCovarianceRows = createTransientView(
      graph,
      `${id}-measurement-covariance`,
      'float32',
      count
    );
    const stableCovariancePartials =
      covarianceTileCount === 1
        ? stableCovarianceRows
        : createTransientView(
            graph,
            `${id}-stable-covariance-partials`,
            'float32',
            count * covarianceTileCount
          );
    const measurementCovariancePartials =
      covarianceTileCount === 1
        ? measurementCovarianceRows
        : createTransientView(
            graph,
            `${id}-measurement-covariance-partials`,
            'float32',
            count * covarianceTileCount
          );
    const covarianceOffsets =
      covarianceTileCount === 1
        ? undefined
        : createTransientView(graph, `${id}-covariance-offsets`, 'uint32', count + 1);
    const usableCount = createTransientView(graph, `${id}-usable-count`, 'uint32', 1);
    const stableCovarianceTotal = createTransientView(graph, `${id}-stable-total`, 'float32', 1);
    const changeValues = createTransientView(graph, `${id}-change-values`, 'float32', count);
    const areaValues = createTransientView(graph, `${id}-area-values`, 'float32', count);
    const validCount = createTransientView(graph, `${id}-valid-count`, 'uint32', 1);
    const changeTotal = createTransientView(graph, `${id}-change-total`, 'float32', 1);
    const areaTotal = createTransientView(graph, `${id}-area-total`, 'float32', 1);
    const measurementCovarianceTotal = createTransientView(
      graph,
      `${id}-measurement-total`,
      'float32',
      1
    );
    const standardDeviations = createTransientView(
      graph,
      `${id}-standard-deviations`,
      'float32',
      count
    );
    const inputs = [
      terrainBinding('reference', maskedReference.values),
      terrainBinding('moving', maskedMoving.values),
      terrainBinding('stable', props.stableMask)
    ];
    const measureDeclarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}
${TERRAIN_FINITE_WGSL}
${getTerrainGroundMeasureWGSL(props.grid, props.cellSizeMode)}`;
    const declarations = `${measureDeclarations}
fn residual(cell: u32) -> f32 { return moving[movingOffset + cell] - reference[referenceOffset + cell]; }
fn usable(cell: u32) -> bool { return stable[stableOffset + cell] != 0u && isFiniteValue(residual(cell)); }
fn correlation(first: u32, second: u32) -> f32 {
 ${
   correlation === 0
     ? 'return select(0.0, 1.0, first == second);'
     : `return exp(-getGroundEdgeLength(first, second) / ${getWGSLFloatLiteral(correlation)});`
 }
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...maskedReference.nodes,
      ...maskedMoving.nodes,
      // Materialize square roots across a dispatch boundary: fast-math can otherwise rewrite
      // sqrt(a) * sqrt(b) as sqrt(a * b), overflowing/underflowing finite covariance inputs.
      createWGSLKernelNode(graph, {
        id: `${id}-standard-deviations`,
        operation,
        invocationCount: count,
        bindings: [
          terrainBinding('variance', props.differenceVariance),
          terrainBinding('deviations', standardDeviations, 'read_write')
        ],
        body: 'deviations[deviationsOffset + index] = sqrt(variance[varianceOffset + index]);'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-usable`,
        operation,
        invocationCount: count,
        declarations,
        bindings: [...inputs, terrainBinding('usableFlags', usableFlags, 'read_write')],
        body: 'usableFlags[usableFlagsOffset + index] = u32(usable(index));'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-covariance-tiles`,
        operation,
        invocationCount: count * covarianceTileCount,
        declarations,
        bindings: [
          ...inputs,
          terrainBinding('variance', props.differenceVariance),
          terrainBinding('deviations', standardDeviations),
          terrainBinding('stableRows', stableCovariancePartials, 'read_write'),
          terrainBinding('measurementRows', measurementCovariancePartials, 'read_write')
        ],
        body: `let first = index / ${covarianceTileCount}u;
var stableCovariance = 0.0; var measurementCovariance = 0.0;
let firstVariance = variance[varianceOffset + first];
let valid = isFiniteValue(residual(first)) && isFiniteValue(firstVariance) && firstVariance >= 0.0;
${
  correlation === 0
    ? `stableCovariance = f32(usable(first));
measurementCovariance = select(0.0, firstVariance, valid);`
    : `let begin = (index % ${covarianceTileCount}u) * ${TERRAIN_COREGISTRATION_COVARIANCE_TILE_SIZE}u;
for (var local = 0u; local < ${TERRAIN_COREGISTRATION_COVARIANCE_TILE_SIZE}u; local++) {
 let second = begin + local; if (second >= ${count}u) { break; }
 let secondVariance = variance[varianceOffset + second];
 let weight = correlation(first, second);
 if (usable(first) && usable(second)) { stableCovariance += weight; }
 if (valid && isFiniteValue(residual(second)) && isFiniteValue(secondVariance) && secondVariance >= 0.0) { measurementCovariance += deviations[deviationsOffset + first] * deviations[deviationsOffset + second] * weight; }
}`
}
stableRows[stableRowsOffset + index] = stableCovariance;
measurementRows[measurementRowsOffset + index] = measurementCovariance;`
      }),
      ...(covarianceOffsets
        ? [
            createWGSLKernelNode(graph, {
              id: `${id}-covariance-offsets`,
              operation,
              invocationCount: count + 1,
              bindings: [terrainBinding('offsets', covarianceOffsets, 'read_write')],
              body: `offsets[offsetsOffset + index] = index * ${covarianceTileCount}u;`
            }),
            createSegmentSumNode(graph, {
              id: `${id}-stable-covariance-rows`,
              operation,
              segmentCount: count,
              input: stableCovariancePartials,
              segmentOffsets: covarianceOffsets,
              output: stableCovarianceRows
            }),
            createSegmentSumNode(graph, {
              id: `${id}-measurement-covariance-rows`,
              operation,
              segmentCount: count,
              input: measurementCovariancePartials,
              segmentOffsets: covarianceOffsets,
              output: measurementCovarianceRows
            })
          ]
        : []),
      ...new GPUReduction({
        id: `${id}-usable-sum`,
        input: usableFlags,
        output: usableCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${id}-stable-sum`,
        input: stableCovarianceRows,
        output: stableCovarianceTotal,
        operation: 'sum'
      }).getCommandNodes(graph)
    ];
    for (const absoluteDeviation of [false, true]) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${absoluteDeviation ? 'deviation' : 'bias'}-rank`,
          operation,
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
          id: `${id}-${absoluteDeviation ? 'deviation' : 'bias'}-median`,
          operation,
          invocationCount: 1,
          declarations,
          bindings: [
            ...inputs,
            terrainBinding('order', sortedCells),
            terrainBinding('usableCount', usableCount),
            terrainBinding('stableTotal', stableCovarianceTotal),
            terrainBinding('fit', props.output.fit, 'read_write')
          ],
          body: `let sampleCount = usableCount[usableCountOffset]; let covariance = stableTotal[stableTotalOffset];
let samples = f32(sampleCount);
var median = getNaN();
if (sampleCount > 0u) {
 let lower = order[orderOffset + (sampleCount - 1u) / 2u]; let upper = order[orderOffset + sampleCount / 2u];
 median = ${absoluteDeviation ? '(abs(residual(lower) - fit[fitOffset]) + abs(residual(upper) - fit[fitOffset])) * 0.5' : '(residual(lower) + residual(upper)) * 0.5'};
}
${
  absoluteDeviation
    ? `let deviation = median * 1.4826022;
fit[fitOffset + 1u] = deviation;
fit[fitOffset + 3u] = select(getNaN(), 1.2533141 * deviation * sqrt(covariance) / max(samples, 1.0), samples >= 2.0);`
    : 'fit[fitOffset] = median; fit[fitOffset + 1u] = getNaN(); fit[fitOffset + 2u] = samples; fit[fitOffset + 3u] = getNaN();'
}`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-apply`,
        operation,
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
let valid = isFiniteValue(difference) && isFiniteValue(cellVariance) && cellVariance >= 0.0 && isFiniteValue(biasError);
validity[validityOffset + index] = u32(valid);
differences[differencesOffset + index * 2u] = select(getNaN(), difference, valid);
differences[differencesOffset + index * 2u + 1u] = select(getNaN(), sqrt(max(cellVariance, 0.0) + biasError * biasError), valid);`
      }),
      // Per-cell planes for the volume sums: GPUReduction needs packed scalar inputs.
      createWGSLKernelNode(graph, {
        id: `${id}-volume-planes`,
        operation,
        invocationCount: count,
        declarations: measureDeclarations,
        bindings: [
          terrainBinding('differences', props.output.differences),
          terrainBinding('validity', props.output.validity),
          terrainBinding('changeValues', changeValues, 'read_write'),
          terrainBinding('areaValues', areaValues, 'read_write')
        ],
        body: `let valid = validity[validityOffset + index] != 0u;
let cellArea = ${isUniform ? '1.0' : 'getGroundCellArea(index)'};
changeValues[changeValuesOffset + index] = select(0.0, differences[differencesOffset + index * 2u] * cellArea, valid);
areaValues[areaValuesOffset + index] = select(0.0, cellArea, valid);`
      }),
      ...new GPUReduction({
        id: `${id}-change-sum`,
        input: changeValues,
        output: changeTotal,
        operation: 'sum'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${id}-valid-sum`,
        input: props.output.validity,
        output: validCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${id}-measurement-sum`,
        input: measurementCovarianceRows,
        mask: props.output.validity,
        output: measurementCovarianceTotal,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    if (!isUniform) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-area-sum`,
          input: areaValues,
          output: areaTotal,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    const area = getWGSLFloatLiteral(
      Math.abs(
        props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
      )
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-volume`,
        operation,
        invocationCount: 1,
        declarations: TERRAIN_FINITE_WGSL,
        bindings: [
          terrainBinding('fit', props.output.fit),
          terrainBinding('validCount', validCount),
          terrainBinding('changeTotal', changeTotal),
          ...(isUniform ? [] : [terrainBinding('areaTotal', areaTotal)]),
          terrainBinding('measurementTotal', measurementCovarianceTotal),
          terrainBinding('volume', props.output.volume, 'read_write')
        ],
        body: `let samples = validCount[validCountOffset];
let biasError = fit[fitOffset + 3u];
let covariance = measurementTotal[measurementTotalOffset];
${
  isUniform
    ? `let validArea = f32(samples) * ${area}; let change = changeTotal[changeTotalOffset] * ${area};
let deviation = sqrt(covariance + f32(samples) * f32(samples) * biasError * biasError) * ${area};`
    : `let validArea = areaTotal[areaTotalOffset]; let meanArea = validArea / f32(max(samples, 1u)); let change = changeTotal[changeTotalOffset];
let deviation = sqrt(covariance * meanArea * meanArea + validArea * validArea * biasError * biasError);`
}
volume[volumeOffset] = select(getNaN(), change, samples > 0u);
volume[volumeOffset + 1u] = select(getNaN(), deviation, samples > 0u);
volume[volumeOffset + 2u] = validArea;`
      })
    );
    return nodes;
  }
}
