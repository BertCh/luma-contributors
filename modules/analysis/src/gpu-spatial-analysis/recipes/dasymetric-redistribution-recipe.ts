// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addPreparedRecipe} from './prepared-recipe';

import {
  validateCapacity,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUReduction, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  GPURasterFractionalCoverage,
  type GPURasterFractionalCoverageProps
} from '../../raster-analysis/gpu-raster-fractional-coverage';
import {
  getRasterBandViews,
  getRasterBufferBandNodes
} from '../../raster-analysis/raster-band-utils';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUArealInterpolation,
  type GPUArealInterpolationDenominator
} from '../areal-interpolation/index';
import {GPUSpatialLag} from '../spatial-weights/index';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createRecipeKernelNode, getRecipeView, validateRecipeBoundary} from './recipe-utils';
import {addRecipeCountNodes, createTransientSpatialWeights, RecipeBuilder} from './recipe-utils';

type FractionalCoverage = GPURasterFractionalCoverageProps['output'];

export type GPUDasymetricRedistributionOutput = {
  targetTotals: GraphDataView<'float32'>;
  /** Sum of eligible source support weights assigned to each target, before normalization. */
  targetSupportWeights: GraphDataView<'float32'>;
  sourceEligibleWeights: GraphDataView<'float32'>;
  sourceTransferredFractions: GraphDataView<'float32'>;
  sourceAllocatedTotals: GraphDataView<'float32'>;
  sourceUnallocatedTotals: GraphDataView<'float32'>;
  /** Complete evidence and positive eligible support; zero-support sources retain their mass. */
  sourceValidity: GraphDataView<'uint32'>;
};

export type GPUDasymetricRedistributionRecipeProps = {
  id?: string;
  /** Planar grid. Ancillary density is per square affine-coordinate unit. */
  grid: GPURasterMetadata;
  source:
    | {
        kind: 'polygons';
        zones: GPUPolygonGeometryFields;
        tupleCapacity: number;
        /** `getGPURasterFractionalCoverageParameterValues(grid)` view: per-frame georeference of `grid`. */
        parameters: GraphDataView<'float32'>;
      }
    | {kind: 'coverage'; coverage: FractionalCoverage};
  /** Nonnegative finite extensive totals in stable source-feature row order. */
  sourceTotals: GraphDataView<'float32'>;
  /** One owner per whole cell; IDs >= targetZoneCount denote unassigned support. */
  targetZones: GraphDataView<'uint32'>;
  targetZoneCount: number;
  /** Calibrated nonnegative density/weight. Zero excludes land; nodata is missing support. */
  ancillary: GPURasterBufferBand;
  /** Optional eligibility mask. Zero excludes a cell even if ancillary data are present. */
  mask?: GraphDataView<'uint32'>;
  /** Default overlap allocates all mass with target support; zone retains uncovered residual. */
  denominator?: GPUArealInterpolationDenominator;
  pairCapacity: number;
  outputs?: Partial<GPUDasymetricRedistributionOutput>;
};

/** Largest `targetZoneCount * tupleCapacity`; the per-target support kernel scans every tuple row once per target. */
const GPU_DASYMETRIC_MAXIMUM_TUPLE_PRODUCT = 4194304;

/**
 * Fractional source support -> ancillary eligibility -> weighted areal interpolation -> mass ledger.
 * Targets own complete raster cells: independent source/target fractions are never multiplied.
 * Source features may overlap: their totals are independent and targetSupportWeights add their
 * measures, not their geometric union. Prepared coverage must contain unique (source, cell) rows,
 * with no repeated halo cells. Uniqueness is a caller precondition, not certified by complete. Concatenate disjoint partition cores BEFORE normalizing; independently
 * normalized tile totals cannot be added. Source totals and all input buffers stay caller-owned.
 *
 * Zero eligible support assigns zero and retains the entire source total. Missing/negative ancillary
 * samples carry no support and remain visible as NaN in intermediates.cellWeights. Incomplete
 * coverage, invalid tuples/source totals, arithmetic overflow or pair overflow invalidate all published numerical outputs;
 * the original per-stage evidence remains available. No CPU readback or fallback is performed.
 */
export function addDasymetricRedistributionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUDasymetricRedistributionRecipeProps
) {
  return addPreparedRecipe(graph, props, declareDasymetricRedistributionRecipe);
}

function declareDasymetricRedistributionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUDasymetricRedistributionRecipeProps
) {
  const id = props.id ?? 'dasymetric-redistribution-recipe';
  const builder = new RecipeBuilder(graph);
  const sourceCount = props.sourceTotals.length;
  const targetCount = props.targetZoneCount;
  const cellCount = props.grid.width * props.grid.height;
  const tupleCapacity =
    props.source.kind === 'polygons'
      ? props.source.tupleCapacity
      : props.source.coverage.zoneIds.length;
  // Bound the explicit target/support summary and require nonempty, aligned buffer shapes.
  validateCapacity(id, 'targetZoneCount', targetCount, 1, GPU_DASYMETRIC_MAXIMUM_TUPLE_PRODUCT);
  validateCapacity(id, 'sourceTotals length', sourceCount, 1);
  validateCapacity(id, 'tupleCapacity', tupleCapacity, 1, GPU_DASYMETRIC_MAXIMUM_TUPLE_PRODUCT);
  validateRequirement(
    id,
    targetCount * tupleCapacity <= GPU_DASYMETRIC_MAXIMUM_TUPLE_PRODUCT,
    `targetZoneCount times tupleCapacity must be at most ${GPU_DASYMETRIC_MAXIMUM_TUPLE_PRODUCT}`
  );
  validateViewLength(id, 'targetZones', props.targetZones, cellCount);
  if (props.mask) validateViewLength(id, 'mask', props.mask, cellCount);
  validateRequirement(
    id,
    !props.denominator || props.denominator === 'zone' || props.denominator === 'overlap',
    "denominator must be 'zone' or 'overlap'"
  );
  const cellArea = Math.abs(
    props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
  );
  validateRequirement(
    id,
    Number.isFinite(Math.fround(cellArea)) && Math.fround(cellArea) > 0,
    'grid cell area must be finite and positive in float32'
  );
  const createView = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => getRecipeView(graph, `${id}-${name}`, format, length);
  let coverage: FractionalCoverage;
  if (props.source.kind === 'polygons') {
    validateViewLength(
      id,
      'source.zones.featureOffsets',
      props.source.zones.featureOffsets,
      sourceCount + 1
    );
    coverage = {
      zoneIds: createView('coverage-sources', 'uint32', tupleCapacity),
      cellIds: createView('coverage-cells', 'uint32', tupleCapacity),
      fractions: createView('coverage-fractions', 'float32', tupleCapacity),
      zoneValidity: createView('coverage-validity', 'uint32', sourceCount),
      status: {
        count: createView('coverage-count', 'uint32', 1),
        requiredCount: createView('coverage-required', 'uint32', 1),
        overflow: createView('coverage-overflow', 'uint32', 1)
      }
    };
    builder.add(
      new GPURasterFractionalCoverage({
        id: `${id}-coverage`,
        grid: props.grid,
        zones: props.source.zones,
        parameters: props.source.parameters,
        output: coverage
      })
    );
  } else {
    coverage = props.source.coverage;
  }
  validateViewLength(id, 'coverage.cellIds', coverage.cellIds, tupleCapacity);
  validateViewLength(id, 'coverage.fractions', coverage.fractions, tupleCapacity);
  validateViewLength(id, 'coverage.zoneValidity', coverage.zoneValidity, sourceCount);
  for (const [name, view] of Object.entries(coverage.status))
    if (view)
      validateRequirement(
        id,
        view.length >= 1,
        `coverage.status.${name} must have at least one row`
      );
  const outputs = {
    targetTotals: getRecipeView(
      graph,
      `${id}-target-totals`,
      'float32',
      targetCount,
      props.outputs?.targetTotals
    ),
    targetSupportWeights: getRecipeView(
      graph,
      `${id}-target-support`,
      'float32',
      targetCount,
      props.outputs?.targetSupportWeights
    ),
    sourceEligibleWeights: getRecipeView(
      graph,
      `${id}-source-eligible`,
      'float32',
      sourceCount,
      props.outputs?.sourceEligibleWeights
    ),
    sourceTransferredFractions: getRecipeView(
      graph,
      `${id}-source-fractions`,
      'float32',
      sourceCount,
      props.outputs?.sourceTransferredFractions
    ),
    sourceAllocatedTotals: getRecipeView(
      graph,
      `${id}-source-allocated`,
      'float32',
      sourceCount,
      props.outputs?.sourceAllocatedTotals
    ),
    sourceUnallocatedTotals: getRecipeView(
      graph,
      `${id}-source-unallocated`,
      'float32',
      sourceCount,
      props.outputs?.sourceUnallocatedTotals
    ),
    sourceValidity: getRecipeView(
      graph,
      `${id}-source-validity`,
      'uint32',
      sourceCount,
      props.outputs?.sourceValidity
    )
  };
  validateRecipeBoundary(
    graph,
    id,
    [
      props.sourceTotals,
      props.targetZones,
      props.mask,
      ...getRasterBandViews(props.ancillary),
      coverage.zoneIds,
      coverage.cellIds,
      coverage.fractions,
      coverage.zoneValidity,
      ...Object.values(coverage.status),
      ...(props.source.kind === 'polygons' ? Object.values(props.source.zones) : [])
    ],
    Object.values(outputs)
  );
  const band = getRasterBufferBandNodes(
    graph,
    `${id}-ancillary`,
    props.ancillary,
    props.grid,
    true
  );
  builder.addGlue('ancillary-band', band.nodes);
  const cellWeights = createView('cell-weights', 'float32', cellCount);
  const cellArithmeticInvalid = createView('cell-arithmetic-invalid', 'uint32', cellCount);
  builder.addGlue(
    'eligibility',
    createRecipeKernelNode(graph, {
      id: `${id}-eligibility`,
      operation: id,
      invocationCount: cellCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'values', view: band.band.storage.values, type: 'f32', access: 'read'},
        {name: 'validity', view: band.band.validity!, type: 'u32', access: 'read'},
        {name: 'weights', view: cellWeights, type: 'f32', access: 'read_write'},
        {name: 'arithmeticInvalid', view: cellArithmeticInvalid, type: 'u32', access: 'read_write'},
        ...(props.mask
          ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
          : [])
      ],
      body: `let value=values[valuesOffset+index];var weight=bitcast<f32>(0x7fc00000u | (index & 0u));
if(validity[validityOffset+index]!=0u&&isFiniteValue(value)&&value>=0.0){weight=value*${cellArea};}
${props.mask ? 'if(mask[maskOffset+index]==0u){weight=0.0;}' : ''}
weights[weightsOffset+index]=weight;arithmeticInvalid[arithmeticInvalidOffset+index]=u32(isFiniteValue(value)&&value>=0.0&&validity[validityOffset+index]!=0u&&!isFiniteValue(weight)${props.mask ? '&&mask[maskOffset+index]!=0u' : ''});`
    })
  );
  const tupleSources = createView('tuple-sources', 'uint32', tupleCapacity);
  const tupleTargets = createView('tuple-targets', 'uint32', tupleCapacity);
  const tupleInvalid = createView('tuple-invalid', 'uint32', tupleCapacity);
  const tupleWeights = createView('tuple-weights', 'float32', tupleCapacity);
  builder.addGlue(
    'tuple-identities',
    createRecipeKernelNode(graph, {
      id: `${id}-tuple-identities`,
      operation: id,
      invocationCount: tupleCapacity,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'sources', view: coverage.zoneIds, type: 'u32', access: 'read'},
        {name: 'cells', view: coverage.cellIds, type: 'u32', access: 'read'},
        {name: 'fractions', view: coverage.fractions, type: 'f32', access: 'read'},
        {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
        {name: 'targets', view: props.targetZones, type: 'u32', access: 'read'},
        {name: 'outputSources', view: tupleSources, type: 'u32', access: 'read_write'},
        {name: 'outputTargets', view: tupleTargets, type: 'u32', access: 'read_write'},
        {name: 'invalid', view: tupleInvalid, type: 'u32', access: 'read_write'}
      ],
      body: `var source=0xffffffffu;var targetZone=0xffffffffu;var invalidTuple=0u;
if(index<count[countOffset]){let cell=cells[cellsOffset+index];let fraction=fractions[fractionsOffset+index];
let valid=sources[sourcesOffset+index]<${sourceCount}u&&cell<${cellCount}u&&isFiniteValue(fraction)&&fraction>=0.0&&fraction<=1.0;
invalidTuple=u32(!valid);if(valid){source=sources[sourcesOffset+index];targetZone=targets[targetsOffset+cell];}}
outputSources[outputSourcesOffset+index]=source;outputTargets[outputTargetsOffset+index]=targetZone;invalid[invalidOffset+index]=invalidTuple;`
    })
  );
  builder.addGlue(
    'tuple-weights',
    createRecipeKernelNode(graph, {
      id: `${id}-tuple-weights`,
      operation: id,
      invocationCount: tupleCapacity,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'sources', view: tupleSources, type: 'u32', access: 'read'},
        {name: 'cells', view: coverage.cellIds, type: 'u32', access: 'read'},
        {name: 'fractions', view: coverage.fractions, type: 'f32', access: 'read'},
        {name: 'weights', view: cellWeights, type: 'f32', access: 'read'},
        {name: 'output', view: tupleWeights, type: 'f32', access: 'read_write'}
      ],
      body: `var weight=0.0;if(sources[sourcesOffset+index]<${sourceCount}u){let value=weights[weightsOffset+cells[cellsOffset+index]];if(isFiniteValue(value)&&value>0.0){weight=value*fractions[fractionsOffset+index];}}output[outputOffset+index]=weight;`
    })
  );
  const pairOverflow = createView('pair-overflow', 'uint32', 1);
  const requiredPairCount = createView('pair-required', 'uint32', 1);
  const weights = createTransientSpatialWeights(
    graph,
    `${id}-weights`,
    targetCount,
    props.pairCapacity,
    // The transfer consumer must read the status written by the areal producer each frame.
    {status: {overflow: pairOverflow, requiredCount: requiredPairCount}}
  );
  const eligibleWeights = createView('eligible-weights', 'float32', sourceCount);
  const transferredFractions = createView('transferred-fractions', 'float32', sourceCount);
  builder.add(
    new GPUArealInterpolation({
      id: `${id}-areal`,
      // Tuple rows are a measure adapter; original grid placement remains in the provenance.
      grid: {
        width: tupleCapacity,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0],
        pixelInterpretation: 'area'
      },
      sourceZones: tupleSources,
      targetZones: tupleTargets,
      sourceZoneCount: sourceCount,
      targetZoneCount: targetCount,
      cellWeights: tupleWeights,
      denominator: props.denominator ?? 'overlap',
      output: {
        weights,
        sourceAreas: eligibleWeights,
        sourceTransferredFractions: transferredFractions,
        status: {overflow: pairOverflow, requiredPairCount}
      }
    })
  );
  const targetTotals = createView('raw-target-totals', 'float32', targetCount);
  builder.add(
    new GPUSpatialLag({
      id: `${id}-transfer`,
      values: props.sourceTotals,
      weights,
      output: targetTotals
    })
  );
  const targetSupport = createView('raw-target-support', 'float32', targetCount);
  builder.addGlue(
    'target-support',
    createRecipeKernelNode(graph, {
      id: `${id}-target-support`,
      operation: id,
      invocationCount: targetCount,
      bindings: [
        {name: 'targets', view: tupleTargets, type: 'u32', access: 'read'},
        {name: 'weights', view: tupleWeights, type: 'f32', access: 'read'},
        {name: 'output', view: targetSupport, type: 'f32', access: 'read_write'}
      ],
      body: `var supportWeight=0.0;for(var row=0u;row<${tupleCapacity}u;row++){if(targets[targetsOffset+row]==index){supportWeight+=weights[weightsOffset+row];}}output[outputOffset+index]=supportWeight;`
    })
  );
  const arithmeticInvalidCount = createView('arithmetic-invalid', 'uint32', 1);
  addRecipeCountNodes(builder, {
    id: `${id}-arithmetic-evidence`,
    operation: id,
    // Cell, source and target rows share one index space; each branch is guarded by its own count.
    rowCount: Math.max(cellCount, sourceCount, targetCount),
    declarations: IS_FINITE_VALUE_WGSL,
    bindings: [
      {name: 'cells', view: cellArithmeticInvalid, type: 'u32', access: 'read'},
      {name: 'eligible', view: eligibleWeights, type: 'f32', access: 'read'},
      {name: 'fractions', view: transferredFractions, type: 'f32', access: 'read'},
      {name: 'sources', view: props.sourceTotals, type: 'f32', access: 'read'},
      {name: 'targets', view: targetTotals, type: 'f32', access: 'read'},
      {name: 'support', view: targetSupport, type: 'f32', access: 'read'}
    ],
    prelude: `var failures=0u;
if(index<${cellCount}u){failures+=cells[cellsOffset+index];}
if(index<${sourceCount}u){let eligibleWeight=eligible[eligibleOffset+index];let fraction=fractions[fractionsOffset+index];let total=sources[sourcesOffset+index];failures+=u32(!isFiniteValue(eligibleWeight));if(eligibleWeight>0.0){failures+=u32(!isFiniteValue(fraction)||!isFiniteValue(total*fraction)||!isFiniteValue(total*(1.0-fraction)));}}
if(index<${targetCount}u){failures+=u32(!isFiniteValue(targets[targetsOffset+index])||!isFiniteValue(support[supportOffset+index]));}`,
    counts: [{name: 'failures', output: arithmeticInvalidCount, value: 'failures'}]
  });
  const coverageComplete = createView('coverage-complete', 'uint32', 1);
  const invalidTupleCount = createView('invalid-tuples', 'uint32', 1);
  builder.add(
    new GPUReduction({
      id: `${id}-invalid-tuples`,
      input: tupleInvalid,
      output: invalidTupleCount,
      operation: 'sum'
    })
  );
  const invalidZoneCount = createView('coverage-invalid-zones', 'uint32', 1);
  addRecipeCountNodes(builder, {
    id: `${id}-coverage-zones`,
    operation: id,
    rowCount: sourceCount,
    bindings: [{name: 'validity', view: coverage.zoneValidity, type: 'u32', access: 'read'}],
    counts: [
      {
        name: 'invalidZones',
        output: invalidZoneCount,
        value: 'u32(validity[validityOffset+index]==0u)'
      }
    ]
  });
  builder.addGlue(
    'coverage-evidence',
    createRecipeKernelNode(graph, {
      id: `${id}-coverage-evidence`,
      operation: id,
      invocationCount: 1,
      bindings: [
        ...(coverage.status.candidateOverflow
          ? [
              {
                name: 'candidateOverflow',
                view: coverage.status.candidateOverflow,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
        {name: 'required', view: coverage.status.requiredCount, type: 'u32', access: 'read'},
        {name: 'overflow', view: coverage.status.overflow, type: 'u32', access: 'read'},
        {name: 'invalidZones', view: invalidZoneCount, type: 'u32', access: 'read'},
        {name: 'invalidTuples', view: invalidTupleCount, type: 'u32', access: 'read'},
        {name: 'complete', view: coverageComplete, type: 'u32', access: 'read_write'}
      ],
      body: `let valid=overflow[overflowOffset]==0u${coverage.status.candidateOverflow ? '&&candidateOverflow[candidateOverflowOffset]==0u' : ''}&&count[countOffset]==required[requiredOffset]&&count[countOffset]<=${tupleCapacity}u&&invalidZones[invalidZonesOffset]==0u;
complete[completeOffset]=u32(valid&&invalidTuples[invalidTuplesOffset]==0u);`
    })
  );
  const complete = createView('complete', 'uint32', 1);
  const invalidSourceCount = createView('invalid-sources', 'uint32', 1);
  const zeroEligibleCount = createView('zero-eligible', 'uint32', 1);
  addRecipeCountNodes(builder, {
    id: `${id}-source-counts`,
    operation: id,
    rowCount: sourceCount,
    declarations: IS_FINITE_VALUE_WGSL,
    bindings: [
      {name: 'totals', view: props.sourceTotals, type: 'f32', access: 'read'},
      {name: 'eligible', view: eligibleWeights, type: 'f32', access: 'read'}
    ],
    prelude: 'let total=totals[totalsOffset+index];',
    counts: [
      {
        name: 'invalidSources',
        output: invalidSourceCount,
        value: 'u32(!isFiniteValue(total)||total<0.0)'
      },
      {name: 'empty', output: zeroEligibleCount, value: 'u32(eligible[eligibleOffset+index]<=0.0)'}
    ]
  });
  builder.addGlue(
    'source-evidence',
    createRecipeKernelNode(graph, {
      id: `${id}-source-evidence`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'coverage', view: coverageComplete, type: 'u32', access: 'read'},
        {name: 'arithmetic', view: arithmeticInvalidCount, type: 'u32', access: 'read'},
        {name: 'overflow', view: pairOverflow, type: 'u32', access: 'read'},
        {name: 'invalidSources', view: invalidSourceCount, type: 'u32', access: 'read'},
        {name: 'complete', view: complete, type: 'u32', access: 'read_write'}
      ],
      body: 'complete[completeOffset]=u32(coverage[coverageOffset]!=0u&&overflow[overflowOffset]==0u&&invalidSources[invalidSourcesOffset]==0u&&arithmetic[arithmeticOffset]==0u);'
    })
  );
  builder.addGlue(
    'source-ledger',
    createRecipeKernelNode(graph, {
      id: `${id}-source-ledger`,
      operation: id,
      invocationCount: sourceCount,
      bindings: [
        {name: 'complete', view: complete, type: 'u32', access: 'read'},
        {name: 'totals', view: props.sourceTotals, type: 'f32', access: 'read'},
        {name: 'eligible', view: eligibleWeights, type: 'f32', access: 'read'},
        {name: 'fractions', view: transferredFractions, type: 'f32', access: 'read'},
        {name: 'allocated', view: outputs.sourceAllocatedTotals, type: 'f32', access: 'read_write'},
        {
          name: 'unallocated',
          view: outputs.sourceUnallocatedTotals,
          type: 'f32',
          access: 'read_write'
        },
        {name: 'validity', view: outputs.sourceValidity, type: 'u32', access: 'read_write'}
      ],
      body: `let valid=complete[completeOffset]!=0u;let supported=eligible[eligibleOffset+index]>0.0;
let fraction=select(0.0,fractions[fractionsOffset+index],supported);let total=totals[totalsOffset+index];let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
allocated[allocatedOffset+index]=select(missing,total*fraction,valid);unallocated[unallocatedOffset+index]=select(missing,total*(1.0-fraction),valid);validity[validityOffset+index]=u32(valid&&supported);`
    })
  );
  for (const [name, source, destination] of [
    ['eligible', eligibleWeights, outputs.sourceEligibleWeights],
    ['fractions', transferredFractions, outputs.sourceTransferredFractions]
  ] as const) {
    builder.addGlue(
      `publish-${name}`,
      createRecipeKernelNode(graph, {
        id: `${id}-publish-${name}`,
        operation: id,
        invocationCount: sourceCount,
        bindings: [
          {name: 'complete', view: complete, type: 'u32', access: 'read'},
          {name: 'source', view: source, type: 'f32', access: 'read'},
          {name: 'output', view: destination, type: 'f32', access: 'read_write'}
        ],
        body: 'output[outputOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),source[sourceOffset+index],complete[completeOffset]!=0u);'
      })
    );
  }
  builder.addGlue(
    'target-ledger',
    createRecipeKernelNode(graph, {
      id: `${id}-target-ledger`,
      operation: id,
      invocationCount: targetCount,
      bindings: [
        {name: 'complete', view: complete, type: 'u32', access: 'read'},
        {name: 'totals', view: targetTotals, type: 'f32', access: 'read'},
        {name: 'rawSupport', view: targetSupport, type: 'f32', access: 'read'},
        {name: 'output', view: outputs.targetTotals, type: 'f32', access: 'read_write'},
        {name: 'support', view: outputs.targetSupportWeights, type: 'f32', access: 'read_write'}
      ],
      body: `let missing=bitcast<f32>(0x7fc00000u | (index & 0u));let valid=complete[completeOffset]!=0u;
output[outputOffset+index]=select(missing,totals[totalsOffset+index],valid);support[supportOffset+index]=select(missing,rawSupport[rawSupportOffset+index],valid);`
    })
  );
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {
      grid: props.grid,
      coverage,
      ancillary: props.ancillary,
      mask: props.mask,
      sourceTotals: props.sourceTotals,
      targetZones: props.targetZones,
      cellWeights,
      tupleSources,
      tupleTargets,
      tupleWeights,
      weights
    },
    status: {
      complete,
      invalidTupleCount,
      invalidSourceCount,
      zeroEligibleCount,
      arithmeticInvalidCount,
      stages: [
        {stage: 'fractional-source-coverage', status: coverage.status},
        {
          stage: 'coverage-validation',
          status: {converged: coverageComplete, invalidCount: invalidTupleCount}
        },
        {
          stage: 'weighted-areal-interpolation',
          status: {overflow: pairOverflow, requiredCount: requiredPairCount}
        },
        {stage: 'source-totals', status: {invalidCount: invalidSourceCount}},
        {stage: 'arithmetic', status: {invalidCount: arithmeticInvalidCount}},
        {
          stage: 'dasymetric-conservation',
          status: {converged: complete, invalidCount: zeroEligibleCount}
        }
      ]
    }
  };
}
