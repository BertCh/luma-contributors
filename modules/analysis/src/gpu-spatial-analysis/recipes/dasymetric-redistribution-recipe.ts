// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
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
import {
  createRecipeKernelNode,
  getRecipeView,
  validateRecipeBoundary
} from './expansion-recipe-utils';
import {createTransientSpatialWeights, RecipeBuilder} from './recipe-utils';

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
    | {kind: 'polygons'; zones: GPUPolygonGeometryFields; tupleCapacity: number}
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
  assert(Number.isSafeInteger(targetCount) && targetCount > 0 && sourceCount > 0);
  assert(Number.isSafeInteger(tupleCapacity) && tupleCapacity > 0);
  assert(targetCount * tupleCapacity <= 4194304);
  assert(
    props.targetZones.length === cellCount && (!props.mask || props.mask.length === cellCount)
  );
  assert(!props.denominator || props.denominator === 'zone' || props.denominator === 'overlap');
  const cellArea = Math.abs(
    props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
  );
  assert(Number.isFinite(Math.fround(cellArea)) && Math.fround(cellArea) > 0);
  const createView = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => getRecipeView(graph, `${id}-${name}`, format, length);
  let coverage: FractionalCoverage;
  if (props.source.kind === 'polygons') {
    assert(props.source.zones.featureOffsets.length === sourceCount + 1);
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
        output: coverage
      })
    );
  } else {
    coverage = props.source.coverage;
  }
  assert(coverage.cellIds.length === tupleCapacity && coverage.fractions.length === tupleCapacity);
  assert(coverage.zoneValidity.length === sourceCount);
  Object.values(coverage.status).forEach(view => assert(view.length === 1));
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
  band.nodes.forEach(node => graph.add(node));
  const cellWeights = createView('cell-weights', 'float32', cellCount);
  const cellArithmeticInvalid = createView('cell-arithmetic-invalid', 'uint32', cellCount);
  graph.add(
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
  graph.add(
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
  graph.add(
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
  const weights = createTransientSpatialWeights(
    graph,
    `${id}-weights`,
    targetCount,
    props.pairCapacity
  );
  const pairOverflow = createView('pair-overflow', 'uint32', 1);
  const requiredPairCount = createView('pair-required', 'uint32', 1);
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
  graph.add(
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
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-arithmetic-evidence`,
      operation: id,
      invocationCount: 1,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'cells', view: cellArithmeticInvalid, type: 'u32', access: 'read'},
        {name: 'eligible', view: eligibleWeights, type: 'f32', access: 'read'},
        {name: 'fractions', view: transferredFractions, type: 'f32', access: 'read'},
        {name: 'sources', view: props.sourceTotals, type: 'f32', access: 'read'},
        {name: 'targets', view: targetTotals, type: 'f32', access: 'read'},
        {name: 'support', view: targetSupport, type: 'f32', access: 'read'},
        {name: 'output', view: arithmeticInvalidCount, type: 'u32', access: 'read_write'}
      ],
      body: `var failures=0u;for(var cell=0u;cell<${cellCount}u;cell++){failures+=cells[cellsOffset+cell];}
for(var source=0u;source<${sourceCount}u;source++){let eligibleWeight=eligible[eligibleOffset+source];let fraction=fractions[fractionsOffset+source];let total=sources[sourcesOffset+source];failures+=u32(!isFiniteValue(eligibleWeight));if(eligibleWeight>0.0){failures+=u32(!isFiniteValue(fraction)||!isFiniteValue(total*fraction)||!isFiniteValue(total*(1.0-fraction)));}}
for(var targetZone=0u;targetZone<${targetCount}u;targetZone++){failures+=u32(!isFiniteValue(targets[targetsOffset+targetZone])||!isFiniteValue(support[supportOffset+targetZone]));}output[outputOffset]=failures;`
    })
  );
  const coverageComplete = createView('coverage-complete', 'uint32', 1);
  const invalidTupleCount = createView('invalid-tuples', 'uint32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-coverage-evidence`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
        {name: 'required', view: coverage.status.requiredCount, type: 'u32', access: 'read'},
        {name: 'overflow', view: coverage.status.overflow, type: 'u32', access: 'read'},
        {name: 'validity', view: coverage.zoneValidity, type: 'u32', access: 'read'},
        {name: 'invalid', view: tupleInvalid, type: 'u32', access: 'read'},
        {name: 'complete', view: coverageComplete, type: 'u32', access: 'read_write'},
        {name: 'invalidCount', view: invalidTupleCount, type: 'u32', access: 'read_write'}
      ],
      body: `var valid=overflow[overflowOffset]==0u&&count[countOffset]==required[requiredOffset]&&count[countOffset]<=${tupleCapacity}u;
for(var source=0u;source<${sourceCount}u;source++){valid=valid&&validity[validityOffset+source]!=0u;}
var totalInvalid=0u;for(var row=0u;row<${tupleCapacity}u;row++){totalInvalid+=invalid[invalidOffset+row];}
invalidCount[invalidCountOffset]=totalInvalid;complete[completeOffset]=u32(valid&&totalInvalid==0u);`
    })
  );
  const complete = createView('complete', 'uint32', 1);
  const invalidSourceCount = createView('invalid-sources', 'uint32', 1);
  const zeroEligibleCount = createView('zero-eligible', 'uint32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-source-evidence`,
      operation: id,
      invocationCount: 1,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'coverage', view: coverageComplete, type: 'u32', access: 'read'},
        {name: 'arithmetic', view: arithmeticInvalidCount, type: 'u32', access: 'read'},
        {name: 'overflow', view: pairOverflow, type: 'u32', access: 'read'},
        {name: 'totals', view: props.sourceTotals, type: 'f32', access: 'read'},
        {name: 'eligible', view: eligibleWeights, type: 'f32', access: 'read'},
        {name: 'complete', view: complete, type: 'u32', access: 'read_write'},
        {name: 'invalid', view: invalidSourceCount, type: 'u32', access: 'read_write'},
        {name: 'zeroEligible', view: zeroEligibleCount, type: 'u32', access: 'read_write'}
      ],
      body: `var invalidSources=0u;var empty=0u;for(var source=0u;source<${sourceCount}u;source++){let total=totals[totalsOffset+source];invalidSources+=u32(!isFiniteValue(total)||total<0.0);empty+=u32(eligible[eligibleOffset+source]<=0.0);}
invalid[invalidOffset]=invalidSources;zeroEligible[zeroEligibleOffset]=empty;complete[completeOffset]=u32(coverage[coverageOffset]!=0u&&overflow[overflowOffset]==0u&&invalidSources==0u&&arithmetic[arithmeticOffset]==0u);`
    })
  );
  graph.add(
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
    graph.add(
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
  graph.add(
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
