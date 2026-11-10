// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validatePredictionViews} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUSpatialCrossValidationProps} from './gpu-spatial-cross-validation';

/**
 * Largest design row count. Fold designs run as ONE serial invocation (k-means, NNDM swaps,
 * kNNDM restarts and the nearest-neighbor tables loop over rows and prediction rows), so 256 is
 * where that invocation stays within a frame. This is the reference tier; the parallel design for
 * each fold design is described in `GPUSpatialCrossValidation`.
 */
export const MAXIMUM_DESIGN_ROW_COUNT = 256;
/** Largest prediction-location count: NNDM and kNNDM compare every row with every location. */
export const MAXIMUM_PREDICTION_ROW_COUNT = 256;
/** Most features per row in environmental k-means, bounding the per-row distance loop. */
const MAXIMUM_FEATURE_COUNT = 8;
/** Most clusters (folds) for environmental and generated kNNDM clustering. */
const MAXIMUM_CLUSTER_COUNT = 16;
/** Most Lloyd iterations in environmental k-means. */
const MAXIMUM_ITERATION_COUNT = 1024;
/** NNDM gives every row its own fold and keeps an exclusion radius per row; 64 bounds that swap loop. */
const MAXIMUM_NNDM_ROW_COUNT = 64;
/** Most kNNDM candidate partitions: each is clustered and scored in the same serial invocation. */
const MAXIMUM_CANDIDATE_COUNT = 64;

/** Finite, reproducible designs over resident rows. Environmental features never include outcomes. */
export type GPUSpatialValidationDesign =
  | {
      /** Rank numeric group IDs and cycle ranks through folds; leave-group-out when groups == folds. */
      mode: 'groups';
      diagnostics?: GraphDataView<'float32'>;
    }
  | {
      mode: 'environmental';
      features: GraphDataView<'float32'>;
      featureCount: number;
      /** Standardize complete feature rows using sample standard deviations. Constant columns contribute zero. */
      standardize?: boolean;
      maximumIterations?: number;
      diagnostics?: GraphDataView<'float32'>;
    }
  | {
      mode: 'nndm';
      predictionPositions: GraphDataView<'float32x2'>;
      maximumDistance: number;
      minimumTrainingFraction?: number;
      /** Four words [status, iterations, objective, selected candidate], then one exclusion radius per row. */
      diagnostics?: GraphDataView<'float32'>;
    }
  | {
      mode: 'knndm';
      predictionPositions: GraphDataView<'float32x2'>;
      /** Explicit candidate partitions, candidate-major row fold IDs. Generated clustering graphs compose directly. */
      candidates?: GraphDataView<'uint32'>;
      /** Default8 deterministic farthest-first/Lloyd restarts when candidates are omitted. */
      candidateCount?: number;
      maximumFoldFraction?: number;
      diagnostics?: GraphDataView<'float32'>;
    };

export function getSpatialDesignInputs(design?: GPUSpatialValidationDesign): GraphDataView[] {
  if (!design || design.mode === 'groups') return [];
  if (design.mode === 'environmental') return [design.features];
  return [
    design.predictionPositions,
    ...(design.mode === 'knndm' && design.candidates ? [design.candidates] : [])
  ];
}
export function validateSpatialValidationDesign(
  id: string,
  props: GPUSpatialCrossValidationProps
): void {
  const design = props.design;
  if (!design) return;
  const rows = props.positions.length;
  validateCapacity(id, 'design rows', rows, 2, MAXIMUM_DESIGN_ROW_COUNT);
  if (design.mode === 'environmental') {
    validateCapacity(id, 'featureCount', design.featureCount, 1, MAXIMUM_FEATURE_COUNT);
    validateCapacity(id, 'cluster count', props.foldCount, 2, MAXIMUM_CLUSTER_COUNT);
    validateCapacity(
      id,
      'maximumIterations',
      design.maximumIterations ?? 100,
      1,
      MAXIMUM_ITERATION_COUNT
    );
    validatePackedView(design.features, ['float32'], id);
    if (design.features.length !== rows * design.featureCount)
      throw new Error(`${id}: feature shape mismatch`);
  }
  if (design.mode === 'nndm' || design.mode === 'knndm') {
    validatePackedView(design.predictionPositions, ['float32x2'], id);
    validateCapacity(
      id,
      'prediction rows',
      design.predictionPositions.length,
      1,
      MAXIMUM_PREDICTION_ROW_COUNT
    );
    if (design.mode === 'nndm') {
      if (
        rows > MAXIMUM_NNDM_ROW_COUNT ||
        props.foldCount !== rows ||
        !Number.isFinite(design.maximumDistance) ||
        design.maximumDistance < 0 ||
        design.maximumDistance > 1e18 ||
        !(
          design.minimumTrainingFraction === undefined ||
          (design.minimumTrainingFraction > 0 && design.minimumTrainingFraction < 1)
        )
      )
        throw new Error(`${id}: invalid NNDM bounds`);
    } else {
      validateCapacity(
        id,
        'candidateCount',
        design.candidateCount ?? 8,
        1,
        MAXIMUM_CANDIDATE_COUNT
      );
      if (!design.candidates)
        validateCapacity(id, 'generated cluster count', props.foldCount, 2, MAXIMUM_CLUSTER_COUNT);
      if (design.candidates) validatePackedView(design.candidates, ['uint32'], id);
      if (
        (design.candidates && design.candidates.length !== rows * (design.candidateCount ?? 8)) ||
        !(
          (design.maximumFoldFraction ?? 0.8) > 1 / props.foldCount &&
          (design.maximumFoldFraction ?? 0.8) < 1
        )
      )
        throw new Error(`${id}: invalid kNNDM candidates or balance limit`);
    }
  }
  if (design.diagnostics) {
    validatePackedView(design.diagnostics, ['float32'], id);
    if (design.diagnostics.length !== 4 + (design.mode === 'nndm' ? rows : 0))
      throw new Error(`${id}: diagnostics shape mismatch`);
  }
  if (design.diagnostics) {
    validatePredictionViews(id, [], [...Object.values(props.output), design.diagnostics]);
  }
  validateGraphOutputsDisjointFromInputs(
    id,
    [...Object.values(props.output), design.diagnostics],
    [...getSpatialDesignInputs(design), props.positions, props.blocks, props.mask]
  );
}

/** Algorithmic NNDM contract follows blockCV4.0.1; no upstream implementation is bundled. */
export function createSpatialValidationDesignNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSpatialCrossValidationProps,
  id: string
): {nodes: GPUCommandNode<Parameters>[]; state: GraphDataView<'float32'>} {
  const design = props.design!;
  const rows = props.positions.length;
  if (design.mode === 'knndm' && !design.candidates) {
    const candidateCount = design.candidateCount ?? 8;
    const candidates = createTransientView(
      graph,
      `${id}-generated-candidates`,
      'uint32',
      rows * candidateCount
    );
    const features = graph.createDataView(props.positions.buffer, {
      format: 'float32',
      length: rows * 2,
      byteOffset: props.positions.byteOffset
    });
    const generated: GPUCommandNode<Parameters>[] = [];
    for (let candidate = 0; candidate < candidateCount; candidate++) {
      const folds = graph.createDataView(candidates.buffer, {
        format: 'uint32',
        length: rows,
        byteOffset: 4 * candidate * rows
      });
      generated.push(
        ...createSpatialValidationDesignNodes(
          graph,
          {
            ...props,
            seed: ((props.seed ?? 0) + candidate * 2654435761) >>> 0,
            design: {mode: 'environmental', features, featureCount: 2, standardize: false},
            output: {...props.output, folds}
          },
          `${id}-candidate-${candidate}`
        ).nodes
      );
    }
    const matched = createSpatialValidationDesignNodes(
      graph,
      {...props, design: {...design, candidates, candidateCount}},
      `${id}-match`
    );
    return {state: matched.state, nodes: [...generated, ...matched.nodes]};
  }
  const state =
    design.diagnostics ??
    createTransientView(
      graph,
      `${id}-design-state`,
      'float32',
      4 + (design.mode === 'nndm' ? rows : 0)
    );
  validateGraphViewsBelongToGraph(id, graph, [...getSpatialDesignInputs(design), state]);
  const declarations = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
fn location(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset+2u*row], positions[positionsOffset+2u*row+1u]); }
fn valid(row: u32) -> bool { let point=location(row); return isFiniteBits(point.x)&&isFiniteBits(point.y)${props.mask ? '&&mask[maskOffset+row]!=0u' : ''}; }
fn rowDistance(first:u32,second:u32)->f32 { return distance(location(first),location(second)); }`;
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32' as const, access: 'read' as const},
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : []),
    {name: 'folds', view: props.output.folds, type: 'u32' as const, access: 'read_write' as const},
    {name: 'state', view: state, type: 'f32' as const, access: 'read_write' as const}
  ];
  const initialize = `state[stateOffset]=0.0; state[stateOffset+1u]=0.0; state[stateOffset+2u]=getNaN(); state[stateOffset+3u]=getNaN(); for(var row=0u;row<${rows}u;row++){folds[foldsOffset+row]=0xffffffffu;}`;
  let body = initialize;
  let extraDeclarations = '';
  if (design.mode === 'groups') {
    bindings.push({name: 'groups', view: props.blocks, type: 'u32', access: 'read'});
    body += `var ordered:array<u32,${rows}>;var count=0u;
for(var row=0u;row<${rows}u;row++){if(!valid(row)){continue;}let group=groups[groupsOffset+row];var index=count;loop{if(index==0u){break;}if(ordered[index-1u]<=group){break;}ordered[index]=ordered[index-1u];index--;}ordered[index]=group;count++;}
for(var row=0u;row<${rows}u;row++){if(!valid(row)){continue;}let group=groups[groupsOffset+row];var rank=0u;for(var other=0u;other<count;other++){if(ordered[other]>=group){break;}if(other==0u){rank++;}else if(ordered[other]!=ordered[other-1u]){rank++;}}folds[foldsOffset+row]=rank%${props.foldCount}u;}`;
  } else if (design.mode === 'environmental') {
    const columns = design.featureCount,
      clusters = props.foldCount;
    bindings.push({name: 'features', view: design.features, type: 'f32', access: 'read'});
    extraDeclarations = `fn complete(row:u32)->bool{if(!valid(row)){return false;}for(var column=0u;column<${columns}u;column++){if(!isFiniteBits(features[featuresOffset+row*${columns}u+column])){return false;}}return true;}`;
    body += `var means:array<f32,${columns}>;var scales:array<f32,${columns}>;var centers:array<f32,${clusters * columns}>;var assignments:array<u32,${rows}>;var sampleCount=0u;
for(var row=0u;row<${rows}u;row++){assignments[row]=0xffffffffu;if(!complete(row)){continue;}sampleCount++;for(var column=0u;column<${columns}u;column++){let value=features[featuresOffset+row*${columns}u+column];let difference=value-means[column];means[column]+=difference/f32(sampleCount);scales[column]+=difference*(value-means[column]);}}
if(sampleCount<${clusters}u){state[stateOffset]=1.0;return;}
for(var column=0u;column<${columns}u;column++){scales[column]=${design.standardize === false ? '1.0;means[column]=0.0' : 'select(1.0,sqrt(scales[column]/f32(sampleCount-1u)),scales[column]>0.0)'};}
var seedRow=${(props.seed ?? 0) % rows}u;for(var offset=0u;offset<${rows}u;offset++){let row=(seedRow+offset)%${rows}u;if(complete(row)){seedRow=row;break;}}
for(var cluster=0u;cluster<${clusters}u;cluster++){
 if(cluster>0u){var farthest=-1.0;for(var row=0u;row<${rows}u;row++){if(!complete(row)){continue;}var nearest=3.402823e38;for(var previous=0u;previous<cluster;previous++){var squared=0.0;for(var column=0u;column<${columns}u;column++){let difference=(features[featuresOffset+row*${columns}u+column]-means[column])/scales[column]-centers[previous*${columns}u+column];squared+=difference*difference;}if(!isFiniteBits(squared)){state[stateOffset]=1.0;return;}nearest=min(nearest,squared);}if(nearest>farthest){farthest=nearest;seedRow=row;}}if(!(farthest>0.0)){state[stateOffset]=2.0;return;}}
 for(var column=0u;column<${columns}u;column++){centers[cluster*${columns}u+column]=(features[featuresOffset+seedRow*${columns}u+column]-means[column])/scales[column];}}
var converged=false;var objective=0.0;
for(var iteration=0u;iteration<${design.maximumIterations ?? 100}u;iteration++){var sums:array<f32,${clusters * columns}>;var counts:array<u32,${clusters}>;var changed=false;objective=0.0;
for(var row=0u;row<${rows}u;row++){if(!complete(row)){continue;}var nearest=3.402823e38;var selected=0u;for(var cluster=0u;cluster<${clusters}u;cluster++){var squared=0.0;for(var column=0u;column<${columns}u;column++){let difference=(features[featuresOffset+row*${columns}u+column]-means[column])/scales[column]-centers[cluster*${columns}u+column];squared+=difference*difference;}if(!isFiniteBits(squared)){state[stateOffset]=1.0;return;}if(squared<nearest){nearest=squared;selected=cluster;}}
changed=changed||assignments[row]!=selected;assignments[row]=selected;objective+=nearest;counts[selected]++;for(var column=0u;column<${columns}u;column++){sums[selected*${columns}u+column]+=(features[featuresOffset+row*${columns}u+column]-means[column])/scales[column];}}
for(var cluster=0u;cluster<${clusters}u;cluster++){if(counts[cluster]==0u){state[stateOffset]=2.0;return;}for(var column=0u;column<${columns}u;column++){centers[cluster*${columns}u+column]=sums[cluster*${columns}u+column]/f32(counts[cluster]);}}
state[stateOffset+1u]=f32(iteration+1u);if(!changed){converged=true;break;}}
if(!isFiniteBits(objective)){state[stateOffset]=1.0;return;}state[stateOffset]=select(3.0,0.0,converged);state[stateOffset+2u]=objective;if(converged){for(var row=0u;row<${rows}u;row++){folds[foldsOffset+row]=assignments[row];}}`;
  } else {
    const predictions = design.predictionPositions.length;
    bindings.push({
      name: 'prediction',
      view: design.predictionPositions,
      type: 'f32',
      access: 'read'
    });
    body += `var predictionDistances:array<f32,${predictions}>; for(var row=0u;row<${rows}u;row++){if(!valid(row)){state[stateOffset]=1.0;return;}}
for(var sample=0u;sample<${predictions}u;sample++){let point=vec2<f32>(prediction[predictionOffset+sample*2u],prediction[predictionOffset+sample*2u+1u]);if(!isFiniteBits(point.x)||!isFiniteBits(point.y)){state[stateOffset]=1.0;return;}var nearest=3.402823e38;for(var row=0u;row<${rows}u;row++){let separation=distance(point,location(row));if(!isFiniteBits(separation)){state[stateOffset]=1.0;return;}nearest=min(nearest,separation);}predictionDistances[sample]=nearest;}`;
    if (design.mode === 'nndm') {
      const distances = createTransientView(graph, `${id}-nndm-distances`, 'float32', rows * rows);
      bindings.push({name: 'distances', view: distances, type: 'f32', access: 'read_write'});
      body += `var nearest:array<f32,${rows}>;var neighbors:array<u32,${rows}>;var remaining:array<u32,${rows}>;
for(var row=0u;row<${rows}u;row++){nearest[row]=3.402823e38;remaining[row]=${rows - 1}u;for(var other=0u;other<${rows}u;other++){let value=select(rowDistance(row,other),-1.0,row==other);if(!isFiniteBits(value)){state[stateOffset]=1.0;return;}distances[distancesOffset+row*${rows}u+other]=value;if(value>=0.0&&value<nearest[row]){nearest[row]=value;neighbors[row]=other;}}}
var radius=3.402823e38;var chosen=0u;for(var row=0u;row<${rows}u;row++){if(nearest[row]<radius){radius=nearest[row];chosen=row;}}
for(var iteration=0u;iteration<${rows * rows + rows}u&&radius<=${getWGSLFloatLiteral(design.maximumDistance)};iteration++){
var trainingBelow=0u;var predictionBelow=0u;for(var row=0u;row<${rows}u;row++){trainingBelow+=u32(nearest[row]<=radius);}for(var sample=0u;sample<${predictions}u;sample++){predictionBelow+=u32(predictionDistances[sample]<=radius);}
let remove=f32(trainingBelow-1u)/${getWGSLFloatLiteral(rows)}>=f32(predictionBelow)/${getWGSLFloatLiteral(predictions)}&&f32(remaining[chosen])/${getWGSLFloatLiteral(rows)}>${getWGSLFloatLiteral(design.minimumTrainingFraction ?? 0.05)};
if(remove){distances[distancesOffset+chosen*${rows}u+neighbors[chosen]]=-1.0;remaining[chosen]--;nearest[chosen]=3.402823e38;for(var other=0u;other<${rows}u;other++){let value=distances[distancesOffset+chosen*${rows}u+other];if(value>=0.0&&value<nearest[chosen]){nearest[chosen]=value;neighbors[chosen]=other;}}}else if(trainingBelow==${rows}u){break;}
var next=3.402823e38;for(var row=0u;row<${rows}u;row++){if(select(nearest[row]>radius,nearest[row]>=radius,remove)&&nearest[row]<next){next=nearest[row];chosen=row;}}radius=next;state[stateOffset+1u]=f32(iteration+1u);}
for(var row=0u;row<${rows}u;row++){folds[foldsOffset+row]=row;state[stateOffset+row+4u]=nearest[row];}`;
    } else {
      bindings.push({name: 'candidates', view: design.candidates!, type: 'u32', access: 'read'});
      body += `var nearest:array<f32,${rows}>;var best=3.402823e38;var selected=0xffffffffu;
for(var candidate=0u;candidate<${design.candidateCount ?? 8}u;candidate++){var counts:array<u32,${props.foldCount}>;var validCandidate=true;for(var row=0u;row<${rows}u;row++){let fold=candidates[candidatesOffset+candidate*${rows}u+row];if(fold>=${props.foldCount}u){validCandidate=false;break;}counts[fold]++;var value=3.402823e38;for(var other=0u;other<${rows}u;other++){if(candidates[candidatesOffset+candidate*${rows}u+other]!=fold){let separation=rowDistance(row,other);if(!isFiniteBits(separation)){state[stateOffset]=1.0;return;}value=min(value,separation);}}nearest[row]=value;}
for(var fold=0u;fold<${props.foldCount}u;fold++){validCandidate=validCandidate&&counts[fold]>0u&&f32(counts[fold])<=${getWGSLFloatLiteral((design.maximumFoldFraction ?? 0.8) * rows)};}if(!validCandidate){continue;}
var previous=0.0;var objective=0.0;for(var step=0u;step<${rows + predictions}u;step++){var next=3.402823e38;var trainBelow=0u;var predictionBelow=0u;for(var row=0u;row<${rows}u;row++){trainBelow+=u32(nearest[row]<=previous);if(nearest[row]>previous){next=min(next,nearest[row]);}}for(var sample=0u;sample<${predictions}u;sample++){predictionBelow+=u32(predictionDistances[sample]<=previous);if(predictionDistances[sample]>previous){next=min(next,predictionDistances[sample]);}}if(next==3.402823e38){break;}objective+=abs(f32(trainBelow)/${getWGSLFloatLiteral(rows)}-f32(predictionBelow)/${getWGSLFloatLiteral(predictions)})*(next-previous);previous=next;}
if(objective<best){best=objective;selected=candidate;}}
state[stateOffset+1u]=${getWGSLFloatLiteral(design.candidateCount ?? 8)};if(selected==0xffffffffu){state[stateOffset]=2.0;return;}state[stateOffset+2u]=best;state[stateOffset+3u]=f32(selected);for(var row=0u;row<${rows}u;row++){folds[foldsOffset+row]=candidates[candidatesOffset+selected*${rows}u+row];}`;
    }
  }
  return {
    state,
    nodes: [
      createWGSLKernelNode(graph, {
        id: `${id}-design`,
        operation: 'GPUSpatialCrossValidation',
        variant: design.mode,
        invocationCount: 1,
        bindings,
        declarations: `${declarations}\n${extraDeclarations}`,
        body
      })
    ]
  };
}
