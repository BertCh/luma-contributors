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
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPULocalKriging} from '../spatial-interpolation/gpu-local-kriging';
import {getGPULocalKrigingParameterValues} from '../spatial-interpolation/prediction-covariance';
import {GPULocalRBF} from '../spatial-interpolation/gpu-local-rbf';
import {getRegressionTileLayout} from '../spatial-regression/regression-row-tiles';
import {getPairwiseSumWGSL} from '../spatial-regression/pairwise-sum-wgsl';
import {validateRegressionAllocations} from '../spatial-regression/spatial-likelihood-allocations';
import {GPUSpatialCrossValidation} from './gpu-spatial-cross-validation';
import {
  GPU_NESTED_VALIDATION_REFERENCE_MAXIMUM_ROW_COUNT,
  type GPUNestedInterpolationValidationProps
} from './gpu-nested-interpolation-validation';

/** Four runtime indices occupy a 256-byte slot so every fold and split has identical offsets. */
const NESTED_PARAMETER_SLOT_LENGTH = 64;

/** Tiled scores and metrics; candidate fits retain exact bounded neighborhood search semantics. */
export function getNestedValidationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNestedInterpolationValidationProps
): GPUCommandNode<Parameters>[] {
  const rows = props.response.length;
  const innerCount = props.innerFoldCount ?? 3;
  const candidateCount = props.candidates.length;
  // A single bounded tile retains the original sum order for existing small default calls.
  const {tileRowCount, tileCount} =
    rows <= GPU_NESTED_VALIDATION_REFERENCE_MAXIMUM_ROW_COUNT && props.mode !== 'tiled'
      ? {tileRowCount: rows, tileCount: 1}
      : getRegressionTileLayout(rows);
  const slotCount = props.foldCount * (1 + candidateCount * (innerCount + 1));
  validateGraphChunkBindingRanges(graph, id, {
    positions: props.positions,
    response: props.response,
    training: props.training,
    testing: props.testing,
    innerBlocks: props.innerBlocks,
    ...props.output
  });
  validateRegressionAllocations(graph.device, id, {
    'fold masks': rows * innerCount,
    'inner block order': rows * 2,
    'fold counts': innerCount * 3,
    'fold statuses': innerCount,
    'candidate outer predictions': candidateCount * rows,
    'candidate inner predictions': candidateCount * innerCount * rows,
    'candidate score partials': candidateCount * tileCount * 3,
    'candidate scores': candidateCount * 3,
    'metric partials': tileCount * 9,
    metrics: 6,
    'interpolation observations/results': rows * 4,
    'runtime parameter slot': 4
  });
  for (const candidate of props.candidates) {
    if (candidate.mode === 'ordinary-kriging') {
      validateRegressionAllocations(graph.device, id, {
        'covariance parameters': getGPULocalKrigingParameterValues({
          covariance: candidate.covariance
        }).length
      });
    }
  }
  validateCapacity(
    id,
    'runtime parameter buffer byteLength',
    slotCount * NESTED_PARAMETER_SLOT_LENGTH * 4,
    1,
    graph.device.limits.maxBufferSize
  );
  const parameterValues = new Uint32Array(slotCount * NESTED_PARAMETER_SLOT_LENGTH);
  let nextSlot = 0;
  const foldSlots: number[] = [];
  const stageSlots: number[][][] = [];
  for (let fold = 0; fold < props.foldCount; fold++) {
    foldSlots.push(nextSlot);
    parameterValues[nextSlot++ * NESTED_PARAMETER_SLOT_LENGTH] = fold;
    const candidates: number[][] = [];
    for (let candidate = 0; candidate < candidateCount; candidate++) {
      const splits: number[] = [];
      for (let split = 0; split <= innerCount; split++) {
        splits.push(nextSlot);
        parameterValues.set(
          [
            fold,
            candidate,
            split === innerCount ? 0 : split,
            split === innerCount ? candidate * rows : (candidate * innerCount + split) * rows
          ],
          nextSlot++ * NESTED_PARAMETER_SLOT_LENGTH
        );
      }
      candidates.push(splits);
    }
    stageSlots.push(candidates);
  }
  const upload = createSpatialStatisticsParameterNodes(graph, `${id}-indices`, parameterValues);
  const nodes = [...upload.nodes];
  const parameters = (slot: number) =>
    graph.createDataView(upload.parameters.buffer, {
      format: 'uint32',
      byteOffset: slot * NESTED_PARAMETER_SLOT_LENGTH * 4,
      length: 4
    });
  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: WGSLKernelBinding['type'] = view.format.startsWith('uint32') ? 'u32' : 'f32'
  ): WGSLKernelBinding => ({name, view, access, type});
  const declarations = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`;
  const view = <Format extends 'float32' | 'uint32' | 'float32x4'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const kernel = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    extraDeclarations = ''
  ) =>
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: 'GPUNestedInterpolationValidation',
        variant: name.replace(/\d+/g, 'slot'),
        invocationCount: count,
        workgroupSize: 64,
        bindings,
        declarations: `${declarations}\n${extraDeclarations}`,
        body
      })
    );
  for (let fold = 0; fold < props.foldCount; fold++) {
    const prefix = `fold-${fold}`;
    const foldParameters = parameters(foldSlots[fold]);
    const parameterBinding = binding('parameters', foldParameters);
    const training = view(`${prefix}-training`, 'uint32', rows);
    const testing = view(`${prefix}-testing`, 'uint32', rows);
    const validTraining = view(`${prefix}-valid`, 'uint32', rows);
    const status = view(`${prefix}-status`, 'uint32', 1);
    kernel(
      `${prefix}-initialize`,
      1,
      [binding('status', status, 'read_write')],
      'status[statusOffset] = 0u;'
    );
    kernel(
      `${prefix}-masks`,
      rows,
      [
        parameterBinding,
        binding('sourceTraining', props.training),
        binding('sourceTesting', props.testing),
        binding('training', training, 'read_write'),
        binding('testing', testing, 'read_write'),
        binding('status', status, 'read_write', 'atomic<u32>')
      ],
      `let source = parameters[parametersOffset] * ${rows}u + index;
let train = sourceTraining[sourceTrainingOffset + source];
let test = sourceTesting[sourceTestingOffset + source];
training[trainingOffset + index] = train;
testing[testingOffset + index] = test;
if (train != 0u && test != 0u) { atomicOr(&status[statusOffset], 1u); }`
    );
    kernel(
      `${prefix}-valid-training`,
      rows,
      [
        binding('training', training),
        binding('response', props.response),
        binding('status', status),
        binding('valid', validTraining, 'read_write')
      ],
      'valid[validOffset + index] = u32(training[trainingOffset + index] != 0u && status[statusOffset] == 0u && isFiniteBits(response[responseOffset + index]));'
    );
    const inner = {
      folds: view(`${prefix}-folds`, 'uint32', rows),
      training: view(`${prefix}-inner-training`, 'uint32', rows * innerCount),
      testing: view(`${prefix}-inner-testing`, 'uint32', rows * innerCount),
      counts: view(`${prefix}-counts`, 'uint32', innerCount * 3),
      status: view(`${prefix}-inner-status`, 'uint32', innerCount)
    };
    nodes.push(
      ...new GPUSpatialCrossValidation({
        id: `${id}-${prefix}-inner`,
        positions: props.positions,
        blocks: props.innerBlocks,
        mask: validTraining,
        foldCount: innerCount,
        seed: props.seed,
        bufferDistance: props.innerBufferDistance,
        output: inner
      }).getCommandNodes(graph)
    );
    const outerPredictions = view(`${prefix}-outer-predictions`, 'float32', candidateCount * rows);
    const innerPredictions = view(
      `${prefix}-inner-predictions`,
      'float32',
      candidateCount * innerCount * rows
    );
    for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex++) {
      const candidate = props.candidates[candidateIndex];
      for (let split = 0; split <= innerCount; split++) {
        const stage = `${prefix}-candidate-${candidateIndex}-split-${split}`;
        const stageParameters = binding(
          'parameters',
          parameters(stageSlots[fold][candidateIndex][split])
        );
        const values = view(
          `${stage}-values`,
          candidate.mode === 'ordinary-kriging' ? 'float32x4' : 'float32',
          rows
        );
        const result = view(`${stage}-result`, 'float32x4', rows);
        const resultStatus = view(`${stage}-status`, 'uint32', rows);
        kernel(
          `${stage}-mask`,
          rows,
          [
            stageParameters,
            binding('mask', split === innerCount ? validTraining : inner.training),
            binding('response', props.response),
            binding('values', values, 'read_write')
          ],
          candidate.mode === 'ordinary-kriging'
            ? `values[valuesOffset + index * 4u] = response[responseOffset + index]; values[valuesOffset + index * 4u + 1u] = 0.0; values[valuesOffset + index * 4u + 2u] = 0.0; values[valuesOffset + index * 4u + 3u] = f32(mask[maskOffset + parameters[parametersOffset + 2u] * ${rows}u + index]);`
            : `values[valuesOffset + index] = select(getNaN(), response[responseOffset + index], mask[maskOffset + parameters[parametersOffset + 2u] * ${rows}u + index] != 0u);`
        );
        if (candidate.mode === 'ordinary-kriging') {
          nodes.push(
            ...new GPULocalKriging({
              id: `${id}-${stage}-fit`,
              positions: props.positions,
              observations: values as GraphDataView<'float32x4'>,
              supports: props.positions,
              mode: 'ordinary',
              covariance: candidate.covariance,
              sampleIndex: 'brute-force',
              neighborCount: props.neighborCount ?? 16,
              output: {values: result, status: resultStatus}
            }).getCommandNodes(graph)
          );
        } else {
          nodes.push(
            ...new GPULocalRBF({
              id: `${id}-${stage}-fit`,
              positions: props.positions,
              values: values as GraphDataView<'float32'>,
              queries: props.positions,
              ...candidate,
              neighborCount: props.neighborCount ?? 16,
              output: {values: result, status: resultStatus}
            }).getCommandNodes(graph)
          );
        }
        kernel(
          `${stage}-collect`,
          rows,
          [
            stageParameters,
            binding('result', result),
            binding('status', resultStatus),
            binding(
              'predictions',
              split === innerCount ? outerPredictions : innerPredictions,
              'read_write'
            )
          ],
          'predictions[predictionsOffset + parameters[parametersOffset + 3u] + index] = select(getNaN(), result[resultOffset + index * 4u], status[statusOffset + index] == 0u);'
        );
      }
    }
    const scorePartials = view(
      `${prefix}-score-partials`,
      'float32',
      candidateCount * tileCount * 3
    );
    const scores = view(`${prefix}-scores`, 'float32', candidateCount * 3);
    const selected = view(`${prefix}-selected`, 'uint32', 1);
    kernel(
      `${prefix}-score-tiles`,
      candidateCount * tileCount,
      [
        binding('response', props.response),
        binding('testing', inner.testing),
        binding('predictions', innerPredictions),
        binding('partial', scorePartials, 'read_write')
      ],
      `let candidate = index / ${tileCount}u; let tile = index % ${tileCount}u;
var squares = 0.0; var count = 0u; var missing = 0u;
for (var split = 0u; split < ${innerCount}u; split++) { for (var row = tile * ${tileRowCount}u; row < min((tile + 1u) * ${tileRowCount}u, ${rows}u); row++) {
if (testing[testingOffset + split * ${rows}u + row] == 0u) { continue; }
let predicted = predictions[predictionsOffset + (candidate * ${innerCount}u + split) * ${rows}u + row]; let observed = response[responseOffset + row];
if (!isFiniteBits(predicted) || !isFiniteBits(observed)) { missing++; continue; }
let error = predicted - observed; squares += error * error; count++;
} }
partial[partialOffset + index * 3u] = squares; partial[partialOffset + index * 3u + 1u] = f32(count); partial[partialOffset + index * 3u + 2u] = f32(missing);`
    );
    kernel(
      `${prefix}-score-merge`,
      candidateCount,
      [binding('partial', scorePartials), binding('scores', scores, 'read_write')],
      `${getPairwiseSumWGSL({count: `${tileCount}u`, name: 'squares', term: `partial[partialOffset + (index * ${tileCount}u + squaresItem) * 3u]`})}
var count = 0.0; var missing = 0.0; for (var tile = 0u; tile < ${tileCount}u; tile++) { count += partial[partialOffset + (index * ${tileCount}u + tile) * 3u + 1u]; missing += partial[partialOffset + (index * ${tileCount}u + tile) * 3u + 2u]; }
scores[scoresOffset + index * 3u] = select(getNaN(), squares / count, count > 0.0 && missing == 0.0); scores[scoresOffset + index * 3u + 1u] = count; scores[scoresOffset + index * 3u + 2u] = missing;`
    );
    kernel(
      `${prefix}-select`,
      1,
      [
        binding('scores', scores),
        binding('selected', selected, 'read_write'),
        binding('status', status, 'read_write')
      ],
      `var best = 3.402823e38; var chosen = 0xffffffffu;
for (var candidate = 0u; candidate < ${candidateCount}u; candidate++) { let score = scores[scoresOffset + candidate * 3u]; if (isFiniteBits(score) && (chosen == 0xffffffffu || score < best)) { best = score; chosen = candidate; } }
selected[selectedOffset] = chosen; if (chosen == 0xffffffffu) { status[statusOffset] |= 2u; }`
    );
    const predictions = view(`${prefix}-predictions`, 'float32', rows);
    kernel(
      `${prefix}-predict`,
      rows,
      [
        binding('selected', selected),
        binding('testing', testing),
        binding('candidates', outerPredictions),
        binding('predictions', predictions, 'read_write')
      ],
      `let chosen = selected[selectedOffset]; var value = getNaN(); if (chosen < ${candidateCount}u && testing[testingOffset + index] != 0u) { value = candidates[candidatesOffset + chosen * ${rows}u + index]; } predictions[predictionsOffset + index] = value;`
    );
    const metricPartials = view(`${prefix}-metric-partials`, 'float32', tileCount * 9);
    const metrics = view(`${prefix}-metrics`, 'float32', 6);
    kernel(
      `${prefix}-metric-tiles`,
      tileCount,
      [
        binding('response', props.response),
        binding('testing', testing),
        binding('predictions', predictions),
        binding('partial', metricPartials, 'read_write')
      ],
      `var count = 0u; var missing = 0u; var squares = 0.0; var absolute = 0.0; var bias = 0.0; var mean = 0.0; var total = 0.0;
for (var row = index * ${tileRowCount}u; row < min((index + 1u) * ${tileRowCount}u, ${rows}u); row++) {
if (testing[testingOffset + row] == 0u) { continue; } let predicted = predictions[predictionsOffset + row]; let observed = response[responseOffset + row];
if (!isFiniteBits(predicted) || !isFiniteBits(observed)) { missing++; continue; }
count++; let error = predicted - observed; squares += error * error; absolute += abs(error); bias += error; let delta = observed - mean; mean += delta / f32(count); total += delta * (observed - mean);
}
let start = partialOffset + index * 9u; partial[start] = f32(count); partial[start + 1u] = f32(missing); partial[start + 2u] = squares; partial[start + 3u] = absolute; partial[start + 4u] = bias; partial[start + 5u] = mean; partial[start + 6u] = total;`
    );
    kernel(
      `${prefix}-metric-merge`,
      1,
      [
        binding('partial', metricPartials),
        binding('metrics', metrics, 'read_write'),
        binding('status', status, 'read_write')
      ],
      `var count = 0.0; var missing = 0.0; var squares = 0.0; var absolute = 0.0; var bias = 0.0; var mean = 0.0; var total = 0.0;
for (var tile = 0u; tile < ${tileCount}u; tile++) { let start = partialOffset + tile * 9u; let tileCount = partial[start]; missing += partial[start + 1u]; squares += partial[start + 2u]; absolute += partial[start + 3u]; bias += partial[start + 4u]; if (tileCount == 0.0) { continue; } if (count == 0.0) { count = tileCount; mean = partial[start + 5u]; total = partial[start + 6u]; continue; } let difference = partial[start + 5u] - mean; let combined = count + tileCount; total += partial[start + 6u] + difference * difference * (count / combined) * tileCount; mean += difference * (tileCount / combined); count = combined; }
let determination = 1.0 - squares / total;
let failed = !isFiniteBits(squares) || !isFiniteBits(absolute) || !isFiniteBits(bias) || !isFiniteBits(mean) || !isFiniteBits(total) || (total > 0.0 && !isFiniteBits(determination));
metrics[metricsOffset] = count; metrics[metricsOffset + 1u] = missing; metrics[metricsOffset + 2u] = select(getNaN(), squares / count, count > 0.0 && !failed); metrics[metricsOffset + 3u] = select(getNaN(), absolute / count, count > 0.0 && !failed); metrics[metricsOffset + 4u] = select(getNaN(), bias / count, count > 0.0 && !failed); metrics[metricsOffset + 5u] = select(getNaN(), determination, total > 0.0 && !failed);
status[statusOffset] |= select(0u, 4u, missing > 0.0) | select(0u, 8u, count + missing == 0.0) | select(0u, 16u, failed);`
    );
    for (const [name, source, target, length] of [
      ['selected', selected, props.output.selected, 1],
      ['status', status, props.output.status, 1],
      ['scores', scores, props.output.scores, candidateCount * 3],
      ['predictions', predictions, props.output.predictions, rows],
      ['metrics', metrics, props.output.metrics, 6]
    ] as const) {
      kernel(
        `${prefix}-publish-${name}`,
        length,
        [parameterBinding, binding('source', source), binding('destination', target, 'read_write')],
        `destination[destinationOffset + parameters[parametersOffset] * ${length}u + index] = source[sourceOffset + index];`
      );
    }
  }
  return nodes;
}
