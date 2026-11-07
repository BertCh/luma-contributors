// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonStudyWindowViews} from '../../analysis-core/contracts/study-window';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {
  getStudyWindowBinding,
  getStudyWindowWGSL,
  getStudyWindowWGSLOptions
} from '../../analysis-core/kernels/study-window-wgsl';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUInhomogeneousPointPatternProps} from './gpu-inhomogeneous-point-pattern';
import {getPolygonTranslationOverlapWGSL} from './polygon-translation-overlap';

/** Resident unordered-pair geometry followed by parallel event/radius partial reductions. */
export function createTranslationPointPatternNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUInhomogeneousPointPatternProps,
  id: string,
  boundary: GraphDataView<'float32'>,
  window: GPUPolygonStudyWindowViews
) {
  const rows = props.positions.length;
  const pairCount = (rows * (rows - 1)) / 2;
  const radiusCount = props.radii.length;
  const pairs = createTransientView(
    graph,
    `${id}-translation-pairs`,
    'float32',
    Math.max(2, 2 * pairCount)
  );
  const partials = createTransientView(
    graph,
    `${id}-translation-partials`,
    'float32',
    rows * radiusCount * 3
  );
  const correlation = props.pairCorrelation;
  const bandwidth = Math.fround(correlation?.bandwidth ?? 1);
  const pointDeclaration = `fn point(row: u32) -> vec2f {
    return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  }`;
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-translation-overlaps`,
      operation: 'GPUInhomogeneousPointPattern',
      variant: 'translation-overlaps',
      invocationCount: Math.max(1, pairCount),
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'boundary', view: boundary, type: 'f32', access: 'read'},
        {name: 'intensity', view: props.output.intensity, type: 'f32', access: 'read'},
        {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
        getStudyWindowBinding(window),
        {name: 'pairs', view: pairs, type: 'f32', access: 'read_write'}
      ],
      declarations: `${IS_FINITE_FLOAT_WGSL}\n${EXACT_ORIENTATION_WGSL}\n${EXACT_GEOMETRY_WGSL}\n${getStudyWindowWGSL(getStudyWindowWGSLOptions(window))}\n${pointDeclaration}\n${getPolygonTranslationOverlapWGSL(window.edges.length / 4, props.maximumOverlapEdgePairs ?? 65536)}`,
      body: `if (index >= ${pairCount}u) { return; }
  var second = u32(floor((1.0 + sqrt(1.0 + 8.0 * f32(index))) * 0.5));
  if (second * (second - 1u) / 2u > index) { second--; }
  if ((second + 1u) * second / 2u <= index) { second++; }
  let first = index - second * (second - 1u) / 2u;
  pairs[pairsOffset + 2u * index] = -1.0;
  pairs[pairsOffset + 2u * index + 1u] = 0.0;
  if (boundary[boundaryOffset + first] < 0.0 || boundary[boundaryOffset + second] < 0.0) { return; }
  let displacement = point(second) - point(first);
  let separation = length(displacement);
  pairs[pairsOffset + 2u * index] = separation;
  if (separation > radii[radiiOffset + ${radiusCount - 1}u] + ${correlation ? bandwidth : 0}f) { return; }
  let firstIntensity = intensity[intensityOffset + first];
  let secondIntensity = intensity[intensityOffset + second];
  if (!(firstIntensity > 0.0 && secondIntensity > 0.0) || !isFiniteFloat(firstIntensity) || !isFiniteFloat(secondIntensity)) { return; }
  let overlap = polygonTranslationOverlap(displacement);
  var weight = -4.0;
  if (overlap == -1.0) { weight = -3.0; }
  if (overlap > 0.0) {
    weight = (1.0 / firstIntensity) / secondIntensity / overlap;
    if (!isFiniteFloat(weight)) { weight = -4.0; }
  }
  pairs[pairsOffset + 2u * index + 1u] = weight;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-translation-partials`,
      operation: 'GPUInhomogeneousPointPattern',
      variant: correlation ? 'kernel-pcf-partials' : 'translation-k-partials',
      invocationCount: rows * radiusCount,
      bindings: [
        {name: 'boundary', view: boundary, type: 'f32', access: 'read'},
        {name: 'intensity', view: props.output.intensity, type: 'f32', access: 'read'},
        {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
        {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
        {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
      ],
      declarations: `${IS_FINITE_FLOAT_WGSL}
fn correlationKernel(offset: f32) -> f32 {
  let standardized = offset / ${bandwidth}f;
  return 0.75 / ${bandwidth}f * max(0.0, 1.0 - standardized * standardized);
}`,
      body: `let focus = index % ${rows}u;
  let radius = radii[radiiOffset + index / ${rows}u];
  var code = 0.0; var count = 0.0; var correlation = 0.0;
  let lambda = intensity[intensityOffset + focus];
  if (boundary[boundaryOffset + focus] < 0.0) { code = 5.0; }
  else if (!(lambda > 0.0) || !isFiniteFloat(lambda)) { code = 2.0; }
  else {
    for (var other = 0u; other < ${rows}u; other++) {
      if (other == focus) { continue; }
      let second = max(focus, other); let first = min(focus, other);
      let pair = second * (second - 1u) / 2u + first;
      let separation = pairs[pairsOffset + 2u * pair];
      if (separation < 0.0 || separation > radius + ${correlation ? bandwidth : 0}f) { continue; }
      let weight = pairs[pairsOffset + 2u * pair + 1u];
      if (weight < 0.0) { code = max(code, -weight); continue; }
      if (separation <= radius) { count += weight; }
      ${
        correlation
          ? `
      let divisor = ${correlation.divisor === 'distance' ? 'separation' : 'radius'};
      if (!(divisor > 0.0)) { code = max(code, 2.0); continue; }
      var kernel = correlationKernel(radius - separation);
      ${correlation.zeroCorrection === 'reflection' ? 'kernel += correlationKernel(radius + separation);' : ''}
      correlation += weight * kernel / (6.283185307179586 * divisor);`
          : ''
      }
    }
  }
  partials[partialsOffset + index * 3u] = count;
  partials[partialsOffset + index * 3u + 1u] = correlation;
  partials[partialsOffset + index * 3u + 2u] = code;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-translation-reduce`,
      operation: 'GPUInhomogeneousPointPattern',
      variant: correlation ? 'kernel-pcf-reduce' : 'translation-k-reduce',
      invocationCount: radiusCount * 256,
      workgroupSize: 256,
      guardIndex: false,
      bindings: [
        {name: 'partials', view: partials, type: 'f32', access: 'read'},
        {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
        {name: 'values', view: props.output.k, type: 'f32', access: 'read_write'},
        {
          name: 'correlation',
          view: props.output.pairCorrelation,
          type: 'f32',
          access: 'read_write'
        },
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: `${IS_FINITE_FLOAT_WGSL}
var<workgroup> partialCounts: array<f32, 256>;
var<workgroup> partialCorrelations: array<f32, 256>;
var<workgroup> partialCodes: array<u32, 256>;
var<workgroup> partialIncluded: array<u32, 256>;`,
      body: `if (workgroupIndex >= ${radiusCount}u) { return; }
  let radiusIndex = workgroupIndex;
  var count = 0.0; var pairCorrelation = 0.0; var code = 0u; var included = 0u;
  for (var focus = localInvocationIndex; focus < ${rows}u; focus += 256u) {
    let base = partialsOffset + (radiusIndex * ${rows}u + focus) * 3u;
    let partialCode = u32(partials[base + 2u]);
    if (partialCode == 5u) { continue; }
    included++;
    code = max(code, partialCode);
    count += partials[base]; pairCorrelation += partials[base + 1u];
  }
  partialCounts[localInvocationIndex] = count;
  partialCorrelations[localInvocationIndex] = pairCorrelation;
  partialCodes[localInvocationIndex] = code;
  partialIncluded[localInvocationIndex] = included;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (localInvocationIndex < stride) {
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialCorrelations[localInvocationIndex] += partialCorrelations[localInvocationIndex + stride];
      partialCodes[localInvocationIndex] = max(partialCodes[localInvocationIndex], partialCodes[localInvocationIndex + stride]);
      partialIncluded[localInvocationIndex] += partialIncluded[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex != 0u) { return; }
  count = partialCounts[0]; pairCorrelation = partialCorrelations[0];
  code = partialCodes[0]; included = partialIncluded[0];
  let radius = radii[radiiOffset + radiusIndex];
  if (!(radius > 0.0) || !isFiniteFloat(radius) || included == 0u || !isFiniteFloat(count) || !isFiniteFloat(pairCorrelation)) { code = 2u; }
  for (var radiusRow = 0u; radiusRow < ${radiusCount}u; radiusRow++) {
    let candidate = radii[radiiOffset + radiusRow];
    if (!isFiniteFloat(candidate) || !(candidate > 0.0)) { code = 2u; }
    if (radiusRow > 0u && !(candidate > radii[radiiOffset + radiusRow - 1u])) { code = 2u; }
  }
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  values[valuesOffset + radiusIndex] = select(invalid, count, code == 0u);
  correlation[correlationOffset + radiusIndex] = select(invalid, pairCorrelation, code == 0u);
  status[statusOffset + radiusIndex] = code;`
    }),
    ...(correlation
      ? []
      : [
          createWGSLKernelNode(graph, {
            id: `${id}-translation-annular`,
            operation: 'GPUInhomogeneousPointPattern',
            variant: 'translation-annular-pcf',
            invocationCount: radiusCount,
            bindings: [
              {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
              {name: 'values', view: props.output.k, type: 'f32', access: 'read'},
              {
                name: 'correlation',
                view: props.output.pairCorrelation,
                type: 'f32',
                access: 'read_write'
              }
            ],
            body: `var previous = 0.0; var previousRadius = 0.0;
  if (index > 0u) { previous = values[valuesOffset + index - 1u]; previousRadius = radii[radiiOffset + index - 1u]; }
  let radius = radii[radiiOffset + index];
  correlation[correlationOffset + index] = (values[valuesOffset + index] - previous) / (3.141592653589793 * (radius * radius - previousRadius * previousRadius));`
          })
        ])
  ];
}
