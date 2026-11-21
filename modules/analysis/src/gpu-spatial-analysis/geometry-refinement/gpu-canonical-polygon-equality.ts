// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateRefinementOutputLength,
  validateRefinementNumber
} from './refinement-output-validation';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRefinementScratchByteLengths} from './refinement-storage-validation';
import {
  getCanonicalEqualityMetricWGSL,
  validateCanonicalEqualityMetric
} from './canonical-equality-metric';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

export type GPUCanonicalPolygonEqualityProps = {
  id?: string;
  left: GPUPolygonGeometryFields;
  right: GPUPolygonGeometryFields;
  /** Absolute tolerance in normalized coordinate units, default zero. */
  tolerance?: number;
  /** Default per-coordinate preserves the existing coordinate-wise comparison. */
  toleranceMetric?: 'per-coordinate' | 'euclidean';
  /** Equal=1, unequal=0, nonfinite=2 per pair of normalized features. */
  output: GraphDataView<'uint32'>;
};

/** Structural coordinate equality for GPUNormalizePolygons outputs, without hashing. */
export class GPUCanonicalPolygonEquality implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUCanonicalPolygonEqualityProps;
  constructor(props: GPUCanonicalPolygonEqualityProps) {
    this.id = props.id ?? 'canonical-polygon-equality';
    props = {...props, left: {...props.left}, right: {...props.right}};
    this.props = props;
    validateCanonicalEqualityMetric(this.id, props.toleranceMetric);
    validateRefinementNumber(
      {id: this.id, contributor: 'GPUCanonicalPolygonEquality'},
      'tolerance',
      props.tolerance ?? 0
    );

    for (const [side, geometry] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      validateSinglePackedView(
        geometry.positions,
        ['float32x2'],
        {id: this.id, contributor: 'GPUCanonicalPolygonEquality'},
        `${side}.positions`
      );
      for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
        validateSinglePackedView(
          geometry[name],
          ['uint32'],
          {id: this.id, contributor: 'GPUCanonicalPolygonEquality'},
          `${side}.${name}`
        );
      }
    }
    validateSinglePackedView(props.output, ['uint32'], `${this.id} output`, {
      id: this.id,
      contributor: 'GPUCanonicalPolygonEquality'
    });
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPUCanonicalPolygonEquality'},
      'right.featureOffsets',
      props.right.featureOffsets,
      props.left.featureOffsets.length,
      'left.featureOffsets.length'
    );
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPUCanonicalPolygonEquality'},
      'output',
      props.output,
      props.left.featureOffsets.length - 1,
      'left.featureOffsets.length - 1'
    );
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.output],
      [...Object.values(props.left), ...Object.values(props.right)]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [...Object.values(props.left), ...Object.values(props.right), props.output].map(
          (view, index) => [`binding.${index}`, view]
        )
      )
    );
    validateRefinementScratchByteLengths(graph, id, {
      spans: props.output.length * 16
    });
    const spans = createTransientView(graph, `${id}-spans`, 'uint32x4', props.output.length);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-structure`,
        operation: 'GPUCanonicalPolygonEquality',
        variant: 'structure',
        invocationCount: props.output.length,
        bindings: [
          {name: 'leftFeatures', view: props.left.featureOffsets, type: 'u32', access: 'read'},
          {name: 'rightFeatures', view: props.right.featureOffsets, type: 'u32', access: 'read'},
          {name: 'leftPolygons', view: props.left.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'rightPolygons', view: props.right.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'leftRings', view: props.left.ringOffsets, type: 'u32', access: 'read'},
          {name: 'rightRings', view: props.right.ringOffsets, type: 'u32', access: 'read'},
          {name: 'spans', view: spans, type: 'u32', access: 'read_write'}
        ],
        body: `let leftFirst = leftFeatures[leftFeaturesOffset + index]; let rightFirst = rightFeatures[rightFeaturesOffset + index];
let leftCount = leftFeatures[leftFeaturesOffset + index + 1u] - leftFirst; let rightCount = rightFeatures[rightFeaturesOffset + index + 1u] - rightFirst;
var equal = leftCount == rightCount;
for (var polygon = 0u; polygon < min(leftCount, rightCount); polygon++) {
  let leftRing = leftPolygons[leftPolygonsOffset + leftFirst + polygon]; let rightRing = rightPolygons[rightPolygonsOffset + rightFirst + polygon];
  let leftLength = leftPolygons[leftPolygonsOffset + leftFirst + polygon + 1u] - leftRing; let rightLength = rightPolygons[rightPolygonsOffset + rightFirst + polygon + 1u] - rightRing;
  if (leftLength != rightLength) { equal = false; }
  for (var ring = 0u; ring < min(leftLength, rightLength); ring++) {
    if (leftRings[leftRingsOffset + leftRing + ring + 1u] - leftRings[leftRingsOffset + leftRing + ring] != rightRings[rightRingsOffset + rightRing + ring + 1u] - rightRings[rightRingsOffset + rightRing + ring]) { equal = false; }
  }
}
let leftStart = leftRings[leftRingsOffset + leftPolygons[leftPolygonsOffset + leftFirst]];
let rightStart = rightRings[rightRingsOffset + rightPolygons[rightPolygonsOffset + rightFirst]];
let leftEnd = leftRings[leftRingsOffset + leftPolygons[leftPolygonsOffset + leftFirst + leftCount]];
spans[spansOffset + index * 4u] = leftStart; spans[spansOffset + index * 4u + 1u] = rightStart;
spans[spansOffset + index * 4u + 2u] = leftEnd - leftStart; spans[spansOffset + index * 4u + 3u] = select(0u, 1u, equal);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-coordinates`,
        operation: 'GPUCanonicalPolygonEquality',
        variant: 'coordinates',
        invocationCount: props.output.length,
        bindings: [
          {name: 'left', view: props.left.positions, type: 'f32', access: 'read'},
          {name: 'right', view: props.right.positions, type: 'f32', access: 'read'},
          {name: 'spans', view: spans, type: 'u32', access: 'read'},
          {name: 'output', view: props.output, type: 'u32', access: 'read_write'}
        ],
        declarations: getCanonicalEqualityMetricWGSL(props.tolerance ?? 0, props.toleranceMetric),
        body: `var result = spans[spansOffset + index * 4u + 3u];
if (result == 1u) {
  let leftStart = spans[spansOffset + index * 4u] * 2u; let rightStart = spans[spansOffset + index * 4u + 1u] * 2u;
  for (var component = 0u; component < spans[spansOffset + index * 4u + 2u] * 2u; component++) {
    let leftValue = left[leftOffset + leftStart + component]; let rightValue = right[rightOffset + rightStart + component];
    if (((bitcast<u32>(leftValue) & 0x7f800000u) == 0x7f800000u) || ((bitcast<u32>(rightValue) & 0x7f800000u) == 0x7f800000u)) { result = 2u; break; }
    if (component % 2u == 0u && !withinTolerance(vec2f(leftValue,left[leftOffset + leftStart + component + 1u]),vec2f(rightValue,right[rightOffset + rightStart + component + 1u]))) { result = 0u; }
  }
}
output[outputOffset + index] = result;`
      })
    ];
  }
}
