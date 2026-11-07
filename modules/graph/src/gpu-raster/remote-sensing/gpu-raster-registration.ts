// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPURasterMetadata} from '../types';
import {validateRasterGrid} from '../raster-band-utils';
import {
  getMaskedRegistrationNodes,
  validateMaskedRegistrationProps
} from './raster-masked-registration';
import {
  GPUFFT2D,
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma-contributors/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../utils/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../utils/gpu-contributor-utils';
import {RASTER_FINITE_WGSL, rasterBinding, validateRasterCount} from './raster-kernel-utils';

export type GPURasterRegistrationProps = {
  id?: string;
  grid: GPURasterMetadata;
  reference: GraphDataView<'float32'>;
  moving: GraphDataView<'float32'>;
  /** Default periodic phase correlation, or zero-padded masked normalized correlation. */
  mode?: 'periodic' | 'masked-linear';
  referenceValidity?: GraphDataView<'uint32'>;
  movingValidity?: GraphDataView<'uint32'>;
  /** Masked mode searches these nonnegative integer column/row shift bounds. */
  maximumShift?: readonly [number, number];
  minimumOverlapFraction?: number;
  minimumOverlapCount?: number;
  minimumCorrelation?: number;
  /** Integer refinement 1–32; precision is 1/upsampleFactor pixel. */
  upsampleFactor?: number;
  /** Relative difference from a nonadjacent peak needed for an unambiguous alignment. */
  ambiguityTolerance?: number;
  output: {
    shift: GraphDataView<'float32x2'>;
    peak: GraphDataView<'float32'>;
    status: GraphDataView<'uint32'>;
    /** Optional masked-mode paired-valid support at the accepted refined shift. */
    overlap?: GraphDataView<'uint32'>;
  };
};
function complexBinding(
  name: string,
  view: GraphDataView<'float32x2'>,
  write = false
): WGSLKernelBinding {
  return {name, view, type: 'f32', access: write ? 'read_write' : 'read'};
}
/** Lower complex views to scalar bindings while retaining complex indexing in the kernel body. */
function createRegistrationKernel<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
): GPUCommandNode<Parameters> {
  let body = props.body;
  let declarations = props.declarations ?? '';
  const bindings = props.bindings.map(binding => {
    if (binding.view.format !== 'float32x2') return binding;
    const {name, view} = binding;
    const pattern = `${name}\\[${name}Offset(?: \\+ ([^\\]]+))?\\]`;
    if (binding.access === 'read_write') {
      body = body.replace(
        new RegExp(`${pattern} = ([^;]+);`, 'g'),
        (_match, index: string | undefined, value: string) =>
          `write${name}(${index ?? '0u'}, ${value});`
      );
      declarations += `\nfn write${name}(index: u32, value: vec2<f32>) { ${name}[${name}Offset + index * 2u] = value.x; ${name}[${name}Offset + index * 2u + 1u] = value.y; }`;
    }
    body = body.replace(
      new RegExp(pattern, 'g'),
      (_match, index: string | undefined) => `read${name}(${index ?? '0u'})`
    );
    declarations += `\nfn read${name}(index: u32) -> vec2<f32> { return vec2<f32>(${name}[${name}Offset + index * 2u], ${name}[${name}Offset + index * 2u + 1u]); }`;
    return {
      ...binding,
      view: graph.createDataView(view.buffer, {
        format: 'float32',
        byteOffset: view.byteOffset,
        length: view.length * 2
      })
    };
  });
  return createWGSLKernelNode(graph, {...props, bindings, declarations, body});
}

/** Periodic unmasked phase correlation with separable localized upsampled inverse DFT.
 * Output (column,row) is the translation to apply to moving to align reference. Dimensions
 * are radix-two 2–2048. Status 1=nonfinite/flat input or unusable spectrum,
 * 2=ambiguous nonadjacent peak.
 * Masked-linear mode instead uses padded normalized cross-correlation and bilinear
 * subpixel refinement, accepts dimensions 2–512, and returns status 3 for inadequate support.
 */
export class GPURasterRegistration implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterRegistrationProps) {
    this.id = props.id ?? 'raster-registration';
    validateRasterGrid(this.id, props.grid);
    assert(['periodic', 'masked-linear'].includes(props.mode ?? 'periodic'));
    if (props.mode === 'masked-linear') {
      validateMaskedRegistrationProps(props, this.id);
      return;
    }
    assert(
      !props.referenceValidity &&
        !props.movingValidity &&
        !props.maximumShift &&
        !props.output.overlap
    );
    assert(
      props.minimumOverlapFraction === undefined &&
        props.minimumOverlapCount === undefined &&
        props.minimumCorrelation === undefined
    );
    for (const dimension of [props.grid.width, props.grid.height]) {
      validateRasterCount(dimension, 2048);
      assert(dimension >= 2 && (dimension & (dimension - 1)) === 0);
    }
    validateRasterCount(props.upsampleFactor ?? 20, 32);
    assert(
      Number.isFinite(props.ambiguityTolerance ?? 1e-4) &&
        (props.ambiguityTolerance ?? 1e-4) >= 0 &&
        (props.ambiguityTolerance ?? 1e-4) < 1
    );
    assert(
      props.reference.length >= props.grid.width * props.grid.height &&
        props.moving.length >= props.grid.width * props.grid.height
    );
    for (const view of [
      props.reference,
      props.moving,
      ...Object.values(props.output).filter(view => view !== undefined)
    ]) {
      validatePackedView(view, [view.format], this.id);
      assert(view.length > 0);
    }
    const outputs = Object.values(props.output).filter(view => view !== undefined);
    validateGraphOutputsDisjointFromInputs(this.id, outputs, [props.reference, props.moving]);
    outputs.forEach((view, index) =>
      validateGraphOutputsDisjointFromInputs(this.id, [view], outputs.slice(index + 1))
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    if (props.mode === 'masked-linear') return getMaskedRegistrationNodes(graph, props, id);
    const count = props.grid.width * props.grid.height;
    const upsample = props.upsampleFactor ?? 20;
    const radius = Math.ceil(upsample / 2);
    const candidates = radius * 2 + 1;
    validateGraphViewsBelongToGraph(id, graph, [
      props.reference,
      props.moving,
      ...Object.values(props.output).filter(view => view !== undefined)
    ]);
    const reference = createTransientView(graph, `${id}-reference`, 'float32x2', count);
    const moving = createTransientView(graph, `${id}-moving`, 'float32x2', count);
    const referenceSpectrum = createTransientView(
      graph,
      `${id}-reference-spectrum`,
      'float32x2',
      count
    );
    const movingSpectrum = createTransientView(graph, `${id}-moving-spectrum`, 'float32x2', count);
    const crossPower = createTransientView(graph, `${id}-cross-power`, 'float32x2', count);
    const spectralScale = createTransientView(graph, `${id}-spectral-scale`, 'float32', 1);
    const correlation = createTransientView(graph, `${id}-correlation`, 'float32x2', count);
    const coarse = createTransientView(graph, `${id}-coarse`, 'float32x2', 1);
    const horizontal = createTransientView(
      graph,
      `${id}-horizontal`,
      'float32x2',
      candidates * props.grid.height
    );
    const refined = createTransientView(graph, `${id}-refined`, 'float32', candidates * candidates);
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createRegistrationKernel(graph, {
        id: `${id}-pack`,
        operation: 'GPURasterRegistration',
        invocationCount: count,
        bindings: [
          rasterBinding('referenceValues', props.reference),
          rasterBinding('movingValues', props.moving),
          complexBinding('reference', reference, true),
          complexBinding('moving', moving, true)
        ],
        body: `reference[referenceOffset + index] = vec2<f32>(referenceValues[referenceValuesOffset + index], 0.0); moving[movingOffset + index] = vec2<f32>(movingValues[movingValuesOffset + index], 0.0);`
      })
    );
    nodes.push(
      ...new GPUFFT2D({
        id: `${id}-reference-fft`,
        input: reference,
        output: referenceSpectrum,
        width: props.grid.width,
        height: props.grid.height
      }).getCommandNodes(graph)
    );
    nodes.push(
      ...new GPUFFT2D({
        id: `${id}-moving-fft`,
        input: moving,
        output: movingSpectrum,
        width: props.grid.width,
        height: props.grid.height
      }).getCommandNodes(graph)
    );
    nodes.push(
      createRegistrationKernel(graph, {
        id: `${id}-spectral-scale`,
        operation: 'GPURasterRegistration',
        invocationCount: 1,
        bindings: [
          complexBinding('reference', referenceSpectrum),
          complexBinding('moving', movingSpectrum),
          rasterBinding('spectralScale', spectralScale, true)
        ],
        body: `var maximum = 0.0; for (var cell = 1u; cell < ${count}u; cell++) { maximum = max(maximum, length(reference[referenceOffset + cell]) * length(moving[movingOffset + cell])); } spectralScale[spectralScaleOffset] = maximum;`
      })
    );
    nodes.push(
      createRegistrationKernel(graph, {
        id: `${id}-cross-power`,
        operation: 'GPURasterRegistration',
        invocationCount: count,
        bindings: [
          complexBinding('reference', referenceSpectrum),
          complexBinding('moving', movingSpectrum),
          complexBinding('crossPower', crossPower, true),
          rasterBinding('spectralScale', spectralScale)
        ],
        body: `let first = reference[referenceOffset + index]; let second = moving[movingOffset + index]; let product = vec2<f32>(dot(first, second), first.y * second.x - first.x * second.y); let magnitude = length(product); crossPower[crossPowerOffset + index] = select(vec2<f32>(0.0), product / max(magnitude, 1e-30), magnitude > max(1e-20, spectralScale[spectralScaleOffset] * 1e-6) && index != 0u);`
      })
    );
    nodes.push(
      ...new GPUFFT2D({
        id: `${id}-inverse`,
        input: crossPower,
        output: correlation,
        direction: 'inverse',
        width: props.grid.width,
        height: props.grid.height
      }).getCommandNodes(graph)
    );
    nodes.push(
      createRegistrationKernel(graph, {
        id: `${id}-coarse-peak`,
        operation: 'GPURasterRegistration',
        invocationCount: 1,
        bindings: [
          rasterBinding('reference', props.reference),
          rasterBinding('moving', props.moving),
          complexBinding('correlation', correlation),
          complexBinding('coarse', coarse, true),
          rasterBinding('status', props.output.status, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var maximum = -3.402823466e38; var selected = 0u; var valid = true; var referenceLow = 3.402823466e38; var referenceHigh = -3.402823466e38; var movingLow = referenceLow; var movingHigh = referenceHigh;
for (var cell = 0u; cell < ${count}u; cell++) { let first = reference[referenceOffset + cell]; let second = moving[movingOffset + cell]; valid = valid && finite(first) && finite(second); referenceLow = min(referenceLow, first); referenceHigh = max(referenceHigh, first); movingLow = min(movingLow, second); movingHigh = max(movingHigh, second); let value = correlation[correlationOffset + cell].x; if (value > maximum) { maximum = value; selected = cell; } }
let position = vec2<u32>(selected % ${props.grid.width}u, selected / ${props.grid.width}u); var secondary = -3.402823466e38;
for (var cell = 0u; cell < ${count}u; cell++) { let other = vec2<u32>(cell % ${props.grid.width}u, cell / ${props.grid.width}u); let distance = vec2<u32>(abs(vec2<i32>(other) - vec2<i32>(position))); let wrapped = min(distance, vec2<u32>(${props.grid.width}u, ${props.grid.height}u) - distance); if (any(wrapped > vec2<u32>(1u))) { secondary = max(secondary, correlation[correlationOffset + cell].x); } }
// Removing DC makes every nonempty finite correlation field have a positive maximum.
// A zero field has no alignment evidence, even when all peaks are adjacent on a 2x2 grid.
var code = 0u; if (!valid || referenceLow == referenceHigh || movingLow == movingHigh || !finite(maximum) || maximum <= 0.0) { code = 1u; } else if (maximum - secondary <= abs(maximum) * ${props.ambiguityTolerance ?? 1e-4}) { code = 2u; }
status[statusOffset] = code; coarse[coarseOffset] = vec2<f32>(f32(select(i32(position.x), i32(position.x) - ${props.grid.width}, position.x > ${props.grid.width / 2}u)), f32(select(i32(position.y), i32(position.y) - ${props.grid.height}, position.y > ${props.grid.height / 2}u)));`
      })
    );
    nodes.push(
      createRegistrationKernel(graph, {
        id: `${id}-horizontal-refinement`,
        operation: 'GPURasterRegistration',
        invocationCount: candidates * props.grid.height,
        bindings: [
          complexBinding('crossPower', crossPower),
          complexBinding('coarse', coarse),
          complexBinding('horizontal', horizontal, true)
        ],
        body: `let candidate = index % ${candidates}u; let row = index / ${candidates}u; let shift = coarse[coarseOffset].x + f32(candidate) / ${upsample}.0 - ${radius / upsample}; var sum = vec2<f32>(0.0);
for (var column = 0u; column < ${props.grid.width}u; column++) { let frequency = select(i32(column), i32(column) - ${props.grid.width}, column >= ${props.grid.width / 2}u); let angle = 6.283185307179586 * f32(frequency) * shift / ${props.grid.width}.0; let value = crossPower[crossPowerOffset + row * ${props.grid.width}u + column]; sum += vec2<f32>(value.x * cos(angle) - value.y * sin(angle), value.x * sin(angle) + value.y * cos(angle)); }
horizontal[horizontalOffset + index] = sum;`
      })
    );
    nodes.push(
      createRegistrationKernel(graph, {
        id: `${id}-vertical-refinement`,
        operation: 'GPURasterRegistration',
        invocationCount: candidates * candidates,
        bindings: [
          complexBinding('horizontal', horizontal),
          complexBinding('coarse', coarse),
          rasterBinding('refined', refined, true)
        ],
        body: `let column = index % ${candidates}u; let candidate = index / ${candidates}u; let shift = coarse[coarseOffset].y + f32(candidate) / ${upsample}.0 - ${radius / upsample}; var sum = 0.0;
for (var row = 0u; row < ${props.grid.height}u; row++) { let frequency = select(i32(row), i32(row) - ${props.grid.height}, row >= ${props.grid.height / 2}u); let angle = 6.283185307179586 * f32(frequency) * shift / ${props.grid.height}.0; let value = horizontal[horizontalOffset + row * ${candidates}u + column]; sum += value.x * cos(angle) - value.y * sin(angle); }
refined[refinedOffset + index] = sum / ${count}.0;`
      })
    );
    nodes.push(
      createRegistrationKernel(graph, {
        id: `${id}-refined-peak`,
        operation: 'GPURasterRegistration',
        invocationCount: 1,
        bindings: [
          rasterBinding('refined', refined),
          complexBinding('coarse', coarse),
          rasterBinding('status', props.output.status),
          complexBinding('shift', props.output.shift, true),
          rasterBinding('peak', props.output.peak, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var maximum = -3.402823466e38; var selected = 0u;
for (var candidate = 0u; candidate < ${candidates * candidates}u; candidate++) { let value = refined[refinedOffset + candidate]; if (value > maximum) { maximum = value; selected = candidate; } }
let valid = status[statusOffset] == 0u; let result = coarse[coarseOffset] + vec2<f32>(f32(selected % ${candidates}u), f32(selected / ${candidates}u)) / ${upsample}.0 - vec2<f32>(${radius / upsample});
shift[shiftOffset] = select(vec2<f32>(missing()), result, valid); peak[peakOffset] = select(missing(), maximum, valid);`
      })
    );
    return nodes;
  }
}
