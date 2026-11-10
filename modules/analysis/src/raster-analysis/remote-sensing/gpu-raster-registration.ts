// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
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
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validatePowerOfTwo,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  createRasterArgmaxNodes,
  createRasterCountNodes,
  rasterBinding
} from './raster-kernel-utils';

export {MAXIMUM_MASKED_REGISTRATION_DIMENSION} from './raster-masked-registration';

/** Periodic mode FFT side length: three complex scratch fields of side squared float32x2 must fit one 128 MiB binding (2048 squared is 32 MiB per field). */
export const MAXIMUM_PERIODIC_REGISTRATION_DIMENSION = 2048;
/** Subpixel refinement window: (2 * ceil(factor / 2) + 1) squared candidates are scored per peak, so the factor bounds that dense window. */
export const MAXIMUM_REGISTRATION_UPSAMPLE_FACTOR = 32;

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
    validateRequirement(
      this.id,
      ['periodic', 'masked-linear'].includes(props.mode ?? 'periodic'),
      "mode must be 'periodic' or 'masked-linear'"
    );
    if (props.mode === 'masked-linear') {
      validateMaskedRegistrationProps(props, this.id);
      return;
    }
    validateRequirement(
      this.id,
      !props.referenceValidity &&
        !props.movingValidity &&
        !props.maximumShift &&
        !props.output.overlap,
      "referenceValidity, movingValidity, maximumShift and output.overlap require mode 'masked-linear'"
    );
    validateRequirement(
      this.id,
      props.minimumOverlapFraction === undefined &&
        props.minimumOverlapCount === undefined &&
        props.minimumCorrelation === undefined,
      "minimumOverlapFraction, minimumOverlapCount and minimumCorrelation require mode 'masked-linear'"
    );
    for (const [name, dimension] of [
      ['grid.width', props.grid.width],
      ['grid.height', props.grid.height]
    ] as const) {
      validatePowerOfTwo(this.id, name, dimension, 2, MAXIMUM_PERIODIC_REGISTRATION_DIMENSION);
    }
    validateCapacity(
      this.id,
      'upsampleFactor',
      props.upsampleFactor ?? 20,
      1,
      MAXIMUM_REGISTRATION_UPSAMPLE_FACTOR
    );
    const ambiguityTolerance = props.ambiguityTolerance ?? 1e-4;
    validateRequirement(
      this.id,
      Number.isFinite(ambiguityTolerance) && ambiguityTolerance >= 0 && ambiguityTolerance < 1,
      'ambiguityTolerance must be a finite number in [0, 1)'
    );
    const cellCount = props.grid.width * props.grid.height;
    validateRequirement(
      this.id,
      props.reference.length >= cellCount && props.moving.length >= cellCount,
      'reference and moving must hold at least grid.width times grid.height values'
    );
    for (const view of [
      props.reference,
      props.moving,
      ...Object.values(props.output).filter(view => view !== undefined)
    ]) {
      validatePackedView(view, [view.format], this.id);
      validateRequirement(this.id, view.length > 0, 'input and output views must not be empty');
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
    const scalarView = (view: GraphDataView<'float32x2'>) =>
      graph.createDataView(view.buffer, {
        format: 'float32',
        byteOffset: view.byteOffset,
        length: view.length * 2
      });
    const scaleIndex = createTransientView(graph, `${id}-spectral-scale-index`, 'uint32', 1);
    // Largest spectral product magnitude, skipping the DC cell (cell 0), as a parallel argmax.
    nodes.push(
      ...createRasterArgmaxNodes(graph, {
        id: `${id}-spectral-scale`,
        operation: 'GPURasterRegistration',
        count,
        bindings: [
          rasterBinding('reference', scalarView(referenceSpectrum)),
          rasterBinding('moving', scalarView(movingSpectrum))
        ],
        candidate: `candidateValue = length(vec2<f32>(reference[referenceOffset + index * 2u], reference[referenceOffset + index * 2u + 1u])) * length(vec2<f32>(moving[movingOffset + index * 2u], moving[movingOffset + index * 2u + 1u])); candidateEligible = index >= 1u;`,
        initialValue: '0.0',
        outputValue: spectralScale,
        outputIndex: scaleIndex
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
    // Input screening by exact integer counts: any nonfinite value, or every value equal to the
    // first (a flat field), means there is no alignment evidence (status 1).
    const nonfiniteCondition = '(bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u';
    // Four counts share one buffer so the status kernel binds them once: reference nonfinite,
    // reference different, moving nonfinite, moving different.
    const screens = createTransientView(graph, `${id}-screens`, 'uint32', 4);
    const screenCell = (cell: number) =>
      graph.createDataView(screens.buffer, {
        format: 'uint32',
        byteOffset: screens.byteOffset + cell * 4,
        length: 1
      });
    (['reference', 'moving'] as const).forEach((name, nameIndex) => {
      const values = name === 'reference' ? props.reference : props.moving;
      const nonfiniteCount = screenCell(nameIndex * 2);
      const differentCount = screenCell(nameIndex * 2 + 1);
      nodes.push(
        ...createRasterCountNodes(graph, {
          id: `${id}-${name}-nonfinite`,
          operation: 'GPURasterRegistration',
          values,
          count,
          condition: nonfiniteCondition,
          output: nonfiniteCount
        }),
        ...createRasterCountNodes(graph, {
          id: `${id}-${name}-different`,
          operation: 'GPURasterRegistration',
          values,
          count,
          condition: 'value != countValues[countValuesOffset]',
          output: differentCount
        })
      );
    });
    const peakValue = createTransientView(graph, `${id}-peak-value`, 'float32', 1);
    const peakIndex = createTransientView(graph, `${id}-peak-index`, 'uint32', 1);
    const secondaryValue = createTransientView(graph, `${id}-secondary-value`, 'float32', 1);
    const secondaryIndex = createTransientView(graph, `${id}-secondary-index`, 'uint32', 1);
    const correlationScalar = scalarView(correlation);
    // Strict maximum (lowest cell wins ties), then the largest value outside the wrapped 3x3
    // neighbourhood of that peak; both are parallel argmax reductions.
    nodes.push(
      ...createRasterArgmaxNodes(graph, {
        id: `${id}-coarse-peak`,
        operation: 'GPURasterRegistration',
        count,
        bindings: [rasterBinding('correlation', correlationScalar)],
        candidate:
          'candidateValue = correlation[correlationOffset + index * 2u]; candidateEligible = true;',
        initialValue: '-3.402823466e38',
        outputValue: peakValue,
        outputIndex: peakIndex
      }),
      ...createRasterArgmaxNodes(graph, {
        id: `${id}-secondary-peak`,
        operation: 'GPURasterRegistration',
        count,
        bindings: [
          rasterBinding('correlation', correlationScalar),
          rasterBinding('peakIndex', peakIndex)
        ],
        candidate: `let selected = peakIndex[peakIndexOffset]; let position = vec2<u32>(selected % ${props.grid.width}u, selected / ${props.grid.width}u);
    let other = vec2<u32>(index % ${props.grid.width}u, index / ${props.grid.width}u); let distance = vec2<u32>(abs(vec2<i32>(other) - vec2<i32>(position))); let wrapped = min(distance, vec2<u32>(${props.grid.width}u, ${props.grid.height}u) - distance);
    candidateValue = correlation[correlationOffset + index * 2u]; candidateEligible = any(wrapped > vec2<u32>(1u));`,
        initialValue: '-3.402823466e38',
        outputValue: secondaryValue,
        outputIndex: secondaryIndex
      }),
      createRegistrationKernel(graph, {
        id: `${id}-coarse-peak-status`,
        operation: 'GPURasterRegistration',
        invocationCount: 1,
        bindings: [
          rasterBinding('peakValue', peakValue),
          rasterBinding('peakIndex', peakIndex),
          rasterBinding('secondaryValue', secondaryValue),
          rasterBinding('screens', screens),
          complexBinding('coarse', coarse, true),
          rasterBinding('status', props.output.status, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `let maximum = peakValue[peakValueOffset]; let secondary = secondaryValue[secondaryValueOffset]; let selected = peakIndex[peakIndexOffset];
let position = vec2<u32>(selected % ${props.grid.width}u, selected / ${props.grid.width}u);
let valid = screens[screensOffset] == 0u && screens[screensOffset + 2u] == 0u;
let isFlat = screens[screensOffset + 1u] == 0u || screens[screensOffset + 3u] == 0u;
// Removing DC makes every nonempty finite correlation field have a positive maximum.
// A zero field has no alignment evidence, even when all peaks are adjacent on a 2x2 grid.
var code = 0u; if (!valid || isFlat || !finite(maximum) || maximum <= 0.0) { code = 1u; } else if (maximum - secondary <= abs(maximum) * ${props.ambiguityTolerance ?? 1e-4}) { code = 2u; }
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
        // Serial over the (2 * ceil(upsampleFactor / 2) + 1) squared refinement candidates, at most
        // 33 x 33 = 1089 values (bounded by MAXIMUM_REGISTRATION_UPSAMPLE_FACTOR), not by raster size.
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
