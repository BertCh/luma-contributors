// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUFFT2D,
  GPU_FFT2D_MIN_DIMENSION,
  GPU_FFT2D_MAX_DIMENSION,
  getGPUFFT2DSupport,
  makeGPUFFT2DStats,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {GPUInvalidStatusPort} from '../../analysis-core/contracts/status';
import {validateGPUStatusPort} from '../../analysis-core/contracts/status';
import {
  validateCommandNodeBudget,
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validatePowerOfTwo,
  validateRequirement,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBand,
  validateRasterBandBelongsToGraph,
  validateRasterGrid,
  validateRasterParameters
} from '../raster-band-utils';
import {RASTER_BAND_CANONICALIZATION_NODE_COUNT} from '../raster-expansion-utils';
import {
  createRasterCountNodes,
  rasterBinding,
  RASTER_FINITE_WGSL
} from '../remote-sensing/raster-kernel-utils';
import {
  GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH,
  MINIMUM_RASTER_SPECTRAL_SIGMA
} from './gpu-raster-spectral-filter-parameters';

/** Published radix-two transform bound; each complex field must fit one storage binding. */
export const MAXIMUM_RASTER_SPECTRAL_FILTER_DIMENSION = GPU_FFT2D_MAX_DIMENSION;
/** Calibration and float nodata are encoded in finite float32 shader literals. */
const MAXIMUM_RASTER_METADATA_FLOAT32_VALUE = 3.4028234663852886e38;
/** Integer nodata must retain the exact raw uint32 or sint32 storage representation. */
const MAXIMUM_RASTER_METADATA_UINT32_VALUE = 0xffffffff;
const MINIMUM_RASTER_METADATA_SINT32_VALUE = -0x80000000;
const MAXIMUM_RASTER_METADATA_SINT32_VALUE = 0x7fffffff;
/** Published raster canonicalization samples one 2D mip, one array layer and one sample. */
const RASTER_SPECTRAL_TEXTURE_LAYOUT_COUNT = 1;
/** Packed raster textures expose four RGBA channels, indexed from zero. */
const MAXIMUM_RASTER_RGBA_CHANNEL = 3;

export type GPURasterSpectralFilterOutput = {
  values: GraphDataView<'float32'>;
  validity: GraphDataView<'uint32'>;
  /** invalidCount counts missing/nonfinite input cells, before global propagation. */
  status?: GPUInvalidStatusPort;
};

export type GPURasterSpectralFilterProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GPURasterBand;
  filter: 'low-pass' | 'high-pass' | 'band-pass' | 'gaussian' | 'notch';
  /** Eight float32 values from getGPURasterSpectralFilterParameterValues. */
  parameters: GraphDataView<'float32'>;
  output: GPURasterSpectralFilterOutput;
};

/**
 * Periodic spectral raster filtering composed from the published GPUFFT2D.
 * Compile-time: grid dimensions, band storage/calibration, filter and view layouts.
 * Per-frame: observations/validity and frequency settings in parameters. Frequencies are in
 * cycles/pixel; no ground distance or affine is used. Input is one complete resident domain,
 * not independently filterable tiles. Any missing input invalidates the entire output, because
 * every Fourier coefficient depends on every cell. Invalid runtime parameters also invalidate
 * the entire output. Padding rows past width * height are neither read nor written.
 */
export class GPURasterSpectralFilter implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPURasterSpectralFilterProps) {
    this.id = props.id ?? 'raster-spectral-filter';
    const {id} = this;
    const count = validateRasterGrid(id, props.grid);
    for (const [name, dimension] of [
      ['grid.width', props.grid.width],
      ['grid.height', props.grid.height]
    ] as const) {
      validatePowerOfTwo(
        id,
        name,
        dimension,
        GPU_FFT2D_MIN_DIMENSION,
        MAXIMUM_RASTER_SPECTRAL_FILTER_DIMENSION
      );
    }
    validateRequirement(
      id,
      ['low-pass', 'high-pass', 'band-pass', 'gaussian', 'notch'].includes(props.filter),
      "filter must be 'low-pass', 'high-pass', 'band-pass', 'gaussian' or 'notch'"
    );
    for (const [name, value] of [
      ['input.scale', props.input.scale ?? 1],
      ['input.offset', props.input.offset ?? 0]
    ] as const) {
      validateFiniteNumber(
        id,
        name,
        value,
        -MAXIMUM_RASTER_METADATA_FLOAT32_VALUE,
        MAXIMUM_RASTER_METADATA_FLOAT32_VALUE
      );
    }
    const {format, noDataValue} = props.input;
    if (noDataValue !== undefined) {
      if (format === 'float32') {
        if (!Number.isNaN(noDataValue)) {
          validateFiniteNumber(
            id,
            'input.noDataValue',
            noDataValue,
            -MAXIMUM_RASTER_METADATA_FLOAT32_VALUE,
            MAXIMUM_RASTER_METADATA_FLOAT32_VALUE
          );
        }
      } else {
        validateCapacity(
          id,
          'input.noDataValue',
          noDataValue,
          format === 'uint32' ? 0 : MINIMUM_RASTER_METADATA_SINT32_VALUE,
          format === 'uint32'
            ? MAXIMUM_RASTER_METADATA_UINT32_VALUE
            : MAXIMUM_RASTER_METADATA_SINT32_VALUE
        );
      }
    }
    validateRasterBand(props.input, props.grid, `${id} input`);
    const {storage} = props.input;
    if (storage.kind === 'buffer') {
      validateSinglePackedView(storage.values, [format], `${id} input.storage.values`);
    } else {
      const textureView = storage.view;
      validateRequirement(
        id,
        textureView.dimension === '2d',
        'input.storage.view.dimension must be 2d'
      );
      for (const [name, value] of [
        ['input.storage.view.mipLevelCount', textureView.mipLevelCount],
        ['input.storage.view.arrayLayerCount', textureView.arrayLayerCount],
        ['input.storage.view.texture.samples', textureView.texture.samples]
      ] as const) {
        validateCapacity(
          id,
          name,
          value,
          RASTER_SPECTRAL_TEXTURE_LAYOUT_COUNT,
          RASTER_SPECTRAL_TEXTURE_LAYOUT_COUNT
        );
      }
      const textureSuffix = {float32: 'float', uint32: 'uint', sint32: 'sint'}[format];
      const textureFormats = [`r32${textureSuffix}`, `rgba32${textureSuffix}`];
      validateRequirement(
        id,
        textureFormats.includes(textureView.format),
        `input.storage.view.format must be ${textureFormats.join(' or ')} for input.format ${format}`
      );
      validateCapacity(
        id,
        'input.storage.channel',
        storage.channel ?? 0,
        0,
        textureView.format.startsWith('rgba') ? MAXIMUM_RASTER_RGBA_CHANNEL : 0
      );
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRasterParameters(id, props.parameters, GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH);
    const inputs = [...getRasterBandViews(props.input), props.parameters];
    const outputs = [
      props.output.values,
      props.output.validity,
      ...Object.values(props.output.status ?? {})
    ].filter(view => view !== undefined);
    if (props.input.validity) {
      validateSinglePackedView(props.input.validity, ['uint32'], `${id} input.validity`);
      validateRequirement(
        id,
        props.input.validity.length >= count,
        `input.validity must hold at least ${count} rows`
      );
    }
    for (const [name, view] of Object.entries({
      values: props.output.values,
      validity: props.output.validity
    })) {
      validateSinglePackedView(
        view,
        [name === 'values' ? 'float32' : 'uint32'],
        `${id} output.${name}`
      );
      validateRequirement(
        id,
        view.length >= count,
        `output.${name} must hold at least ${count} rows`
      );
    }
    validateGPUStatusPort(id, props.output.status ?? {});
    for (const [name, view] of Object.entries(props.output.status ?? {})) {
      if (view) validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
    }
    validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
    outputs.forEach((view, index) =>
      validateGraphOutputsDisjointFromInputs(id, [view], outputs.slice(index + 1))
    );
    // Published contiguous FFT dispatch count; allowance covers published band conversion.
    this.estimatedCommandNodeCount =
      RASTER_BAND_CANONICALIZATION_NODE_COUNT +
      5 +
      2 * makeGPUFFT2DStats(props.grid.width, props.grid.height).dispatchCountPerEncode;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props.grid;
    const count = width * height;
    // Before allocating scratch or creating nodes, reject device-specific binding limits.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'complex scratch',
      count * 8,
      'reduce grid.width or grid.height'
    );
    const support = getGPUFFT2DSupport(graph.device, {width, height});
    validateRequirement(
      id,
      support.supported,
      `grid.width and grid.height must fit device FFT limits: ${support.reason}`
    );
    validateRasterBandBelongsToGraph(id, graph, props.input, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      props.output.values,
      props.output.validity,
      ...Object.values(props.output.status ?? {})
    ]);
    // Published raster canonicalization requires exact grid lengths. Borrow only the raster
    // prefix of padded caller views, preserving their offsets and band metadata.
    let input: GPURasterBand = {
      ...props.input,
      validity: props.input.validity
        ? graph.createDataView(props.input.validity.buffer, {
            format: props.input.validity.format,
            byteOffset: props.input.validity.byteOffset,
            length: count
          })
        : undefined
    };
    if (input.storage.kind === 'buffer') {
      const {values} = input.storage;
      input = {
        ...input,
        storage: {
          ...input.storage,
          values: graph.createDataView(values.buffer, {
            format: values.format,
            byteOffset: values.byteOffset,
            length: count
          })
        }
      } as GPURasterBand;
    }
    const source = getRasterBufferBandNodes(graph, id, input, props.grid, true);
    const packed = createTransientView(graph, `${id}-packed`, 'float32x2', count);
    const spectrum = createTransientView(graph, `${id}-spectrum`, 'float32x2', count);
    const filtered = createTransientView(graph, `${id}-filtered`, 'float32x2', count);
    const result = createTransientView(graph, `${id}-result`, 'float32x2', count);
    const invalidCount = createTransientView(graph, `${id}-invalid-count`, 'uint32', 1);
    const scalar = (view: GraphDataView<'float32x2'>) =>
      graph.createDataView(view.buffer, {
        format: 'float32',
        byteOffset: view.byteOffset,
        length: count * 2
      });
    const nodes: GPUCommandNode<Parameters>[] = [
      ...source.nodes,
      ...createRasterCountNodes(graph, {
        id: `${id}-invalid`,
        operation: 'GPURasterSpectralFilter',
        values: source.band.validity!,
        count,
        condition: 'value == 0u',
        output: invalidCount
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pack`,
        operation: 'GPURasterSpectralFilter',
        invocationCount: count,
        bindings: [
          rasterBinding('source', source.band.storage.values),
          rasterBinding('sourceValidity', source.band.validity!),
          rasterBinding('packed', scalar(packed), true)
        ],
        body: `packed[packedOffset + index * 2u] = select(0.0, source[sourceOffset + index], sourceValidity[sourceValidityOffset + index] != 0u); packed[packedOffset + index * 2u + 1u] = 0.0;`
      }),
      ...new GPUFFT2D({
        id: `${id}-forward`,
        input: packed,
        output: spectrum,
        width,
        height
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-response`,
        operation: 'GPURasterSpectralFilter',
        invocationCount: count,
        bindings: [
          rasterBinding('spectrum', scalar(spectrum)),
          rasterBinding('filtered', scalar(filtered), true),
          rasterBinding('parameters', props.parameters)
        ],
        body: `let column = index % ${width}u; let row = index / ${width}u;
// Evaluate conjugate bins at one shared index, so their responses are bit-identical.
let conjugateIndex = ((${height}u - row) % ${height}u) * ${width}u + ((${width}u - column) % ${width}u);
let frequencyIndex = min(index, conjugateIndex);
let frequencyColumn = frequencyIndex % ${width}u; let frequencyRow = frequencyIndex / ${width}u;
let frequency = vec2<f32>(f32(select(i32(frequencyColumn), i32(frequencyColumn) - ${width}, frequencyColumn >= ${width / 2}u)) / ${width}.0, f32(select(i32(frequencyRow), i32(frequencyRow) - ${height}, frequencyRow >= ${height / 2}u)) / ${height}.0);
let radius = length(frequency); let cutoff = parameters[parametersOffset];
let lowerCutoff = parameters[parametersOffset + 1u]; let upperCutoff = parameters[parametersOffset + 2u]; let sigma = parameters[parametersOffset + 3u];
let center = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
let notchRadius = parameters[parametersOffset + 6u];
let firstDistance = abs(frequency - center); let secondDistance = abs(frequency + center);
// At Nyquist, +/-0.5 are one frequency. Subtract the center's magnitude directly instead
// of rounding 1 - abs(-0.5 - center), which can incorrectly cross a notch boundary.
let nyquistAxes = abs(frequency) == vec2<f32>(0.5);
let nyquistDistance = vec2<f32>(0.5) - abs(center);
let firstWrappedDistance = select(min(firstDistance, vec2<f32>(1.0) - firstDistance), nyquistDistance, nyquistAxes);
let secondWrappedDistance = select(min(secondDistance, vec2<f32>(1.0) - secondDistance), nyquistDistance, nyquistAxes);
let notchDistance = min(length(firstWrappedDistance), length(secondWrappedDistance));
let response = ${getResponseWGSL(props.filter)};
filtered[filteredOffset + index * 2u] = spectrum[spectrumOffset + index * 2u] * response;
filtered[filteredOffset + index * 2u + 1u] = spectrum[spectrumOffset + index * 2u + 1u] * response;`
      }),
      ...new GPUFFT2D({
        id: `${id}-inverse`,
        input: filtered,
        output: result,
        direction: 'inverse',
        width,
        height
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterSpectralFilter',
        invocationCount: count,
        bindings: [
          rasterBinding('result', scalar(result)),
          rasterBinding('invalidCount', invalidCount),
          rasterBinding('parameters', props.parameters),
          rasterBinding('values', props.output.values, true),
          rasterBinding('validity', props.output.validity, true),
          ...(props.output.status?.invalidCount
            ? [rasterBinding('reportedInvalidCount', props.output.status.invalidCount, true)]
            : [])
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var validParameters = true;
for (var parameterIndex = 0u; parameterIndex < 7u; parameterIndex++) { validParameters = validParameters && finite(parameters[parametersOffset + parameterIndex]); }
let maximumFrequency = 0.7071067811865476;
let cutoff = parameters[parametersOffset]; let lower = parameters[parametersOffset + 1u]; let upper = parameters[parametersOffset + 2u]; let sigma = parameters[parametersOffset + 3u];
validParameters = validParameters && cutoff >= 0.0 && cutoff <= maximumFrequency && lower >= 0.0 && upper >= lower && upper <= maximumFrequency && sigma >= ${MINIMUM_RASTER_SPECTRAL_SIGMA} && sigma <= maximumFrequency && abs(parameters[parametersOffset + 4u]) <= 0.5 && abs(parameters[parametersOffset + 5u]) <= 0.5 && parameters[parametersOffset + 6u] >= 0.0 && parameters[parametersOffset + 6u] <= maximumFrequency;
let value = result[resultOffset + index * 2u]; let valid = invalidCount[invalidCountOffset] == 0u && validParameters && finite(value);
values[valuesOffset + index] = select(missing(), value, valid); validity[validityOffset + index] = u32(valid);
${props.output.status?.invalidCount ? 'if (index == 0u) { reportedInvalidCount[reportedInvalidCountOffset] = invalidCount[invalidCountOffset]; }' : ''}`
      })
    ];
    return nodes;
  }
}

function getResponseWGSL(filter: GPURasterSpectralFilterProps['filter']): string {
  switch (filter) {
    case 'low-pass':
      return 'select(0.0, 1.0, radius <= cutoff)';
    case 'high-pass':
      return 'select(0.0, 1.0, radius > cutoff)';
    case 'band-pass':
      return 'select(0.0, 1.0, radius >= lowerCutoff && radius <= upperCutoff)';
    case 'gaussian':
      return 'exp(-0.5 * dot(frequency / sigma, frequency / sigma))';
    case 'notch':
      return 'select(0.0, 1.0, notchDistance > notchRadius)';
  }
}
