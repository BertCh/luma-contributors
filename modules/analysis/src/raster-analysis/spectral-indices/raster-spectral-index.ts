// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPURasterBandMath, type GPURasterBandMathOperation} from '@luma.gl/graph/gpu-raster';
import {
  GPURasterTextureToBuffer,
  type GPURasterBand,
  type GPURasterTextureBand,
  type GPURasterBufferBand
} from '@luma.gl/graph/gpu-raster';
import {
  captureGraphCommandNodes,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  snapshotRasterBand,
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterTextureBand,
  validateRasterBandBelongsToGraph
} from '../raster-band-utils';
import {validateRasterStorageBindings} from '../raster-grid/raster-storage-preflight';

/** Capture the validated policy, output ports and named source descriptors once. @internal */
export function snapshotSpectralIndexProps<Props extends SpectralIndexProps>(
  props: Props,
  fields: readonly (keyof Props)[]
): Props {
  const captured = snapshotConstructionFields(props, [
    'id',
    'width',
    'height',
    'output',
    'outputValidity',
    'epsilon',
    'clamp',
    ...fields
  ]);
  const bands = fields
    .filter(field => !['method', 'soilAdjustment'].includes(String(field)))
    .map(field => [field, Object.freeze(snapshotRasterBand(captured[field] as GPURasterBand))]);
  return Object.freeze({
    ...captured,
    ...Object.fromEntries(bands),
    ...(captured.clamp ? {clamp: Object.freeze([...captured.clamp])} : {})
  }) as Props;
}

/** Shared output and denominator policy for dimensionless reflectance indices. */
export type SpectralIndexProps = {
  id?: string;
  width: number;
  height: number;
  /** Caller-owned float32 index; invalid pixels receive canonical quiet NaNs. */
  output: GraphDataView<'float32'>;
  /** Caller-owned canonical zero/one intersection of the required source validity domains. */
  outputValidity: GraphDataView<'uint32'>;
  /** Rejects the calibrated formula denominator when its absolute value is <= epsilon. Default 0. */
  epsilon?: number;
  /** Explicit optional output clamp; no range clipping is applied by default. */
  clamp?: readonly [number, number];
};

/** Common ownership and calibration validation, delegated to the underlying public primitive. */
export abstract class RasterSpectralIndex<Properties extends SpectralIndexProps> {
  readonly id: string;
  readonly props: Readonly<Properties>;
  private readonly sourceBands: GPURasterBand[];
  private readonly convertedBands = new WeakMap<object, Map<GPURasterBand, GPURasterBufferBand>>();
  private readonly commandNodes = new WeakMap<object, readonly GPUCommandNode[]>();

  /** Returns the ordered public band-math passes without scheduling them. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const cached = this.commandNodes.get(graph);
    if (cached) return cached as readonly GPUCommandNode<Parameters>[];
    for (const band of this.sourceBands) validateRasterBandBelongsToGraph(this.id, graph, band, []);
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        ...this.sourceBands.flatMap(getRasterBandViews),
        this.props.output,
        this.props.outputValidity
      ],
      {'canonical band planes': this.props.width * this.props.height * 4}
    );
    const nodes = Object.freeze(captureGraphCommandNodes(graph, () => this.addIndexNodes(graph)));
    this.commandNodes.set(graph, nodes as readonly GPUCommandNode[]);
    return nodes;
  }

  protected abstract addIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): void;

  protected constructor(props: Properties, defaultId: string, bands: GPURasterBand[]) {
    this.id = props.id ?? defaultId;
    this.props = props;
    this.sourceBands = bands;
    if (!Number.isFinite(Math.fround(props.epsilon ?? 0)) || (props.epsilon ?? 0) < 0)
      throw new Error(`${this.id} epsilon must be finite and non-negative`);
    if (
      props.clamp &&
      (props.clamp.length !== 2 ||
        !props.clamp.every(value => Number.isFinite(Math.fround(value))) ||
        props.clamp[0] > props.clamp[1])
    )
      throw new Error(`${this.id} clamp must contain an ordered finite range`);
    for (const band of bands) {
      // Validate every original source against both caller outputs before allocating transients.
      // In particular, the final divide alone cannot detect aliases with an earlier source.
      if (band.storage.kind === 'buffer') {
        const input = band as GPURasterBufferBand;
        new GPURasterBandMath({...props, id: this.id, left: input, right: input, operation: 'add'});
      } else {
        validateRasterTextureBand(this.id, band, this.getGrid());
        new GPURasterTextureToBuffer({
          id: this.id,
          input: band as GPURasterTextureBand,
          output: props.output,
          outputValidity: props.outputValidity,
          applyCalibration: true
        });
      }
      validateGraphOutputsDisjointFromInputs(
        this.id,
        {output: props.output, outputValidity: props.outputValidity},
        Object.fromEntries(getRasterBandViews(band).map((view, index) => [`input ${index}`, view]))
      );
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      {output: props.output},
      {outputValidity: props.outputValidity}
    );
  }

  private getGrid() {
    return {
      width: this.props.width,
      height: this.props.height,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
  }

  private resolveBand<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    band: GPURasterBand
  ): GPURasterBufferBand {
    let converted = this.convertedBands.get(graph);
    if (!converted) {
      converted = new Map();
      this.convertedBands.set(graph, converted);
    }
    const cached = converted.get(band);
    if (cached) return cached;
    const source = getRasterBufferBandNodes(
      graph,
      `${this.id}-source-${converted.size}`,
      band,
      this.getGrid(),
      false
    );
    for (const node of source.nodes) graph.add(node);
    converted.set(band, source.band);
    return source.band;
  }

  protected addStage<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    name: string,
    left: GPURasterBand,
    right: GPURasterBand,
    operation: GPURasterBandMathOperation
  ): GPURasterBufferBand<'float32'> {
    const id = `${this.id}-${name}`;
    const length = this.props.width * this.props.height;
    const values = createTransientView(graph, `${id}-values`, 'float32', length);
    const validity = createTransientView(graph, `${id}-validity`, 'uint32', length);
    new GPURasterBandMath({
      id,
      width: this.props.width,
      height: this.props.height,
      left: this.resolveBand(graph, left),
      right: this.resolveBand(graph, right),
      operation,
      output: values,
      outputValidity: validity
    }).addToGraph(graph);
    return {id, format: 'float32', storage: {kind: 'buffer', values}, validity};
  }

  protected addOutput<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    left: GPURasterBand,
    right: GPURasterBand,
    operation: 'divide' | 'normalized-difference'
  ): void {
    new GPURasterBandMath({
      ...this.props,
      id: this.id,
      left: this.resolveBand(graph, left),
      right: this.resolveBand(graph, right),
      operation
    }).addToGraph(graph);
  }
}
