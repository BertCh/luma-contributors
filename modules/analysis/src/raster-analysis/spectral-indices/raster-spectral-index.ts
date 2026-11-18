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
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {
  captureGraphCommandNodes,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterBand} from '../raster-band-utils';

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
    .map(field => [
      field,
      Object.freeze(snapshotRasterBand(captured[field] as GPURasterBufferBand))
    ]);
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
  readonly props: Properties;
  private readonly commandNodes = new WeakMap<object, readonly GPUCommandNode[]>();

  /** Returns the ordered public band-math passes without scheduling them. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const cached = this.commandNodes.get(graph);
    if (cached) return cached as readonly GPUCommandNode<Parameters>[];
    const nodes = Object.freeze(captureGraphCommandNodes(graph, () => this.addIndexNodes(graph)));
    this.commandNodes.set(graph, nodes as readonly GPUCommandNode[]);
    return nodes;
  }

  protected abstract addIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): void;

  protected constructor(props: Properties, defaultId: string, bands: GPURasterBufferBand[]) {
    this.id = props.id ?? defaultId;
    this.props = props;
    for (const band of bands) {
      // Validate every original source against both caller outputs before allocating transients.
      // In particular, the final divide alone cannot detect aliases with an earlier source.
      new GPURasterBandMath({...props, id: this.id, left: band, right: band, operation: 'add'});
      validateGraphOutputsDisjointFromInputs(
        this.id,
        {output: props.output, outputValidity: props.outputValidity},
        {values: band.storage.values, validity: band.validity}
      );
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      {output: props.output},
      {outputValidity: props.outputValidity}
    );
  }

  protected addStage<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    name: string,
    left: GPURasterBufferBand,
    right: GPURasterBufferBand,
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
      left,
      right,
      operation,
      output: values,
      outputValidity: validity
    }).addToGraph(graph);
    return {id, format: 'float32', storage: {kind: 'buffer', values}, validity};
  }

  protected addOutput<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    left: GPURasterBufferBand,
    right: GPURasterBufferBand,
    operation: 'divide' | 'normalized-difference'
  ): void {
    new GPURasterBandMath({...this.props, id: this.id, left, right, operation}).addToGraph(graph);
  }
}
