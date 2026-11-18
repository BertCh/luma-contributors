// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {
  RasterSpectralIndex,
  snapshotSpectralIndexProps,
  type SpectralIndexProps
} from './raster-spectral-index';

/** SAVI bands must calibrate to unitless reflectance before the soil adjustment is applied. */
export type GPURasterSAVIProps = SpectralIndexProps & {
  nearInfrared: GPURasterBufferBand;
  red: GPURasterBufferBand;
  /** Soil brightness adjustment L in [0, 1]. Defaults to 0.5; zero gives the NDVI formula. */
  soilAdjustment?: number;
};

/** Soil-adjusted vegetation index `(1 + L) * (NIR - red)/(NIR + red + L)`. */
export class GPURasterSAVI
  extends RasterSpectralIndex<GPURasterSAVIProps>
  implements GPUCommandNodeProducer
{
  readonly soilAdjustment: number;

  constructor(props: GPURasterSAVIProps) {
    props = snapshotSpectralIndexProps(props, ['nearInfrared', 'red', 'soilAdjustment']);
    super(props, 'gpu-raster-savi', [props.nearInfrared, props.red]);
    this.soilAdjustment = props.soilAdjustment ?? 0.5;
    if (
      !Number.isFinite(this.soilAdjustment) ||
      this.soilAdjustment < 0 ||
      this.soilAdjustment > 1
    ) {
      throw new Error(`${this.id} soilAdjustment must be finite and in [0, 1]`);
    }
  }

  /** Adds three public band-math passes, with graph-owned intermediate values and masks. */
  protected addIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const {nearInfrared, red} = this.props;
    const numerator = this.addStage(graph, 'numerator', nearInfrared, red, 'subtract');
    const denominator = this.addStage(graph, 'denominator', nearInfrared, red, 'add');
    this.addOutput(
      graph,
      {...numerator, scale: 1 + this.soilAdjustment},
      {...denominator, offset: this.soilAdjustment},
      'divide'
    );
  }
}
