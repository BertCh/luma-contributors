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

/** EVI bands must calibrate to unitless reflectance, not radiance, percent, or scaled integers. */
export type GPURasterEVIProps = SpectralIndexProps & {
  nearInfrared: GPURasterBufferBand;
  red: GPURasterBufferBand;
  blue: GPURasterBufferBand;
};

/**
 * Enhanced vegetation index `2.5 * (NIR - red)/(NIR + 6 * red - 7.5 * blue + 1)`.
 * Six public band-math passes materialize calibration before applying the red/blue coefficients.
 * Each intermediate propagates validity; only the final denominator is tested against epsilon.
 */
export class GPURasterEVI
  extends RasterSpectralIndex<GPURasterEVIProps>
  implements GPUCommandNodeProducer
{
  constructor(props: GPURasterEVIProps) {
    props = snapshotSpectralIndexProps(props, ['nearInfrared', 'red', 'blue']);
    super(props, 'gpu-raster-evi', [props.nearInfrared, props.red, props.blue]);
  }

  protected addIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const {nearInfrared} = this.props;
    // Adding a zero-calibrated copy preserves exact raw nodata/masks and materializes reflectance
    // before coefficients. Folding coefficients into raw scale/offset would change this order.
    const red = this.addStage(
      graph,
      'calibrate-red',
      this.props.red,
      {...this.props.red, scale: 0, offset: 0},
      'add'
    );
    const blue = this.addStage(
      graph,
      'calibrate-blue',
      this.props.blue,
      {...this.props.blue, scale: 0, offset: 0},
      'add'
    );
    const numerator = this.addStage(graph, 'numerator', nearInfrared, red, 'subtract');
    const weightedSum = this.addStage(
      graph,
      'weighted-sum',
      nearInfrared,
      {...red, scale: 6},
      'add'
    );
    const denominator = this.addStage(
      graph,
      'denominator',
      weightedSum,
      {...blue, scale: 7.5},
      'subtract'
    );
    this.addOutput(graph, {...numerator, scale: 2.5}, {...denominator, offset: 1}, 'divide');
  }
}
