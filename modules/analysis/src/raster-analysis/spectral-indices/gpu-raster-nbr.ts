// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {
  RasterSpectralIndex,
  snapshotSpectralIndexProps,
  type SpectralIndexProps
} from './raster-spectral-index';

/** Normalized burn ratio using near infrared and the approximately 2.2 micrometer SWIR2 band. */
export type GPURasterNBRProps = SpectralIndexProps & {
  nearInfrared: GPURasterBand;
  shortwaveInfrared2: GPURasterBand;
};

/** Calibrated `(NIR - SWIR2)/(NIR + SWIR2)`; outputs the ratio, not a scaled integer product. */
export class GPURasterNBR
  extends RasterSpectralIndex<GPURasterNBRProps>
  implements GPUCommandNodeProducer
{
  constructor(props: GPURasterNBRProps) {
    props = snapshotSpectralIndexProps(props, ['nearInfrared', 'shortwaveInfrared2']);
    super(props, 'gpu-raster-nbr', [props.nearInfrared, props.shortwaveInfrared2]);
  }

  protected addIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    this.addOutput(
      graph,
      this.props.nearInfrared,
      this.props.shortwaveInfrared2,
      'normalized-difference'
    );
  }
}
