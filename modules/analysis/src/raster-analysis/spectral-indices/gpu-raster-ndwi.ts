// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  RasterSpectralIndex,
  snapshotSpectralIndexProps,
  type SpectralIndexProps
} from './raster-spectral-index';

/**
 * Explicit NDWI convention: Gao vegetation liquid water versus McFeeters open water (OTB NDWI2).
 * The original Gao bands center around 0.86 and 1.24 micrometers; a different shortwave band is
 * a sensor-specific adaptation. McFeeters uses visible green and near infrared.
 */
export type GPURasterNDWIProps = SpectralIndexProps &
  (
    | {
        method: 'gao';
        nearInfrared: GPURasterBufferBand;
        shortwaveInfrared: GPURasterBufferBand;
      }
    | {
        method: 'mcfeeters';
        green: GPURasterBufferBand;
        nearInfrared: GPURasterBufferBand;
      }
  );

/** Calibrated `(green - NIR)/(green + NIR)` or `(NIR - SWIR)/(NIR + SWIR)`. */
export class GPURasterNDWI
  extends RasterSpectralIndex<GPURasterNDWIProps>
  implements GPUCommandNodeProducer
{
  constructor(props: GPURasterNDWIProps) {
    const captured = snapshotConstructionFields(
      props as GPURasterNDWIProps & {
        green?: GPURasterBufferBand;
        shortwaveInfrared?: GPURasterBufferBand;
      },
      ['id', 'width', 'height', 'output', 'outputValidity', 'epsilon', 'clamp', 'method']
    );
    if (captured.method !== 'gao' && captured.method !== 'mcfeeters')
      throw new Error('GPURasterNDWI requires an explicit gao or mcfeeters method');
    const sources = snapshotConstructionFields(
      props as GPURasterNDWIProps & {
        green?: GPURasterBufferBand;
        shortwaveInfrared?: GPURasterBufferBand;
      },
      ['nearInfrared', captured.method === 'gao' ? 'shortwaveInfrared' : 'green']
    );
    props = snapshotSpectralIndexProps({...captured, ...sources}, [
      'method',
      'nearInfrared',
      captured.method === 'gao' ? 'shortwaveInfrared' : 'green'
    ]);
    super(props, 'gpu-raster-ndwi', [
      props.nearInfrared,
      props.method === 'gao' ? props.shortwaveInfrared : props.green
    ]);
  }

  /** Adds one public band-math pass; sensor selection and quality-bit decoding stay caller-owned. */
  protected addIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const props = this.props;
    this.addOutput(
      graph,
      props.method === 'gao' ? props.nearInfrared : props.green,
      props.method === 'gao' ? props.shortwaveInfrared : props.nearInfrared,
      'normalized-difference'
    );
  }
}
