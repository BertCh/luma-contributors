// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {GPURasterBandMath, type GPURasterBandMathOperation} from './gpu-raster-band-math';
import type {GPURasterBufferBand} from './types';

/** Shared output and denominator policy for dimensionless reflectance indices. */
type SpectralIndexProps = {
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

/** Normalized burn ratio using near infrared and the approximately 2.2 micrometer SWIR2 band. */
export type GPURasterNBRProps = SpectralIndexProps & {
  nearInfrared: GPURasterBufferBand;
  shortwaveInfrared2: GPURasterBufferBand;
};

/** EVI bands must calibrate to unitless reflectance, not radiance, percent, or scaled integers. */
export type GPURasterEVIProps = SpectralIndexProps & {
  nearInfrared: GPURasterBufferBand;
  red: GPURasterBufferBand;
  blue: GPURasterBufferBand;
};

/** SAVI bands must calibrate to unitless reflectance before the soil adjustment is applied. */
export type GPURasterSAVIProps = SpectralIndexProps & {
  nearInfrared: GPURasterBufferBand;
  red: GPURasterBufferBand;
  /** Soil brightness adjustment L in [0, 1]. Defaults to 0.5; zero gives the NDVI formula. */
  soilAdjustment?: number;
};

/** Common ownership and calibration validation, delegated to the underlying public primitive. */
abstract class GPURasterSpectralIndex<Properties extends SpectralIndexProps> {
  readonly id: string;
  readonly props: Properties;

  protected constructor(props: Properties, defaultId: string, bands: GPURasterBufferBand[]) {
    this.id = props.id ?? defaultId;
    this.props = props;
    for (const band of bands) {
      // Validate every original source against both caller outputs before allocating transients.
      // In particular, the final divide alone cannot detect aliases with an earlier source.
      new GPURasterBandMath({...props, id: this.id, left: band, right: band, operation: 'add'});
    }
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

/** Calibrated `(green - NIR)/(green + NIR)` or `(NIR - SWIR)/(NIR + SWIR)`. */
export class GPURasterNDWI extends GPURasterSpectralIndex<GPURasterNDWIProps> {
  constructor(props: GPURasterNDWIProps) {
    if (props.method !== 'gao' && props.method !== 'mcfeeters') {
      throw new Error('GPURasterNDWI requires an explicit gao or mcfeeters method');
    }
    super(props, 'gpu-raster-ndwi', [
      props.nearInfrared,
      props.method === 'gao' ? props.shortwaveInfrared : props.green
    ]);
  }

  /** Adds one public band-math pass; sensor selection and quality-bit decoding stay caller-owned. */
  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const props = this.props;
    this.addOutput(
      graph,
      props.method === 'gao' ? props.nearInfrared : props.green,
      props.method === 'gao' ? props.shortwaveInfrared : props.nearInfrared,
      'normalized-difference'
    );
  }
}

/** Calibrated `(NIR - SWIR2)/(NIR + SWIR2)`; outputs the ratio, not a scaled integer product. */
export class GPURasterNBR extends GPURasterSpectralIndex<GPURasterNBRProps> {
  constructor(props: GPURasterNBRProps) {
    super(props, 'gpu-raster-nbr', [props.nearInfrared, props.shortwaveInfrared2]);
  }

  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    this.addOutput(
      graph,
      this.props.nearInfrared,
      this.props.shortwaveInfrared2,
      'normalized-difference'
    );
  }
}

/**
 * Enhanced vegetation index `2.5 * (NIR - red)/(NIR + 6 * red - 7.5 * blue + 1)`.
 * Six public band-math passes materialize calibration before applying the red/blue coefficients.
 * Each intermediate propagates validity; only the final denominator is tested against epsilon.
 */
export class GPURasterEVI extends GPURasterSpectralIndex<GPURasterEVIProps> {
  constructor(props: GPURasterEVIProps) {
    super(props, 'gpu-raster-evi', [props.nearInfrared, props.red, props.blue]);
  }

  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
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

/** Soil-adjusted vegetation index `(1 + L) * (NIR - red)/(NIR + red + L)`. */
export class GPURasterSAVI extends GPURasterSpectralIndex<GPURasterSAVIProps> {
  readonly soilAdjustment: number;

  constructor(props: GPURasterSAVIProps) {
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
  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
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
