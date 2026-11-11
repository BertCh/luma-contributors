// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  resolveRasterOperationBand,
  type RasterOperationBand
} from '../../raster-analysis/raster-grid/raster-operation-band';
import {GPUTerrainPeakProminence} from './gpu-terrain-peak-prominence';

/** Buffer-backed band only: texture storage cannot pass through the published mixed executor. */
export type TerrainPeakProminenceOperationProps = {
  id?: string;
  grid: GPURasterMetadata;
  elevation: RasterOperationBand;
  output: {prominence: string; keySaddle?: string; parentPeak?: string};
};

/** Buffer operation form; input contents are per-frame, grid and band metadata compile-time. */
export class GPUTerrainPeakProminenceOperation extends GPUContributorOperation {
  constructor(props: TerrainPeakProminenceOperationProps) {
    const id = props.id ?? 'terrain-peak-prominence';
    const length = props.grid.width * props.grid.height;
    super({
      id,
      inputs: {
        values: props.elevation.storage.values,
        ...(props.elevation.validity !== undefined ? {validity: props.elevation.validity} : {})
      },
      outputs: {
        prominence: {value: props.output.prominence, format: 'float32', length},
        ...(props.output.keySaddle !== undefined
          ? {keySaddle: {value: props.output.keySaddle, format: 'float32' as const, length}}
          : {}),
        ...(props.output.parentPeak !== undefined
          ? {parentPeak: {value: props.output.parentPeak, format: 'uint32' as const, length}}
          : {})
      },
      createContributor: getView =>
        new GPUTerrainPeakProminence({
          id,
          grid: props.grid,
          elevation: resolveRasterOperationBand(props.elevation, getView),
          output: {
            prominence: getView<'float32'>('prominence'),
            keySaddle:
              props.output.keySaddle !== undefined ? getView<'float32'>('keySaddle') : undefined,
            parentPeak:
              props.output.parentPeak !== undefined ? getView<'uint32'>('parentPeak') : undefined
          }
        })
    });
  }
}
