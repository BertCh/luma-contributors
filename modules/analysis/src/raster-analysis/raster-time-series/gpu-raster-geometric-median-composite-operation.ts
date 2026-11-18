// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURasterGeometricMedianComposite} from './gpu-raster-geometric-median-composite';
import {getGPURasterGeometricMedianCompositeOutputLayout} from './raster-geometric-median-output-layout';

/** Graph value names replacing views, with the contributor's fixed-count recursion settings. */
export type RasterGeometricMedianCompositeOperationProps = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  bandCount: number;
  iterationCount: number;
  values: string;
  /** Per-observation uint32 input; nonzero keeps the observation. */
  mask?: string;
  parameters: string;
  output: {composite: string; status: string; lastStep?: string};
};

/** Vector-only operation form of the geometric median contributor. */
export class GPURasterGeometricMedianCompositeOperation extends GPUContributorOperation {
  constructor(props: RasterGeometricMedianCompositeOperationProps) {
    const id = props.id ?? 'raster-geometric-median-composite';
    const lastStep = props.output.lastStep;
    const outputLayout = getGPURasterGeometricMedianCompositeOutputLayout({
      ...props,
      id,
      lastStep: lastStep !== undefined
    });
    super({
      id,
      inputs: {
        values: props.values,
        parameters: props.parameters,
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs: {
        composite: {
          value: props.output.composite,
          ...outputLayout.composite
        },
        status: {value: props.output.status, ...outputLayout.status},
        ...(lastStep !== undefined && outputLayout.lastStep
          ? {lastStep: {value: lastStep, ...outputLayout.lastStep}}
          : {})
      },
      createContributor: getView =>
        new GPURasterGeometricMedianComposite({
          ...props,
          id,
          values: getView<'float32'>('values'),
          parameters: getView<'float32'>('parameters'),
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          output: {
            composite: getView<'float32'>('composite'),
            status: getView<'uint32'>('status'),
            lastStep: lastStep !== undefined ? getView<'float32'>('lastStep') : undefined
          }
        })
    });
  }
}
