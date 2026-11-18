// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  resolveRasterOperationBand,
  type RasterOperationBand
} from '../raster-grid/raster-operation-band';
import {
  GPURasterFractionalZonalDistribution,
  type GPURasterFractionalZonalDistributionProps,
  type GPURasterFractionalZonalDistributionOutput
} from '../gpu-raster-fractional-zonal-distribution';

/** Operation ports of the fractional distribution, all packed scalar vectors. */
export type RasterFractionalZonalDistributionOperationProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: RasterOperationBand;
  zoneCount: number;
  /** Explicit length of the probabilities graph value. */
  probabilityCount: number;
  observationWeights?: string;
  cellAreas?: string;
  parameters?: string;
  /** Default uniform; non-uniform ground areas require an axis-aligned grid and no cellAreas. */
  cellSizeMode?: GPURasterFractionalZonalDistributionProps['cellSizeMode'];
  quantileMethod?: 'exactextract' | 'inverse-cdf';
  coverage: {
    zoneIds: string;
    cellIds: string;
    fractions: string;
    zoneValidity: string;
    status: {count: string; requiredCount: string; overflow: string; candidateOverflow?: string};
  };
  probabilities: string;
  output: {[Port in keyof GPURasterFractionalZonalDistributionOutput]: string};
};

/** Operation form of the vector-backed fractional zonal distribution contributor. */
export class GPURasterFractionalZonalDistributionOperation extends GPUContributorOperation {
  constructor(props: RasterFractionalZonalDistributionOperationProps) {
    const id = props.id ?? 'raster-fractional-zonal-distribution';
    super({
      id,
      inputs: {
        values: props.input.storage.values,
        probabilities: props.probabilities,
        zoneIds: props.coverage.zoneIds,
        cellIds: props.coverage.cellIds,
        fractions: props.coverage.fractions,
        zoneValidity: props.coverage.zoneValidity,
        count: props.coverage.status.count,
        requiredCount: props.coverage.status.requiredCount,
        overflow: props.coverage.status.overflow,
        ...(props.coverage.status.candidateOverflow !== undefined
          ? {candidateOverflow: props.coverage.status.candidateOverflow}
          : {}),
        ...(props.input.validity !== undefined ? {inputValidity: props.input.validity} : {}),
        ...(props.observationWeights !== undefined
          ? {observationWeights: props.observationWeights}
          : {}),
        ...(props.cellAreas !== undefined ? {cellAreas: props.cellAreas} : {}),
        ...(props.parameters !== undefined ? {parameters: props.parameters} : {})
      },
      outputs: Object.fromEntries(
        Object.entries(props.output).map(([port, value]) => [
          port,
          {
            value,
            format:
              port === 'validity' || port === 'status' || port === 'quantileValidity'
                ? ('uint32' as const)
                : ('float32' as const),
            length:
              props.zoneCount *
              (port === 'quantiles' || port === 'quantileValidity' ? props.probabilityCount : 1)
          }
        ])
      ),
      createContributor: getView =>
        new GPURasterFractionalZonalDistribution({
          ...props,
          id,
          cellSizeMode: props.cellSizeMode,
          input: resolveRasterOperationBand(props.input, port =>
            getView(port === 'validity' ? 'inputValidity' : port)
          ),
          probabilities: getView<'float32'>('probabilities'),
          observationWeights:
            props.observationWeights !== undefined
              ? getView<'float32'>('observationWeights')
              : undefined,
          cellAreas: props.cellAreas !== undefined ? getView<'float32'>('cellAreas') : undefined,
          parameters: props.parameters !== undefined ? getView<'float32'>('parameters') : undefined,
          coverage: {
            zoneIds: getView<'uint32'>('zoneIds'),
            cellIds: getView<'uint32'>('cellIds'),
            fractions: getView<'float32'>('fractions'),
            zoneValidity: getView<'uint32'>('zoneValidity'),
            status: {
              count: getView<'uint32'>('count'),
              requiredCount: getView<'uint32'>('requiredCount'),
              overflow: getView<'uint32'>('overflow'),
              candidateOverflow:
                props.coverage.status.candidateOverflow !== undefined
                  ? getView<'uint32'>('candidateOverflow')
                  : undefined
            }
          },
          output: {
            means: getView<'float32'>('means'),
            populationVariances: getView<'float32'>('populationVariances'),
            sampleVariances: getView<'float32'>('sampleVariances'),
            effectiveSampleSizes: getView<'float32'>('effectiveSampleSizes'),
            weightSums: getView<'float32'>('weightSums'),
            squaredWeightSums: getView<'float32'>('squaredWeightSums'),
            coveredArea: getView<'float32'>('coveredArea'),
            validArea: getView<'float32'>('validArea'),
            quantiles: getView<'float32'>('quantiles'),
            quantileValidity: getView<'uint32'>('quantileValidity'),
            validity: getView<'uint32'>('validity'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
