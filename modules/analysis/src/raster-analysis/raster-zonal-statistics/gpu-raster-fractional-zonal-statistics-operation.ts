// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPURasterFractionalZonalStatistics,
  type GPURasterFractionalZonalStatisticsProps
} from '../gpu-raster-fractional-coverage';
import {
  resolveRasterOperationBand,
  type RasterOperationBand
} from '../raster-grid/raster-operation-band';

/** Logical graph values in place of the contributor's buffer views. */
export type RasterFractionalZonalStatisticsOperationProps = Pick<
  GPURasterFractionalZonalStatisticsProps,
  'id' | 'grid' | 'zoneCount' | 'quantity'
> & {
  input: RasterOperationBand;
  parameters?: string;
  coverage: {
    zoneIds: string;
    cellIds: string;
    fractions: string;
    zoneValidity: string;
    status: {count: string; requiredCount: string; overflow: string};
  };
  output: {totals: string; means: string; coveredArea: string; validArea: string; validity: string};
};

/** Buffer operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPURasterFractionalZonalStatisticsOperation extends GPUContributorOperation {
  constructor(props: RasterFractionalZonalStatisticsOperationProps) {
    super({
      id: props.id ?? 'fractional-zonal-statistics',
      inputs: {
        values: props.input.storage.values,
        ...(props.input.validity !== undefined ? {inputValidity: props.input.validity} : {}),
        zoneIds: props.coverage.zoneIds,
        cellIds: props.coverage.cellIds,
        fractions: props.coverage.fractions,
        zoneValidity: props.coverage.zoneValidity,
        count: props.coverage.status.count,
        requiredCount: props.coverage.status.requiredCount,
        overflow: props.coverage.status.overflow,
        ...(props.parameters !== undefined ? {parameters: props.parameters} : {})
      },
      outputs: {
        totals: {value: props.output.totals, format: 'float32', length: props.zoneCount},
        means: {value: props.output.means, format: 'float32', length: props.zoneCount},
        coveredArea: {value: props.output.coveredArea, format: 'float32', length: props.zoneCount},
        validArea: {value: props.output.validArea, format: 'float32', length: props.zoneCount},
        validity: {value: props.output.validity, format: 'uint32', length: props.zoneCount}
      },
      createContributor: getView =>
        new GPURasterFractionalZonalStatistics({
          ...props,
          input: resolveRasterOperationBand(props.input, port =>
            getView(port === 'validity' ? 'inputValidity' : port)
          ),
          parameters: props.parameters !== undefined ? getView<'float32'>('parameters') : undefined,
          coverage: {
            zoneIds: getView<'uint32'>('zoneIds'),
            cellIds: getView<'uint32'>('cellIds'),
            fractions: getView<'float32'>('fractions'),
            zoneValidity: getView<'uint32'>('zoneValidity'),
            status: {
              count: getView<'uint32'>('count'),
              requiredCount: getView<'uint32'>('requiredCount'),
              overflow: getView<'uint32'>('overflow')
            }
          },
          output: {
            totals: getView<'float32'>('totals'),
            means: getView<'float32'>('means'),
            coveredArea: getView<'float32'>('coveredArea'),
            validArea: getView<'float32'>('validArea'),
            validity: getView<'uint32'>('validity')
          }
        })
    });
  }
}
