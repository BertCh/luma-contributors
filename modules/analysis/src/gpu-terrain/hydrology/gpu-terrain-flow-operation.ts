// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  resolveRasterOperationBand,
  type RasterOperationBand
} from '../../raster-analysis/raster-grid/raster-operation-band';
import {
  GPUTerrainFlow,
  type GPUTerrainFlowProps,
  type GPUTerrainFlowStatus
} from './gpu-terrain-flow';

/** Buffer-backed graph-value props shared by the GPU operation and its CPU twin. */
export type TerrainFlowOperationProps = Omit<
  GPUTerrainFlowProps,
  'elevation' | 'parameters' | 'runoff' | 'flatContinuation' | 'output'
> & {
  elevation: RasterOperationBand;
  /** float32 value packed by getGPUTerrainFlowParameterValues; at least eight rows. */
  parameters: string;
  /** Optional float32 value, exactly one weight per cell. */
  runoff?: string;
  output: {
    filledElevation?: string;
    flowDirections?: string;
    cellClasses?: string;
    accumulation?: string;
    streams?: string;
    status?: {[Port in keyof GPUTerrainFlowStatus]?: string};
  };
};

/**
 * Buffer-backed terrain flow as a graph operation. Compile with
 * GPU_CONTRIBUTOR_OPERATION_LOWERING. Elevation, validity, parameters and runoff contents are
 * per-frame; other props are compile-time. Texture bands and optional flatContinuation are
 * omitted: the mixed executor cannot represent textures or values read and written in one stage.
 */
export class GPUTerrainFlowOperation extends GPUContributorOperation {
  constructor(props: TerrainFlowOperationProps) {
    const id = props.id ?? 'terrain-flow';
    const length = props.grid.width * props.grid.height;
    const {output} = props;
    super({
      id,
      inputs: {
        values: props.elevation.storage.values,
        parameters: props.parameters,
        ...(props.elevation.validity !== undefined ? {validity: props.elevation.validity} : {}),
        ...(props.runoff !== undefined ? {runoff: props.runoff} : {})
      },
      outputs: {
        ...Object.fromEntries(
          (['filledElevation', 'flowDirections', 'cellClasses', 'accumulation', 'streams'] as const)
            .filter(port => output[port] !== undefined)
            .map(port => [
              port,
              {
                value: output[port]!,
                format:
                  port === 'filledElevation' || port === 'accumulation' ? 'float32' : 'uint32',
                length
              }
            ])
        ),
        ...Object.fromEntries(
          Object.entries(output.status ?? {}).map(([port, value]) => [
            port,
            {value, format: 'uint32', length: 1}
          ])
        )
      },
      createContributor: getView =>
        new GPUTerrainFlow({
          ...props,
          id,
          elevation: resolveRasterOperationBand(props.elevation, getView),
          parameters: getView<'float32'>('parameters'),
          runoff: props.runoff !== undefined ? getView<'float32'>('runoff') : undefined,
          output: {
            filledElevation:
              output.filledElevation !== undefined
                ? getView<'float32'>('filledElevation')
                : undefined,
            flowDirections:
              output.flowDirections !== undefined ? getView<'uint32'>('flowDirections') : undefined,
            cellClasses:
              output.cellClasses !== undefined ? getView<'uint32'>('cellClasses') : undefined,
            accumulation:
              output.accumulation !== undefined ? getView<'float32'>('accumulation') : undefined,
            streams: output.streams !== undefined ? getView<'uint32'>('streams') : undefined,
            status: Object.fromEntries(
              Object.keys(output.status ?? {}).map(port => [port, getView<'uint32'>(port)])
            )
          }
        })
    });
  }
}
