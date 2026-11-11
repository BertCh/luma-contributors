// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSolarPosition} from './gpu-solar-position';

/** Packed positions and getGPUSolarPositionParameterValues parameters, named as graph values. */
export type SolarPositionOperationProps = {
  id?: string;
  positions: string;
  parameters: string;
  /** Explicit number of float32x2 position rows. */
  rowCount: number;
  refraction?: boolean;
  output: {azimuth?: string; altitude?: string; daylight?: string};
};

/** GPU solar position as an operation; all props are compile-time except the input contents. */
export class GPUSolarPositionOperation extends GPUContributorOperation {
  constructor(props: SolarPositionOperationProps) {
    const id = props.id ?? 'solar-position';
    super({
      id,
      inputs: {positions: props.positions, parameters: props.parameters},
      outputs: Object.fromEntries(
        (['azimuth', 'altitude', 'daylight'] as const)
          .filter(name => props.output[name] !== undefined)
          .map(name => [
            name,
            {
              value: props.output[name]!,
              format: name === 'daylight' ? ('uint32' as const) : ('float32' as const),
              length: props.rowCount
            }
          ])
      ),
      createContributor: getView =>
        new GPUSolarPosition({
          id,
          positions: getView<'float32x2'>('positions'),
          parameters: getView<'float32'>('parameters'),
          refraction: props.refraction,
          output: {
            azimuth: props.output.azimuth !== undefined ? getView<'float32'>('azimuth') : undefined,
            altitude:
              props.output.altitude !== undefined ? getView<'float32'>('altitude') : undefined,
            daylight:
              props.output.daylight !== undefined ? getView<'uint32'>('daylight') : undefined
          }
        })
    });
  }
}
