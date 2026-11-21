// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSolarPosition} from './gpu-solar-position';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {TerrainAtLeastOne} from '../hydrology/terrain-output';

/** Packed positions and getGPUSolarPositionParameterValues parameters, named as graph values. */
export type SolarPositionOperationProps = {
  id?: string;
  positions: string;
  parameters: string;
  /** Explicit number of float32x2 position rows. */
  rowCount: number;
  refraction?: boolean;
  output: TerrainAtLeastOne<{azimuth?: string; altitude?: string; daylight?: string}>;
};

/** @internal Capture shared CPU/GPU topology, including inherited declared getters. */
export function snapshotSolarPositionOperationProps(
  props: SolarPositionOperationProps
): SolarPositionOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'positions',
    'parameters',
    'rowCount',
    'refraction',
    'output'
  ]);
  const output = snapshotConstructionFields(captured.output, ['azimuth', 'altitude', 'daylight']);
  validateRequirement(
    captured.id ?? 'solar-position',
    output.azimuth !== undefined || output.altitude !== undefined || output.daylight !== undefined,
    'output must contain azimuth, altitude or daylight; received an empty selection'
  );
  return Object.freeze({...captured, output});
}

/** GPU solar position as an operation; all props are compile-time except the input contents. */
export class GPUSolarPositionOperation extends GPUContributorOperation {
  constructor(props: SolarPositionOperationProps) {
    props = snapshotSolarPositionOperationProps(props);
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
      createContributor: getView => {
        const output = {
          azimuth: props.output.azimuth !== undefined ? getView<'float32'>('azimuth') : undefined,
          altitude:
            props.output.altitude !== undefined ? getView<'float32'>('altitude') : undefined,
          daylight: props.output.daylight !== undefined ? getView<'uint32'>('daylight') : undefined
        };
        return new GPUSolarPosition({
          id,
          positions: getView<'float32x2'>('positions'),
          parameters: getView<'float32'>('parameters'),
          refraction: props.refraction,
          output:
            output.azimuth !== undefined
              ? {...output, azimuth: output.azimuth}
              : output.altitude !== undefined
                ? {...output, altitude: output.altitude}
                : {...output, daylight: getView<'uint32'>('daylight')}
        });
      }
    });
  }
}
