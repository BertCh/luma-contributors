// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateOption,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {TerrainAtLeastOne} from './terrain-output';
import {snapshotRasterGrid} from '../../raster-analysis/raster-grid/raster-construction-snapshot';
import {
  resolveRasterOperationBand,
  snapshotRasterOperationBand,
  type RasterOperationBand
} from '../../raster-analysis/raster-grid/raster-operation-band';
import {
  GPUTerrainFlow,
  type GPUTerrainFlowOutput,
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
  output: TerrainAtLeastOne<
    {
      filledElevation?: string;
      flowDirections?: string;
      cellClasses?: string;
      accumulation?: string;
      streams?: string;
      status?: {[Port in keyof GPUTerrainFlowStatus]?: string};
    },
    'filledElevation' | 'flowDirections' | 'cellClasses' | 'accumulation' | 'streams'
  >;
};

/** @internal Capture declared fields before either operation validates or declares ports. */
export function snapshotTerrainFlowOperationProps(
  props: TerrainFlowOperationProps
): TerrainFlowOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'grid',
    'cellSizeMode',
    'fillDepressions',
    'resolveFlats',
    'flowRouting',
    'accumulationUnits',
    'maximumFillIterations',
    'maximumFlatIterations',
    'maximumAccumulationIterations',
    'elevation',
    'parameters',
    'runoff',
    'output'
  ]);
  const capturedGrid = snapshotRasterGrid(captured.grid);
  const output = snapshotConstructionFields(captured.output, [
    'filledElevation',
    'flowDirections',
    'cellClasses',
    'accumulation',
    'streams',
    'status'
  ]);
  return Object.freeze({
    ...captured,
    grid: capturedGrid,
    elevation: snapshotRasterOperationBand(captured.elevation),
    output: {
      ...output,
      status:
        output.status &&
        snapshotConstructionFields(output.status, [
          'fillConverged',
          'flatsConverged',
          'accumulationConverged',
          'fillIterationCount',
          'flatsIterationCount',
          'accumulationIterationCount'
        ])
    }
  });
}

/** @internal The logical selection follows the same route relationships as the contributor. */
export function validateTerrainFlowOperationOutput(props: TerrainFlowOperationProps): void {
  const id = props.id ?? 'terrain-flow';
  const {output} = props;
  const status = output.status;
  validateRequirement(
    id,
    [
      output.filledElevation,
      output.flowDirections,
      output.cellClasses,
      output.accumulation,
      output.streams
    ].some(value => value !== undefined),
    'output requires at least one per-cell output'
  );
  validateRequirement(
    id,
    !(
      output.filledElevation !== undefined ||
      status?.fillConverged !== undefined ||
      status?.fillIterationCount !== undefined
    ) || Boolean(props.fillDepressions),
    'output.filledElevation and fill status require fillDepressions'
  );
  validateRequirement(
    id,
    !(status?.flatsConverged !== undefined || status?.flatsIterationCount !== undefined) ||
      Boolean(props.resolveFlats),
    'output.status flat words require resolveFlats'
  );
  validateRequirement(
    id,
    !(
      status?.accumulationConverged !== undefined ||
      status?.accumulationIterationCount !== undefined
    ) ||
      output.accumulation !== undefined ||
      output.streams !== undefined,
    'output.status accumulation words require output.accumulation or output.streams'
  );
}

/**
 * Buffer-backed terrain flow as a graph operation. Compile with
 * GPU_CONTRIBUTOR_OPERATION_LOWERING. Elevation, validity, parameters and runoff contents are
 * per-frame; other props are compile-time. Texture bands and optional flatContinuation are
 * omitted: the mixed executor cannot represent textures or values read and written in one stage.
 */
export class GPUTerrainFlowOperation extends GPUContributorOperation {
  constructor(props: TerrainFlowOperationProps) {
    props = snapshotTerrainFlowOperationProps(props);
    const id = props.id ?? 'terrain-flow';
    if (props.flowRouting !== undefined)
      validateOption(id, 'flowRouting', props.flowRouting, [
        'd8',
        'd-infinity',
        'mfd-freeman',
        'mfd-quinn'
      ]);
    if (props.cellSizeMode !== undefined)
      validateOption(id, 'cellSizeMode', props.cellSizeMode, [
        'uniform',
        'web-mercator',
        'geographic'
      ]);
    if (props.accumulationUnits !== undefined)
      validateOption(id, 'accumulationUnits', props.accumulationUnits, ['cells', 'area']);
    validateTerrainFlowOperationOutput(props);
    const length = props.grid.width * props.grid.height;
    const {output} = props;
    const statusEntries = Object.entries(output.status ?? {}).filter(
      ([, value]) => value !== undefined
    );
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
          statusEntries.map(([port, value]) => [port, {value, format: 'uint32', length: 1}])
        )
      },
      createContributor: getView => {
        const views = {
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
          status: Object.fromEntries(statusEntries.map(([port]) => [port, getView<'uint32'>(port)]))
        };
        // The named selection was validated above. Preserve a definite selected writer without
        // asking a resolver (which may allocate) for the same port a second time.
        const contributorOutput: GPUTerrainFlowOutput =
          views.filledElevation !== undefined
            ? {...views, filledElevation: views.filledElevation}
            : views.flowDirections !== undefined
              ? {...views, flowDirections: views.flowDirections}
              : views.cellClasses !== undefined
                ? {...views, cellClasses: views.cellClasses}
                : views.accumulation !== undefined
                  ? {...views, accumulation: views.accumulation}
                  : {...views, streams: views.streams!};
        return new GPUTerrainFlow({
          ...props,
          id,
          elevation: resolveRasterOperationBand(props.elevation, getView),
          parameters: getView<'float32'>('parameters'),
          runoff: props.runoff !== undefined ? getView<'float32'>('runoff') : undefined,
          output: contributorOutput
        });
      }
    });
  }
}
