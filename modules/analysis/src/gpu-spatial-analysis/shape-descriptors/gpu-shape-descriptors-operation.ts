// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RequiredGeometryOutput} from '../group-geometry/required-geometry-output';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {validateShapeDescriptorMeasurements} from './shape-descriptors-measurement-validation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  GPUShapeDescriptors,
  type GPUShapeDescriptorsProps,
  type GPUShapeDescriptorsOutput
} from './gpu-shape-descriptors';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';

/** Graph names in place of polygon/parameter views, with explicit output row count. */
export type ShapeDescriptorsOperationProps = Pick<
  GPUShapeDescriptorsProps,
  'id' | 'spatialContext' | 'convexityMethod' | 'maximumVerticesPerFeature' | 'momentMethod'
> & {
  /** CPU host interpretation, defaulting to resident-capacity for graph inputs. */
  storage?: 'compact' | 'resident-capacity';
  positions: string;
  ringOffsets: string;
  polygonOffsets: string;
  featureOffsets: string;
  parameters: string;
  featureCount: number;
  output: RequiredGeometryOutput<{[Column in keyof GPUShapeDescriptorsOutput]?: string}>;
};

/** @internal Capture supported columns by name for both operation backends. */
export function snapshotShapeDescriptorsOperation(
  props: ShapeDescriptorsOperationProps
): ShapeDescriptorsOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'convexityMethod',
    'maximumVerticesPerFeature',
    'momentMethod',
    'storage',
    'positions',
    'ringOffsets',
    'polygonOffsets',
    'featureOffsets',
    'parameters',
    'featureCount',
    'output'
  ]);
  const output = snapshotConstructionFields(snapshot.output, [
    'areas',
    'perimeters',
    'polsbyPopper',
    'schwartzberg',
    'elongation',
    'orientation',
    'convexity',
    'clockwise',
    'sliver'
  ]);
  validateRequirement(
    snapshot.id ?? 'shape-descriptors',
    Object.values(output).some(value => value !== undefined),
    'output must name at least one terminal column'
  );
  validateRequirement(
    snapshot.id ?? 'shape-descriptors',
    snapshot.storage === undefined || ['compact', 'resident-capacity'].includes(snapshot.storage),
    `storage must be compact or resident-capacity; received ${snapshot.storage}`
  );
  const spatialContext = resolveGPUSpatialContext(
    snapshot.id ?? 'shape-descriptors',
    snapshot.spatialContext
  );
  validateShapeDescriptorMeasurements(snapshot.id ?? 'shape-descriptors', spatialContext, output);
  return Object.freeze({...snapshot, spatialContext, output});
}

/** Operation form of GPUShapeDescriptors; uses GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUShapeDescriptorsOperation extends GPUContributorOperation {
  constructor(props: ShapeDescriptorsOperationProps) {
    props = snapshotShapeDescriptorsOperation(props);
    const id = props.id ?? 'shape-descriptors';
    super({
      emptyOutputPorts: Object.keys(props.output),
      id,
      inputs: {
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        featureOffsets: props.featureOffsets,
        parameters: props.parameters
      },
      outputs: Object.fromEntries(
        Object.entries(props.output)
          .filter(([, value]) => value !== undefined)
          .map(([port, value]) => [
            port,
            {
              value: value!,
              format: port === 'clockwise' || port === 'sliver' ? 'uint32' : 'float32',
              length: props.featureCount
            }
          ])
      ),
      createContributor: getView =>
        new GPUShapeDescriptors({
          ...props,
          id,
          positions: getView<'float32x2'>('positions'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          polygonOffsets: getView<'uint32'>('polygonOffsets'),
          featureOffsets: getView<'uint32'>('featureOffsets'),
          parameters: getView<'float32'>('parameters'),
          output: Object.fromEntries(
            Object.entries(props.output)
              .filter(([, value]) => value !== undefined)
              .map(([port]) => [
                port,
                getView(port) as GraphDataView<'float32'> | GraphDataView<'uint32'>
              ])
          ) as GPUShapeDescriptorsOutput
        })
    });
  }
}
