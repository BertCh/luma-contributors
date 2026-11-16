// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
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
  positions: string;
  ringOffsets: string;
  polygonOffsets: string;
  featureOffsets: string;
  parameters: string;
  featureCount: number;
  output: {[Column in keyof GPUShapeDescriptorsOutput]?: string};
};

/** Operation form of GPUShapeDescriptors; uses GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUShapeDescriptorsOperation extends GPUContributorOperation {
  constructor(props: ShapeDescriptorsOperationProps) {
    const id = props.id ?? 'shape-descriptors';
    super({
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
          )
        })
    });
  }
}
