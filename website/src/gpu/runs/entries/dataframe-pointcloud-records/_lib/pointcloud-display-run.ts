// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {DemoTheme} from '../../../../demos/kit';
import type {RunBuilder, RunBuffer} from '../../../define-run';
import {
  createPointCloudDisplayCopyNode,
  createPointCloudGroundClassNode
} from './pointcloud-display';

export function addPointCloudDisplayCopy(
  builder: RunBuilder,
  id: string,
  source: RunBuffer,
  status: RunBuffer,
  length: number,
  options: {
    componentStart?: number;
    components?: 1 | 2;
    invalidBits?: number;
    statusIndex?: number;
  } = {}
): RunBuffer {
  const output = builder.allocate(id, options.components === 2 ? 'float32x2' : 'float32', length);
  builder.graph.add(
    createPointCloudDisplayCopyNode(builder.graph, {
      id,
      source: source.view,
      status: status.view as GraphDataView<'uint32'>,
      output: output.view,
      ...options
    })
  );
  return output;
}

/** Per-point ground filter class (see createPointCloudGroundClassNode) as a drawable float buffer. */
export function addGroundClasses(
  builder: RunBuilder,
  mask: RunBuffer,
  status: RunBuffer,
  length: number,
  unclassifiedBits: number
): RunBuffer {
  const output = builder.allocate('ground-classes', 'float32', length);
  builder.graph.add(
    createPointCloudGroundClassNode(builder.graph, {
      id: 'ground-classes',
      mask: mask.view,
      status: status.view,
      output: output.view,
      unclassifiedBits
    })
  );
  return output;
}

/** Colors for addGroundClasses: removed blue, ground orange, and both faded when provisional. */
export function groundClassLook(theme: DemoTheme) {
  const [removed, ground] = theme.categories;
  return {
    ramp: 'categorical' as const,
    categories: [removed, ground, [...removed, 0.3] as const, [...ground, 0.3] as const]
  };
}
