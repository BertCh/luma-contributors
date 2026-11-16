// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';

/** Finds the first count level's input without depending on a particular kernel builder. */
export function getPairCountOverflowPartials(node: GPUCommandNode): GraphDataView<'uint32'> {
  const inputs = (node.resources ?? []).filter(
    resource => 'buffer' in resource && resource.usage === 'storage-read'
  );
  if (inputs.length !== 1 || !('buffer' in inputs[0])) {
    throw new Error(`${node.id} must declare exactly one count input`);
  }
  const view = inputs[0].buffer;
  if (!(view instanceof GraphDataView) || view.format !== 'uint32') {
    throw new Error(`${node.id} count input must be a uint32 data view`);
  }
  return view as GraphDataView<'uint32'>;
}
