// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Capture the optional output map and iteration status without cloning the graph view. @internal */
export function snapshotSURIterationOutput<Output extends {iteration?: {status: object}}>(
  id: string,
  output: Output
): Output {
  validateRequirement(
    id,
    !('iterationCount' in output),
    'output.iterationCount was removed; use output.iteration.status.iterationCount'
  );
  return Object.freeze({
    ...output,
    ...(output.iteration
      ? {iteration: Object.freeze({status: Object.freeze({...output.iteration.status})})}
      : {})
  });
}
