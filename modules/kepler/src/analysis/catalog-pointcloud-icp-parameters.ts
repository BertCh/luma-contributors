// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const POINT_CLOUD_ICP_ID = 'pointcloud-icp';

export const POINT_CLOUD_ICP_MAXIMUM_ITERATIONS_PARAMETER = Object.freeze({
  name: 'maximumIterations',
  label: 'Maximum registration iterations',
  kind: 'integer',
  defaultValue: 10,
  minimum: 1,
  maximum: 30
} as const);

export const REMOVED_POINT_CLOUD_ICP_ITERATIONS_MESSAGE =
  'iterations was removed; use maximumIterations';

/** Targeted diagnostic for the retired catalog parameter. */
export function getRemovedPointCloudIcpParameterMessage(
  operationId: string | undefined,
  parameterName: string
): string | undefined {
  if (operationId === POINT_CLOUD_ICP_ID && parameterName === 'iterations')
    return REMOVED_POINT_CLOUD_ICP_ITERATIONS_MESSAGE;
  return undefined;
}
