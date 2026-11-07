// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getLinearDispatchCount} from '../../analysis-core/linear-dispatch';

/** Workgroup size shared by the linear rigid-body kernels. */
export const RIGID_BODY_LINEAR_WORKGROUP_SIZE = 64;

/** Returns the x workgroup count of a one-dimensional rigid-body dispatch on `device`. */
export function getRigidBodyDispatchCount(
  operationName: string,
  elementCount: number,
  device: Pick<Device, 'limits'>,
  workgroupSize: number = RIGID_BODY_LINEAR_WORKGROUP_SIZE
): number {
  return getLinearDispatchCount(
    operationName,
    elementCount,
    workgroupSize,
    device.limits.maxComputeWorkgroupsPerDimension
  );
}
