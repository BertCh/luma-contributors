// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type * as arrow from 'apache-arrow';
import {makeDecimal128WordVectors as uploadDecimal128WordVectors} from '@luma-contributors/analysis/dataframe-analysis';

export {
  aggregateDecimal128Column,
  aggregateDecimal128OnGpu,
  makeDecimal128SelectionExpression
} from '@luma-contributors/analysis/dataframe-analysis';

/**
 * Compatibility entry for the example. The public uploader handles Arrow's already-sliced
 * values and owns its buffers until the caller destroys the vectors (or calls destroy()).
 * No unused validity buffer is allocated; retain that optional legacy cleanup slot.
 */
export function makeDecimal128WordVectors(
  device: Device,
  column: arrow.Vector
): ReturnType<typeof uploadDecimal128WordVectors> & {
  unusedValidity: ReturnType<typeof uploadDecimal128WordVectors>['validity'];
} {
  return {
    ...uploadDecimal128WordVectors(device, column as arrow.Vector<arrow.Decimal>),
    unusedValidity: undefined
  };
}
