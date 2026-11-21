// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getExactFloat32Mean} from './value-cube-oracle';

/** Publishes one mean per hand-enumerated composition bin, including NaN for empty bins. */
export function getValueCubeMeanValues(bins: readonly (readonly number[])[]): Float32Array {
  return Float32Array.from(bins, values => getExactFloat32Mean(values));
}
