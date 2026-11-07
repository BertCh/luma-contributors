// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MeuseValidationOptions} from './meuse-validation.compute';

/** Stable, response-specific log-concentration ranges shared by map layers and legends. */
export const MEUSE_LOG_RESPONSE_RANGES: Record<
  MeuseValidationOptions['response'],
  [number, number]
> = {
  zinc: [4.7, 7.6],
  copper: [2.6, 4.9],
  lead: [3.6, 6.5],
  cadmium: [-1.7, 3]
};
