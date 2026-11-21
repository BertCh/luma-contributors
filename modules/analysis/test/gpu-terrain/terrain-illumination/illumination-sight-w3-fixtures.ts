// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';

/** A single eastward rise for the clockwise-bin regression. */
export const POINT_PROFILE_GRID = makeGPURasterMetadata({width: 4, height: 3, cellSize: 1});
export const POINT_PROFILE_ELEVATION = new Float32Array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
