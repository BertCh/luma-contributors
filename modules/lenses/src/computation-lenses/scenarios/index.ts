// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers every lens scenario (side-effect imports). Append-only: one line per entry.
// The array result view and shared core formulas that most scenarios render with.
import '../../computation-views/packs/gpu-core/index';
import './gpu-spatial-analysis/index';
import './gpu-network/index';
import './gpu-graph/index';
import './gpu-terrain/index';
import './gpu-raster/index';
import './gpu-dataframe/index';
