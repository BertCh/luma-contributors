// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import process from 'process';

// Kepler's browser dependencies include Node's util shim. Initialize its browser process
// implementation before evaluating the application dependency graph (including util).
Object.assign(globalThis, {process});
void import('./main');
