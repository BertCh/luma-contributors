// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// This synchronous fixture is reached only through a scenario's lazy load(). Vite removes static
// JSON attributes when serving JavaScript; native Node retains them and CommonJS uses require.
import roads from '../../data/n-chicago-loop-roads.json' with {type: 'json'};
export default roads;
