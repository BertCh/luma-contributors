// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Static attributes preserve synchronous construction fixtures in native Node and Vite.
import crashes from '../../data/n-chicago-loop-crashes.json' with {type: 'json'};
import odFlows from '../../data/n-chicago-od-flows.json' with {type: 'json'};
import polylines from '../../data/n-chicago-loop-polylines.json' with {type: 'json'};
import traces from '../../data/n-chicago-loop-traces.json' with {type: 'json'};
export {crashes, odFlows, polylines, traces};
