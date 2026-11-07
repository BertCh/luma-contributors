// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {computeColumnStats, timeToUnixMilli} from '@kepler.gl/utils';
import {setColumnStatsComputer} from './resident-column-statistics';
import {setTimeToUnixMilliConverter} from './native-time-converter';

// Host registration: node-tested modules must not value-import @kepler.gl/utils because it loads a
// second deck.gl copy. Importing this module from the browser entry installs the native functions.
setColumnStatsComputer(computeColumnStats);
setTimeToUnixMilliConverter(timeToUnixMilli);
