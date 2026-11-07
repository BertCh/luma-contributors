// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import naturalBreaks from '@kepler.gl/utils/jenks';
import {computeNativeJenksBreaks, type NativeJenksWorkerRequest} from './native-jenks-breaks';

/** Runs the pinned native Jenks off the main thread so exact breaks never block rendering. */
self.onmessage = (event: MessageEvent<NativeJenksWorkerRequest>) => {
  self.postMessage(computeNativeJenksBreaks(event.data, naturalBreaks));
};
