// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Text helpers of the regression runs: bar names written up the bars, so six or more names fit a
// narrow chart panel.

import type {RunLabel, RunText, WorldRectangle} from '../../../text';

/**
 * One name per bar, rotated 90 degrees inside the plot so the names need no room beside the bars:
 * written up from the baseline, or with `from: 'top'` hanging down from the top of the plot (for
 * bars that stay short, which leaves the lower part to a reference rule).
 */
export function createBarNames(
  rectangle: WorldRectangle,
  names: readonly string[],
  from: 'bottom' | 'top' = 'bottom'
): RunText {
  const [x0, y0, x1, y1] = rectangle;
  const labels: RunLabel[] = names.map((name, index) => ({
    text: name,
    at: [x0 + ((index + 0.5) / names.length) * (x1 - x0), from === 'top' ? y1 : y0],
    anchor: from === 'top' ? 'right' : 'left',
    angle: 90,
    offset: [0, from === 'top' ? -4 : 4],
    role: 'tick',
    tone: 'ink',
    priority: 1
  }));
  return {labels};
}
