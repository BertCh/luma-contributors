// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A small text badge in a canvas corner: plain lines of text on a translucent panel, for a count or
// a note that has no color to key (the legend helper always draws a swatch).

import {getTextWidth, type RunLabel, type RunRule, type RunText} from '../../../text';

const LINE_HEIGHT = 13;

export type BadgeOptions = {
  lines: readonly string[];
  /** Default `bottom-left`. */
  corner?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  /** Distance from the canvas edges in text pixels. Default 6. */
  inset?: number;
};

export function createBadge({lines, corner = 'bottom-left', inset = 6}: BadgeOptions): RunText {
  const width = Math.max(...lines.map(line => getTextWidth(line, 'label')));
  const height = lines.length * LINE_HEIGHT;
  const isRight = corner.endsWith('right');
  const isBottom = corner.startsWith('bottom');
  const at: readonly [number, number] = [isRight ? 1 : 0, isBottom ? 1 : 0];
  const left = isRight ? -(inset + width) : inset;
  const top = isBottom ? inset + height : -inset;
  const padding = 4;
  const rules: RunRule[] = [
    {
      from: at,
      to: at,
      space: 'canvas',
      fromOffset: [left - padding, top - height / 2],
      toOffset: [left + width + padding, top - height / 2],
      width: height + 2 * padding,
      tone: 'background',
      opacity: 0.78
    }
  ];
  const labels: RunLabel[] = lines.map((text, line) => ({
    text,
    at,
    space: 'canvas',
    anchor: 'left',
    offset: [left, top - (line + 0.5) * LINE_HEIGHT],
    role: 'label',
    priority: 4
  }));
  return {labels, rules};
}
