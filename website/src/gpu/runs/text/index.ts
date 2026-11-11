// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Text in run pictures: labels, swatches and rules drawn on the GPU from a generated font atlas,
// so posters and live pictures show the same labels. A run lists them as a `text` layer in `draw`
// (after the wiring in define-run.ts); the helpers in ./labels build the common cases.

import type {RunText} from './text-layout';

export {getFontPixels, loadFontPixels} from './font';
export {FONT_LICENSE, FONT_METRICS} from './font-metrics';
export {getTextPalette, TextLayer, type TextLayerOptions} from './text-layer';
export {
  CARD_DISPLAY_WIDTH,
  getRoleSize,
  getSwatchWidth,
  getTextScale,
  getTextWidth,
  layoutText,
  measureText,
  type RunLabel,
  type RunRule,
  type RunText,
  type TextAnchor,
  type TextLayout,
  type TextRole,
  type TextSpace,
  type TextTone
} from './text-layout';
export {
  createBarValueLabels,
  createChartAxes,
  createLegend,
  createMatrixTicks,
  DEFAULT_CHART_FRACTIONS,
  formatTick,
  getChartRectangle,
  getNiceTicks,
  mergeText,
  type BarLabelsOptions,
  type CategoryAxis,
  type ChartAxes,
  type ChartAxesOptions,
  type LegendEntry,
  type LegendOptions,
  type MatrixTicksOptions,
  type ValueAxis,
  type WorldRectangle
} from './labels';

/** The layer descriptor a run puts in `draw`, usually last: `{layer: 'text', ...createLegend(...)}`. */
export type TextRunLayer = {
  layer: 'text';
  /** Input text: drawn fainter and kept in the inputs-only render. */
  context?: boolean;
  alpha?: number;
  /** Swipe gestures: keep only the part left or right of world x, like the other layers. */
  clip?: {side: 'left' | 'right'; x: number};
} & RunText;
