// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerExtension} from '@deck.gl/core';
import {PathStyleExtension} from '@deck.gl/extensions';

/**
 * Dashed strokes for the native editor (Kepler uses `PathStyleExtension({dash: true})`).
 *
 * deck.gl 10 ships WGSL for `PathStyleExtension` (segment-relative dashes in `widths` units, butt
 * and round caps, gap picking), so the host uses it as is on WebGPU and WebGL. This module only
 * keeps the previous status contract: a shader or pipeline failure disables the dash for later
 * editor layers, which then render solid strokes.
 */

let disabledReason: string | null = null;

/** Stop installing the dash extension (for example after a shader build failure). */
export function disableEditorDashExtension(reason: string): void {
  disabledReason = reason;
}

export function getEditorDashStatus(): {enabled: boolean; reason: string | null} {
  return {enabled: disabledReason === null, reason: disabledReason};
}

export function resetEditorDashStatusForTest(): void {
  disabledReason = null;
}

/**
 * Extensions for the resident editor. Empty (solid strokes) once the dash was disabled, so a
 * failed build degrades to the previous solid-stroke behavior.
 */
export function makeEditorDashExtensions(): LayerExtension[] {
  return disabledReason === null ? [new PathStyleExtension({dash: true})] : [];
}
