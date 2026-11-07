// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {setSelectedFeature} from '@kepler.gl/actions';
import type {ResidentDrawingPanelProps} from './resident-drawing-panel';

export type NativeFeature = Parameters<typeof setSelectedFeature>[0] & object;

/**
 * Kepler's editor state types features by its own `Feature` shape while the drawing panel declares
 * the smaller structural view it needs. The two describe the same GeoJSON features; these adapters
 * are the single place the shapes are reconciled.
 */
export function toDrawnFeatures(
  features: readonly object[] | undefined
): ResidentDrawingPanelProps['features'] {
  return (features ?? []) as unknown as ResidentDrawingPanelProps['features'];
}

export function toDrawnFeature(
  feature: object | null | undefined
): ResidentDrawingPanelProps['selectedFeature'] {
  return (feature ?? null) as unknown as ResidentDrawingPanelProps['selectedFeature'];
}

export function toSelectedNativeFeature(
  feature: object | null
): Parameters<typeof setSelectedFeature>[0] {
  return feature as unknown as Parameters<typeof setSelectedFeature>[0];
}

export function toNativeFeature(feature: object): NativeFeature {
  return feature as NativeFeature;
}
