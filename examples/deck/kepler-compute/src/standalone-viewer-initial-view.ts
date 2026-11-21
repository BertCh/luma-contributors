// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewerConfig} from './standalone-viewer-config';
import type {ViewerBuildResult} from './standalone-viewer-layers';
import type {ViewerViewState} from './standalone-viewer-panes';
import {getViewStateForBounds} from './standalone-viewer-model';
import {WebMercatorViewport} from '@deck.gl/core';

/** Resolve the saved camera first, then fit the visible layer geometry. */
export function getViewerInitialViewState(
  saved: ViewerConfig['mapState'],
  result: Pick<ViewerBuildResult, 'bounds' | 'geographicFit'> | null,
  width: number,
  height: number
): ViewerViewState {
  if (saved && Number.isFinite(saved.zoom))
    return {
      longitude: saved.longitude ?? 0,
      latitude: saved.latitude ?? 0,
      zoom: saved.zoom ?? 1,
      pitch: saved.pitch ?? 0,
      bearing: saved.bearing ?? 0
    };
  const fit = result?.geographicFit;
  if (fit) {
    const viewport = new WebMercatorViewport({
      width: Math.max(width, 1),
      height: Math.max(height, 1)
    }).fitBounds(
      [
        [fit.longitudeCenterDegrees - fit.longitudeSpanDegrees / 2, fit.southDegrees],
        [fit.longitudeCenterDegrees + fit.longitudeSpanDegrees / 2, fit.northDegrees]
      ],
      {maxZoom: 20}
    );
    return {longitude: viewport.longitude, latitude: viewport.latitude, zoom: viewport.zoom};
  }
  return result?.bounds
    ? getViewStateForBounds(result.bounds, width, height)
    : {longitude: 0, latitude: 0, zoom: 1};
}
