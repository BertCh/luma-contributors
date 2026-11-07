// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MediaExportHost, VideoViewState} from './native-video-export';

type KeplerStoreLike = {
  getState(): {
    keplerGl: Record<
      string,
      | {
          mapState?: Partial<VideoViewState>;
          visState?: {animationConfig?: {domain?: number[] | null; currentTime?: number | null}};
        }
      | undefined
    >;
  };
  dispatch(action: unknown): unknown;
};

/** Bind the media exporters to the compute store. Dispatches only map viewport and animation time. */
export function makeKeplerMediaExportHost(
  store: KeplerStoreLike,
  instanceId: string,
  actions: {
    updateViewState(viewState: VideoViewState): unknown;
    setAnimationTime(time: number): unknown;
  }
): MediaExportHost {
  const read = () => store.getState().keplerGl[instanceId];
  return {
    getCanvas: () =>
      document.querySelector<HTMLCanvasElement>(
        '.maps canvas:not(.maplibregl-canvas):not(.mapboxgl-canvas)'
      ),
    getViewState: () => {
      const mapState = read()?.mapState ?? {};
      return {
        longitude: mapState.longitude ?? 0,
        latitude: mapState.latitude ?? 0,
        zoom: mapState.zoom ?? 0,
        bearing: mapState.bearing ?? 0,
        pitch: mapState.pitch ?? 0
      };
    },
    setViewState: viewState => void store.dispatch(actions.updateViewState(viewState)),
    getAnimationRange: () => {
      const domain = read()?.visState?.animationConfig?.domain;
      return domain && domain.length === 2 && domain[1] > domain[0] ? [domain[0], domain[1]] : null;
    },
    getAnimationTime: () => read()?.visState?.animationConfig?.currentTime ?? null,
    setAnimationTime: time => void store.dispatch(actions.setAnimationTime(time)),
    // Deck presents asynchronously; two animation frames cover state commit, render and submit.
    waitForRender: () =>
      new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
  };
}
