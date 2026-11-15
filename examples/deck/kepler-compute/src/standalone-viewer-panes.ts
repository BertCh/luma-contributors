// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, _GlobeView as GlobeView, MapView, type DeckProps} from '@deck.gl/core';
import {SolidPolygonLayer} from '@deck.gl/layers';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

import {clamp} from '@math.gl/core';

/**
 * Map panes of the offline viewer: one pane (single), two side-by-side panes with synchronized
 * views (dual) or two stacked panes separated by a draggable divider (swipe), each a deck canvas.
 * Globe mode swaps MapView for GlobeView. The optional maplibre basemap is created only when a
 * style URL is configured and the page is online; otherwise panes keep the plain background.
 */
export type ViewerSplitMode = 'single' | 'dual' | 'swipe';

export type ViewerViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch?: number;
  bearing?: number;
};

export type ViewerPanesOptions = {
  container: HTMLElement;
  mode: ViewerSplitMode;
  isGlobe: boolean;
  background: string;
  basemapStyleUrl: string | null;
  isOnline: boolean;
  swipePercentage: number;
  initialViewState: ViewerViewState;
  deckProps: DeckProps<GlobeView | MapView>;
  onViewStateChange(viewState: ViewerViewState, width: number, height: number): void;
};

type Pane = {element: HTMLElement; deck: Deck<GlobeView | MapView>; basemap: maplibregl.Map | null};

export type ViewerPanes = {
  paneCount: number;
  hasBasemap: boolean;
  /** One layer list per pane (a single list is cloned into every pane). */
  setLayers(layersByPane: any[][]): void;
  getSize(): {width: number; height: number};
};

function createSphereLayer(background: string): any {
  const shade = background.match(/\d+/g)?.map(Number);
  const color =
    shade && shade.length >= 3 ? shade.slice(0, 3).map(v => Math.min(255, v + 25)) : [30, 40, 55];
  return new SolidPolygonLayer({
    id: '__globe-sphere',
    data: [
      [
        [-180, 90],
        [0, 90],
        [180, 90],
        [180, -90],
        [0, -90],
        [-180, -90]
      ]
    ],
    getPolygon: polygon => polygon,
    getFillColor: [color[0], color[1], color[2], 255],
    pickable: false
  });
}

export function createViewerPanes(options: ViewerPanesOptions): ViewerPanes {
  const {container, mode} = options;
  container.textContent = '';
  const useBasemap = Boolean(options.basemapStyleUrl) && options.isOnline && !options.isGlobe;
  const paneCount = mode === 'single' ? 1 : 2;
  let viewState: ViewerViewState = options.initialViewState;
  const panes: Pane[] = [];
  const swipePercentage = clamp(options.swipePercentage, 0, 100);

  const applyViewState = (next: ViewerViewState): void => {
    viewState = next;
    for (const pane of panes) {
      pane.deck.setProps({viewState: next});
      pane.basemap?.jumpTo({
        center: [next.longitude, next.latitude],
        zoom: next.zoom,
        bearing: next.bearing ?? 0,
        pitch: next.pitch ?? 0
      });
    }
  };

  for (let index = 0; index < paneCount; index++) {
    const element = document.createElement('div');
    element.className = 'pane';
    element.style.cssText = 'position:absolute;top:0;bottom:0;overflow:hidden;';
    if (mode === 'dual') {
      element.style.left = index === 0 ? '0' : '50%';
      element.style.width = '50%';
      if (index === 1) element.style.borderLeft = '2px solid #fff8';
    } else {
      element.style.left = '0';
      element.style.width = '100%';
      if (mode === 'swipe' && index === 1)
        element.style.clipPath = `inset(0 0 0 ${swipePercentage}%)`;
    }
    element.style.background = index === 0 || mode === 'dual' ? options.background : 'transparent';
    container.appendChild(element);

    let basemap: maplibregl.Map | null = null;
    if (useBasemap && (mode !== 'swipe' || index === 0)) {
      const basemapElement = document.createElement('div');
      basemapElement.style.cssText = 'position:absolute;inset:0;';
      element.appendChild(basemapElement);
      try {
        basemap = new maplibregl.Map({
          container: basemapElement,
          style: options.basemapStyleUrl!,
          center: [viewState.longitude, viewState.latitude],
          zoom: viewState.zoom,
          bearing: viewState.bearing ?? 0,
          pitch: viewState.pitch ?? 0,
          interactive: false,
          attributionControl: {compact: true}
        });
      } catch {
        basemap = null;
      }
    }

    const deckElement = document.createElement('div');
    deckElement.style.cssText = 'position:absolute;inset:0;';
    element.appendChild(deckElement);
    const deck = new Deck<GlobeView | MapView>({
      ...options.deckProps,
      parent: deckElement,
      views: options.isGlobe
        ? new GlobeView({id: 'globe', resolution: 36})
        : new MapView({id: 'map', repeat: true}),
      viewState,
      controller: true,
      style: {background: 'transparent'},
      onViewStateChange: ({viewState: next}) => {
        applyViewState(next as ViewerViewState);
        options.onViewStateChange(
          next as ViewerViewState,
          element.clientWidth,
          element.clientHeight
        );
      }
    });
    panes.push({element, deck, basemap});
  }

  if (mode === 'swipe') {
    const divider = document.createElement('div');
    divider.style.cssText =
      'position:absolute;top:0;bottom:0;width:6px;margin-left:-3px;background:#fffc;cursor:ew-resize;z-index:2;';
    divider.style.left = `${swipePercentage}%`;
    divider.onpointerdown = event => {
      divider.setPointerCapture(event.pointerId);
      divider.onpointermove = move => {
        const percentage = clamp((move.clientX / container.clientWidth) * 100, 0, 100);
        divider.style.left = `${percentage}%`;
        panes[1].element.style.clipPath = `inset(0 0 0 ${percentage}%)`;
      };
    };
    divider.onpointerup = () => (divider.onpointermove = null);
    container.appendChild(divider);
  }

  if (useBasemap) {
    const attribution = document.createElement('div');
    attribution.className = 'basemap-attribution';
    attribution.textContent = `Basemap style: ${new URL(options.basemapStyleUrl!).hostname} (online only)`;
    container.appendChild(attribution);
  }

  return {
    paneCount,
    hasBasemap: useBasemap,
    setLayers(layersByPane) {
      panes.forEach((pane, index) => {
        const source = layersByPane[index] ?? layersByPane[0] ?? [];
        // Each deck owns its layer instances, so share definitions by cloning.
        const layers = source.map(layer => layer.clone({}));
        pane.deck.setProps({
          layers: options.isGlobe ? [createSphereLayer(options.background), ...layers] : layers
        });
      });
    },
    getSize() {
      return {width: panes[0].element.clientWidth, height: panes[0].element.clientHeight};
    }
  };
}
