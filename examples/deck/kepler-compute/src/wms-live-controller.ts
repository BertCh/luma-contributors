// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WMSLayer as DeckWmsLayer} from '@kepler.gl/deckgl-layers';
import {transformMercatorToLngLat} from './raster-reprojection';
import {
  buildWmsGetFeatureInfoUrl,
  classifyWmsResponseError,
  isWmsTimeSupported,
  parseWmsFeatureInfoResponse,
  parseWmsTimeExtent,
  type WmsCrs,
  type WmsVersion
} from './wms-identify';

type ImageSourceLike = {vendorParameters?: Record<string, unknown>};
type DeckWmsInstance = {
  props: {id: string};
  context: {viewport: unknown};
  state: {
    imageSource?: ImageSourceLike;
    lastRequestParameters?: unknown;
    inFlightRequestParameters?: unknown;
  };
  loadImage(viewport: unknown, reason: string): Promise<void>;
};
type DeckWmsPrototype = {
  _createImageSource(props: unknown): ImageSourceLike;
  [key: symbol]: boolean | undefined;
};

type RequestParametersLike = {
  bbox?: number[];
  width: number;
  height: number;
  layers: string[];
  srs: string;
};

type KeplerWmsLayerLike = {
  id: string;
  type: string;
  meta?: {tilesetDataUrl?: string; wmsVersion?: string};
  deckLayerRef?: {state?: {lastRequestParameters?: RequestParametersLike}};
};

export type WmsLiveIdentifyRow = {name: string; value: string};

export type WmsLiveState = Readonly<{
  layerId: string;
  time: string | null;
  lastIdentifyStatus: 'none' | 'ok' | 'empty' | 'service-exception' | 'http-error' | 'network';
  lastIdentifyUrl: string | null;
  lastMapUrlTime: string | null;
}>;

const PATCH_MARKER = Symbol.for('luma.kepler.wmsLive');
const IDENTIFY_FORMATS = ['application/json', 'application/vnd.ogc.gml', 'text/plain'];

/** Kepler names the deck layer `${keplerLayerId}-WMSLayer` (plus `-globe` in globe mode). */
function getKeplerLayerId(deckLayerId: string): string {
  return deckLayerId.replace(/-globe$/, '').replace(/-WMSLayer$/, '');
}

function normalizeWmsVersion(version: string | undefined): WmsVersion {
  return version === '1.1.1' ? '1.1.1' : '1.3.0';
}

/**
 * Live WMS integration for the WebGPU host. Kepler renders a WMS layer as one viewport image
 * (deck `BitmapLayer`, a plain texture, so the draw path is the host's WebGPU BitmapLayer).
 * This controller adds what Kepler lacks: a TIME parameter injected into every GetMap and
 * GetFeatureInfo request, and a GetFeatureInfo identify that tries JSON, GML then text, handles
 * 1.1.1 X/Y versus 1.3.0 I/J, and reports service exceptions instead of silently returning nothing.
 */
export class WmsLiveController {
  private readonly timeByLayerId = new Map<string, string>();
  private readonly states = new Map<string, WmsLiveState>();

  constructor(
    private readonly options: {
      getLayers(): readonly unknown[];
      /** Live deck layer instances (for example `deck.layerManager.getLayers()`). */
      getDeckLayers(): readonly unknown[];
      fetchText?(url: string): Promise<{status: number; text: string}>;
    }
  ) {
    this.installPrototypePatches();
  }

  private installPrototypePatches(): void {
    const prototype = DeckWmsLayer.prototype as unknown as DeckWmsPrototype;
    if (prototype[PATCH_MARKER]) return;
    const original = prototype._createImageSource;
    const controller = this;
    prototype._createImageSource = function createImageSource(
      this: DeckWmsInstance,
      props: unknown
    ) {
      const imageSource = original.call(this, props);
      controller.applyTime(imageSource, getKeplerLayerId(this.props.id));
      return imageSource;
    };
    prototype[PATCH_MARKER] = true;
  }

  /** Patches `getWMSFeatureInfo` on the prototype of a Kepler WMS layer instance (once). */
  installIdentify(sampleKeplerLayer: object): void {
    const prototype = Object.getPrototypeOf(sampleKeplerLayer) as {
      getWMSFeatureInfo(x: number, y: number): Promise<WmsLiveIdentifyRow[] | null>;
      [key: symbol]: boolean | undefined;
    };
    if (prototype[PATCH_MARKER]) return;
    const controller = this;
    prototype.getWMSFeatureInfo = function getWMSFeatureInfo(
      this: KeplerWmsLayerLike,
      pixelX: number,
      pixelY: number
    ) {
      return controller.identifyPixel(this, pixelX, pixelY);
    };
    prototype[PATCH_MARKER] = true;
  }

  private applyTime(imageSource: ImageSourceLike, keplerLayerId: string): void {
    const time = this.timeByLayerId.get(keplerLayerId);
    const {TIME: _previous, ...rest} = imageSource.vendorParameters ?? {};
    imageSource.vendorParameters = time ? {...rest, TIME: time} : rest;
  }

  getTime(layerId: string): string | null {
    return this.timeByLayerId.get(layerId) ?? null;
  }

  /**
   * Sets (or clears with null) the TIME dimension for a layer, validating against a capabilities
   * extent when one is supplied, then forces the live deck layer to refetch its image.
   * Returns false when the time is outside the extent.
   */
  setTime(layerId: string, time: string | null, extentText?: string): boolean {
    if (time && extentText && !isWmsTimeSupported(parseWmsTimeExtent(extentText), time))
      return false;
    if (time) this.timeByLayerId.set(layerId, time);
    else this.timeByLayerId.delete(layerId);
    this.patchState(layerId, {time});
    for (const candidate of this.options.getDeckLayers()) {
      const instance = candidate as DeckWmsInstance;
      if (!instance?.props?.id?.startsWith(layerId) || !instance.state?.imageSource) continue;
      if (getKeplerLayerId(instance.props.id) !== layerId) continue;
      this.applyTime(instance.state.imageSource, layerId);
      // Same view, new time: clear the coverage cache or loadImage would skip the request.
      instance.state.lastRequestParameters = undefined;
      instance.state.inFlightRequestParameters = undefined;
      void instance.loadImage(instance.context.viewport, 'time changed');
    }
    return true;
  }

  getState(layerId: string): WmsLiveState {
    return (
      this.states.get(layerId) ?? {
        layerId,
        time: null,
        lastIdentifyStatus: 'none',
        lastIdentifyUrl: null,
        lastMapUrlTime: null
      }
    );
  }

  private patchState(layerId: string, patch: Partial<WmsLiveState>): void {
    this.states.set(layerId, {...this.getState(layerId), ...patch});
  }

  /** GetFeatureInfo for a pixel of the last loaded image; rows feed Kepler's WMS tooltip. */
  async identifyPixel(
    layer: KeplerWmsLayerLike,
    pixelX: number,
    pixelY: number
  ): Promise<WmsLiveIdentifyRow[] | null> {
    const request = layer.deckLayerRef?.state?.lastRequestParameters;
    const serviceUrl = layer.meta?.tilesetDataUrl;
    if (!request?.bbox || !serviceUrl) return null;
    const crs = (request.srs === 'EPSG:3857' ? 'EPSG:3857' : 'EPSG:4326') as WmsCrs;
    const [west, south, east, north] =
      crs === 'EPSG:3857'
        ? [
            ...transformMercatorToLngLat(request.bbox[0], request.bbox[1]),
            ...transformMercatorToLngLat(request.bbox[2], request.bbox[3])
          ]
        : request.bbox;
    const time = this.timeByLayerId.get(layer.id);
    const fetchText =
      this.options.fetchText ??
      (async (url: string) => {
        const response = await fetch(url);
        return {status: response.status, text: await response.text()};
      });
    let lastUrl = '';
    for (const infoFormat of IDENTIFY_FORMATS) {
      lastUrl = buildWmsGetFeatureInfoUrl({
        serviceUrl,
        version: normalizeWmsVersion(layer.meta?.wmsVersion),
        layers: request.layers,
        crs,
        extent: [west, south, east, north],
        width: request.width,
        height: request.height,
        pixelX,
        pixelY,
        infoFormat,
        time
      });
      let response: {status: number; text: string};
      try {
        response = await fetchText(lastUrl);
      } catch {
        this.patchState(layer.id, {lastIdentifyStatus: 'network', lastIdentifyUrl: lastUrl});
        return [{name: 'WMS', value: 'Feature info request failed (network)'}];
      }
      const classification = classifyWmsResponseError({
        status: response.status,
        bodyText: response.text
      });
      if (classification.kind === 'http-error' || classification.kind === 'service-exception') {
        // Some servers only accept one info format; try the next before reporting.
        if (infoFormat !== IDENTIFY_FORMATS[IDENTIFY_FORMATS.length - 1]) continue;
        this.patchState(layer.id, {
          lastIdentifyStatus: classification.kind,
          lastIdentifyUrl: lastUrl
        });
        return [
          {
            name: 'WMS',
            value:
              classification.exception?.message ??
              `Feature info request failed (${response.status})`
          }
        ];
      }
      const parsed = parseWmsFeatureInfoResponse(response.text);
      if (parsed.features.length === 0) continue;
      this.patchState(layer.id, {lastIdentifyStatus: 'ok', lastIdentifyUrl: lastUrl});
      const isSingle = parsed.features.length === 1;
      const rows = parsed.features.flatMap(feature =>
        feature.attributes.map(attribute => ({
          name: isSingle ? attribute.name : `${feature.name}.${attribute.name}`,
          value: attribute.value
        }))
      );
      if (time) rows.push({name: 'TIME', value: time});
      return rows;
    }
    this.patchState(layer.id, {lastIdentifyStatus: 'empty', lastIdentifyUrl: lastUrl});
    return null;
  }
}
