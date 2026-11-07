// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {TripsLayer} from '@deck.gl/geo-layers';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {webgl2Adapter} from '@luma.gl/core/webgl';
import {
  createViewerPanes,
  type ViewerSplitMode,
  type ViewerViewState
} from './standalone-viewer-panes';
import {
  createIconAtlas,
  fetchDefaultIconDefinitions,
  type ViewerIconAtlas
} from './standalone-viewer-icons';
import {isViewerOnline} from './standalone-viewer-raster';
import {getViewStateForBounds} from './standalone-viewer-model';
import {readViewerConfig} from './standalone-viewer-config';
import {createViewerDataset, type ViewerDataset} from './standalone-viewer-data';
import {buildViewerLayers, type ViewerBuildResult} from './standalone-viewer-layers';
import {renderFilterControls, renderLegends} from './standalone-viewer-panels';

/**
 * Offline viewer: decodes the embedded Arrow sources and restores the saved layers, styles, filters
 * and legends on CPU-side scales. It imports only pure host modules (layer plans, legend models,
 * cell-layer types); the GPU-resident layers, DuckDB and the Kepler store are deliberately excluded.
 */
const status = document.getElementById('status')!;

function decodeBase64(text: string): Uint8Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function downloadBundle(json: string): void {
  const url = URL.createObjectURL(new Blob([json], {type: 'application/json'}));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'kepler-workspace.bundle.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function verifyDigest(bytes: Uint8Array, expected: string): Promise<boolean> {
  if (!globalThis.crypto?.subtle) return true; // insecure context: digest check unavailable
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource));
  return Array.from(hash, byte => byte.toString(16).padStart(2, '0')).join('') === expected;
}

async function main(): Promise<void> {
  const element = document.getElementById('luma-kepler-bundle');
  if (!element?.textContent) {
    status.textContent = 'No embedded workspace bundle found.';
    return;
  }
  const json = element.textContent;
  const bundle = JSON.parse(json);
  if (bundle.format !== 'luma-kepler-sandbox' || bundle.version !== 1) {
    status.textContent = 'Unsupported bundle format.';
    return;
  }
  const datasets = new Map<string, ViewerDataset>();
  for (const source of bundle.sources)
    datasets.set(
      source.datasetId,
      createViewerDataset(source.datasetId, arrow.tableFromIPC(decodeBase64(source.data)))
    );
  // Native-only tables (DATA2): checksummed Arrow IPC with no sources[] entry.
  const rejectedNative: string[] = [];
  for (const source of bundle.nativeOnlySources ?? []) {
    try {
      const bytes = decodeBase64(source.data);
      if (bytes.byteLength !== source.byteLength || !(await verifyDigest(bytes, source.sha256)))
        throw new Error('digest mismatch');
      datasets.set(
        source.datasetId,
        createViewerDataset(source.datasetId, arrow.tableFromIPC(bytes))
      );
    } catch {
      rejectedNative.push(source.label ?? source.datasetId);
    }
  }
  const config = readViewerConfig(bundle.state?.nativeConfig);
  // Bundles without saved layers get one default point layer per dataset with lat/lng columns.
  if (!config.layers.length) addDefaultPointLayers(config, datasets);

  const jsonCache = new Map<string, unknown[]>();
  const parseJson = (datasetId: string, fieldName: string, row: number): unknown => {
    const key = `${datasetId}\u0000${fieldName}`;
    let parsed = jsonCache.get(key);
    if (!parsed) {
      parsed = new Array(datasets.get(datasetId)!.rowCount);
      jsonCache.set(key, parsed);
    }
    if (parsed[row] === undefined) {
      const value = datasets.get(datasetId)!.readValue(fieldName, row);
      try {
        parsed[row] = typeof value === 'string' ? JSON.parse(value) : (value ?? null);
      } catch {
        parsed[row] = null;
      }
    }
    return parsed[row];
  };

  const map = document.getElementById('map')!;
  map.style.background = config.backgroundColor;
  const legendPanel = document.getElementById('legends')!;
  const filterPanel = document.getElementById('filters')!;
  let currentTime = config.animationCurrentTime ?? 0;

  // Icon meshes: custom catalog embedded in the bundle now, default CDN catalog when online.
  const customIcons = Array.isArray(bundle.iconCatalog?.icons)
    ? bundle.iconCatalog.icons
    : Array.isArray(bundle.iconCatalog)
      ? bundle.iconCatalog
      : [];
  let iconAtlas: ViewerIconAtlas | null = safeCreateIconAtlas(customIcons);

  const mapState = config.mapState;
  const isGlobe = mapState?.mapViewMode === 'globe' || Boolean(mapState?.globe?.enabled);
  const splitMode: ViewerSplitMode =
    config.splitMapLayers.length >= 2 || mapState?.isSplit
      ? mapState?.mapSplitMode === 'SWIPE_COMPARE'
        ? 'swipe'
        : 'dual'
      : 'single';
  let view: ViewerViewState = {longitude: 0, latitude: 0, zoom: 1};
  let viewSize = {width: map.clientWidth, height: map.clientHeight};
  const basemapStyleUrl = bundle.viewer?.basemapStyleUrl ?? config.basemapStyleUrl;

  function rebuild(): ViewerBuildResult {
    const built = buildViewerLayers(config.layers, {
      datasets,
      filters: config.filters,
      currentTime,
      parseJson,
      nativeConfig: bundle.state?.nativeConfig,
      iconAtlas,
      view: {...view, width: viewSize.width, height: viewSize.height}
    });
    renderLegends(legendPanel, built);
    return built;
  }

  // First build only to find bounds and the initial view; rebuilt once the panes exist.
  view =
    mapState && Number.isFinite(mapState.zoom)
      ? {
          longitude: mapState.longitude ?? 0,
          latitude: mapState.latitude ?? 0,
          zoom: mapState.zoom ?? 1,
          pitch: mapState.pitch ?? 0,
          bearing: mapState.bearing ?? 0
        }
      : {longitude: 0, latitude: 0, zoom: 1};
  let result: ViewerBuildResult = rebuild();
  if (!(mapState && Number.isFinite(mapState.zoom)) && result.bounds) {
    view = getViewStateForBounds(result.bounds, map.clientWidth, map.clientHeight);
    result = rebuild();
  }
  let animationStart = performance.now();

  const tooltipRow = (info: any): number | null => {
    const object = info.object;
    if (typeof object === 'number') return object;
    if (typeof object?.properties?.__row === 'number') return object.properties.__row;
    return null;
  };
  // Deck 9.4 aggregation layers (heatmap, CPU grid/hexagon) support WebGPU, so it stays first.
  const adapters = [webgpuAdapter, webgl2Adapter];
  let lastClusterZoom = Math.round(view.zoom);
  const panes = createViewerPanes({
    container: map,
    mode: splitMode,
    isGlobe,
    background: config.backgroundColor,
    basemapStyleUrl,
    isOnline: isViewerOnline(),
    swipePercentage: mapState?.swipeComparePercentage ?? 50,
    initialViewState: view,
    onViewStateChange: (next, width, height) => {
      view = next;
      viewSize = {width, height};
      // Cluster layers depend on the rounded zoom level only.
      if (result.isZoomDependent && Math.round(next.zoom) !== lastClusterZoom) {
        lastClusterZoom = Math.round(next.zoom);
        applyFilters();
      }
    },
    deckProps: {
      getTooltip: (info: any) => {
        if (info.object?.tooltipText) return {text: info.object.tooltipText};
        const row = tooltipRow(info);
        const datasetId = info.layer ? result.pickTargets.get(info.layer.id) : undefined;
        const dataset = datasetId ? datasets.get(datasetId) : undefined;
        if (row === null || !dataset) return null;
        const fields = config.tooltipFields[datasetId!] ?? dataset.fieldNames.slice(0, 8);
        return {
          text: fields
            .filter(name => dataset.fieldNames.includes(name) && !/geojson|geometry/i.test(name))
            .map(name => `${name}: ${String(dataset.readValue(name, row))}`)
            .join('\n')
        };
      },
      deviceProps: {type: 'best-available', adapters}
    }
  });
  viewSize = panes.getSize();

  /** Split panes show the layers their saved split settings make visible. */
  const getPaneLayers = (layers: any[]): any[][] =>
    splitMode === 'single'
      ? [layers]
      : [0, 1].map(pane =>
          layers.filter(layer => config.splitMapLayers[pane]?.[layer.id] !== false)
        );
  const applyLayers = (layers: any[]) => panes.setLayers(getPaneLayers(layers));
  const applyFilters = () => {
    result = rebuild();
    applyLayers(result.layers);
    animationStart = performance.now();
  };
  applyFilters();
  renderFilterControls(filterPanel, config.filters, datasets, applyFilters);

  if (customIcons.length === 0 || result.layers.length) {
    // Online: fetch the default catalog and redraw icon layers with the real meshes.
    if (config.layers.some(layer => layer.type === 'icon') && isViewerOnline())
      void fetchDefaultIconDefinitions().then(defaults => {
        if (!defaults.length) return;
        const atlas = safeCreateIconAtlas([...defaults, ...customIcons]);
        if (atlas) {
          iconAtlas = atlas;
          applyFilters();
        }
      });
  }

  if (result.tripDomain) {
    const [start, end] = result.tripDomain;
    const durationMilliseconds = 10_000;
    const tick = (now: number) => {
      const fraction = ((now - animationStart) % durationMilliseconds) / durationMilliseconds;
      currentTime = start + (end - start) * fraction;
      applyLayers(
        result.layers.map(layer =>
          layer instanceof TripsLayer ? layer.clone({currentTime}) : layer
        )
      );
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  const drawn = result.layers.length;
  status.textContent = drawn
    ? `Rendering ${drawn} layer(s)${panes.hasBasemap ? ' with an online basemap' : ' offline (no basemap)'}${splitMode !== 'single' ? `; ${splitMode} split map` : ''}${isGlobe ? '; globe view' : ''}${rejectedNative.length ? `; native tables rejected: ${rejectedNative.join(', ')}` : ''}. `
    : 'No drawable layers found in this bundle. ';
  const button = document.createElement('button');
  button.textContent = 'Download bundle JSON';
  button.onclick = () => downloadBundle(json);
  status.appendChild(button);
}

function addDefaultPointLayers(
  config: ReturnType<typeof readViewerConfig>,
  datasets: Map<string, ViewerDataset>
): void {
  for (const dataset of datasets.values()) {
    const find = (names: string[]) =>
      dataset.fieldNames.find(name => names.includes(name.toLowerCase().replace(/[^a-z]/g, '')));
    const longitude = find(['lng', 'lon', 'long', 'longitude', 'x']);
    const latitude = find(['lat', 'latitude', 'y']);
    if (!longitude || !latitude) continue;
    config.layers.push({
      id: dataset.datasetId,
      type: 'point',
      label: dataset.datasetId,
      datasetId: dataset.datasetId,
      isVisible: true,
      color: [0, 200, 255],
      columns: {lng: longitude, lat: latitude},
      visConfig: {radius: 4, opacity: 0.8},
      channels: {}
    });
  }
}

function safeCreateIconAtlas(definitions: any[]): ViewerIconAtlas | null {
  try {
    return createIconAtlas(definitions);
  } catch {
    return null;
  }
}

void main();
