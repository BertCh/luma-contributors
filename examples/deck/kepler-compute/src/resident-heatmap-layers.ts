// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LayerClasses, type LayerClassesType, type Layer as KeplerLayer} from '@kepler.gl/layers';
import type {Layer} from '@deck.gl/core';
import {makeResidentHeatmapLayer, type ResidentHeatmapInputs} from './resident-heatmap-layer';

export type ResidentHeatmapBinding = ResidentHeatmapInputs & {
  columns: {lat: string; lng: string};
  weightColumn?: string;
};

// Native Kepler narrows the renderer return type to its own Deck subclass. The resident adapter
// preserves its configuration path while returning the borrowed-vector subclass instead.
export interface NativeHeatmapLayer
  extends Omit<InstanceType<LayerClassesType['heatmap']>, 'renderLayer' | 'hasLayerData'> {
  hasLayerData(layerData: unknown): boolean;
  formatLayerData(
    ...parameters: Parameters<InstanceType<LayerClassesType['heatmap']>['formatLayerData']>
  ): Record<string, unknown>;
  updateLayerMeta(
    ...parameters: Parameters<InstanceType<LayerClassesType['heatmap']>['updateLayerMeta']>
  ): void;
  updateLayerDomain(...parameters: Parameters<KeplerLayer['updateLayerDomain']>): KeplerLayer;
  renderLayer(...parameters: Parameters<KeplerLayer['renderLayer']>): Layer[];
}

export type ResidentHeatmapLayerOptions = {
  isResidentDataset(datasetId: string): boolean;
  resolveHeatmap(
    datasetId: string,
    layer: NativeHeatmapLayer
  ): {binding?: ResidentHeatmapBinding; reason?: string};
  getBounds?(datasetId: string): [number, number, number, number] | undefined;
  onUnsupported(layerId: string, reason: string | null): void;
  onRenderError?(layer: NativeHeatmapLayer, error: Error): void;
};

type RenderOptions = {
  idx?: number;
  visible?: boolean;
  mapState?: {zoom: number; globe?: {enabled?: boolean}};
  interactionConfig?: {brush?: {enabled?: boolean}};
};
const RESIDENT_HEATMAP_DATA = Object.freeze({residentHeatmap: true});

/** Native controls and density framing, with no native CPU formatting or accessor evaluation. */
export function createResidentHeatmapLayerClasses(
  options: ResidentHeatmapLayerOptions
): Pick<LayerClassesType, 'heatmap'> {
  const diagnostics = new Map<string, string | null>();
  const isResident = (layer: NativeHeatmapLayer): boolean =>
    Boolean(layer.config.dataId && options.isResidentDataset(layer.config.dataId));
  const report = (layer: NativeHeatmapLayer, reason: string | null): void => {
    if (diagnostics.get(layer.id) === reason) return;
    diagnostics.set(layer.id, reason);
    options.onUnsupported(layer.id, reason);
  };
  const updateMetadata = (layer: NativeHeatmapLayer): void => {
    const bounds = layer.config.dataId ? options.getBounds?.(layer.config.dataId) : undefined;
    if (bounds) layer.meta = {...layer.meta, bounds};
  };
  // Kepler's declarations reference a nested Deck copy; Vite deduplicates the actual runtime.
  const HeatmapBase = LayerClasses.heatmap as unknown as new (
    props: ConstructorParameters<LayerClassesType['heatmap']>[0]
  ) => NativeHeatmapLayer;

  class ResidentHeatmapLayer extends HeatmapBase {
    static findDefaultLayerProps(
      ...parameters: Parameters<typeof LayerClasses.heatmap.findDefaultLayerProps>
    ): ReturnType<typeof LayerClasses.heatmap.findDefaultLayerProps> {
      return options.isResidentDataset(parameters[0].id)
        ? {props: []}
        : LayerClasses.heatmap.findDefaultLayerProps(...parameters);
    }

    override hasLayerData(layerData: unknown): boolean {
      return isResident(this) ? layerData === RESIDENT_HEATMAP_DATA : super.hasLayerData(layerData);
    }

    override formatLayerData(
      ...parameters: Parameters<InstanceType<LayerClassesType['heatmap']>['formatLayerData']>
    ): Record<string, unknown> {
      if (!isResident(this)) return super.formatLayerData(...parameters);
      updateMetadata(this);
      return RESIDENT_HEATMAP_DATA;
    }

    override updateLayerMeta(
      ...parameters: Parameters<InstanceType<LayerClassesType['heatmap']>['updateLayerMeta']>
    ): void {
      if (!isResident(this)) return super.updateLayerMeta(...parameters);
      updateMetadata(this);
    }

    override updateLayerDomain(
      ...parameters: Parameters<KeplerLayer['updateLayerDomain']>
    ): KeplerLayer {
      return isResident(this) ? this : super.updateLayerDomain(...parameters);
    }

    override renderLayer(renderOptions: RenderOptions = {}): Layer[] {
      if (!isResident(this)) return super.renderLayer(renderOptions);
      if (!this.config.isVisible || renderOptions.visible === false) return [];
      // Native heatmaps ignore the hover brush, so it needs no refusal.
      if (this.config.columnMode !== 'points') {
        report(this, 'Resident heatmaps currently require explicit point coordinate columns.');
        return [];
      }
      const {binding, reason} = options.resolveHeatmap(this.config.dataId!, this);
      if (!binding) {
        report(this, reason ?? null);
        return [];
      }
      if (
        this.config.columns.lat?.value !== binding.columns.lat ||
        this.config.columns.lng?.value !== binding.columns.lng ||
        (this.config.weightField?.name ?? undefined) !== binding.weightColumn
      ) {
        report(this, 'The selected heatmap coordinate or weight field has no resident binding.');
        return [];
      }
      report(this, null);
      const nativeLayers = super.renderLayer({
        ...renderOptions,
        data: RESIDENT_HEATMAP_DATA,
        gpuFilter: null,
        visible: renderOptions.visible ?? true,
        mapState: renderOptions.mapState ?? {zoom: 14}
      });
      return nativeLayers.map(nativeLayer =>
        makeResidentHeatmapLayer(
          nativeLayer.clone({
            id: `${this.id}-resident-heatmap`,
            ...{nativeLayerId: this.id},
            onError: (error: Error) => {
              if (options.onRenderError) options.onRenderError(this, error);
              else report(this, error.message);
              return true;
            }
          }),
          binding
        )
      );
    }
  }
  // Widening is localized to the native constructor boundary; configuration/static metadata remain native.
  return {heatmap: ResidentHeatmapLayer as unknown as LayerClassesType['heatmap']};
}
