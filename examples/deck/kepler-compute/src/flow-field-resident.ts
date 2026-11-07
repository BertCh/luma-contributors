// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer as DeckLayer} from '@deck.gl/core';
import type {Layer as KeplerLayer} from '@kepler.gl/layers';
import type {FlowFieldLayerConfig, FlowFieldResolution} from './flow-field-bindings';
import {getFlowFieldPalette} from './flow-field-plan';

export type ResidentFlowFieldHooks = {
  isResident: (layer: KeplerLayer) => boolean;
  report: (layer: KeplerLayer, reason: string | null) => void;
  resolve: (layer: KeplerLayer) => FlowFieldResolution | undefined;
};

// biome-ignore lint/suspicious/noExplicitAny: mixin constructors require any[] parameters
type LayerConstructor = new (...parameters: any[]) => KeplerLayer;

/**
 * Resident Flow Field layer: the pinned native layer supplies configuration, controls and
 * metadata; rendering is a GPU particle simulation fed by the retained source's vector field.
 * `Base` is the resident-guarded native class so non-resident datasets keep native behavior.
 */
export function createResidentFlowFieldLayerClass<Base extends LayerConstructor>(
  Base: Base,
  hooks: ResidentFlowFieldHooks
): Base {
  class ResidentFlowFieldLayer extends Base {
    override updateLayerMeta(...parameters: unknown[]): void {
      if (!hooks.isResident(this)) {
        Reflect.apply(Base.prototype.updateLayerMeta, this, parameters);
        return;
      }
      const bounds = hooks.resolve(this)?.bounds;
      if (bounds) this.meta = {...this.meta, bounds};
    }
    override renderLayer(...parameters: unknown[]): DeckLayer[] {
      if (!hooks.isResident(this))
        return Reflect.apply(Base.prototype.renderLayer, this, parameters) as DeckLayer[];
      const config = this.config as unknown as FlowFieldLayerConfig & {isVisible: boolean};
      const resolution = hooks.resolve(this);
      const reason =
        resolution?.reason ??
        (resolution?.limits.length ? `Rendered with limits: ${resolution.limits.join(' ')}` : null);
      hooks.report(this, reason ?? (resolution ? null : 'No resident flow field source.'));
      if (!resolution?.runtime) return [];
      const debugLayers = config.visConfig.debugGrid
        ? (resolution.getDebugLayers?.(Math.max(0, config.visConfig.elevationMultiplier ?? 1)) ??
          [])
        : [];
      return [
        resolution.runtime.createLayer({
          id: `${this.id}-resident-flow-field`,
          color: config.color,
          visible: config.isVisible !== false,
          speedPalette: config.visConfig.colorBySpeed
            ? getFlowFieldPalette(config.visConfig.colorRange)
            : undefined,
          elevationMultiplier: Math.max(0, config.visConfig.elevationMultiplier ?? 1)
        }),
        ...debugLayers
      ];
    }
  }
  return ResidentFlowFieldLayer;
}
