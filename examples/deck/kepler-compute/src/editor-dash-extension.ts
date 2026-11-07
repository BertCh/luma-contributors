// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, LayerContext, LayerExtension} from '@deck.gl/core';
import {PathStyleExtension} from '@deck.gl/extensions';
import type {ShaderModule} from '@luma.gl/shaders';

/**
 * WGSL dashed strokes for the native editor (Kepler uses `PathStyleExtension({dash: true})`).
 *
 * Deck 9.4's PathStyleExtension injects GLSL only. This subclass keeps its accessors, attribute
 * management and `dashGapPickable` prop, and on a WebGPU device swaps its GLSL injections for WGSL
 * ones that anchor on literal text of Deck's `path-layer.wgsl`. Supported subset: segment-relative
 * dashes in `widths` units, butt or round caps, hard (non anti-aliased) dash edges, gap picking.
 * Not supported: `dashJustified`, `dashMode: 'path'`, `offset`, other `dashUnits`, scatterplot and
 * text-background outlines (those sublayers are left undashed). On WebGL it defers to Deck.
 */

const PATH_ATTRIBUTES_ANCHOR = '@location(12) rowIndexes: u32,';
const PATH_VARYINGS_ANCHOR = '@location(7) clipCoordinates: vec2<f32>,';
const PATH_VERTEX_ANCHOR = 'varyings.clipCoordinates = geometry.position.xy;';

/** Literal anchors that must occur exactly once in Deck's path-layer WGSL. */
export const EDITOR_DASH_WGSL_ANCHORS = [
  PATH_ATTRIBUTES_ANCHOR,
  PATH_VARYINGS_ANCHOR,
  PATH_VERTEX_ANCHOR
] as const;

const pathStyleModuleWGSL = /* wgsl */ `\
struct pathStyleUniforms {
  dashAlignMode: f32,
  dashGapPickable: i32,
  dashUnits: i32,
};

@group(0) @binding(auto) var<uniform> pathStyle: pathStyleUniforms;
`;

export const editorPathStyleModule: ShaderModule = {
  name: 'pathStyle',
  source: pathStyleModuleWGSL,
  uniformTypes: {dashAlignMode: 'f32', dashGapPickable: 'i32', dashUnits: 'i32'}
};

/** Injections keyed by literal anchor text (inserted after the anchor) or stage hook. */
export const EDITOR_DASH_WGSL_INJECTIONS: Record<string, string> = {
  [PATH_ATTRIBUTES_ANCHOR]: '\n  @location(13) instanceDashArrays: vec2<f32>,',
  [PATH_VARYINGS_ANCHOR]: '\n  @location(9) vDashArray: vec2<f32>,',
  [PATH_VERTEX_ANCHOR]: '\n  varyings.vDashArray = attributes.instanceDashArrays;',
  'fs:#main-start': `
  let dashSolidLength = varyings.vDashArray.x;
  let dashGapLength = varyings.vDashArray.y;
  let dashUnitLength = dashSolidLength + dashGapLength;
  if (dashUnitLength > 0.0 && dashGapLength > 0.0) {
    let dashPosition = varyings.vPathPosition.y;
    let dashPhase = dashPosition - dashUnitLength * floor(dashPosition / dashUnitLength);
    var isInDashGap = dashPhase > dashSolidLength;
    if (isInDashGap && path.capType > 0.5) {
      // Rounded caps: the dash end is an arc, so resolve the 2D distance to the nearest dash.
      isInDashGap = length(vec2<f32>(
        min(dashPhase - dashSolidLength, dashUnitLength - dashPhase),
        varyings.vPathPosition.x
      )) > 1.0;
    }
    let isGapPickable = picking.isActive > 0.5 && pathStyle.dashGapPickable != 0;
    if (isInDashGap && !isGapPickable) {
      discard;
    }
  }
`
};

let disabledReason: string | null = null;

/** Stop installing the WGSL dash extension (for example after a shader build failure). */
export function disableEditorDashExtension(reason: string): void {
  disabledReason = reason;
}

export function getEditorDashStatus(): {enabled: boolean; reason: string | null} {
  return {enabled: disabledReason === null, reason: disabledReason};
}

export function resetEditorDashStatusForTest(): void {
  disabledReason = null;
}

function isPathLayer(layer: Layer): boolean {
  return 'pathTesselator' in (layer.state ?? {});
}

type EditorDashLayer = Layer<Record<string, unknown>>;

export class EditorWGSLDashExtension extends PathStyleExtension {
  static override extensionName = 'EditorWGSLDashExtension';

  constructor() {
    super({dash: true});
  }

  override isEnabled(layer: EditorDashLayer): boolean {
    return disabledReason === null && isPathLayer(layer);
  }

  override getShaders(this: EditorDashLayer, extension: this): any {
    if (!extension.isEnabled(this)) return null;
    if (this.context?.device?.type !== 'webgpu') {
      return PathStyleExtension.prototype.getShaders.call(this, extension);
    }
    return {
      modules: [editorPathStyleModule],
      defines: {DASH_ENABLED: true},
      inject: EDITOR_DASH_WGSL_INJECTIONS
    };
  }

  override initializeState(this: EditorDashLayer, context: LayerContext, extension: this): void {
    if (!extension.isEnabled(this)) return;
    PathStyleExtension.prototype.initializeState.call(this, context, extension);
  }
}

/**
 * Extensions for the resident editor. Empty (solid strokes) once the WGSL dash was disabled, so a
 * failed build degrades to the previous solid-stroke behavior.
 */
export function makeEditorDashExtensions(): LayerExtension[] {
  return disabledReason === null ? [new EditorWGSLDashExtension()] : [];
}
