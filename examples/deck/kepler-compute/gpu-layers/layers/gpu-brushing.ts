// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerContext} from '@deck.gl/core';
import type {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';

/**
 * Native Kepler hover brushing for WebGPU layers: render-only, never a data predicate. Mirrors
 * Deck's BrushingExtension: a row is visible when its position (or, for connections, either
 * endpoint) lies within the radius in meters of the mouse position, measured in common space.
 * The visibility is folded into the existing per-instance `selected` flag, so no varying, CPU
 * attribute or extra draw is added. Layers compile the module only when brushing is enabled.
 */
export type GPUBrushingProps = {
  /** Defaults to false; changing it rebuilds the layer model. */
  brushingEnabled?: boolean;
  /** Meters. Native Kepler passes the brush size in kilometers times 1000. */
  brushingRadius?: number;
};

export const gpuBrushing = {
  name: 'brushing',
  uniformTypes: {
    enabled: 'i32',
    radius: 'f32',
    mousePosition: 'vec2<f32>'
  },
  source: /* wgsl */ `
struct brushingUniforms {
  enabled: i32,
  radius: f32,
  mousePosition: vec2<f32>,
};
@group(3) @binding(auto) var<uniform> brushing: brushingUniforms;

fn brushing_isPointInRange(position: vec2<f32>) -> bool {
  if (brushing.enabled == 0) { return true; }
  let sourceCommon = project_position_vec3_f64(vec3<f32>(position, 0.0), vec3<f32>(0.0)).xy;
  let mouseCommon = project_position_vec3_f64(vec3<f32>(brushing.mousePosition, 0.0), vec3<f32>(0.0)).xy;
  return length((mouseCommon - sourceCommon) / project.commonUnitsPerMeter.xy) <= brushing.radius;
}

fn brushing_arePointsInRange(sourcePosition: vec2<f32>, targetPosition: vec2<f32>) -> bool {
  return brushing_isPointInRange(sourcePosition) || brushing_isPointInRange(targetPosition);
}
`
} as const satisfies ShaderModule;

/**
 * Per-vertex brushing for polygon and path shaders. Native sublayers interpolate the visibility
 * between vertices, so shaders write `brushingVisibility(position.xy)` to a smooth varying and
 * discard below 0.5, exactly like Deck's `brushing_isVisible`. Replaces the
 * `/*BRUSHING_VISIBILITY*\/` token, so disabled layers neither bind nor compile the module.
 */
export function applyGPUBrushingVisibility(source: string, enabled: boolean): string {
  return source.replace(
    '/*BRUSHING_VISIBILITY*/',
    enabled
      ? 'fn brushingVisibility(position: vec2<f32>) -> f32 { return select(0.0, 1.0, brushing_isPointInRange(position)); }'
      : 'fn brushingVisibility(position: vec2<f32>) -> f32 { return 1.0; }'
  );
}

/** Replaces the layer's selection output so brushing hides rows without touching the mask. */
export function applyGPUBrushingToShader(
  source: string,
  enabled: boolean,
  visibleExpression: string
): string {
  if (!enabled) return source;
  const marker = 'output.selected = selectionMask;';
  // The layer shader lost its selection output marker; brushing cannot be attached.
  if (!source.includes(marker)) throw new Error('Brushing requires a selection output');
  return source.replace(
    marker,
    `output.selected = select(0u, selectionMask, ${visibleExpression});`
  );
}

/** Attaches brushing to a layer whose selected output is not the plain selection-mask copy. */
export function replaceGPUBrushingMarker(
  source: string,
  enabled: boolean,
  marker: string,
  replacement: string
): string {
  if (!enabled) return source;
  // The layer shader lost its selection output marker; brushing cannot be attached.
  if (!source.includes(marker)) throw new Error('Brushing requires a selection output');
  return source.replace(marker, replacement);
}

/** Updates the module uniforms from Deck's current pointer; outside the viewport nothing is hidden. */
export function setGPUBrushingProps(
  model: Pick<Model, 'shaderInputs'>,
  context: LayerContext,
  props: GPUBrushingProps
): void {
  if (!props.brushingEnabled) return;
  const {viewport, mousePosition} = context;
  const enabled = Boolean(mousePosition && viewport.containsPixel(mousePosition));
  const [longitude, latitude] = enabled
    ? viewport.unproject([mousePosition!.x - viewport.x, mousePosition!.y - viewport.y])
    : [0, 0];
  model.shaderInputs.setProps({
    brushing: {
      enabled: enabled ? 1 : 0,
      radius: props.brushingRadius ?? 10000,
      mousePosition: [longitude, latitude]
    }
  });
}
