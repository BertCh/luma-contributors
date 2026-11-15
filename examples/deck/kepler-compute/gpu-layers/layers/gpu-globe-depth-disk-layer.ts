// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Layer,
  project32,
  _GlobeViewport,
  type Color,
  type LayerContext,
  type LayerProps
} from '@deck.gl/core';
import type {RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';
import {vec3, type NumberArray3} from '@math.gl/core';
import {getGlobeDepthDiskGeometry} from './globe-tile-occlusion';

export type GPUGlobeDepthDiskLayerProps = LayerProps & {
  /** Disk color, drawn behind everything. Defaults to opaque black. */
  fillColor?: Color;
};

type GlobeDepthDiskUniforms = {
  center: readonly [number, number, number];
  right: readonly [number, number, number];
  up: readonly [number, number, number];
  color: readonly [number, number, number, number];
};

const globeDepthDisk = {
  name: 'globeDepthDisk',
  uniformTypes: {center: 'vec3<f32>', right: 'vec3<f32>', up: 'vec3<f32>', color: 'vec4<f32>'},
  source: /* wgsl */ `struct globeDepthDiskUniforms {
  center: vec3<f32>,
  right: vec3<f32>,
  up: vec3<f32>,
  color: vec4<f32>,
};
@group(0) @binding(auto) var<uniform> globeDepthDisk: globeDepthDiskUniforms;`
} as const satisfies ShaderModule<GlobeDepthDiskUniforms>;

const SOURCE = /* wgsl */ `
struct DiskVaryings {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};
@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> DiskVaryings {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  var output: DiskVaryings;
  output.position = project_common_position_to_clipspace(
    vec4<f32>(globeDepthDisk.center + globeDepthDisk.right * corner.x + globeDepthDisk.up * corner.y, 1.0)
  );
  output.local = corner;
  return output;
}
@fragment fn fragmentMain(input: DiskVaryings) -> @location(0) vec4<f32> {
  if (length(input.local) > 1.0) { discard; }
  return globeDepthDisk.color;
}`;

/**
 * WGSL replacement for Kepler's GLSL globe depth disk. Deck's GlobeView has no occluder, so
 * far-side basemap tiles and other depth-tested geometry draw through the planet. This layer
 * fills the depth buffer with the globe's silhouette cross-section: a camera-facing disk at the
 * tangent-cone plane (distance R squared over camera distance from the center, radius
 * R*sqrt(1-(R/d)^2)). Anything behind it fails the depth test. Draw it before basemap tiles.
 * It does not pick and is a no-op outside a globe view.
 */
export class GPUGlobeDepthDiskLayer extends Layer<GPUGlobeDepthDiskLayerProps> {
  static override layerName = 'GPUGlobeDepthDiskLayer';
  static override defaultProps = {fillColor: [0, 0, 0, 255], pickable: false};
  declare state: {model: Model};

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    this.setState({
      model: new Model(device, {
        ...this.getShaders({source: SOURCE, modules: [project32, globeDepthDisk]}),
        id: `${this.id}-model`,
        topology: 'triangle-list',
        vertexCount: 6,
        parameters: {depthCompare: 'less-equal', depthWriteEnabled: true, cullMode: 'none'}
      })
    });
  }

  override getModels(): Model[] {
    return this.state.model ? [this.state.model] : [];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const viewport = this.context.viewport;
    if (!(viewport instanceof _GlobeViewport)) return;
    const camera = viewport.cameraPosition;
    const distance = Math.hypot(camera[0], camera[1], camera[2]);
    const disk = getGlobeDepthDiskGeometry(distance);
    // Inside the sphere there is no silhouette.
    if (!disk) return;
    const forward = vec3.normalize([0, 0, 0], camera as NumberArray3);
    // Right = up-axis cross forward; fall back near the poles' degenerate view.
    const right = vec3.normalize([0, 0, 0], [-forward[1], forward[0], 0]);
    const up = vec3.cross([0, 0, 0], forward, right);
    const {shift, radius} = disk;
    const fillColor = this.props.fillColor ?? [0, 0, 0, 255];
    this.state.model.shaderInputs.setProps({
      globeDepthDisk: {
        center: [forward[0] * shift, forward[1] * shift, forward[2] * shift],
        right: [right[0] * radius, right[1] * radius, right[2] * radius],
        up: [up[0] * radius, up[1] * radius, up[2] * radius],
        color: [
          fillColor[0] / 255,
          fillColor[1] / 255,
          fillColor[2] / 255,
          (fillColor[3] ?? 255) / 255
        ]
      }
    });
    this.state.model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    this.state.model?.destroy();
    super.finalizeState(context);
  }
}
