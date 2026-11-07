// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Layer, color, project32, type LayerContext, type LayerProps} from '@deck.gl/core';
import {Buffer, type Device, type Texture} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {ShaderModule} from '@luma.gl/shaders';

/** Shader mapping of Kepler's globe density mesh: the weights texture sampled over the sphere. */
const GLOBE_HEATMAP_WGSL = /* wgsl */ `\
struct GlobeHeatmapAttributes {
  @location(0) positions: vec3<f32>,
  @location(1) texCoords: vec2<f32>,
};
struct GlobeHeatmapVaryings {
  @builtin(position) position: vec4<f32>,
  @location(0) textureCoordinates: vec2<f32>,
  @location(1) intensityMinimum: f32,
  @location(2) intensityMaximum: f32,
  @location(3) commonPosition: vec3<f32>,
};
@vertex
fn vertexMain(attributes: GlobeHeatmapAttributes) -> GlobeHeatmapVaryings {
  var output: GlobeHeatmapVaryings;
  let projected = project_position_to_clipspace_and_commonspace(attributes.positions, vec3<f32>(0.0), vec3<f32>(0.0));
  output.position = projected.clipPosition;
  output.commonPosition = projected.commonPosition.xyz;
  // WebGPU render-target textures use a top-left origin.
  output.textureCoordinates = vec2<f32>(attributes.texCoords.x, 1.0 - attributes.texCoords.y);
  let maxWeights = textureLoad(maxTexture, vec2<i32>(0), 0);
  var maximumValue = select(maxWeights.r, maxWeights.g, globeHeatmap.aggregationMode > 0.5);
  var minimumValue = maximumValue * globeHeatmap.threshold;
  if (globeHeatmap.colorDomain.y > 0.0) {
    maximumValue = globeHeatmap.colorDomain.y;
    minimumValue = globeHeatmap.colorDomain.x;
  }
  output.intensityMaximum = globeHeatmap.intensity / maximumValue;
  output.intensityMinimum = globeHeatmap.intensity / minimumValue;
  return output;
}
@fragment
fn fragmentMain(varyings: GlobeHeatmapVaryings) -> @location(0) vec4<f32> {
  // Deck GlobeView has no occluder: discard fragments past the horizon.
  if (dot(varyings.commonPosition, project.cameraPosition - varyings.commonPosition) <= 0.0) { discard; }
  let weights = textureSample(weightsTexture, weightsTextureSampler, varyings.textureCoordinates);
  var weight = weights.r;
  if (globeHeatmap.aggregationMode > 0.5) {
    weight /= max(1.0, weights.a);
  }
  var rampColor = textureSample(colorTexture, colorTextureSampler, vec2<f32>(clamp(weight * varyings.intensityMaximum, 0.0, 1.0), 0.5));
  if (weight <= 0.0) {
    discard;
  }
  rampColor.a *= min(weight * varyings.intensityMinimum, 1.0) * layer.opacity;
  return deckgl_premultiplied_alpha(rampColor);
}
`;

const GLOBE_HEATMAP_UNIFORM_WGSL = /* wgsl */ `\
struct globeHeatmapUniforms {
  aggregationMode: f32,
  colorDomain: vec2<f32>,
  intensity: f32,
  threshold: f32,
};
@group(0) @binding(auto) var<uniform> globeHeatmap: globeHeatmapUniforms;
@group(0) @binding(auto) var colorTexture: texture_2d<f32>;
@group(0) @binding(auto) var colorTextureSampler: sampler;
@group(0) @binding(auto) var maxTexture: texture_2d<f32>;
@group(0) @binding(auto) var weightsTexture: texture_2d<f32>;
@group(0) @binding(auto) var weightsTextureSampler: sampler;
`;

type GlobeHeatmapUniformProps = {
  aggregationMode: number;
  colorDomain: Readonly<[number, number]>;
  intensity: number;
  threshold: number;
  colorTexture: Texture;
  maxTexture: Texture;
  weightsTexture: Texture;
};

const globeHeatmapUniforms = {
  name: 'globeHeatmap',
  source: GLOBE_HEATMAP_UNIFORM_WGSL,
  uniformTypes: {
    aggregationMode: 'f32',
    colorDomain: 'vec2<f32>',
    intensity: 'f32',
    threshold: 'f32'
  }
} as const satisfies ShaderModule<GlobeHeatmapUniformProps>;

/** Triangle-list vertices on the density frame, ready for upload. */
export type GlobeHeatmapMesh = {
  positions: Float32Array;
  texCoords: Float32Array;
  vertexCount: number;
};

/**
 * Tessellates the density frame `[west, south, east, north]` at `resolution` degrees, like Kepler's
 * `createGlobeHeatmapMesh`. Each vertex samples the density texture at its Web Mercator position
 * within the frame (`projectToFrame`). The result is a de-indexed triangle list.
 */
export function createGlobeHeatmapMesh(
  worldBounds: readonly [number, number, number, number],
  projectToFrame: (longitude: number, latitude: number) => readonly [number, number],
  resolution: number
): GlobeHeatmapMesh {
  const [west, south, east, north] = worldBounds;
  const columnCount = Math.max(2, Math.ceil(Math.abs(east - west) / resolution) + 1);
  const rowCount = Math.max(2, Math.ceil(Math.abs(north - south) / resolution) + 1);
  const gridPositions = new Float32Array(columnCount * rowCount * 3);
  const gridTexCoords = new Float32Array(columnCount * rowCount * 2);
  for (let column = 0; column < columnCount; column++) {
    for (let row = 0; row < rowCount; row++) {
      const longitude = west + ((east - west) * column) / (columnCount - 1);
      const latitude = south + ((north - south) * row) / (rowCount - 1);
      const vertex = column * rowCount + row;
      gridPositions.set([longitude, latitude, 0], vertex * 3);
      gridTexCoords.set(projectToFrame(longitude, latitude), vertex * 2);
    }
  }
  const vertexCount = (columnCount - 1) * (rowCount - 1) * 6;
  const positions = new Float32Array(vertexCount * 3);
  const texCoords = new Float32Array(vertexCount * 2);
  let output = 0;
  const emit = (vertex: number): void => {
    positions.set(gridPositions.subarray(vertex * 3, vertex * 3 + 3), output * 3);
    texCoords.set(gridTexCoords.subarray(vertex * 2, vertex * 2 + 2), output * 2);
    output++;
  };
  for (let column = 1; column < columnCount; column++) {
    for (let row = 1; row < rowCount; row++) {
      const vertex = column * rowCount + row;
      const left = vertex - rowCount;
      emit(left);
      emit(left - 1);
      emit(vertex - 1);
      emit(left);
      emit(vertex - 1);
      emit(vertex);
    }
  }
  return {positions, texCoords, vertexCount};
}

export type ResidentGlobeHeatmapTriangleLayerProps = LayerProps &
  GlobeHeatmapUniformProps & {mesh: GlobeHeatmapMesh};

/** Draws the density texture on the tessellated sphere mesh with the native ramp mapping. */
export class ResidentGlobeHeatmapTriangleLayer extends Layer<ResidentGlobeHeatmapTriangleLayerProps> {
  static override layerName = 'ResidentGlobeHeatmapTriangleLayer';

  declare state: {model?: Model; positions?: Buffer; texCoords?: Buffer; mesh?: GlobeHeatmapMesh};

  override getShaders() {
    return super.getShaders({
      source: GLOBE_HEATMAP_WGSL,
      modules: [project32, color, globeHeatmapUniforms]
    });
  }

  override initializeState({device}: LayerContext): void {
    this.getAttributeManager()?.remove(['instancePickingColors']);
    this.setState({model: this.createModel(device)});
  }

  private createModel(device: Device): Model {
    const {mesh} = this.props;
    const positions = device.createBuffer({
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: mesh.positions
    });
    const texCoords = device.createBuffer({
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: mesh.texCoords
    });
    this.setState({positions, texCoords, mesh});
    return new Model(device, {
      ...this.getShaders(),
      id: this.props.id,
      attributes: {positions, texCoords},
      bufferLayout: [
        {name: 'positions', format: 'float32x3'},
        {name: 'texCoords', format: 'float32x2'}
      ],
      topology: 'triangle-list',
      vertexCount: mesh.vertexCount
    });
  }

  override updateState(): void {
    const {model, mesh} = this.state;
    if (model && this.props.mesh !== mesh) {
      // A new frame or resolution replaces the vertex data; counts may differ.
      this.state.positions?.destroy();
      this.state.texCoords?.destroy();
      model.destroy();
      this.setState({model: this.createModel(this.context.device)});
    }
  }

  override finalizeState(context: LayerContext): void {
    super.finalizeState(context);
    this.state.model?.destroy();
    this.state.positions?.destroy();
    this.state.texCoords?.destroy();
  }

  override draw(): void {
    const {model} = this.state;
    if (!model) return;
    const {
      aggregationMode,
      colorDomain,
      intensity,
      threshold,
      colorTexture,
      maxTexture,
      weightsTexture
    } = this.props;
    model.shaderInputs.setProps({
      globeHeatmap: {
        aggregationMode,
        colorDomain,
        intensity,
        threshold,
        colorTexture,
        maxTexture,
        weightsTexture
      }
    });
    model.draw(this.context.renderPass);
  }
}
