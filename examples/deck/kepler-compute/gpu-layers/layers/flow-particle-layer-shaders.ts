// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ShaderModule} from '@luma.gl/shaders';

export const flowStreak = {
  name: 'flowStreak',
  source: `struct FlowStreakUniforms { bounds: vec4<f32>, boundsLow: vec4<f32>, appearance: vec4<f32>, color: vec4<f32>, speed: vec4<f32>, palette0: vec4<f32>, palette1: vec4<f32>, palette2: vec4<f32>, palette3: vec4<f32>, palette4: vec4<f32>, palette5: vec4<f32>, palette6: vec4<f32>, palette7: vec4<f32> };
  @group(3) @binding(0) var<uniform> flowStreak: FlowStreakUniforms;
  @group(3) @binding(1) var flowState: texture_2d<f32>;
  @group(3) @binding(2) var flowPreviousState: texture_2d<f32>;
  @group(3) @binding(3) var flowVelocity: texture_2d<f32>;
  @group(3) @binding(4) var flowAltitude: texture_2d<f32>;`,
  vs: `layout(std140) uniform flowStreakUniforms { vec4 bounds;
  vec4 boundsLow;
  vec4 appearance;
  vec4 color;
  vec4 speed;
  vec4 palette0;
  vec4 palette1;
  vec4 palette2;
  vec4 palette3;
  vec4 palette4;
  vec4 palette5;
  vec4 palette6;
  vec4 palette7; } flowStreak;
  uniform highp sampler2D flowState; uniform highp sampler2D flowPreviousState;
  uniform highp sampler2D flowVelocity; uniform highp sampler2D flowAltitude;`,
  fs: `layout(std140) uniform flowStreakUniforms { vec4 bounds;
  vec4 boundsLow;
  vec4 appearance;
  vec4 color;
  vec4 speed;
  vec4 palette0;
  vec4 palette1;
  vec4 palette2;
  vec4 palette3;
  vec4 palette4;
  vec4 palette5;
  vec4 palette6;
  vec4 palette7; } flowStreak;`,
  bindingLayout: [
    {name: 'flowStreak', group: 3},
    {name: 'flowState', group: 3, visibility: 1},
    {name: 'flowPreviousState', group: 3, visibility: 1},
    {name: 'flowVelocity', group: 3, visibility: 1},
    {name: 'flowAltitude', group: 3, visibility: 1}
  ],
  uniformTypes: {
    bounds: 'vec4<f32>',
    boundsLow: 'vec4<f32>',
    appearance: 'vec4<f32>',
    color: 'vec4<f32>',
    speed: 'vec4<f32>',
    palette0: 'vec4<f32>',
    palette1: 'vec4<f32>',
    palette2: 'vec4<f32>',
    palette3: 'vec4<f32>',
    palette4: 'vec4<f32>',
    palette5: 'vec4<f32>',
    palette6: 'vec4<f32>',
    palette7: 'vec4<f32>'
  },
  defaultUniforms: {
    bounds: [0, 0, 1, 1],
    boundsLow: [0, 0, 0, 0],
    appearance: [1.5, 1, 1, 1],
    color: [0.4, 0.9, 1, 0.9],
    speed: [0, 0, 0, 1],
    palette0: [1, 1, 1, 1],
    palette1: [1, 1, 1, 1],
    palette2: [1, 1, 1, 1],
    palette3: [1, 1, 1, 1],
    palette4: [1, 1, 1, 1],
    palette5: [1, 1, 1, 1],
    palette6: [1, 1, 1, 1],
    palette7: [1, 1, 1, 1]
  }
} as const satisfies ShaderModule;

export const SOURCE = /* wgsl */ `struct FlowVertex {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) pickingColor: vec3<f32>,
  @location(1) coordinates: vec2<f32>,
  @location(2) visible: f32,
  @location(3) tint: vec4<f32>,
};
fn getPaletteColor(index: u32) -> vec4<f32> {
  switch (index) {
    case 0u: { return flowStreak.palette0; }
    case 1u: { return flowStreak.palette1; }
    case 2u: { return flowStreak.palette2; }
    case 3u: { return flowStreak.palette3; }
    case 4u: { return flowStreak.palette4; }
    case 5u: { return flowStreak.palette5; }
    case 6u: { return flowStreak.palette6; }
    default: { return flowStreak.palette7; }
  }
}
// Normalized speed (0..1) stored in the velocity texel's w channel, nearest texel.
fn sampleSpeed(position: vec2<f32>) -> f32 {
  let dimensions = vec2<i32>(textureDimensions(flowVelocity));
  let cell = clamp(vec2<i32>(round(position * vec2<f32>(dimensions - vec2<i32>(1)))), vec2<i32>(0), dimensions - vec2<i32>(1));
  return textureLoad(flowVelocity, cell, 0).w;
}
// Terrain height in meters, bilinear over cells that carry an altitude (z is the validity flag).
fn sampleAltitude(position: vec2<f32>) -> f32 {
  if (flowStreak.speed.z < 0.5) { return 0.0; }
  let dimensions = vec2<i32>(textureDimensions(flowAltitude));
  let gridPosition = clamp(position, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dimensions - vec2<i32>(1));
  let lower = vec2<i32>(floor(gridPosition));
  let upper = min(lower + vec2<i32>(1), dimensions - vec2<i32>(1));
  let fraction = fract(gridPosition);
  let a = textureLoad(flowAltitude, lower, 0);
  let b = textureLoad(flowAltitude, vec2<i32>(upper.x, lower.y), 0);
  let c = textureLoad(flowAltitude, vec2<i32>(lower.x, upper.y), 0);
  let d = textureLoad(flowAltitude, upper, 0);
  let wa = a.z * (1.0 - fraction.x) * (1.0 - fraction.y);
  let wb = b.z * fraction.x * (1.0 - fraction.y);
  let wc = c.z * (1.0 - fraction.x) * fraction.y;
  let wd = d.z * fraction.x * fraction.y;
  let total = wa + wb + wc + wd;
  if (total <= 0.0) { return 0.0; }
  return (a.x * wa + b.x * wb + c.x * wc + d.x * wd) / total * flowStreak.speed.w;
}
fn projectParticle(position: vec2<f32>) -> vec4<f32> {
  let delta = position * flowStreak.bounds.zw;
  let high = flowStreak.bounds.xy + delta;
  let virtualDelta = high - flowStreak.bounds.xy;
  let low = (flowStreak.bounds.xy - (high - virtualDelta)) + (delta - virtualDelta)
    + flowStreak.boundsLow.xy + position * flowStreak.boundsLow.zw;
  return project_position_to_clipspace(vec3<f32>(high, flowStreak.appearance.z + sampleAltitude(position)), vec3<f32>(low, 0.0), vec3<f32>(0.0));
}
@vertex fn vertexMain(@location(0) corner: vec2<f32>, @builtin(instance_index) identifier: u32) -> FlowVertex {
  let dimensions = textureDimensions(flowState);
  let coordinates = vec2<i32>(i32(identifier % dimensions.x), i32(identifier / dimensions.x));
  let particle = textureLoad(flowState, coordinates, 0);
  let previous = textureLoad(flowPreviousState, coordinates, 0);
  let currentClip = projectParticle(particle.xy);
  let previousClip = projectParticle(previous.xy);
  let currentScreen = currentClip.xy / max(currentClip.w, 0.00001);
  let sameParticle = particle.w == previous.w && previous.z >= 0.0 && previousClip.w > 0.0;
  // Project the tail on the particle plane so the river cannot hide it at oblique angles.
  let tailClip = projectParticle(select(particle.xy, mix(particle.xy, previous.xy, flowStreak.appearance.y), sameParticle));
  let tailScreen = tailClip.xy / max(tailClip.w, 0.00001);
  let motion = (currentScreen - tailScreen) * project.viewportSize * 0.5;
  let distance = length(motion);
  let direction = select(vec2<f32>(0.0, 1.0), motion / max(distance, 0.00001), distance > 0.00001);
  let width = max(flowStreak.appearance.x, 0.0) * project.devicePixelRatio;
  let centerClip = mix(tailClip, currentClip, corner.x);
  let offset = direction * (width * 0.5 - (1.0 - corner.x) * max(0.0, width - distance)) + vec2<f32>(-direction.y, direction.x) * corner.y * width * 0.5;
  var output: FlowVertex;
  output.position = vec4<f32>(centerClip.xy + offset * 2.0 / project.viewportSize * centerClip.w, centerClip.zw);
  output.coordinates = corner;
  output.visible = select(0.0, 1.0, particle.z >= 0.0 && currentClip.w > 0.0 && width > 0.0);
  output.pickingColor = picking_getPickingColorFromIndex(identifier);
  let bandCount = max(flowStreak.speed.y, 1.0);
  let band = u32(clamp(floor(clamp(sampleSpeed(particle.xy), 0.0, 1.0) * bandCount - 0.000001), 0.0, bandCount - 1.0));
  let bandColor = getPaletteColor(band);
  output.tint = select(flowStreak.color, vec4<f32>(bandColor.rgb, flowStreak.color.a), flowStreak.speed.x > 0.5);
  return output;
}
@fragment fn fragmentMain(input: FlowVertex) -> @location(0) vec4<f32> {
  if (input.visible < 0.5) { discard; }
  if (picking.isActive > 0.5) { return vec4<f32>(input.pickingColor, 1.0); }
  let coverage = 1.0 - smoothstep(0.6, 1.0, abs(input.coordinates.y));
  var color = input.tint.rgb;
  if (picking.isHighlightActive > 0.5 && distance(input.pickingColor, picking_normalizeColor(picking.highlightedObjectColor)) < 0.00001) {
    color = mix(color, picking.highlightColor.rgb, picking.highlightColor.a);
  }
  return vec4<f32>(color, input.tint.a * flowStreak.appearance.w * coverage * mix(0.25, 1.0, input.coordinates.x));
}
`;
export const VERTEX_SHADER = /* glsl */ `#version 300 es
in vec2 corner;
out vec2 coordinates;
out float visible;
out vec4 tint;
vec4 getPaletteColor(int index) {
  if (index == 0) return flowStreak.palette0;
  if (index == 1) return flowStreak.palette1;
  if (index == 2) return flowStreak.palette2;
  if (index == 3) return flowStreak.palette3;
  if (index == 4) return flowStreak.palette4;
  if (index == 5) return flowStreak.palette5;
  if (index == 6) return flowStreak.palette6;
  return flowStreak.palette7;
}
float sampleSpeed(vec2 position) {
  ivec2 dimensions = textureSize(flowVelocity, 0);
  ivec2 cell = clamp(ivec2(floor(position * vec2(dimensions - ivec2(1)) + 0.5)), ivec2(0), dimensions - ivec2(1));
  return texelFetch(flowVelocity, cell, 0).w;
}
float sampleAltitude(vec2 position) {
  if (flowStreak.speed.z < 0.5) return 0.0;
  ivec2 dimensions = textureSize(flowAltitude, 0);
  vec2 gridPosition = clamp(position, vec2(0.0), vec2(1.0)) * vec2(dimensions - ivec2(1));
  ivec2 lower = ivec2(floor(gridPosition));
  ivec2 upper = min(lower + ivec2(1), dimensions - ivec2(1));
  vec2 fraction = fract(gridPosition);
  vec4 a = texelFetch(flowAltitude, lower, 0);
  vec4 b = texelFetch(flowAltitude, ivec2(upper.x, lower.y), 0);
  vec4 c = texelFetch(flowAltitude, ivec2(lower.x, upper.y), 0);
  vec4 d = texelFetch(flowAltitude, upper, 0);
  float wa = a.z * (1.0 - fraction.x) * (1.0 - fraction.y);
  float wb = b.z * fraction.x * (1.0 - fraction.y);
  float wc = c.z * (1.0 - fraction.x) * fraction.y;
  float wd = d.z * fraction.x * fraction.y;
  float total = wa + wb + wc + wd;
  if (total <= 0.0) return 0.0;
  return (a.x * wa + b.x * wb + c.x * wc + d.x * wd) / total * flowStreak.speed.w;
}
vec4 projectParticle(vec2 position) {
  vec2 delta = position * flowStreak.bounds.zw;
  vec2 high = flowStreak.bounds.xy + delta;
  vec2 virtualDelta = high - flowStreak.bounds.xy;
  vec2 low = (flowStreak.bounds.xy - (high - virtualDelta)) + (delta - virtualDelta)
    + flowStreak.boundsLow.xy + position * flowStreak.boundsLow.zw;
  return project_position_to_clipspace(vec3(high, flowStreak.appearance.z + sampleAltitude(position)), vec3(low, 0.0), vec3(0.0));
}
void main() {
  ivec2 dimensions = textureSize(flowState, 0);
  ivec2 texel = ivec2(gl_InstanceID % dimensions.x, gl_InstanceID / dimensions.x);
  vec4 particle = texelFetch(flowState, texel, 0);
  vec4 previous = texelFetch(flowPreviousState, texel, 0);
  vec4 currentClip = projectParticle(particle.xy);
  vec4 previousClip = projectParticle(previous.xy);
  vec2 currentScreen = currentClip.xy / max(currentClip.w, 0.00001);
  bool sameParticle = particle.w == previous.w && previous.z >= 0.0 && previousClip.w > 0.0;
  // Project the tail on the particle plane so the river cannot hide it at oblique angles.
  vec4 tailClip = projectParticle(sameParticle ? mix(particle.xy, previous.xy, flowStreak.appearance.y) : particle.xy);
  vec2 tailScreen = tailClip.xy / max(tailClip.w, 0.00001);
  vec2 motion = (currentScreen - tailScreen) * project.viewportSize * 0.5;
  float distance = length(motion);
  vec2 direction = distance > 0.00001 ? motion / distance : vec2(0.0, 1.0);
  float width = max(flowStreak.appearance.x, 0.0) * project.devicePixelRatio;
  vec4 centerClip = mix(tailClip, currentClip, corner.x);
  vec2 offset = direction * (width * 0.5 - (1.0 - corner.x) * max(0.0, width - distance)) + vec2(-direction.y, direction.x) * corner.y * width * 0.5;
  gl_Position = vec4(centerClip.xy + offset * 2.0 / project.viewportSize * centerClip.w, centerClip.zw);
  coordinates = corner;
  visible = particle.z >= 0.0 && currentClip.w > 0.0 && width > 0.0 ? 1.0 : 0.0;
  float bandCount = max(flowStreak.speed.y, 1.0);
  int band = int(clamp(floor(clamp(sampleSpeed(particle.xy), 0.0, 1.0) * bandCount - 0.000001), 0.0, bandCount - 1.0));
  vec4 bandColor = getPaletteColor(band);
  tint = flowStreak.speed.x > 0.5 ? vec4(bandColor.rgb, flowStreak.color.a) : flowStreak.color;
  geometry.pickingColor = picking_getPickingColorFromIndex(float(gl_InstanceID));
  DECKGL_FILTER_GL_POSITION(gl_Position, geometry);
  vec4 color = vec4(1.0);
  DECKGL_FILTER_COLOR(color, geometry);
}
`;
export const FRAGMENT_SHADER = /* glsl */ `#version 300 es
precision highp float;
in vec2 coordinates;
in float visible;
in vec4 tint;
out vec4 fragmentColor;
void main() {
  if (visible < 0.5) discard;
  float coverage = 1.0 - smoothstep(0.6, 1.0, abs(coordinates.y));
  fragmentColor = vec4(tint.rgb, tint.a * flowStreak.appearance.w * coverage * mix(0.25, 1.0, coordinates.x));
  DECKGL_FILTER_COLOR(fragmentColor, geometry);
}
`;
