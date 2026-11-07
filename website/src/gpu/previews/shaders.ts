// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shaders for the preview renderer, in WGSL (WebGPU) and GLSL ES 3.00 (WebGL2).
//
// One uniform block per draw: timing = (seconds, 0, still, 0), view = (clip per unit x, y, 0, 0),
// anim = layer specific, extra = reserved. "still" is 1 for reduced motion: render a finished frame.

const WGSL_UNIFORMS = /* wgsl */ `
struct SceneUniforms {
  timing: vec4<f32>,
  view: vec4<f32>,
  anim: vec4<f32>,
  extra: vec4<f32>
};
@group(0) @binding(0) var<uniform> sceneUniforms: SceneUniforms;
`;

const GLSL_UNIFORMS = /* glsl */ `
uniform sceneUniforms {
  vec4 timing;
  vec4 view;
  vec4 anim;
  vec4 extra;
};
`;

// Sprites: instanced glowing discs. anim = (pulse, turntable speed, turntable angle, perspective).
export const SPRITE_WGSL = /* wgsl */ `${WGSL_UNIFORMS}
struct VertexInputs {
  @location(0) corner: vec2<f32>,
  @location(1) center: vec3<f32>,
  @location(2) color: vec4<f32>,
  @location(3) shape: vec2<f32>
};
struct FragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>
};

@vertex
fn vertexMain(inputs: VertexInputs) -> FragmentInputs {
  var outputs: FragmentInputs;
  let time = sceneUniforms.timing.x;
  let angle = sceneUniforms.anim.z + time * sceneUniforms.anim.y;
  let local = vec3<f32>(inputs.center.x - 80.0, inputs.center.y - 50.0, inputs.center.z);
  let rotatedX = local.x * cos(angle) - local.z * sin(angle);
  let depth = local.x * sin(angle) + local.z * cos(angle);
  let scale = 1.0 / (1.0 + depth * sceneUniforms.anim.w);
  let pulse = 1.0 + sceneUniforms.anim.x * sin(time * 2.0 + inputs.shape.y * 6.2831853);
  let radius = inputs.shape.x * 2.5 * pulse * scale;
  let scenePosition = vec2<f32>(rotatedX * scale, local.y * scale) + inputs.corner * radius;
  outputs.position = vec4<f32>(scenePosition.x * sceneUniforms.view.x, -scenePosition.y * sceneUniforms.view.y, 0.0, 1.0);
  outputs.corner = inputs.corner;
  outputs.color = inputs.color;
  return outputs;
}

@fragment
fn fragmentMain(inputs: FragmentInputs) -> @location(0) vec4<f32> {
  let ringDistance = length(inputs.corner);
  let glow = exp(-ringDistance * ringDistance * 4.0);
  let core = 1.0 - smoothstep(0.2, 0.3, ringDistance);
  let intensity = min(1.0, glow * 0.7 + core) * (1.0 - smoothstep(0.7, 1.0, ringDistance));
  return vec4<f32>(inputs.color.rgb, inputs.color.a * intensity);
}
`;

export const SPRITE_VERTEX_GLSL = /* glsl */ `#version 300 es
precision highp float;
${GLSL_UNIFORMS}
in vec2 corner;
in vec3 center;
in vec4 color;
in vec2 shape;
out vec2 vCorner;
out vec4 vColor;
void main() {
  float time = timing.x;
  float angle = anim.z + time * anim.y;
  vec3 local = vec3(center.x - 80.0, center.y - 50.0, center.z);
  float rotatedX = local.x * cos(angle) - local.z * sin(angle);
  float depth = local.x * sin(angle) + local.z * cos(angle);
  float scale = 1.0 / (1.0 + depth * anim.w);
  float pulse = 1.0 + anim.x * sin(time * 2.0 + shape.y * 6.2831853);
  float radius = shape.x * 2.5 * pulse * scale;
  vec2 scenePosition = vec2(rotatedX * scale, local.y * scale) + corner * radius;
  gl_Position = vec4(scenePosition.x * view.x, -scenePosition.y * view.y, 0.0, 1.0);
  vCorner = corner;
  vColor = color;
}
`;

export const SPRITE_FRAGMENT_GLSL = /* glsl */ `#version 300 es
precision highp float;
in vec2 vCorner;
in vec4 vColor;
out vec4 fragmentColor;
void main() {
  float ringDistance = length(vCorner);
  float glow = exp(-ringDistance * ringDistance * 4.0);
  float core = 1.0 - smoothstep(0.2, 0.3, ringDistance);
  float intensity = min(1.0, glow * 0.7 + core) * (1.0 - smoothstep(0.7, 1.0, ringDistance));
  fragmentColor = vec4(vColor.rgb, vColor.a * intensity);
}
`;

// Meshes (stroke ribbons and fill triangles). anim = (mode, speed, trail length, 0);
// mode 0 static, 1 flow along the path, 2 draw in, 3 pulse.
export const MESH_WGSL = /* wgsl */ `${WGSL_UNIFORMS}
struct VertexInputs {
  @location(0) position: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) track: vec3<f32>
};
struct FragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) track: vec3<f32>
};

@vertex
fn vertexMain(inputs: VertexInputs) -> FragmentInputs {
  var outputs: FragmentInputs;
  outputs.position = vec4<f32>(
    (inputs.position.x - 80.0) * sceneUniforms.view.x,
    -(inputs.position.y - 50.0) * sceneUniforms.view.y,
    0.0,
    1.0
  );
  outputs.color = inputs.color;
  outputs.track = inputs.track;
  return outputs;
}

@fragment
fn fragmentMain(inputs: FragmentInputs) -> @location(0) vec4<f32> {
  let time = sceneUniforms.timing.x;
  let still = sceneUniforms.timing.z > 0.5;
  let mode = sceneUniforms.anim.x;
  let speed = sceneUniforms.anim.y;
  let trail = max(sceneUniforms.anim.z, 0.05);
  let along = inputs.track.x;
  let phase = inputs.track.y;
  var rgb = inputs.color.rgb;
  var alpha = inputs.color.a;
  if (mode > 0.5 && mode < 1.5) {
    var head = fract(time * speed + phase);
    if (still) { head = fract(phase + 0.4); }
    let behind = fract(head - along);
    alpha = alpha * (0.2 + 0.8 * exp(-behind * 5.0 / trail));
    rgb = rgb + rgb * 0.8 * exp(-behind * 18.0 / trail);
  } else if (mode > 1.5 && mode < 2.5) {
    let cycle = fract(time * speed + phase * 0.3) * 1.35;
    var visible = 1.0 - smoothstep(cycle - 0.04, cycle, along);
    visible = visible * (1.0 - smoothstep(1.05, 1.35, cycle));
    if (still) { visible = 1.0; }
    alpha = alpha * visible;
  } else if (mode > 2.5) {
    alpha = alpha * (0.65 + 0.35 * sin(6.2831853 * (time * speed + phase)));
  }
  alpha = alpha * (1.0 - smoothstep(0.6, 1.0, abs(inputs.track.z)));
  return vec4<f32>(rgb, alpha);
}
`;

export const MESH_VERTEX_GLSL = /* glsl */ `#version 300 es
precision highp float;
${GLSL_UNIFORMS}
in vec2 position;
in vec4 color;
in vec3 track;
out vec4 vColor;
out vec3 vTrack;
void main() {
  gl_Position = vec4((position.x - 80.0) * view.x, -(position.y - 50.0) * view.y, 0.0, 1.0);
  vColor = color;
  vTrack = track;
}
`;

export const MESH_FRAGMENT_GLSL = /* glsl */ `#version 300 es
precision highp float;
${GLSL_UNIFORMS}
in vec4 vColor;
in vec3 vTrack;
out vec4 fragmentColor;
void main() {
  float time = timing.x;
  bool still = timing.z > 0.5;
  float mode = anim.x;
  float speed = anim.y;
  float trail = max(anim.z, 0.05);
  float along = vTrack.x;
  float phase = vTrack.y;
  vec3 rgb = vColor.rgb;
  float alpha = vColor.a;
  if (mode > 0.5 && mode < 1.5) {
    float head = fract(time * speed + phase);
    if (still) { head = fract(phase + 0.4); }
    float behind = fract(head - along);
    alpha = alpha * (0.2 + 0.8 * exp(-behind * 5.0 / trail));
    rgb = rgb + rgb * 0.8 * exp(-behind * 18.0 / trail);
  } else if (mode > 1.5 && mode < 2.5) {
    float cycle = fract(time * speed + phase * 0.3) * 1.35;
    float visible = 1.0 - smoothstep(cycle - 0.04, cycle, along);
    visible = visible * (1.0 - smoothstep(1.05, 1.35, cycle));
    if (still) { visible = 1.0; }
    alpha = alpha * visible;
  } else if (mode > 2.5) {
    alpha = alpha * (0.65 + 0.35 * sin(6.2831853 * (time * speed + phase)));
  }
  alpha = alpha * (1.0 - smoothstep(0.6, 1.0, abs(vTrack.z)));
  fragmentColor = vec4(rgb, alpha);
}
`;

// Field: textured quad. fieldTexture = (value, phase or gradient x, gradient y), rampTexture = colors.
// anim = (style, levels, speed, strength); style 0 ramp, 1 contours, 2 hillshade, 3 expand.
// extra = (light angle or cutoff, 1 when fixed, 0, 0).
export const FIELD_WGSL = /* wgsl */ `${WGSL_UNIFORMS}
@group(0) @binding(1) var fieldTexture: texture_2d<f32>;
@group(0) @binding(2) var fieldTextureSampler: sampler;
@group(0) @binding(3) var rampTexture: texture_2d<f32>;
@group(0) @binding(4) var rampTextureSampler: sampler;

struct VertexInputs {
  @location(0) corner: vec2<f32>
};
struct FragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>
};

@vertex
fn vertexMain(inputs: VertexInputs) -> FragmentInputs {
  var outputs: FragmentInputs;
  let uv = inputs.corner * 0.5 + vec2<f32>(0.5, 0.5);
  let scenePosition = uv * vec2<f32>(160.0, 100.0);
  outputs.position = vec4<f32>(
    (scenePosition.x - 80.0) * sceneUniforms.view.x,
    -(scenePosition.y - 50.0) * sceneUniforms.view.y,
    0.0,
    1.0
  );
  outputs.uv = uv;
  return outputs;
}

fn getRamp(value: f32) -> vec3<f32> {
  return textureSampleLevel(rampTexture, rampTextureSampler, vec2<f32>(clamp(value, 0.0, 1.0), 0.5), 0.0).rgb;
}

@fragment
fn fragmentMain(inputs: FragmentInputs) -> @location(0) vec4<f32> {
  let time = sceneUniforms.timing.x;
  let still = sceneUniforms.timing.z > 0.5;
  let style = sceneUniforms.anim.x;
  let levels = sceneUniforms.anim.y;
  let speed = sceneUniforms.anim.z;
  let strength = sceneUniforms.anim.w;
  let sampled = textureSample(fieldTexture, fieldTextureSampler, inputs.uv);
  let value = sampled.r;
  var rgb = vec3<f32>(0.0);
  if (style < 0.5) {
    rgb = getRamp(value + strength * sin(time * 1.2 + sampled.g * 6.2831853));
  } else if (style < 1.5) {
    let scaled = value * levels;
    let width = fwidth(scaled);
    let nearest = round(scaled);
    let line = 1.0 - smoothstep(0.4 * width, 1.4 * width, abs(scaled - nearest));
    let cycle = fract(time * speed);
    var visible = 1.0 - smoothstep(cycle * 1.3 * levels - 0.5, cycle * 1.3 * levels + 0.5, nearest);
    visible = visible * (1.0 - smoothstep(0.85, 1.0, cycle));
    if (still) { visible = 1.0; }
    var weight = 0.6;
    if (fract(nearest / 4.0) < 0.01) { weight = 1.0; }
    rgb = getRamp(value) * 0.16 + getRamp(nearest / levels) * line * visible * weight * 1.3;
  } else if (style < 2.5) {
    let gradient = (sampled.gb - vec2<f32>(0.5, 0.5)) * 8.0;
    let normal = normalize(vec3<f32>(-gradient.x, -gradient.y, 1.0));
    var angle = 2.2 + time * speed;
    if (sceneUniforms.extra.y > 0.5) { angle = sceneUniforms.extra.x; }
    let light = normalize(vec3<f32>(cos(angle) * 0.6, sin(angle) * 0.6, 0.55));
    let shade = clamp(dot(normal, light), 0.0, 1.0);
    rgb = getRamp(value) * (0.35 + 0.9 * shade);
  } else {
    let stopAt = select(0.55, sceneUniforms.extra.x, sceneUniforms.extra.y > 0.5);
    var cutoff = 0.12 + (stopAt - 0.12) * clamp(fract(time * speed) * 1.25, 0.0, 1.0);
    if (still) { cutoff = stopAt; }
    let inside = 1.0 - smoothstep(cutoff - 0.004, cutoff + 0.004, value);
    let scaled = value * levels * 2.0;
    let band = 1.0 - smoothstep(0.0, 2.0 * fwidth(scaled), abs(scaled - round(scaled)));
    var color = getRamp(value / cutoff * 0.9);
    color = color * (0.85 + 0.15 * band);
    rgb = mix(getRamp(value) * 0.12, color, inside) + vec3<f32>(1.0, 0.95, 0.85) * exp(-abs(value - cutoff) * 70.0) * 0.55;
  }
  return vec4<f32>(rgb, 1.0);
}
`;

export const FIELD_VERTEX_GLSL = /* glsl */ `#version 300 es
precision highp float;
${GLSL_UNIFORMS}
in vec2 corner;
out vec2 vUv;
void main() {
  vec2 uv = corner * 0.5 + vec2(0.5);
  vec2 scenePosition = uv * vec2(160.0, 100.0);
  gl_Position = vec4((scenePosition.x - 80.0) * view.x, -(scenePosition.y - 50.0) * view.y, 0.0, 1.0);
  vUv = uv;
}
`;

export const FIELD_FRAGMENT_GLSL = /* glsl */ `#version 300 es
precision highp float;
${GLSL_UNIFORMS}
uniform sampler2D fieldTexture;
uniform sampler2D rampTexture;
in vec2 vUv;
out vec4 fragmentColor;

vec3 getRamp(float value) {
  return texture(rampTexture, vec2(clamp(value, 0.0, 1.0), 0.5)).rgb;
}

void main() {
  float time = timing.x;
  bool still = timing.z > 0.5;
  float style = anim.x;
  float levels = anim.y;
  float speed = anim.z;
  float strength = anim.w;
  vec4 sampled = texture(fieldTexture, vUv);
  float value = sampled.r;
  vec3 rgb = vec3(0.0);
  if (style < 0.5) {
    rgb = getRamp(value + strength * sin(time * 1.2 + sampled.g * 6.2831853));
  } else if (style < 1.5) {
    float scaled = value * levels;
    float width = fwidth(scaled);
    float nearest = round(scaled);
    float line = 1.0 - smoothstep(0.4 * width, 1.4 * width, abs(scaled - nearest));
    float cycle = fract(time * speed);
    float visible = 1.0 - smoothstep(cycle * 1.3 * levels - 0.5, cycle * 1.3 * levels + 0.5, nearest);
    visible = visible * (1.0 - smoothstep(0.85, 1.0, cycle));
    if (still) { visible = 1.0; }
    float weight = 0.6;
    if (fract(nearest / 4.0) < 0.01) { weight = 1.0; }
    rgb = getRamp(value) * 0.16 + getRamp(nearest / levels) * line * visible * weight * 1.3;
  } else if (style < 2.5) {
    vec2 gradient = (sampled.gb - vec2(0.5)) * 8.0;
    vec3 normal = normalize(vec3(-gradient.x, -gradient.y, 1.0));
    float angle = 2.2 + time * speed;
    if (extra.y > 0.5) { angle = extra.x; }
    vec3 light = normalize(vec3(cos(angle) * 0.6, sin(angle) * 0.6, 0.55));
    float shade = clamp(dot(normal, light), 0.0, 1.0);
    rgb = getRamp(value) * (0.35 + 0.9 * shade);
  } else {
    float stopAt = extra.y > 0.5 ? extra.x : 0.55;
    float cutoff = 0.12 + (stopAt - 0.12) * clamp(fract(time * speed) * 1.25, 0.0, 1.0);
    if (still) { cutoff = stopAt; }
    float inside = 1.0 - smoothstep(cutoff - 0.004, cutoff + 0.004, value);
    float scaled = value * levels * 2.0;
    float band = 1.0 - smoothstep(0.0, 2.0 * fwidth(scaled), abs(scaled - round(scaled)));
    vec3 color = getRamp(value / cutoff * 0.9);
    color = color * (0.85 + 0.15 * band);
    rgb = mix(getRamp(value) * 0.12, color, inside) + vec3(1.0, 0.95, 0.85) * exp(-abs(value - cutoff) * 70.0) * 0.55;
  }
  fragmentColor = vec4(rgb, 1.0);
}
`;
