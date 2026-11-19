import{Q as e}from"./dist-BusamGlx.js";import{t}from"./model-CeCyfMN-.js";import{a as n,i as r,n as i,o as a,r as o,s}from"./illustrative-preview-sPPEPO9l.js";var c=`
struct SceneUniforms {
  timing: vec4<f32>,
  view: vec4<f32>,
  anim: vec4<f32>,
  extra: vec4<f32>
};
@group(0) @binding(0) var<uniform> sceneUniforms: SceneUniforms;
`,l=`
uniform sceneUniforms {
  vec4 timing;
  vec4 view;
  vec4 anim;
  vec4 extra;
};
`,u=`${c}
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
`,d=`#version 300 es
precision highp float;
${l}
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
`,f=`#version 300 es
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
`,p=`${c}
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
`,m=`#version 300 es
precision highp float;
${l}
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
`,h=`#version 300 es
precision highp float;
${l}
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
`,g=`${c}
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
`,_=`#version 300 es
precision highp float;
${l}
in vec2 corner;
out vec2 vUv;
void main() {
  vec2 uv = corner * 0.5 + vec2(0.5);
  vec2 scenePosition = uv * vec2(160.0, 100.0);
  gl_Position = vec4((scenePosition.x - 80.0) * view.x, -(scenePosition.y - 50.0) * view.y, 0.0, 1.0);
  vUv = uv;
}
`,v=`#version 300 es
precision highp float;
${l}
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
`,y={blend:!0,blendColorOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one`,blendAlphaOperation:`add`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one`},b={...y,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaDstFactor:`one-minus-src-alpha`},x={ramp:0,contours:1,hillshade:2,expand:3},S={static:0,flow:1,draw:2,pulse:3},C=64;function w(e){let t=e.length,n=0;for(let r=0;r<t;r++){let i=e[(r+1)%t];n+=e[r][0]*i[1]-i[0]*e[r][1]}let r=n>=0?1:-1,i=(t,n,r)=>(e[n][0]-e[t][0])*(e[r][1]-e[t][1])-(e[n][1]-e[t][1])*(e[r][0]-e[t][0]),a=e.map((e,t)=>t),o=[],s=t*t;for(;a.length>3&&s-->0;){let e=!1;for(let t=0;t<a.length;t++){let n=a[(t+a.length-1)%a.length],s=a[t],c=a[(t+1)%a.length];if(!(i(n,s,c)*r<=1e-9)&&!a.some(e=>e===n||e===s||e===c?!1:i(n,s,e)*r>=0&&i(s,c,e)*r>=0&&i(c,n,e)*r>=0)){o.push(n,s,c),a.splice(t,1),e=!0;break}}if(!e)break}if(a.length===3)o.push(...a);else for(let e=1;e+1<a.length;e++)o.push(a[0],a[e],a[e+1]);return o}function T(e,t){let n=[];for(let r of e){let e=t.roles[r.role];for(let t of w(r.ring))n.push(r.ring[t][0],r.ring[t][1],e[0],e[1],e[2],r.alpha??.5,0,r.phase??0,0)}return new Float32Array(n)}function E(e,t){let n=[];for(let r of e){let e=r.points.filter((e,t,n)=>t===0||e[0]!==n[t-1][0]||e[1]!==n[t-1][1]);if(e.length<2)continue;let i=t.roles[r.role],a=Math.max(r.width*.5,.18)*1.25,o=[0];for(let t=1;t<e.length;t++)o.push(o[t-1]+Math.hypot(e[t][0]-e[t-1][0],e[t][1]-e[t-1][1]));let s=o[o.length-1]||1,c=[];for(let t=0;t<e.length;t++){let n=e[Math.max(t-1,0)],r=e[Math.min(t+1,e.length-1)],i=e[t],l=t>0?[i[0]-n[0],i[1]-n[1]]:[r[0]-i[0],r[1]-i[1]],u=t<e.length-1?[r[0]-i[0],r[1]-i[1]]:l,d=Math.hypot(l[0],l[1])||1,f=Math.hypot(u[0],u[1])||1,p=[-l[1]/d,l[0]/d],m=[-u[1]/f,u[0]/f],h=p[0]+m[0],g=p[1]+m[1],_=Math.hypot(h,g)||1;h/=_,g/=_;let v=a/Math.max(h*p[0]+g*p[1],.5);c.push([i[0]+h*v,i[1]+g*v,i[0]-h*v,i[1]-g*v,o[t]/s])}let l=(e,t,a,o)=>n.push(e,t,i[0],i[1],i[2],r.alpha??.9,a,r.phase??0,o);for(let e=0;e+1<c.length;e++){let t=c[e],n=c[e+1];l(t[0],t[1],t[4],1),l(t[2],t[3],t[4],-1),l(n[0],n[1],n[4],1),l(n[0],n[1],n[4],1),l(t[2],t[3],t[4],-1),l(n[2],n[3],n[4],-1)}}return new Float32Array(n)}function D(e){let t=e.length/9,n=new Float32Array(t*2),r=new Float32Array(t*4),i=new Float32Array(t*3);for(let a=0;a<t;a++)n.set(e.subarray(a*9,a*9+2),a*2),r.set(e.subarray(a*9+2,a*9+6),a*4),i.set(e.subarray(a*9+6,a*9+9),a*3);return{position:n,color:r,track:i,count:t}}var O=class{device;palette;renderers=[];buffers=[];textures=[];cornerBuffer;uniformData=new Float32Array(16);constructor(e,t,n){this.device=e,this.palette=n,this.cornerBuffer=this.createVertexBuffer(new Float32Array([-1,-1,1,-1,-1,1,1,1]));for(let e of t)if(e.type===`field`)this.renderers.push(this.createFieldRenderer(e));else if(e.type===`sprites`)this.renderers.push(this.createSpriteRenderer(e));else{let t=this.createMeshRenderer(e);t&&this.renderers.push(t)}}createVertexBuffer(t){let n=this.device.createBuffer({usage:e.VERTEX|e.COPY_DST,data:t});return this.buffers.push(n),n}createUniformBuffer(){let t=this.device.createBuffer({usage:e.UNIFORM|e.COPY_DST,data:new Float32Array(16)});return this.buffers.push(t),t}createModel(e,n,r){let i={id:e,...r,isInstanced:r.instanceCount!==void 0,disableWarnings:!0};return this.device.type===`webgpu`?new t(this.device,{...i,source:n.wgsl}):new t(this.device,{...i,vs:n.vertex,fs:n.fragment})}createLayerRenderer(e,t,n,r=[],i=()=>[0,0]){return{model:e,update:(e,r,a,o)=>{let s=this.uniformData;s.set([r?0:e,0,+!!r,0,a,o,0,0,...n(),...i(),0,0]),t.write(s)},destroy:()=>{e.destroy();for(let e of r)e.destroy()}}}createSpriteRenderer(e){let t=e.items.length,n=new Float32Array(t*3),r=new Float32Array(t*4),i=new Float32Array(t*2);e.items.forEach((e,t)=>{let a=this.palette.roles[e.role];n.set([e.x,e.y,e.z??0],t*3),r.set([a[0],a[1],a[2],e.alpha??.9],t*4),i.set([e.radius,e.phase??0],t*2)});let a=this.createUniformBuffer(),o=this.createModel(`preview-sprites`,{wgsl:u,vertex:d,fragment:f},{bufferLayout:[{name:`corner`,format:`float32x2`},{name:`center`,format:`float32x3`,stepMode:`instance`},{name:`color`,format:`float32x4`,stepMode:`instance`},{name:`shape`,format:`float32x2`,stepMode:`instance`}],attributes:{corner:this.cornerBuffer,center:this.createVertexBuffer(n),color:this.createVertexBuffer(r),shape:this.createVertexBuffer(i)},topology:`triangle-strip`,vertexCount:4,instanceCount:t,bindings:{sceneUniforms:a},parameters:y}),s=e.turntable;return this.createLayerRenderer(o,a,()=>[e.pulse??0,s?.speed??0,s?.angle??0,s?.perspective??0])}createMeshRenderer(e){let t=D(e.type===`strokes`?E(e.items,this.palette):T(e.items,this.palette));if(t.count===0)return null;let n=this.createUniformBuffer(),r=this.createModel(`preview-${e.type}`,{wgsl:p,vertex:m,fragment:h},{bufferLayout:[{name:`position`,format:`float32x2`},{name:`color`,format:`float32x4`},{name:`track`,format:`float32x3`}],attributes:{position:this.createVertexBuffer(t.position),color:this.createVertexBuffer(t.color),track:this.createVertexBuffer(t.track)},topology:`triangle-list`,vertexCount:t.count,bindings:{sceneUniforms:n},parameters:e.blend===`alpha`?b:y}),i=S[e.animation??`static`],a=e.type===`strokes`?e.speed??.3:.3,o=e.type===`strokes`?e.trail??1:1;return this.createLayerRenderer(r,n,()=>[i,a,o,0])}createFieldRenderer(e){let{columns:t,rows:n,values:r}=e,a=new Uint8Array(t*n*4),o=e.strength??.05,c=(e,i)=>r[s(i,0,n-1)*t+s(e,0,t-1)];for(let r=0;r<n;r++)for(let n=0;n<t;n++){let i=(r*t+n)*4,o=c(n,r);if(a[i]=Math.round(o*255),e.style===`hillshade`){let t=e.strength??5;a[i+1]=Math.round((s((c(n+1,r)-c(n-1,r))*t,-.5,.5)+.5)*255),a[i+2]=Math.round((s((c(n,r+1)-c(n,r-1))*t,-.5,.5)+.5)*255)}else a[i+1]=Math.round((o*2.5-Math.floor(o*2.5))*255);a[i+3]=255}let l=new Uint8Array(256);for(let t=0;t<C;t++){let n=i(e.ramp,t/63);l.set([n[0]*255,n[1]*255,n[2]*255,255],t*4)}let u=e.filter===`nearest`?`nearest`:`linear`,d=this.device.createTexture({width:t,height:n,format:`rgba8unorm`,data:a,sampler:{minFilter:u,magFilter:u,addressModeU:`clamp-to-edge`,addressModeV:`clamp-to-edge`}}),f=this.device.createTexture({width:C,height:1,format:`rgba8unorm`,data:l,sampler:{minFilter:`linear`,magFilter:`linear`,addressModeU:`clamp-to-edge`,addressModeV:`clamp-to-edge`}});this.textures.push(d,f);let p=this.createUniformBuffer(),m=this.createModel(`preview-field`,{wgsl:g,vertex:_,fragment:v},{bufferLayout:[{name:`corner`,format:`float32x2`}],attributes:{corner:this.cornerBuffer},topology:`triangle-strip`,vertexCount:4,bindings:{sceneUniforms:p,fieldTexture:d,rampTexture:f},parameters:{}}),h=x[e.style];return this.createLayerRenderer(m,p,()=>[h,e.levels??1,e.speed??.1,e.style===`ramp`?o:0],[d,f],()=>e.style===`hillshade`&&e.light!==void 0?[e.light,1]:e.style===`expand`&&e.cutoff!==void 0?[e.cutoff,1]:[0,0])}render(e,t,n,r){let i=Math.max(n/160,r/100),a=i/n*2,o=i/r*2;for(let n of this.renderers)n.update(e,t,a,o);let s=this.palette.background,c=this.device.beginRenderPass({clearColor:[s[0],s[1],s[2],1]});for(let e of this.renderers)e.model.draw(c);c.end(),this.device.submit()}destroy(){for(let e of this.renderers)e.destroy();for(let e of this.buffers)e.destroy()}};function k(e,t){return Math.max(e/160,t/100)}var A=2;function j(e,t){return i=>{let s=i.device.getDefaultCanvasContext(),c=o(),l=a(e,t),u=l!==`none`,d={...n(e,t),...i.parameters},f=null,p=``,m=!1,h=r(e,i.seed,d,null,t),g=new O(i.device,h.layers,c),_=h.layers.some(e=>e.type===`sprites`&&e.turntable),v=new Set([`spin`,`depth`]),y=0,b=0,x=-1,S=()=>{m=!1;let n=r(e,i.seed,d,f,t);g.destroy(),h=n,g=new O(i.device,h.layers,c)};return{render(e){let[t,n]=s.getDrawingBufferSize();if(t===0||n===0)return;let r=e/1e3;if(_){let e=x<0||i.reducedMotion?0:r-x;y+=(d.spin??0)*Math.PI/180*e,x=r;for(let e of h.layers)e.type===`sprites`&&e.turntable&&(e.turntable.angle=.5+y+b,e.turntable.perspective=(d.depth??4.5)/1e3)}m&&S(),g.render(r,i.reducedMotion,t,n)},setPointer(e){if(!u)return;let[t,n]=s.getDrawingBufferSize();if(_){b=e&&t>0?(e[0]/t-.5)*Math.PI:0;return}let r=``,i=null;if(e&&t>0&&n>0){let a=k(t,n);i=[Math.round((80+(e[0]-t/2)/a)/A)*A,Math.round((50+(e[1]-n/2)/a)/A)*A],r=i.join(`,`)}else if(l===`sticky`)return;r!==p&&(p=r,f=i,m=!0)},setParameters(e){Object.assign(d,e),(!_||Object.keys(e).some(e=>!v.has(e)))&&(m=!0)},destroy(){g.destroy()}}}}export{j as createPreviewScene};