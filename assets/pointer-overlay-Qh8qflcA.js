import{Q as e}from"./dist-BusamGlx.js";import{t}from"./gpu-command-graph-BdzdmV6u.js";import{t as n}from"./model-CeCyfMN-.js";function r(e){let t=e>>>0||1;return()=>{t=t+1831565813>>>0;let e=Math.imul(t^t>>>15,1|t);return e=e+Math.imul(e^e>>>7,61|e)^e,((e^e>>>14)>>>0)/4294967296}}function i(e){let t=Number.parseInt(e.slice(1),16);return[(t>>16&255)/255,(t>>8&255)/255,(t&255)/255]}var a=[`#5aa9ff`,`#ffb454`,`#ff6b6b`,`#4fd1c5`,`#8bd450`,`#f5d547`,`#c792ea`,`#ff9ec7`],o=[`#2563c9`,`#d9730d`,`#d1343b`,`#12857a`,`#4a8f1f`,`#b8940b`,`#8a4fc0`,`#d1487f`];function s(e){return e===`dark`?{background:[.043,.059,.09,1],ramp:[`#0b0f1a`,`#2a1a63`,`#7a1f8f`,`#d63f6b`,`#ff9a3c`,`#fff1a8`].map(i),diverging:[`#3b7dff`,`#7fb2ff`,`#2a3350`,`#2a3350`,`#ff9a7a`,`#ff4d4d`].map(i),categories:a.map(i),ink:i(`#e8ecf5`),muted:i(`#5b6580`)}:{background:[.965,.957,.937,1],ramp:[`#f6f4ef`,`#fbe3b0`,`#f6b26b`,`#e5603b`,`#a8204e`,`#3b0f5e`].map(i),diverging:[`#1f4fc4`,`#7fa4e8`,`#e7e3da`,`#e7e3da`,`#ee9a82`,`#c4242b`].map(i),categories:o.map(i),ink:i(`#1c2230`),muted:i(`#9aa0ad`)}}var c=e.STORAGE|e.COPY_DST|e.COPY_SRC;function l(e,t){return e.createBuffer({data:t,usage:c})}function u(e,t){return e.createBuffer({byteLength:Math.max(t,1)*4,usage:c})}function d(e,t,n,r,i){let a=e.importBuffer({id:t,byteLength:n.byteLength,usage:n.usage},n);return e.createDataView(a,{format:r,length:i})}var f=class e{device;compiled;constructor(e,t){this.device=e,this.compiled=t}static async create(n,r,i){let a=new t(n,{id:r});return i(a),new e(n,await a.compileAsync())}run(){let e=this.device.createCommandEncoder({id:`${this.compiled.id}-encoder`});this.compiled.encode(e,{parameters:void 0}),this.device.submit(e.finish())}destroy(){this.compiled.destroy()}};function p(e){let t=e.worldWidth*(e.longitudeScale??1)/e.worldHeight,n=e.width/e.height;return t>n?[1,n/t]:[t/n,1]}function m(e,t){let[n,r]=p(e),i=t[0]/e.width*2-1,a=1-t[1]/e.height*2;return[(e.origin?.[0]??0)+(i/n+1)/2*e.worldWidth,(e.origin?.[1]??0)+(a/r+1)/2*e.worldHeight]}var h=`
struct Look {
  scale: vec2f,
  world: vec2f,
  size: vec2f,
  range: vec2f,
  mode: f32,
  radius: f32,
  alpha: f32,
  hillshade: f32,
  grid: vec2f,
  elevation: f32,
  smoothing: f32,
  background: vec4f,
  light: vec4f,
  stops: array<vec4f, 6>,
  colors: array<vec4f, 16>,
  // x: curve (0 linear, 1 log2), y: fade alpha with value (0 or 1), z: quantize to this many bands,
  // w: dots only, scale the radius with the value (0 or 1)
  extra: vec4f,
  origin: vec2f
};
@group(0) @binding(0) var<uniform> look: Look;

fn worldToClip(point: vec2f) -> vec2f {
  return ((point - look.origin) / look.world * 2.0 - 1.0) * look.scale;
}

fn isValid(value: f32) -> bool {
  return abs(value) < 1.0e20;
}

fn sampleRamp(position: f32) -> vec3f {
  let scaled = clamp(position, 0.0, 1.0) * 5.0;
  let index = min(u32(floor(scaled)), 4u);
  return mix(look.stops[index].rgb, look.stops[index + 1u].rgb, scaled - f32(index));
}

// mode 0: sequential ramp, 1: diverging ramp (range is symmetric about zero), 2: categorical,
// 3: cyclic hue wheel (angles).
fn colorize(value: f32) -> vec4f {
  if (!isValid(value)) {
    return vec4f(look.background.rgb, 0.0);
  }
  if (look.mode > 2.5) {
    // Cyclic hue wheel for angles: the range spans one full turn; values below it are muted.
    if (value < look.range.x) {
      return vec4f(look.background.rgb, 0.0);
    }
    let turn = fract((value - look.range.x) / max(look.range.y - look.range.x, 1.0e-6));
    return vec4f(0.55 + 0.4 * cos(6.28318 * (turn + vec3f(0.0, 0.3333, 0.6667))), 1.0);
  }
  if (look.mode > 1.5) {
    let index = u32(clamp(round(value), 0.0, 15.0));
    return vec4f(look.colors[index].rgb, look.colors[index].a);
  }
  var shaped = value;
  if (look.extra.x > 0.5) {
    shaped = log2(max(value, 1.0));
  }
  
  var span = look.range.y - look.range.x;
  if (abs(span) < 1.0e-12) {
    span = 1.0e-12;
  }
  var position = clamp((shaped - look.range.x) / span, 0.0, 1.0);
  if (look.extra.z > 0.5) {
    position = floor(position * look.extra.z) / look.extra.z;
  }
  let fade = select(1.0, position, look.extra.y > 0.5);
  return vec4f(sampleRamp(position), fade);
}
`,g=e=>`
@group(0) @binding(${e}) var<storage, read> values: array<f32>;
fn readValue(index: u32) -> f32 {
  return values[index];
}
`,_=e=>`
@group(0) @binding(${e}) var<storage, read> values: array<u32>;
fn readValue(index: u32) -> f32 {
  let raw = values[index];
  return select(f32(raw % 16u), 1.0e30, raw == 0xffffffffu);
}
`;function v(e,t){return e===`uint32`?_(t):g(t)}var y=e=>`
${h}
${v(e,1)}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  var output: VertexOutput;
  output.position = vec4f((corner * 2.0 - 1.0) * look.scale, 0.0, 1.0);
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}

fn cellValue(column: i32, row: i32) -> f32 {
  let columns = i32(look.grid.x);
  let rows = i32(look.grid.y);
  let clampedColumn = clamp(column, 0, columns - 1);
  let clampedRow = clamp(row, 0, rows - 1);
  return readValue(u32(clampedRow * columns + clampedColumn));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let cell = input.uv * look.grid;
  let column = i32(floor(cell.x));
  let row = i32(floor(cell.y));
  var value = cellValue(column, row);
  if (look.smoothing > 0.5 && isValid(value)) {
    let shifted = cell - vec2f(0.5);
    let base = vec2i(floor(shifted));
    let fraction = shifted - floor(shifted);
    let v00 = cellValue(base.x, base.y);
    let v10 = cellValue(base.x + 1, base.y);
    let v01 = cellValue(base.x, base.y + 1);
    let v11 = cellValue(base.x + 1, base.y + 1);
    if (isValid(v00) && isValid(v10) && isValid(v01) && isValid(v11)) {
      value = mix(mix(v00, v10, fraction.x), mix(v01, v11, fraction.x), fraction.y);
    }
  }
  var color = colorize(value);
  if (look.hillshade > 0.0 && color.a > 0.0) {
    let slopeX = (cellValue(column + 1, row) - cellValue(column - 1, row)) * look.elevation;
    let slopeY = (cellValue(column, row + 1) - cellValue(column, row - 1)) * look.elevation;
    let normal = normalize(vec3f(-slopeX, slopeY, 1.0));
    let shade = clamp(dot(normal, normalize(look.light.xyz)), 0.0, 1.0);
    color = vec4f(color.rgb * mix(1.0, 0.45 + 0.8 * shade, look.hillshade), color.a);
  }
  return vec4f(color.rgb, color.a * look.alpha);
}
`,b=e=>`
${h}
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
${v(e,2)}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) value: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let center = worldToClip(positions[instanceIndex]);
  let value = readValue(instanceIndex);
  var radius = look.radius;
  if (look.extra.w > 0.5) {
    // Size by value: from 30% of the radius at the low end of the range to all of it at the high end.
    var amount = 0.0;
    if (isValid(value)) {
      amount = clamp((value - look.range.x) / max(look.range.y - look.range.x, 1.0e-12), 0.0, 1.0);
    }
    radius = look.radius * (0.3 + 0.7 * amount);
  }
  var output: VertexOutput;
  output.position = vec4f(center + corner * radius * 2.0 / look.size, 0.0, 1.0);
  output.local = corner;
  output.value = value;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.local);
  if (distance > 1.0) {
    discard;
  }
  var color = colorize(input.value);
  // Invalid values take the solid color. A value faded to nothing stays invisible.
  if (look.mode < -0.5 || (color.a == 0.0 && look.extra.y < 0.5)) {
    color = look.light;
  }
  let edge = 1.0 - smoothstep(1.0 - 2.0 / max(look.radius, 1.0), 1.0, distance);
  let rim = smoothstep(0.7, 1.0, distance) * look.hillshade;
  let rgb = mix(color.rgb, look.background.rgb, rim * 0.6);
  return vec4f(rgb, color.a * edge * look.alpha);
}
`,x=`
${h}
@group(0) @binding(1) var<storage, read> nodes: array<vec2f>;
@group(0) @binding(2) var<storage, read> edges: array<vec2u>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
fn readValue(index: u32) -> f32 {
  return values[index];
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) value: f32,
  @location(1) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let edge = edges[instanceIndex];
  let start = worldToClip(nodes[edge.x]);
  let end = worldToClip(nodes[edge.y]);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  var output: VertexOutput;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  let startValue = readValue(edge.x);
  let endValue = readValue(edge.y);
  output.value = select(max(startValue, endValue), mix(startValue, endValue, along[vertexIndex]), isValid(startValue) && isValid(endValue));
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (color.a == 0.0) {
    return vec4f(look.light.rgb, look.light.a * look.alpha);
  }
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(color.rgb, color.a * feather * look.alpha);
}
`,S=`
${h}
@group(0) @binding(1) var<storage, read> segments: array<vec4f>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.across = 0.0;
  if (instanceIndex >= counts[0]) {
    output.position = vec4f(2.0, 2.0, 0.0, 1.0);
    return output;
  }
  let segment = segments[instanceIndex];
  let start = worldToClip(segment.xy);
  let end = worldToClip(segment.zw);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`,C=`
${h}
@group(0) @binding(1) var<storage, read> nodes: array<vec2f>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.across = 0.0;
  if (instanceIndex + 1u >= counts[0]) {
    output.position = vec4f(2.0, 2.0, 0.0, 1.0);
    return output;
  }
  let startPoint = nodes[ids[instanceIndex]];
  let endPoint = nodes[ids[instanceIndex + 1u]];
  let start = worldToClip(startPoint);
  let end = worldToClip(endPoint);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`,w=`
${h}
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.across = 0.0;
  let triangle = instanceIndex / 3u;
  if (triangle >= counts[0]) {
    output.position = vec4f(2.0, 2.0, 0.0, 1.0);
    return output;
  }
  let corner = instanceIndex % 3u;
  let first = triangles[triangle * 3u + corner];
  let second = triangles[triangle * 3u + (corner + 1u) % 3u];
  let start = worldToClip(positions[first]);
  let end = worldToClip(positions[second]);
  var along = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
  let halfSize = look.size * 0.5;
  let startPixels = start * halfSize;
  let endPixels = end * halfSize;
  let direction = normalize(endPixels - startPixels + vec2f(1.0e-6, 0.0));
  let normal = vec2f(-direction.y, direction.x);
  let pixels = mix(startPixels, endPixels, along[vertexIndex]) + normal * side[vertexIndex] * look.radius;
  output.position = vec4f(pixels / halfSize, 0.0, 1.0);
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`,T={blend:!0,blendColorOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaOperation:`add`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},E=class{lookBuffer;model;lookValues=new Float32Array(120);device;constructor(t,r,i,a,o){this.device=t,this.lookBuffer=t.createBuffer({byteLength:480,usage:e.UNIFORM|e.COPY_DST}),this.model=new n(t,{id:r,source:i,topology:`triangle-list`,vertexCount:6,instanceCount:o,bindings:{look:this.lookBuffer,...a},parameters:{...T}})}update(e,t,n={}){let r=this.lookValues,[i,a]=p(e),o=n.range??[0,1];r.set([i,a,e.worldWidth,e.worldHeight,e.width,e.height],0),r.set(o,6),r[8]=n.mode??0,r[9]=n.radius??3,r[10]=n.alpha??1,r[11]=n.hillshade??0,this.writeGrid(r),r[14]=n.elevation??1,r[15]=+!!n.smoothing,r.set(t.background,16);let s=n.lightDirection??[-.6,.6,.7],c=n.solid??[t.ink[0],t.ink[1],t.ink[2],1];r.set(n.mode===-1||this.usesSolid?c:[...s,0],20);let l=n.ramp??t.ramp;for(let e=0;e<6;e++)r.set([...l[Math.min(e,l.length-1)],1],24+e*4);let u=n.categories??t.categories;for(let e=0;e<16;e++){let t=u[e%u.length];r.set([t[0],t[1],t[2],t.length===4?t[3]:1],48+e*4)}r.set([+!!n.logarithmic,+!!n.fade,n.bands??0,+!!n.sizeByValue],112),r.set(e.origin??[0,0],116),this.lookBuffer.write(r)}get usesSolid(){return!1}setInstanceCount(e){this.model.setInstanceCount(e)}writeGrid(e){}draw(e){this.model.draw(e)}destroy(){this.model.destroy(),this.lookBuffer.destroy()}},D=class extends E{columns;rows;constructor(e,t,n,r,i=`float32`){super(e,`demo-raster`,y(i),{values:t},1),this.columns=n,this.rows=r}writeGrid(e){e[12]=this.columns,e[13]=this.rows}},O=class extends E{constructor(e,t,n,r,i=`float32`){super(e,`demo-dots`,b(i),{positions:t,values:n},r)}get usesSolid(){return!0}},k=class extends E{constructor(e,t,n,r,i){super(e,`demo-segments`,x,{nodes:t,edges:n,values:r},i)}get usesSolid(){return!0}},A=class extends E{constructor(e,t,n,r){super(e,`demo-segment-list`,S,{segments:t,counts:n},r)}get usesSolid(){return!0}},j=class extends E{constructor(e,t,n,r,i){super(e,`demo-path`,C,{nodes:t,ids:n,counts:r},i)}get usesSolid(){return!0}},M=class extends E{constructor(e,t,n,r,i){super(e,`demo-triangle-edges`,w,{positions:t,triangles:n,counts:r},i*3)}get usesSolid(){return!0}};function N(e,t,n){let r=e.beginRenderPass({clearColor:[...t.background]});for(let e of n)e.draw(r);r.end(),e.submit()}var P=class{draw;handle=0;constructor(e){this.draw=e}request(){this.handle||=requestAnimationFrame(()=>{this.handle=0,this.draw()})}flush(){this.cancel(),this.draw()}cancel(){this.handle&&cancelAnimationFrame(this.handle),this.handle=0}};function F(e){if(e.type!==`webgpu`)throw Error(`WebGPU required`)}function I(e,t,n){return Math.min(n,Math.max(t,e))}function L(e,t){let n=Math.round((t-e.min)/e.step);return I(Number((e.min+n*e.step).toPrecision(12)),e.min,e.max)}function R(e,t){let n=e.find(e=>e.id===t);if(!n)throw Error(`Unknown interaction parameter "${t}"`);return n}function z(e,t,n=`linear`){let r=I((t-.05)/.9,0,1);return n===`logarithmic`&&e.min>0?L(e,e.min*(e.max/e.min)**+r):L(e,e.min+(e.max-e.min)*r)}function B(e,t,n=`linear`){return .05+I(n===`logarithmic`&&e.min>0?Math.log(t/e.min)/Math.log(e.max/e.min):(t-e.min)/(e.max-e.min||1),0,1)*.9}function V(e,t=2){return Number.isInteger(e)?String(e):e.toFixed(t)}function H(e,t){if(e.options)return`${e.label} ${e.options[Math.round(t)]}`;let n=(String(e.step).split(`.`)[1]??``).length,r=e.unit??``;return`${e.label.replace(/^Data: /,``)} ${t.toFixed(n)}${/^[A-Za-z]/.test(r)?` `:``}${r}`}function U(e){return typeof e==`string`?{parameter:e,scale:`linear`,flip:!1}:{parameter:e.parameter,scale:e.scale??`linear`,flip:!!e.flip}}function W(e,t){let n=t.x?U(t.x):null,r=t.y?U(t.y):null,i=n?R(e,n.parameter):null,a=r?R(e,r.parameter):null;return{kind:t.kind,hint:t.hint,rest:()=>[n&&i?G(B(i,i.default,n.scale),n.flip):.5,r&&a?1-G(B(a,a.default,r.scale),r.flip):.5],resolve(e,o){let s={};n&&i&&(s[i.id]=z(i,G(e.unitX,n.flip),n.scale)),r&&a&&(s[a.id]=z(a,G(1-e.unitY,r.flip),r.scale));let c=t.decorate?.({...o.values,...s},e,o),l=c?.readout??[i,a].filter(e=>!!e).map(e=>H(e,s[e.id])).join(` · `);return{...c,values:s,readout:l}}}}function G(e,t){return t?1-e:e}function K(e,t){let n=R(e,t.angle),r=t.distance?R(e,t.distance.parameter):null,i=t.convention??`compass`,a=e=>t.anchor?.(e)??[e.world.width/2,e.world.height/2],o=e=>t.distance?.reach??e.world.height*.45,s=e=>i===`compass`?(90-e)*Math.PI/180:e*Math.PI/180;return{kind:`aim`,hint:t.hint,resolve(e,s){let c=a(s),l=e.x-c[0],u=e.y-c[1],d=Math.hypot(l,u),f={};if(d>s.pixelSize*4){let e=Math.atan2(u,l)*180/Math.PI,t=i===`compass`?90-e:e;if(n.max-n.min>=359)t=n.min+((t-n.min)%360+360)%360;else{for(;t>n.max&&t-360>=n.min;)t-=360;for(;t<n.min&&t+360<=n.max;)t+=360}f[n.id]=L(n,t)}if(r&&t.distance){let e=I(d/o(s),0,1),n=t.distance.invert?1-e:e;f[r.id]=L(r,r.min+(r.max-r.min)*n)}let p={...s.values,...f};return{values:f,glyphs:[{shape:`arrow`,from:c,to:[e.x,e.y]},{shape:`dot`,at:c}],readout:t.readout?.(p)??[n,r].filter(e=>!!e).map(e=>H(e,p[e.id])).join(` · `)}},rest(e){let i=a(e),c=s(n.default),l=o(e)*.6;if(r&&t.distance){let n=(r.default-r.min)/(r.max-r.min||1);l=o(e)*(t.distance.invert?1-n:n)}let u=i[0]+Math.cos(c)*l,d=i[1]+Math.sin(c)*l,[f,p]=e.toUnit([u,d]);return[I(f,0,1),I(p,0,1)]}}}var q=1024;function J(){return{segments:new Float32Array(q*4),segmentCount:0,dots:new Float32Array(64),dotCount:0}}function Y(e,t,n,r,i=1){t.segmentCount=0,t.dotCount=0;let a=n*7,o=(e,n,r,i)=>{t.segmentCount>=1024||(t.segments.set([e,n,r,i],t.segmentCount*4),t.segmentCount++)},s=(e,t,n,r,s)=>{if(!s)return o(e,t,n,r);let c=Math.hypot((n-e)/i,r-t),l=Math.max(1,Math.min(64,Math.round(c/a)));for(let i=0;i<l;i+=2){let a=i/l,s=Math.min(1,(i+1)/l);o(e+(n-e)*a,t+(r-t)*a,e+(n-e)*s,t+(r-t)*s)}},c=(e,n)=>{t.dotCount>=32||(t.dots.set([e,n],t.dotCount*2),t.dotCount++)};for(let t of e)switch(t.shape){case`crosshair`:{let e=(t.size??9)*n,r=3*n,[a,s]=t.at;o(a-e*i,s,a-r*i,s),o(a+r*i,s,a+e*i,s),o(a,s-e,a,s-r),o(a,s+r,a,s+e);break}case`dot`:c(t.at[0],t.at[1]);break;case`ring`:for(let e=0;e<64;e++){if(t.dashed&&e%2==1)continue;let n=e/64*Math.PI*2,r=(e+1)/64*Math.PI*2;o(t.at[0]+Math.cos(n)*t.radius*i,t.at[1]+Math.sin(n)*t.radius,t.at[0]+Math.cos(r)*t.radius*i,t.at[1]+Math.sin(r)*t.radius)}break;case`box`:{let[e,n]=t.minimum,[r,i]=t.maximum;s(e,n,r,n,t.dashed),s(r,n,r,i,t.dashed),s(r,i,e,i,t.dashed),s(e,i,e,n,t.dashed);break}case`line`:s(t.from[0],t.from[1],t.to[0],t.to[1],t.dashed);break;case`arrow`:{let[e,r]=t.from,[a,s]=t.to;o(e,r,a,s);let c=(a-e)/i,l=s-r,u=Math.hypot(c,l);if(u>n*6){let e=9*n,t=c/u,r=l/u;for(let n of[-1,1]){let c=-t*.9+n*-r*.45,l=-r*.9+n*t*.45;o(a,s,a+c*e*i,s+l*e)}}break}case`curtain`:o(t.x,r.minimumY,t.x,r.maximumY);break;case`polyline`:{let e=Math.floor(t.points.length/2),n=t.closed?e:e-1;for(let r=0;r<n;r++){let n=(r+1)%e;s(t.points[r*2],t.points[r*2+1],t.points[n*2],t.points[n*2+1],t.dashed)}break}}}var X=360,Z=.58;function Q(e,t,n){let[r,i]=p(e),a=((t-(e.origin?.[0]??0))/e.worldWidth*2-1)*r,o=((n-(e.origin?.[1]??0))/e.worldHeight*2-1)*i;return[(a+1)/2*e.width,(1-o)/2*e.height]}function $(e){let[t,n]=p(e),r=e.width/X,i=e.worldHeight/(e.height*n),a=e.worldWidth/(e.width*t);return{glyphScale:r,pixelSize:i*r,aspectX:a/i}}var ee=`
${h}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) clip: vec2f
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var output: VertexOutput;
  output.position = vec4f(corners[vertexIndex], 0.0, 1.0);
  output.clip = corners[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let world = (input.clip / look.scale + 1.0) * 0.5 * look.world + look.origin;
  let offset = (world - look.range) / vec2f(max(look.hillshade, 1.0e-6), 1.0);
  let amount = smoothstep(look.elevation, look.elevation * 1.8, length(offset));
  return vec4f(look.background.rgb, amount * look.alpha);
}
`,te=class extends E{constructor(e){super(e,`pointer-spotlight`,ee,{},1)}};function ne(e,t,n,r,i,a){let o=1/0,s=1/0,c=-1/0,l=-1/0,u=(e,t)=>{e<o&&(o=e),e>c&&(c=e),t<s&&(s=t),t>l&&(l=t)};for(let n=0;n<t;n++)u(e[n*4],e[n*4+1]),u(e[n*4+2],e[n*4+3]);for(let e=0;e<r;e++)u(n[e*2],n[e*2+1]);if(o>c)return null;let d=(c-o)/2/i,f=(l-s)/2;return{at:[(o+c)/2,(s+l)/2],radius:Math.max(a,Math.hypot(d,f)*1.15)}}var re=class{device;geometry=J();resources=null;spotlight=null;spotlightAmount=0;aspectX=1;constructor(e){this.device=e}set(e,t,n){if(e.length===0&&!this.resources&&(n===null||n===`auto`))return;let{pixelSize:r,aspectX:i}=$(t),[a,o]=m(t,[0,0]),[s,c]=m(t,[t.width,t.height]);Y(e,this.geometry,r,{minimumX:a,minimumY:c,maximumX:s,maximumY:o},i),this.aspectX=i,this.spotlight=n===`auto`?ne(this.geometry.segments,this.geometry.segmentCount,this.geometry.dots,this.geometry.dotCount,i,(o-c)/6):n;let l=this.getResources();l.segments.write(this.geometry.segments),l.count.write(Uint32Array.of(this.geometry.segmentCount)),l.dots.write(this.geometry.dots),l.dotLayer.setInstanceCount(this.geometry.dotCount)}fadeSpotlight(e,t){let n=e&&this.spotlight?1:0;if(this.spotlightAmount===n)return!1;let r=t/160;return this.spotlightAmount=n>this.spotlightAmount?Math.min(n,this.spotlightAmount+r):Math.max(n,this.spotlightAmount-r),!0}setSpotlightOn(e){this.spotlightAmount=e&&this.spotlight?1:0}getLayers(e,t){let n=this.resources;if(!n)return[];let r=[];if(this.spotlight&&this.spotlightAmount>0&&(n.spotlight.update(e,t,{range:this.spotlight.at,elevation:this.spotlight.radius,hillshade:this.aspectX,alpha:Z*this.spotlightAmount}),r.push(n.spotlight)),this.geometry.segmentCount===0&&this.geometry.dotCount===0)return r;let{glyphScale:i}=$(e),a=[t.ink[0],t.ink[1],t.ink[2],.96],o=[t.background[0],t.background[1],t.background[2],.7];return n.halo.update(e,t,{mode:-1,solid:o,radius:2.1*i}),n.ink.update(e,t,{mode:-1,solid:a,radius:.85*i}),n.dotLayer.update(e,t,{mode:-1,solid:a,radius:2.8*i,hillshade:1}),this.geometry.segmentCount>0&&r.push(n.halo,n.ink),this.geometry.dotCount>0&&r.push(n.dotLayer),r}destroy(){let e=this.resources;e&&(e.halo.destroy(),e.ink.destroy(),e.dotLayer.destroy(),e.spotlight.destroy(),e.segments.destroy(),e.count.destroy(),e.dots.destroy(),e.values.destroy(),this.resources=null)}getResources(){if(this.resources)return this.resources;let e=u(this.device,q*4),t=l(this.device,Uint32Array.of(0)),n=u(this.device,64),r=u(this.device,32);return this.resources={segments:e,count:t,dots:n,values:r,halo:new A(this.device,e,t,q),ink:new A(this.device,e,t,q),dotLayer:new O(this.device,n,r,32,`float32`),spotlight:new te(this.device)},this.resources}};export{r as A,u as C,v as D,s as E,m as O,F as S,N as T,j as _,W as a,A as b,V as c,L as d,E as f,h as g,f as h,K as i,d as k,B as l,P as m,$ as n,I as o,O as p,Q as r,R as s,re as t,z as u,D as v,l as w,M as x,k as y};