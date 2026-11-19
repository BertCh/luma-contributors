import{r as e}from"./rolldown-runtime-XKyaHtl9.js";import{t}from"./react-DlmbjZJV.js";import{t as n}from"./link-D9V6xy_k.js";import{t as r}from"./jsx-runtime-LKYXHABd.js";import{a as i,i as a,n as o,r as s,t as c}from"./topics-BYMS4mpg.js";import{L as l,k as u}from"./index-DIfTuwM1.js";import{t as d}from"./gpu-canvas-rTkYWLOu.js";import{a as f,c as p,d as m,i as h,l as g,o as _,r as v,s as y,u as b}from"./lens-graph-panel-fUMNGNry.js";var x=e(t(),1),ee={webgpu:`WebGPU`,"luma.gl":`luma.gl`,gridlight:`this library`},S=[{id:`buffer`,term:`Buffer`,definition:`A block of GPU memory. A graph imports a buffer the caller owns, or allocates a transient one as scratch.`,owner:`luma.gl`,topic:`data-on-the-gpu`,code:`graph.importBuffer()`},{id:`view`,term:`View`,definition:`A typed, non-owning range of a graph buffer: a format and a row count. Every input and output of an operation is a view.`,owner:`luma.gl`,topic:`data-on-the-gpu`,code:`GraphDataView`},{id:`format`,term:`Format`,definition:`How the bytes of one row are laid out, such as two 32-bit floats per point.`,owner:`luma.gl`,topic:`data-on-the-gpu`,code:`'float32x2'`},{id:`offsets`,term:`Offsets`,definition:`An array that says where each variable-length row starts in a flat array of values. Polygons, neighbor lists and networks all use it.`,owner:`gridlight`,topic:`data-on-the-gpu`,code:`ringOffsets, offsets`},{id:`mask`,term:`Mask and validity`,definition:`One uint32 per row. A nonzero mask keeps an input row; a nonzero validity marks an output row as computed.`,owner:`gridlight`,topic:`data-on-the-gpu`,code:`mask, validity`},{id:`compact-output`,term:`Compact output`,definition:`A variable-length result in a fixed-capacity buffer, with the count written, the count required and an overflow flag.`,owner:`gridlight`,topic:`data-on-the-gpu`,code:`GPUCompactOutput`},{id:`kernel`,term:`Kernel`,definition:`One WGSL compute program. The GPU runs it once per element, for thousands of elements at the same time.`,owner:`luma.gl`,topic:`kernels`,code:`Kernel`},{id:`invocation`,term:`Invocation`,definition:`One run of a kernel. It receives an index and computes the element at that index.`,owner:`webgpu`,topic:`kernels`},{id:`workgroup`,term:`Workgroup`,definition:`A block of invocations launched together. The library picks a power-of-two size from 32 to 256.`,owner:`webgpu`,topic:`kernels`,code:`@workgroup_size`},{id:`dispatch`,term:`Dispatch`,definition:`The command that launches a kernel over a grid of workgroups.`,owner:`webgpu`,topic:`kernels`,code:`dispatchWorkgroups`},{id:`index-guard`,term:`Index guard`,definition:`The first line of every kernel: invocations past the element count return without touching memory.`,owner:`gridlight`,topic:`kernels`,code:`if (index >= count) { return; }`},{id:`contributor`,term:`Contributor`,definition:`A class for one operation. Its constructor validates the props; its getCommandNodes returns the kernels.`,owner:`gridlight`,topic:`contract`,code:`GPUKernelDensity`},{id:`recipe`,term:`Recipe`,definition:`A function that adds several contributors to a graph in order and returns their outputs, intermediates and status.`,owner:`gridlight`,topic:`workflows`,code:`addHotSpotAnalysisRecipe`},{id:`command-graph`,term:`Command graph`,definition:`The list of kernels of a computation and the buffers each one reads and writes. It compiles once and encodes every frame.`,owner:`luma.gl`,topic:`command-graph`,code:`GPUCommandGraph`},{id:`command-node`,term:`Command node`,definition:`One step of a command graph: a compute kernel, a copy or a render pass.`,owner:`luma.gl`,topic:`command-graph`},{id:`hazard`,term:`Hazard`,definition:`An ordering the compiler infers when two kernels touch the same buffer: read after write, write after read, or write after write.`,owner:`luma.gl`,topic:`command-graph`},{id:`compile`,term:`Compile`,definition:`Orders the kernels by their hazards, plans buffer lifetimes and creates the pipelines. It happens once.`,owner:`luma.gl`,topic:`scheduling`,code:`graph.compile()`},{id:`encode`,term:`Encode`,definition:`Records the compiled passes into a command encoder. It happens every frame and does no analysis.`,owner:`luma.gl`,topic:`scheduling`,code:`compiled.encode()`},{id:`parameter-buffer`,term:`Parameter buffer`,definition:`A small GPU buffer of per-frame numbers such as a bandwidth. Writing it changes the result without recompiling.`,owner:`gridlight`,topic:`scheduling`,code:`GPUParameterBuffer`},{id:`reduction`,term:`Reduction`,definition:`Folds a column to one value, such as a sum or a maximum, in a tree of passes.`,owner:`luma.gl`,topic:`building-blocks`,code:`GPUReduction`},{id:`scan`,term:`Scan`,definition:`A prefix sum: element i becomes the total of everything before it. It turns counts into write positions.`,owner:`luma.gl`,topic:`building-blocks`,code:`GPUScan`},{id:`compaction`,term:`Compaction`,definition:`Keeps the flagged rows and packs them to the front, using a scan to place each one.`,owner:`luma.gl`,topic:`building-blocks`,code:`GPUCompaction`},{id:`sort`,term:`Sort`,definition:`A radix sort: for each digit, count, scan and scatter. Stable, so ties keep row order.`,owner:`luma.gl`,topic:`building-blocks`,code:`GPUSort`},{id:`grid-index`,term:`Grid index`,definition:`Points bucketed by cell, so a kernel visits only the neighbors near its element and never every pair.`,owner:`luma.gl`,topic:`building-blocks`,code:`GPUGridIndex`},{id:`round`,term:`Round`,definition:`One pass of an iterative method. Rounds are separate kernels added when the graph is built, up to a budget.`,owner:`gridlight`,topic:`iteration`,code:`maximumIterations`},{id:`gate`,term:`Gate`,definition:`A dispatch whose size is read from a GPU buffer, through an indirect dispatch of luma.gl. A round with nothing left to do runs zero workgroups.`,owner:`gridlight`,topic:`iteration`,code:`createWGSLDispatchGate`},{id:`status`,term:`Status`,definition:`Named outputs a kernel writes about its own run, such as overflow, converged or invalidCount. The caller decides whether to read them.`,owner:`gridlight`,topic:`contract`,code:`output.status`},{id:`readback`,term:`Readback`,definition:`Copying a GPU buffer to the CPU. It is asynchronous, the application owns it, and no operation needs it to finish.`,owner:`luma.gl`,topic:`anatomy`,code:`GPUReadbackRing`},{id:`snapshot-tap`,term:`Snapshot tap`,definition:`A copy node that captures chosen buffers of a run so a record can show their values.`,owner:`gridlight`,topic:`lenses`,code:`GPUSnapshotTap`},{id:`record`,term:`Computation record`,definition:`Plain data that describes one compiled graph and its captured values: kernels, buffers, hazards, formulas, choices.`,owner:`gridlight`,topic:`lenses`,code:`createGPUComputationRecord`},{id:`lens`,term:`Lens`,definition:`A recorded scenario of one operation, drawn as its graph, its formula and the data it produced.`,owner:`gridlight`,topic:`lenses`}];function te(e){return S.find(t=>t.id===e)}var C=r();function w({topic:e,children:t,tryIt:r}){let o=i(e.part),s=c.findIndex(t=>t.slug===e.slug)+1,{previous:u,next:d}=a(e.slug);return(0,C.jsxs)(`main`,{className:`page topic-page`,children:[(0,C.jsx)(l,{eyebrow:(0,C.jsxs)(C.Fragment,{children:[(0,C.jsx)(n,{to:`/how-it-works`,className:`topic-page__home`,children:`How it works`}),(0,C.jsx)(`span`,{"aria-hidden":`true`,children:`/`}),(0,C.jsx)(`span`,{children:o.title}),(0,C.jsxs)(`span`,{className:`topic-page__position`,children:[s,` of `,c.length]})]}),title:e.title,lede:e.lead}),t,r?.length?(0,C.jsx)(re,{links:r}):null,(0,C.jsxs)(`nav`,{className:`topic-pager`,"aria-label":`Previous and next page`,children:[u?(0,C.jsx)(T,{topic:u,direction:`previous`}):(0,C.jsx)(`span`,{}),d?(0,C.jsx)(T,{topic:d,direction:`next`}):(0,C.jsx)(`span`,{})]}),(0,C.jsx)(ne,{current:e.slug})]})}function T({topic:e,direction:t}){return(0,C.jsxs)(n,{to:`/how-it-works/$topic`,params:{topic:e.slug},className:`topic-pager__link`,"data-direction":t,children:[(0,C.jsx)(`span`,{className:`topic-pager__label`,children:t===`next`?`Next`:`Previous`}),(0,C.jsx)(`span`,{className:`topic-pager__title`,children:e.title}),(0,C.jsx)(`span`,{className:`topic-pager__lead`,children:e.lead})]})}function ne({current:e}){return(0,C.jsx)(`nav`,{className:`topic-map`,"aria-label":`All how-it-works pages`,children:o.map(t=>(0,C.jsxs)(`div`,{className:`topic-map__part`,children:[(0,C.jsx)(`span`,{className:`topic-map__title`,children:t.title}),(0,C.jsx)(`ul`,{children:c.filter(e=>e.part===t.id).map(t=>(0,C.jsx)(`li`,{children:(0,C.jsx)(n,{to:`/how-it-works/$topic`,params:{topic:t.slug},"aria-current":t.slug===e?`page`:void 0,children:t.title})},t.slug))})]},t.id))})}function re({links:e}){return(0,C.jsxs)(`section`,{className:`try-it`,"aria-label":`Use it`,children:[(0,C.jsx)(`h2`,{className:`try-it__title`,children:`Use it`}),(0,C.jsx)(`ul`,{className:`try-it__list`,children:e.map(e=>(0,C.jsxs)(`li`,{children:[(0,C.jsx)(ie,{link:e}),e.note?(0,C.jsx)(`span`,{className:`try-it__note`,children:e.note}):null]},`${e.kind}:${e.kind===`function`?e.id:e.slug}`))})]})}function ie({link:e}){if(e.kind===`function`){let t=u(e.id);return(0,C.jsxs)(n,{to:`/fn/$id`,params:{id:e.id},className:`try-it__link`,children:[(0,C.jsx)(`span`,{className:`try-it__kind`,children:`function`}),t?.title??e.id]})}return e.kind===`reference`?(0,C.jsxs)(n,{to:`/docs/$`,params:{_splat:e.slug},className:`try-it__link`,children:[(0,C.jsx)(`span`,{className:`try-it__kind`,children:`reference`}),e.label]}):(0,C.jsxs)(n,{to:`/how-it-works/$topic`,params:{topic:e.slug},className:`try-it__link`,children:[(0,C.jsx)(`span`,{className:`try-it__kind`,children:`explainer`}),s(e.slug)?.title??e.slug]})}function ae({children:e}){return(0,C.jsx)(`p`,{className:`topic-prose`,children:e})}function oe({points:e}){return(0,C.jsxs)(`aside`,{className:`in-short`,"aria-label":`In short`,children:[(0,C.jsx)(`span`,{className:`in-short__title`,children:`In short`}),(0,C.jsx)(`ul`,{children:e.map((e,t)=>(0,C.jsx)(`li`,{children:e},t))})]})}function se({id:e,children:t}){let r=te(e);return(0,C.jsx)(n,{to:`/how-it-works/$topic`,params:{topic:`glossary`},hash:`term-${e}`,className:`term`,title:r?.definition,children:t??r?.term.toLowerCase()??e})}function ce({children:e,caption:t,notice:n}){return(0,C.jsxs)(`figure`,{className:`meta-figure`,children:[e,t||n?(0,C.jsxs)(`figcaption`,{className:`meta-figure__caption`,children:[n?(0,C.jsxs)(`strong`,{className:`meta-figure__notice`,children:[n,` `]}):null,t]}):null]})}function le({scenarioId:e}){return(0,C.jsxs)(`p`,{className:`meta-figure__caption`,children:[`No recorded run for `,e,`.`]})}function ue({stats:e}){return(0,C.jsx)(`dl`,{className:`stat-strip`,children:e.map((e,t)=>(0,C.jsxs)(`div`,{children:[(0,C.jsx)(`dt`,{children:e.label}),(0,C.jsx)(`dd`,{children:e.value})]},t))})}function de(e){return e.replaceAll(`&`,`&amp;`).replaceAll(`<`,`&lt;`).replaceAll(`>`,`&gt;`)}function fe(e){return e.html?e.html:`<pre class="shiki"><code>${e.code.split(`
`).map(e=>`<span class="line">${de(e)}</span>`).join(`
`)}</code></pre>`}function E({example:e,lines:t,label:n}){let r=(0,x.useId)();return(0,C.jsxs)(`div`,{className:`example-code`,"data-scope":r,"data-marked":t?`true`:void 0,children:[n?(0,C.jsx)(`div`,{className:`example-code__label`,children:n}):null,t?(0,C.jsx)(`style`,{children:`[data-scope="${r}"] .line:nth-child(n+${t[0]}):nth-child(-n+${t[1]}){background:var(--color-accent-soft) !important;box-shadow:inset 3px 0 0 var(--color-accent);opacity:1}`}):null,(0,C.jsx)(`div`,{dangerouslySetInnerHTML:{__html:fe(e)}})]})}function pe({example:e,steps:t,active:n,onActive:r,label:i}){let a=t.find(e=>e.id===n)??t[0];return(0,C.jsxs)(`div`,{className:`code-walk`,children:[(0,C.jsx)(E,{example:e,lines:a?.lines,label:i}),(0,C.jsx)(`ol`,{className:`code-walk__steps`,children:t.map(e=>(0,C.jsx)(`li`,{children:(0,C.jsxs)(`button`,{type:`button`,className:`code-walk__step`,"aria-pressed":e.id===a?.id,onClick:()=>r(e.id),onPointerEnter:t=>{t.pointerType===`mouse`&&r(e.id)},onFocus:()=>r(e.id),children:[(0,C.jsx)(`strong`,{children:e.title}),(0,C.jsx)(`span`,{children:e.text})]})},e.id))})]})}var me=[{id:`buffers`,title:`Buffers`,topic:`data-on-the-gpu`},{id:`build`,title:`Build`,topic:`command-graph`},{id:`compile`,title:`Compile`,topic:`scheduling`},{id:`encode`,title:`Encode`,topic:`scheduling`},{id:`dispatch`,title:`Dispatch`,topic:`kernels`},{id:`readback`,title:`Read back`,topic:`anatomy`}];function he({stages:e}){return(0,C.jsx)(`ol`,{className:`run-locator`,"aria-label":`Stages of a run; this page covers the marked ones`,children:me.map(t=>(0,C.jsx)(`li`,{"data-here":e.includes(t.id)?`true`:void 0,children:(0,C.jsx)(n,{to:`/how-it-works/$topic`,params:{topic:t.topic},children:t.title})},t.id))})}var D=32,O=`
struct Look {
  grid: vec4f,
  clock: vec4f,
  size: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f,
  ink: vec4f,
  pointer: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
${h}
`,ge=`
${O}
@group(0) @binding(1) var<storage, read> cells: array<f32>;

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
  output.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}

fn cellValue(column: i32, row: i32) -> f32 {
  let columns = i32(look.grid.x);
  let rows = i32(look.grid.y);
  return cells[u32(clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1))];
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let shifted = input.uv * look.grid.xy - vec2f(0.5);
  let base = vec2i(floor(shifted));
  let fraction = shifted - floor(shifted);
  let value = mix(
    mix(cellValue(base.x, base.y), cellValue(base.x + 1, base.y), fraction.x),
    mix(cellValue(base.x, base.y + 1), cellValue(base.x + 1, base.y + 1), fraction.x),
    fraction.y
  );
  let presence = look.pointer.z;
  // Hover locks the band to the value under the probe and widens it slightly.
  let level = mix(fract(look.clock.x * 0.09), look.pointer.w, presence);
  let width = mix(0.025, 0.032, presence);
  let band = exp(-pow((value - level) / width, 2.0)) * max(look.clock.y, presence);
  let body = sampleRamp(value);
  let visible = smoothstep(0.02, 0.28, value);
  var color = mix(look.background.rgb, body, visible);
  color = mix(color, look.ink.rgb, band * mix(0.6, 0.75, presence));
  // Probe ring: thin antialiased circle at the eased pointer position (framebuffer pixels).
  let ringDistance = abs(length(input.position.xy - look.pointer.xy) - look.grid.z);
  let ring = (1.0 - smoothstep(0.5, 1.5, ringDistance)) * presence;
  color = mix(color, look.signal.rgb, ring);
  return vec4f(color, 1.0);
}
`,_e=`
${O}
@group(0) @binding(1) var<storage, read> points: array<vec4f>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) corner: vec2f,
  @location(1) value: f32,
  @location(2) kind: f32
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let point = points[instanceIndex];
  let corner = corners[vertexIndex];
  let sweep = exp(-pow((point.x - fract(look.clock.x * 0.12)) / 0.07, 2.0));
  // Points near the pointer swell with a gaussian falloff (sigma is six ring radii).
  let pointerOffset = (vec2f(point.x, point.y) - look.pointer.xy / look.size.xy) * look.size.xy;
  let swell = exp(-dot(pointerOffset, pointerOffset) / pow(look.grid.z * 6.0, 2.0)) * look.pointer.z;
  let radius = look.clock.z * (1.0 + 0.7 * sweep * select(0.0, 1.0, point.w > 0.5)) * (1.0 + 0.8 * swell);
  let center = vec2f(point.x * 2.0 - 1.0, 1.0 - point.y * 2.0);
  var output: VertexOutput;
  output.position = vec4f(center + corner * radius * 2.0 / look.size.xy, 0.0, 1.0);
  output.corner = corner;
  output.value = point.z;
  output.kind = point.w;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.corner);
  if (distance > 1.0) {
    discard;
  }
  let edge = 1.0 - smoothstep(0.7, 1.0, distance);
  if (input.kind > 1.5) {
    return vec4f(look.signal.rgb, edge);
  }
  if (input.kind > 0.5) {
    return vec4f(sampleRamp(input.value), edge);
  }
  return vec4f(look.ink.rgb, edge * 0.5);
}
`;function ve(e){return async()=>({default:k(e)})}function k(e){return({device:t,theme:n,width:r,height:i,reducedMotion:a})=>new A(t,e,n,r,i,a)}var A=class{device;colors;look=new Float32Array(D);lookBuffer;buffers=[];rasterModel;pointModel;rasterSize;pointRadius;reducedMotion;pointer;rasterValues;rasterColumns;width;height;constructor(e,t,n,r,i,a){if(y(e),this.device=e,this.colors=m(n),this.width=r,this.height=i,this.reducedMotion=a,this.pointer=new _(a),this.lookBuffer=b(e,D),this.rasterSize=t.raster?[t.raster.width,t.raster.height]:[1,1],this.rasterColumns=this.rasterSize[0],t.raster){let r=Uint8Array.from(atob(t.raster.grid),e=>e.charCodeAt(0)),i=Float32Array.from(r,e=>e/255),a=g(e,i);this.rasterValues=i,this.buffers.push(a),this.rasterModel=p(e,{id:`map-raster`,source:ge,bindings:{look:this.lookBuffer,cells:a},instanceCount:1,blendMode:`opaque`,theme:n})}let o=[];t.layers.forEach((e,t)=>{for(let n=0;n<e.positions.length/2;n++){let r=e.values?.[n],i=r===void 0?t>0&&e.positions.length<=64?2:0:1;o.push(e.positions[n*2],e.positions[n*2+1],r??0,i)}});let s=o.length/4;if(this.pointRadius=Math.min(7,Math.max(2.2,120/Math.sqrt(Math.max(1,s)))),s>0){let t=g(e,Float32Array.from(o));this.buffers.push(t),this.pointModel=p(e,{id:`map-points`,source:_e,bindings:{look:this.lookBuffer,points:t},instanceCount:s,theme:n})}}setPointer(e){this.pointer.set(e)}sampleRaster(e,t){let n=this.rasterValues;if(!n)return 0;let[r,i]=this.rasterSize,a=e/this.width*r-.5,o=t/this.height*i-.5,s=Math.floor(a),c=Math.floor(o),l=a-s,u=o-c,d=(e,t)=>n[Math.min(Math.max(t,0),i-1)*this.rasterColumns+Math.min(Math.max(e,0),r-1)],f=d(s,c)*(1-l)+d(s+1,c)*l,p=d(s,c+1)*(1-l)+d(s+1,c+1)*l;return f*(1-u)+p*u}render(e){this.pointer.update(e/1e3);let t=typeof window>`u`?1:window.devicePixelRatio||1,n=(this.reducedMotion?f:e)/1e3,r=this.look;r.set([this.rasterSize[0],this.rasterSize[1],10*t,0],0),r.set([n,+!this.reducedMotion,this.pointRadius*.55*(this.width/700>1?1.3:1),0],4),r.set([this.width,this.height,0,0],8),r.set(this.colors.background,12),r.set(this.colors.accent,16),r.set(this.colors.signal,20),r.set(this.colors.ink,24),r.set([this.pointer.x,this.pointer.y,this.pointer.presence,Math.max(.06,this.sampleRaster(this.pointer.x,this.pointer.y))],28),this.lookBuffer.write(r);let i=this.device.beginRenderPass({clearColor:this.colors.background});this.rasterModel?.draw(i),this.pointModel?.draw(i),i.end(),this.device.submit()}resize(e,t){this.width=e,this.height=t}destroy(){this.rasterModel?.destroy(),this.pointModel?.destroy(),this.lookBuffer.destroy();for(let e of this.buffers)e.destroy()}},j={recordCount:427,packs:{"gpu-dataframe":26,"gpu-network":43,"gpu-raster":54,"gpu-spatial-analysis":258,"gpu-terrain":46},kernels:37919,copies:587,gatedKernels:11203,dependencies:{raw:128377,war:34591,waw:3032},buffers:24980,bufferBytes:501552408,bufferBytesKind:`lower-bound`,snapshots:3499,formulas:986,recordsWithFormulas:255,kernelsPerRecord:{minimum:1,median:27,maximum:885},smallest:{id:`gpu-dataframe.scalar-scale`,kernels:1},largest:{id:`gpu-dataframe.ripley-envelope`,kernels:885},recordBytes:50393969,viewModelBytes:8029075},M=`modules/analysis/src/gpu-terrain/terrain-decode/gpu-terrain-rgb-decode.ts`;function N(e){let t=Math.floor(4294967295/e)+1;return[{text:`override INVOCATION_COUNT: u32;`,tag:`setup`},{text:`override WGSL_KERNEL_DISPATCH_X: u32;`,tag:`setup`},{text:`override WGSL_KERNEL_DISPATCH_Y: u32;`,tag:`setup`},{text:`const inputWordsOffset: u32 = 0u;`,tag:`setup`},{text:`@group(0) @binding(0) var<storage, read> inputWords: array<u32>;`,tag:`setup`},{text:`const outputValuesOffset: u32 = 0u;`,tag:`setup`},{text:`@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;`,tag:`setup`},{text:`const outputValidityOffset: u32 = 0u;`,tag:`setup`},{text:`@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;`,tag:`setup`},{text:`const WIDTH: u32 = 32u;`,tag:`setup`},{text:`const HEIGHT: u32 = 32u;`,tag:`setup`},{text:`fn isEncodedPixelValid(red: u32, green: u32, blue: u32, alpha: u32) -> bool {`,tag:`math`},{text:`  if (alpha == 0u) { return false; }`,tag:`math`},{text:`  `,tag:`math`},{text:`  return true;`,tag:`math`},{text:`}`,tag:`math`},{text:`// Integer code N = R * 65536 + G * 256 + B < 2^24, then exactly one float operation sequence.`,tag:`plain`},{text:`fn decodeHeight(red: u32, green: u32, blue: u32) -> f32 {`,tag:`math`},{text:`  let code = (red << 16u) | (green << 8u) | blue;`,tag:`math`},{text:`  return f32(code) * 0.00390625 - 32768.0;`,tag:`math`},{text:`}`,tag:`math`},{text:`fn isHeightInRange(height: f32) -> bool {`,tag:`math`},{text:`  return height >= -11000.0 && height <= 9000.0;`,tag:`math`},{text:`}`,tag:`math`},{text:`fn applySeaClamp(height: f32) -> f32 {`,tag:`math`},{text:`  `,tag:`math`},{text:`  return height;`,tag:`math`},{text:`}`,tag:`math`},{text:``,tag:`plain`},{text:`@compute @workgroup_size(${e})`,tag:`shape`},{text:`fn main(`,tag:`plain`},{text:`  @builtin(local_invocation_index) localInvocationIndex: u32,`,tag:`index`},{text:`  @builtin(workgroup_id) workgroupId: vec3<u32>`,tag:`index`},{text:`) {`,tag:`plain`},{text:`  _ = &inputWords;`,tag:`plain`},{text:`  _ = &outputValues;`,tag:`plain`},{text:`  _ = &outputValidity;`,tag:`plain`},{text:`  _ = INVOCATION_COUNT;`,tag:`plain`},{text:`  let workgroupIndex =`,tag:`index`},{text:`    (workgroupId.z * WGSL_KERNEL_DISPATCH_Y + workgroupId.y) * WGSL_KERNEL_DISPATCH_X +`,tag:`index`},{text:`    workgroupId.x;`,tag:`index`},{text:`  if (workgroupIndex >= ${t}u) { return; }`,tag:`index`},{text:`  let index = workgroupIndex * ${e}u + localInvocationIndex;`,tag:`index`},{text:`  if (index >= INVOCATION_COUNT) {`,tag:`guard`},{text:`    return;`,tag:`guard`},{text:`  }`,tag:`guard`},{text:`  let pixelIndex = index;`,tag:`math`},{text:`  let word = inputWords[inputWordsOffset + pixelIndex];`,tag:`read`},{text:`  let red = word & 0xffu;`,tag:`math`},{text:`  let green = (word >> 8u) & 0xffu;`,tag:`math`},{text:`  let blue = (word >> 16u) & 0xffu;`,tag:`math`},{text:`  let alpha = word >> 24u;`,tag:`math`},{text:`  var isValid = isEncodedPixelValid(red, green, blue, alpha);`,tag:`math`},{text:`  `,tag:`plain`},{text:`  let height = decodeHeight(red, green, blue);`,tag:`math`},{text:`  isValid = isValid && isHeightInRange(height);`,tag:`math`},{text:`  outputValidity[outputValidityOffset + pixelIndex] = select(0u, 1u, isValid);`,tag:`write`},{text:`  outputValues[outputValuesOffset + pixelIndex] =`,tag:`write`},{text:`    select(bitcast<f32>(0x7fc00000u | (pixelIndex & 0u)), applySeaClamp(height), isValid);`,tag:`write`},{text:`}`,tag:`plain`}]}var P=[8,16,32,64,128,256],F={columns:32,rows:32,invocationCount:1e3};function ye(e,t){let n=t*e,r=n+e-1,i=Math.min(r,F.invocationCount-1);return{first:n,last:i,guarded:Math.max(0,r-i),activeCount:Math.max(0,i-n+1)}}var I=65535;function be(e){return Math.min(256,2**Math.ceil(Math.log2(Math.max(e,32))))}function xe(e,t,n=I){let r=Math.max(1,Math.ceil(e/t)),i=Math.min(r,n),a=Math.min(Math.ceil(r/i),n);return{x:i,y:a,z:Math.ceil(r/i/a),workgroupCount:r}}function Se(e){return{workgroupSize:e,hovered:-1}}var L=32,R=4.5,Ce=6.5,we=.12,Te=.22,Ee=.1,De=`
struct Look {
  grid: vec4f,
  clock: vec4f,
  view: vec4f,
  view2: vec4f,
  background: vec4f,
  signal: vec4f,
  ink: vec4f,
  pointer: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
${h}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  return vec4f(corners[vertexIndex] * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let columns = look.grid.x;
  let rows = look.grid.y;
  let size = look.grid.z;
  let count = look.grid.w;
  let hovered = look.clock.y;
  let front = look.clock.z;
  let concurrency = look.clock.w;
  let focus = look.pointer.x;
  let outlineStrength = look.pointer.y;
  let local = (position.xy - vec2f(look.view.w, look.view2.x)) / look.view.z;
  if (local.x < 0.0 || local.y < 0.0 || local.x >= columns || local.y >= rows) {
    return vec4f(0.0);
  }
  let cell = floor(local);
  let inCell = fract(local);
  let index = cell.y * columns + cell.x;
  let workgroup = floor(index / size);
  let workgroupCount = ceil(count / size);
  let offset = abs(inCell - vec2f(0.5));
  let distance = max(offset.x, offset.y);
  let body = 1.0 - smoothstep(0.38, 0.45, distance);
  let hole = 1.0 - smoothstep(0.27, 0.34, distance);

  if (index >= count) {
    return vec4f(look.signal.rgb, (body - hole) * 0.85);
  }

  var color = sampleRamp(workgroup / max(workgroupCount - 1.0, 1.0));
  color = mix(color, look.ink.rgb, select(0.0, 0.32, (u32(workgroup) & 1u) == 1u));
  let baseColor = color;
  var alpha = 0.82;
  if (hovered >= 0.0 && focus >= 1.0) {
    alpha = select(0.16, 1.0, abs(hovered - workgroup) < 0.5);
  } else {
    let age = front - workgroup;
    if (age >= 0.0 && age < concurrency) {
      color = mix(color, vec3f(1.0), 0.55);
      alpha = 1.0;
    } else if (age < 0.0) {
      alpha = 0.3;
    }
    // A fading pointer hover blends from the sweep toward the hovered look.
    if (hovered >= 0.0) {
      let hoveredAlpha = select(0.16, 1.0, abs(hovered - workgroup) < 0.5);
      alpha = mix(alpha, hoveredAlpha, focus);
      color = mix(color, baseColor, focus);
    }
  }
  // The first invocation of a workgroup carries a bar on its left edge.
  let isStart = (u32(index) % u32(size)) == 0u;
  let bar = select(0.0, 1.0 - smoothstep(0.1, 0.16, inCell.x), isStart);
  color = mix(color, look.ink.rgb, bar * 0.9);
  // Pointer hover outlines the hovered workgroup's span: a thin line on each cell edge whose
  // neighbour belongs to another workgroup, is guarded, or lies outside the grid.
  var outline = 0.0;
  if (outlineStrength > 0.0 && abs(hovered - workgroup) < 0.5) {
    for (var side = 0; side < 4; side++) {
      let isHorizontal = side < 2;
      let direction = select(-1.0, 1.0, (side & 1) == 1);
      let neighbor = cell + select(vec2f(0.0, direction), vec2f(direction, 0.0), isHorizontal);
      let neighborIndex = neighbor.y * columns + neighbor.x;
      let isOutside = neighbor.x < 0.0 || neighbor.y < 0.0 || neighbor.x >= columns || neighbor.y >= rows;
      let isOther = isOutside || neighborIndex >= count || floor(neighborIndex / size) != workgroup;
      let along = select(inCell.y, inCell.x, isHorizontal);
      let edgeDistance = select(along, 1.0 - along, direction > 0.0);
      let line = 1.0 - smoothstep(0.03, 0.09, edgeDistance);
      outline = max(outline, select(0.0, line, isOther));
    }
  }
  let lineAlpha = outline * outlineStrength * 0.9;
  return vec4f(mix(color, look.ink.rgb, lineAlpha), max(body * alpha, lineAlpha));
}
`;function z(e){return async()=>({default:Oe(e)})}function Oe(e){return({device:t,theme:n,width:r,height:i,reducedMotion:a})=>new ke(t,e,n,r,i,a)}var ke=class{device;store;colors;look=new Float32Array(L);lookBuffer;model;reducedMotion;pointer;focus=0;pointerWorkgroup=-1;lastFocusSeconds=-1;width;height;constructor(e,t,n,r,i,a){y(e),this.device=e,this.store=t,this.colors=m(n),this.width=r,this.height=i,this.reducedMotion=a,this.pointer=new _(a),this.lookBuffer=b(e,L),this.model=p(e,{id:`kernel-tiling`,source:De,bindings:{look:this.lookBuffer},instanceCount:1,theme:n})}render(e){let{columns:t,rows:n,invocationCount:r}=F,i=this.store.workgroupSize,a=Math.ceil(r/i),o=(this.reducedMotion?f:e)/1e3,s=o%Ce,c=Math.max(2,Math.round(a/8)),l=this.reducedMotion||s>=R?1e6:s/R*(a+c),u=Math.min(this.width/t,this.height/n),d=(this.width-u*t)/2,p=(this.height-u*n)/2,m=e/1e3;this.pointer.update(m),this.updateFocus(m,i,u,d,p);let h=this.store.hovered>=0,g=h?this.store.hovered:this.focus>0?this.pointerWorkgroup:-1,_=h?1:this.focus,v=this.look;v.set([t,n,i,r],0),v.set([o,g,l,c],4),v.set([this.width,this.height,u,d],8),v.set([p,0,0,0],12),v.set(this.colors.background,16),v.set(this.colors.signal,20),v.set(this.colors.ink,24),v.set([_,h?0:_,0,0],28),this.lookBuffer.write(v);let y=this.device.beginRenderPass({clearColor:this.colors.background});this.model.draw(y),y.end(),this.device.submit()}setPointer(e){this.pointer.set(e)}updateFocus(e,t,n,r,i){let{columns:a,rows:o,invocationCount:s}=F,c=this.pointer.target,l=!1;if(c){let e=Math.floor((c[0]-r)/n),u=Math.floor((c[1]-i)/n),d=u*a+e;e>=0&&e<a&&u>=0&&u<o&&d<s&&(l=!0,this.pointerWorkgroup=Math.floor(d/t))}let u=this.lastFocusSeconds<0?0:e-this.lastFocusSeconds;this.lastFocusSeconds=e;let d=+!!l;if(this.reducedMotion||u<=0||u>Ee){this.focus=d;return}let f=1-Math.exp(-u/(d>this.focus?we:Te));this.focus+=(d-this.focus)*f,Math.abs(d-this.focus)<.002&&(this.focus=d)}resize(e,t){this.width=e,this.height=t}destroy(){this.model.destroy(),this.lookBuffer.destroy()}},Ae={columns:16,rows:8,laneSide:8,invocationCount:7781,concurrency:7},B=32,je={none:0,dispatch:1,trail:2},Me=.6,Ne=.45,Pe=.25,V=-1e6,H=10,Fe=6.5,Ie=`
struct Look {
  grid: vec4f,
  clock: vec4f,
  size: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f,
  highlight: vec4f,
  pointer: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;
@group(0) @binding(1) var<storage, read> touched: array<f32>;

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
  output.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let columns = look.grid.x;
  let rows = look.grid.y;
  let laneSide = look.grid.z;
  let invocationCount = look.grid.w;
  let progress = look.clock.y;
  let concurrency = look.clock.z;
  let fade = look.clock.w;

  let cellUv = input.uv * vec2f(columns, rows);
  let cell = floor(cellUv);
  let inCell = fract(cellUv);
  let workgroup = cell.y * columns + cell.x;
  let age = progress - workgroup;

  let gap = 0.09;
  let inner = (inCell - vec2f(gap)) / (1.0 - 2.0 * gap);
  var color = look.background.rgb;
  var alpha = 0.0;

  // Workgroup frame.
  let frameDistance = max(abs(inCell.x - 0.5), abs(inCell.y - 0.5));
  let frame = smoothstep(0.5 - gap * 0.55, 0.5 - gap * 0.45, frameDistance) *
    (1.0 - smoothstep(0.5 - gap * 0.2, 0.5 - gap * 0.05, frameDistance));
  var frameStrength = 0.14;
  if (age >= 0.0 && age < concurrency) {
    frameStrength = 0.9;
  } else if (age >= concurrency) {
    frameStrength = 0.3 * fade;
  }

  // Pointer interaction state; everything below is gated by mode, so mode 0 is the plain picture.
  let mode = look.size.z;
  let isDispatch = mode > 0.5 && mode < 1.5;
  let isTrail = mode > 1.5;
  let presence = look.pointer.z;
  let isHovered = isDispatch && workgroup == look.size.w;
  let pointerOffset = cell + vec2f(0.5) - look.pointer.xy;
  let glow = select(0.0, exp(-dot(pointerOffset, pointerOffset) / 1.6) * presence, isDispatch);
  var heat = 0.0;
  if (isTrail) {
    heat = exp(-max(look.pointer.w - touched[u32(workgroup)], 0.0) / 0.8);
    frameStrength = max(frameStrength * (1.0 - 0.3 * presence), heat);
  }
  if (isDispatch) {
    frameStrength = frameStrength * (1.0 - 0.5 * presence) + glow * 0.6;
    if (isHovered) {
      frameStrength = mix(frameStrength, 1.0, presence);
    }
  }
  color = look.accent.rgb;
  alpha = frame * frameStrength;
  let frameAlpha = alpha;

  if (inner.x >= 0.0 && inner.x < 1.0 && inner.y >= 0.0 && inner.y < 1.0) {
    let lanePosition = inner * laneSide;
    let lane = floor(lanePosition);
    let inLane = fract(lanePosition) - vec2f(0.5);
    let laneIndex = lane.y * laneSide + lane.x;
    let laneCount = laneSide * laneSide;
    let invocation = workgroup * laneCount + laneIndex;
    let radius = length(inLane);
    let disc = 1.0 - smoothstep(0.27, 0.36, radius);
    let ring = disc - (1.0 - smoothstep(0.17, 0.26, radius));
    let guarded = invocation >= invocationCount;

    if (guarded) {
      color = look.signal.rgb;
      alpha = max(alpha, ring * 0.7 * select(0.4, 1.0, age >= 0.0));
    } else if (age < 0.0) {
      color = look.accent.rgb;
      alpha = max(alpha, disc * 0.16);
    } else if (age < concurrency) {
      let reach = clamp(age / concurrency, 0.0, 1.0);
      let lit = select(0.0, 1.0, laneIndex / laneCount <= reach);
      color = mix(look.accent.rgb, vec3f(1.0), 0.35 * lit);
      alpha = max(alpha, disc * mix(0.2, 1.0, lit));
    } else {
      color = look.accent.rgb;
      alpha = max(alpha, disc * 0.5 * fade);
    }

    if (isDispatch) {
      if (!guarded) {
        alpha = alpha * (1.0 - 0.5 * presence) + disc * glow * 0.35;
      }
      if (isHovered) {
        // Lanes fill in launch order up to reach; guarded lanes show as hollow signal rings.
        let reach = look.pointer.w;
        let lanePosition = (laneIndex + 1.0) / laneCount;
        let isLit = lanePosition <= reach;
        let isFront = isLit && reach < 1.0 && lanePosition > reach - 1.0 / laneCount;
        var hoverColor = look.signal.rgb;
        var hoverAlpha = max(frameAlpha, ring * 0.95);
        if (!guarded) {
          hoverColor = mix(look.accent.rgb, look.highlight.rgb, select(0.0, select(0.35, 0.75, isFront), isLit));
          hoverAlpha = max(frameAlpha, disc * select(0.2, 1.0, isLit));
        }
        color = mix(color, hoverColor, presence);
        alpha = mix(alpha, hoverAlpha, presence);
      }
    }

    if (isTrail) {
      if (guarded) {
        color = mix(color, mix(look.signal.rgb, look.highlight.rgb, 0.4), heat);
        alpha = mix(alpha, max(alpha, ring), heat);
      } else {
        alpha = mix(alpha * (1.0 - 0.3 * presence), max(alpha, disc), heat);
        color = mix(color, mix(look.accent.rgb, look.highlight.rgb, 0.55), heat);
      }
    }
  }
  return vec4f(color, clamp(alpha, 0.0, 1.0));
}
`;function Le(e,t={}){return async()=>({default:Re(e,t)})}function Re(e,t={}){return({device:n,theme:r,width:i,height:a,reducedMotion:o})=>new Be(n,e,t.interaction??`none`,r,i,a,o)}function ze(e,t){if(t===`dark`)return[1,1,1,1];let[n,r,i]=e.accent,[a,o,s]=e.ink;return[(n+a)/2,(r+o)/2,(i+s)/2,1]}var Be=class{device;grid;interaction;colors;highlight;look=new Float32Array(B);lookBuffer;lastTouched;touchedBuffer;model;reducedMotion;pointer;width;height;hoveredIndex=-1;launchSeconds=0;wasActive=!1;previousCell=null;hasTouches=!1;constructor(e,t,n,r,i,a,o){y(e),this.device=e,this.grid=t,this.interaction=n,this.colors=m(r),this.highlight=ze(this.colors,r),this.width=i,this.height=a,this.reducedMotion=o,this.pointer=new _(o),this.lookBuffer=b(e,B),this.lastTouched=new Float32Array(t.columns*t.rows).fill(V),this.touchedBuffer=g(e,this.lastTouched),this.model=p(e,{id:`workgroup-grid`,source:Ie,bindings:{look:this.lookBuffer,touched:this.touchedBuffer},instanceCount:1,theme:r})}setPointer(e){this.pointer.set(e)}render(e){let t=e/1e3;this.pointer.update(t);let n=(this.reducedMotion?f*1.4:e)/1e3,r=n%H,i=this.grid.columns*this.grid.rows,a=Math.min(1,r/Fe)*(i+this.grid.concurrency),o=r<9?1:Math.max(0,H-r),s=0;this.interaction===`dispatch`?s=this.updateDispatch(t):this.interaction===`trail`&&(this.updateTrail(t),s=t);let c=this.look;c.set([this.grid.columns,this.grid.rows,this.grid.laneSide,this.grid.invocationCount],0),c.set([n,a,this.grid.concurrency,o],4),c.set([this.width,this.height,je[this.interaction],this.hoveredIndex],8),c.set(this.colors.background,12),c.set(this.colors.accent,16),c.set(this.colors.signal,20),c.set(this.highlight,24),c.set([this.pointer.x/this.width*this.grid.columns,this.pointer.y/this.height*this.grid.rows,this.pointer.presence,s],28),this.lookBuffer.write(c);let l=this.device.beginRenderPass({clearColor:this.colors.background});this.model.draw(l),l.end(),this.device.submit()}updateDispatch(e){let{columns:t,rows:n}=this.grid,r=this.pointer.isActive;if(this.pointer.target){let r=Math.floor(this.pointer.target[0]/this.width*t),i=Math.floor(this.pointer.target[1]/this.height*n);if(r>=0&&r<t&&i>=0&&i<n){let n=i*t+r;n===this.hoveredIndex&&this.wasActive||(this.launchSeconds=e),this.hoveredIndex=n}}if(this.wasActive=r,this.reducedMotion)return 1;let i=(e-this.launchSeconds)%Me;return Math.min(1,i/Ne)}updateTrail(e){let{columns:t,rows:n}=this.grid,r=!1,i=(i,a)=>{let o=Math.floor(i),s=Math.floor(a);o<0||o>=t||s<0||s>=n||(this.lastTouched[s*t+o]=e,r=!0,this.hasTouches=!0)};if(this.reducedMotion)this.hasTouches&&(this.lastTouched.fill(V),this.hasTouches=!1,r=!0),this.pointer.isActive&&i(this.pointer.x/this.width*t,this.pointer.y/this.height*n);else if(this.pointer.isActive){let e=this.pointer.x/this.width*t,r=this.pointer.y/this.height*n,[a,o]=this.previousCell??[e,r],s=Math.max(1,Math.ceil(Math.hypot(e-a,r-o)/Pe));for(let t=0;t<=s;t++){let n=t/s;i(a+(e-a)*n,o+(r-o)*n)}this.previousCell=[e,r]}else this.previousCell=null;this.reducedMotion&&(this.previousCell=null),r&&this.touchedBuffer.write(this.lastTouched)}resize(e,t){this.width=e,this.height=t}destroy(){this.model.destroy(),this.lookBuffer.destroy(),this.touchedBuffer.destroy()}},U=[{name:`@luma-contributors/kepler`,role:`kepler.gl adapters`,owner:`ours`},{name:`@luma-contributors/lenses`,role:`records to pictures`,owner:`ours`},{name:`@luma-contributors/analysis`,role:`GPU operations`,owner:`ours`},{name:`@luma.gl/graph`,role:`kernels, command graphs`,owner:`upstream`},{name:`@luma.gl/core`,role:`device, buffers, passes`,owner:`upstream`},{name:`@luma.gl/shaders`,role:`shader modules`,owner:`upstream`}],W=188,G=58,K=40,q=22,J=46,Y=44+U.length*W+(U.length-1)*K,X=138,Ve=640,He=400,Z=168,Ue=92,Q=[34,236,438],We=[64,244],Ge=Q[1]-Q[0]-Z;function $(e=`card`){let t=U.length-1,n=Q.length,r=U.map((r,i)=>{let a=e===`card`,o=Math.floor(i/n),s=o===0?i:n*2-1-i;return{x:a?Q[s]:q+i*228,y:a?We[o]:J,width:a?Z:W,height:a?Ue:G,kind:r.owner===`ours`?0:3,progress:i/t}}),i=r.slice(0,-1).map((i,a)=>{let o=r[a+1],s=i.y+i.height/2,c=e===`card`?Ge:K,l={x0:i.x+i.width,y0:s,x1:o.x,y1:s,controlX0:i.x+i.width+c/2,controlY0:s,controlX1:o.x-c/2,controlY1:s,weight:.7,progressStart:a/t,progressEnd:(a+1)/t,fromNode:a,toNode:a+1};if(e===`row`||o.y===i.y)return e===`card`&&a>=n?{...l,x0:i.x,x1:o.x+o.width,controlX0:i.x-c/2,controlX1:o.x+o.width+c/2}:l;let u=i.x+i.width/2,d=(i.y+i.height+o.y)/2;return{...l,x0:u,y0:i.y+i.height,x1:o.x+o.width/2,y1:o.y,controlX0:u,controlY0:d,controlX1:o.x+o.width/2,controlY1:d}});return e===`card`?{width:Ve,height:He,nodes:r,edges:i}:{width:Y,height:X,nodes:r,edges:i}}function Ke({version:e}){let t=(0,x.useMemo)(()=>v($(`row`)),[]),n=e=>q+e*228-10;return(0,C.jsx)(`div`,{className:`meta-canvas`,children:(0,C.jsx)(`div`,{className:`meta-scroll`,children:(0,C.jsxs)(`div`,{className:`meta-stage`,style:{minWidth:900,aspectRatio:`${Y} / ${X}`},children:[(0,C.jsx)(`div`,{className:`meta-stage__canvas`,"aria-hidden":`true`,children:(0,C.jsx)(d,{scene:t,fallback:null,seed:1,label:`A call descending the package stack`,aspectRatio:Y/X})}),(0,C.jsxs)(`svg`,{className:`meta-stage__svg`,viewBox:`0 0 ${Y} ${X}`,role:`img`,"aria-label":`kepler depends on lenses, which depend on analysis, which builds on luma.gl graph, core and shaders`,children:[(0,C.jsx)(`rect`,{className:`layer-group`,x:n(0),y:14,width:664,height:118,rx:12}),(0,C.jsx)(`rect`,{className:`layer-group`,x:n(3),y:14,width:664,height:118,rx:12}),(0,C.jsx)(`text`,{className:`layer-group-label`,x:n(0)+12,y:34,children:`this repository`}),(0,C.jsx)(`text`,{className:`layer-group-label`,x:n(3)+12,y:34,children:e?`published luma.gl ${e}`:`published luma.gl`}),U.slice(0,-1).map((e,t)=>{let n=q+t*228+W;return(0,C.jsx)(`path`,{className:`layer-arrow`,d:`M${n+4} 75 H${n+K-5} M${n+K-11} 70 L${n+K-4} 75 L${n+K-11} 80`},`arrow-${e.name}`)}),U.map((e,t)=>(0,C.jsxs)(`g`,{className:`layer-node`,"data-owner":e.owner,transform:`translate(${q+t*228} ${J})`,children:[(0,C.jsx)(`rect`,{width:W,height:G,rx:9}),(0,C.jsx)(`text`,{className:`layer-node__title`,x:12,y:25,children:e.name.replace(`@luma-contributors/`,``)}),(0,C.jsx)(`text`,{className:`layer-node__meta`,x:12,y:43,children:e.role})]},e.name))]})]})})})}export{he as C,S as D,w as E,ee as O,ae as S,se as T,pe as _,z as a,oe as b,M as c,ye as d,be as f,ve as g,j as h,Le as i,I as l,N as m,$ as n,Se as o,xe as p,Ae as r,F as s,Ke as t,P as u,E as v,ue as w,le as x,ce as y};