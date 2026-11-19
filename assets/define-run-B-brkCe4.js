import{Q as e}from"./dist-BusamGlx.js";import{t}from"./gpu-command-graph-BdzdmV6u.js";import{C as n}from"./index-DIfTuwM1.js";import{A as r,C as i,D as a,E as o,O as s,S as c,_ as l,b as u,f as d,g as f,h as p,k as m,m as h,n as g,p as _,r as ee,t as te,v,w as y,x as ne}from"./pointer-overlay-Qh8qflcA.js";import{t as b}from"./preload-helper-Ehsq8Wa4.js";var x=class{buffer;ring;state=`reserved`;copiedByteOffset=0;copiedByteLength;cancelled=!1;constructor(e,t){this.ring=e,this.buffer=t,this.copiedByteLength=t.byteLength}copyFrom(e,t,n={}){this.assertReserved();let r=n.sourceOffset??0,i=n.destinationOffset??0,a=n.byteLength??t.byteLength-r;S(t,this.buffer,r,i,a),e.copyBufferToBuffer({sourceBuffer:t,sourceOffset:r,destinationBuffer:this.buffer,destinationOffset:i,size:a}),this.copiedByteOffset=i,this.copiedByteLength=a,this.state=`encoded`}markEncoded(e={}){this.assertReserved();let t=e.byteOffset??0,n=e.byteLength??this.buffer.byteLength-t;C(this.buffer.byteLength,t,n,`readback`),this.copiedByteOffset=t,this.copiedByteLength=n,this.state=`encoded`}async read(){if(this.state!==`encoded`)throw Error(`GPU readback ticket must be encoded before reading`);this.state=`reading`;try{let e=await this.buffer.readAsync(this.copiedByteOffset,this.copiedByteLength);if(this.cancelled)throw Error(`GPU readback ticket was cancelled`);return e}finally{this.release()}}cancel(){if(this.state===`reserved`)this.cancelled=!0,this.release();else if(this.state===`reading`)this.cancelled=!0;else if(this.state===`encoded`)throw Error(`Start reading an encoded GPU readback ticket before cancelling it`)}assertReserved(){if(this.state!==`reserved`)throw Error(`GPU readback ticket has already been used`)}release(){this.state!==`released`&&(this.state=`released`,this.ring.release(this.buffer))}},re=class{byteLength;slotCount;buffers;availableBuffers;waiters=[];destroyed=!1;constructor(t,n){let r=n.slotCount??3;if(C(2**53-1,0,n.byteLength,`ring`),!Number.isSafeInteger(r)||r<=0)throw Error(`GPUReadbackRing slotCount must be a positive integer`);this.byteLength=n.byteLength,this.slotCount=r;let i=n.id??`gpu-readback-ring`;this.buffers=Array.from({length:r},(r,a)=>t.createBuffer({id:`${i}-slot-${a}`,byteLength:n.byteLength,usage:e.COPY_DST|e.MAP_READ})),this.availableBuffers=[...this.buffers],t.lost.then(()=>this.destroy())}get availableSlotCount(){return this.availableBuffers.length}tryAcquire(){if(this.destroyed)return null;let e=this.availableBuffers.shift();return e?new x(this,e):null}acquire(){let e=this.tryAcquire();return e?Promise.resolve(e):this.destroyed?Promise.reject(Error(`GPUReadbackRing has been destroyed`)):new Promise((e,t)=>this.waiters.push({resolve:e,reject:t}))}destroy(){if(!this.destroyed){this.destroyed=!0;for(let e of this.waiters.splice(0))e.reject(Error(`GPUReadbackRing has been destroyed`));for(let e of this.availableBuffers.splice(0))e.destroy()}}release(e){if(this.destroyed){e.destroy();return}let t=this.waiters.shift();t?t.resolve(new x(this,e)):this.availableBuffers.push(e)}};function S(e,t,n,r,i){if(e.device!==t.device)throw Error(`GPU readback source and staging buffers must belong to the same device`);C(e.byteLength,n,i,`source`),C(t.byteLength,r,i,`destination`)}function C(e,t,n,r){if(!Number.isSafeInteger(t)||!Number.isSafeInteger(n)||t<0||n<=0||t%4!=0||n%4!=0||t+n>e)throw Error(`${r} byte range must be positive, four-byte aligned, and in bounds`)}var w=e=>`
${f}
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> owners: array<u32>;
${a(e,4)}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) value: f32
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let point = positions[indices[vertexIndex]];
  var output: VertexOutput;
  output.position = vec4f(worldToClip(point), 0.0, 1.0);
  output.value = readValue(owners[vertexIndex / 3u]);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (look.mode < -0.5 || color.a == 0.0) {
    color = look.light;
  }
  return vec4f(color.rgb, color.a * look.alpha);
}
`,T=class extends d{constructor(e,t,n,r,i,a,o){super(e,`run-mesh`,w(o),{positions:t,indices:n,owners:r,values:i},1),this.model.setVertexCount(Math.max(a,1)*3)}get usesSolid(){return!0}},E=e=>`
${f}
@group(0) @binding(1) var<storage, read> nodes: array<vec2f>;
@group(0) @binding(2) var<storage, read> edges: array<vec2u>;
${a(e,3)}

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
  if (look.mode > 1.5) {
    output.value = select(endValue, startValue, isValid(startValue));
  } else {
    output.value = select(max(startValue, endValue), mix(startValue, endValue, along[vertexIndex]), isValid(startValue) && isValid(endValue));
  }
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (look.mode < -0.5 || color.a == 0.0) {
    return vec4f(look.light.rgb, look.light.a * look.alpha);
  }
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(color.rgb, color.a * feather * look.alpha);
}
`,ie=class extends d{constructor(e,t,n,r,i,a){super(e,`run-edges`,E(a),{nodes:t,edges:n,values:r},i)}get usesSolid(){return!0}},D=`
// plot[0]: rectangle in world units (x0, y0, x1, y1); plot[1]: x range (min, max) for scatter.
@group(0) @binding(2) var<uniform> plot: array<vec4f, 2>;

fn plotPoint(x: f32, normalizedY: f32) -> vec2f {
  let world = vec2f(mix(plot[0].x, plot[0].z, x), mix(plot[0].y, plot[0].w, normalizedY));
  return worldToClip(world);
}

fn normalizeValue(value: f32) -> f32 {
  var span = look.range.y - look.range.x;
  if (abs(span) < 1.0e-12) {
    span = 1.0e-12;
  }
  return clamp((value - look.range.x) / span, 0.0, 1.0);
}
`,ae=e=>`
${f}
${a(e,1)}
${D}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) value: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let count = f32(arrayLength(&values));
  let value = readValue(instanceIndex);
  let height = select(0.0, normalizeValue(value), isValid(value));
  let x = (f32(instanceIndex) + 0.1 + corner.x * 0.8) / count;
  var output: VertexOutput;
  output.position = vec4f(plotPoint(x, corner.y * height), 0.0, 1.0);
  output.value = value;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color = colorize(input.value);
  if (look.mode < -0.5 || color.a == 0.0) {
    color = look.light;
  }
  return vec4f(color.rgb, color.a * look.alpha);
}
`,oe=e=>`
${f}
${a(e,1)}
${D}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) across: f32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let count = f32(arrayLength(&values));
  let first = readValue(instanceIndex);
  let second = readValue(instanceIndex + 1u);
  let start = plotPoint((f32(instanceIndex) + 0.5) / count, normalizeValue(first));
  let end = plotPoint((f32(instanceIndex) + 1.5) / count, normalizeValue(second));
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
  output.across = side[vertexIndex];
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let feather = 1.0 - smoothstep(0.6, 1.0, abs(input.across));
  return vec4f(look.light.rgb, look.light.a * feather * look.alpha);
}
`,O=e=>`
${f}
${a(e,1)}
${D}
@group(0) @binding(3) var<storage, read> xValues: array<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f
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
  var spanX = plot[1].y - plot[1].x;
  if (abs(spanX) < 1.0e-12) {
    spanX = 1.0e-12;
  }
  let x = clamp((xValues[instanceIndex] - plot[1].x) / spanX, 0.0, 1.0);
  let center = plotPoint(x, normalizeValue(readValue(instanceIndex)));
  var output: VertexOutput;
  output.position = vec4f(center + corner * look.radius * 2.0 / look.size, 0.0, 1.0);
  output.local = corner;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.local);
  if (distance > 1.0) {
    discard;
  }
  let edge = 1.0 - smoothstep(1.0 - 2.0 / max(look.radius, 1.0), 1.0, distance);
  return vec4f(look.light.rgb, look.light.a * edge * look.alpha);
}
`,se=class extends d{plotBuffer;plotValues=new Float32Array(8);constructor(t,n,r,i,a,o){let s=t.createBuffer({byteLength:32,usage:e.UNIFORM|e.COPY_DST}),c=n===`bars`?ae(o):n===`line`?oe(o):O(o);super(t,`run-chart-${n}`,c,n===`scatter`?{values:r,plot:s,xValues:i}:{values:r,plot:s},n===`line`?Math.max(a-1,0):a),this.plotBuffer=s}get usesSolid(){return!0}setPlot(e,t){this.plotValues.set(e,0),this.plotValues.set(t,4),this.plotBuffer.write(this.plotValues)}destroy(){super.destroy(),this.plotBuffer.destroy()}},ce=null,k=null;function le(){return ce}function ue(){return k??=b(async()=>{let{FONT_PIXELS:e}=await import(`./font-pixels-BiY0CLDt.js`);return{FONT_PIXELS:e}},[]).then(({FONT_PIXELS:e})=>{let t=atob(e.data),n=new Uint8Array(Math.ceil(t.length/4)*4);for(let e=0;e<t.length;e++)n[e]=t.charCodeAt(e);return ce=new Uint32Array(n.buffer),ce}),k}var A={em:24,spread:6,width:512,height:180,ascent:.75,descent:.25,capHeight:.694,xHeight:.444,glyphs:[[32,459,156,13,13,-.2708,.2708,.25],[33,375,38,16,30,-.1667,.9688,.319],[34,107,156,21,19,-.2396,.9688,.5],[35,238,0,31,35,-.2187,.9688,.833],[36,368,0,23,33,-.2292,1.0208,.5],[37,391,0,31,33,-.2187,1.0208,.833],[38,478,0,29,31,-.2292,.9896,.758],[39,128,156,16,19,-.1875,.9688,.278],[40,32,0,20,37,-.1979,1.0208,.389],[41,52,0,20,37,-.2187,1.0208,.389],[42,337,129,22,24,-.2083,1.0208,.5],[43,260,99,30,29,-.2187,.8542,.778],[44,144,156,16,19,-.1875,.375,.278],[45,324,156,20,15,-.2604,.5313,.333],[46,270,156,16,16,-.1875,.375,.278],[47,72,0,23,37,-.2187,1.0208,.5],[48,0,38,24,31,-.2396,.9583,.5],[49,391,38,22,30,-.1979,.9583,.5],[50,413,38,23,30,-.2292,.9479,.5],[51,24,38,23,31,-.2292,.9583,.5],[52,290,99,24,29,-.25,.9271,.5],[53,436,38,24,30,-.2396,.9271,.5],[54,460,38,23,30,-.2292,.9479,.5],[55,483,38,23,30,-.2292,.9271,.5],[56,47,38,23,31,-.2292,.9583,.5],[57,0,69,23,30,-.2292,.9479,.5],[58,359,129,16,24,-.1875,.7188,.278],[59,439,99,16,27,-.1875,.7188,.278],[60,455,99,28,27,-.1979,.8125,.778],[61,160,156,30,19,-.2187,.6458,.778],[62,483,99,28,27,-.1979,.8125,.778],[63,23,69,22,30,-.2187,.9792,.472],[64,70,38,27,31,-.2187,.9792,.667],[65,45,69,28,30,-.25,.9688,.667],[66,73,69,26,30,-.1875,.9688,.667],[67,97,38,26,31,-.2187,.9792,.639],[68,99,69,27,30,-.1875,.9688,.722],[69,126,69,25,30,-.1875,.9688,.597],[70,151,69,24,30,-.1875,.9688,.569],[71,123,38,27,31,-.2187,.9792,.667],[72,175,69,26,30,-.1875,.9688,.708],[73,201,69,16,30,-.1875,.9688,.278],[74,150,38,22,31,-.2292,.9688,.472],[75,217,69,27,30,-.1875,.9688,.694],[76,244,69,23,30,-.1875,.9688,.542],[77,267,69,30,30,-.1875,.9688,.875],[78,297,69,26,30,-.1875,.9688,.708],[79,172,38,29,31,-.2187,.9896,.736],[80,323,69,25,30,-.1875,.9688,.639],[81,339,0,29,34,-.2187,.9896,.736],[82,348,69,26,30,-.1875,.9688,.646],[83,201,38,24,31,-.2292,.9896,.556],[84,374,69,28,30,-.2396,.9688,.681],[85,225,38,26,31,-.1875,.9688,.688],[86,402,69,29,30,-.2604,.9688,.667],[87,431,69,36,30,-.2604,.9688,.944],[88,467,69,29,30,-.2604,.9688,.667],[89,0,99,29,30,-.2708,.9688,.667],[90,29,99,26,30,-.2187,.9688,.611],[91,95,0,18,37,-.1771,1.0208,.289],[92,113,0,23,37,-.2187,1.0208,.5],[93,136,0,18,37,-.25,1.0208,.289],[94,190,156,22,18,-.1979,.9688,.5],[95,344,156,25,15,-.2708,.2396,.5],[97,136,129,23,25,-.2396,.7396,.481],[98,251,38,23,31,-.1979,.9688,.517],[99,159,129,23,25,-.2396,.7396,.444],[100,55,99,23,30,-.2396,.9688,.517],[101,182,129,23,25,-.25,.7396,.444],[102,78,99,21,30,-.25,.9792,.306],[103,314,99,25,29,-.25,.7292,.5],[104,99,99,22,30,-.1979,.9688,.517],[105,121,99,16,30,-.2083,.9583,.239],[106,269,0,20,35,-.3333,.9583,.267],[107,137,99,23,30,-.1979,.9688,.489],[108,160,99,16,30,-.1979,.9688,.239],[109,375,129,29,24,-.1979,.7292,.794],[110,404,129,22,24,-.1979,.7292,.517],[111,205,129,24,25,-.25,.7396,.5],[112,339,99,23,29,-.1979,.7292,.517],[113,362,99,23,29,-.2396,.7292,.517],[114,426,129,20,24,-.1979,.7292,.342],[115,229,129,22,25,-.25,.7396,.383],[116,0,129,21,27,-.2604,.8438,.361],[117,446,129,22,24,-.1979,.7188,.517],[118,468,129,24,24,-.2604,.7188,.461],[119,0,156,29,24,-.2604,.7188,.683],[120,29,156,25,24,-.2708,.7188,.461],[121,385,99,24,29,-.2604,.7188,.461],[122,54,156,23,24,-.25,.7188,.435],[123,154,0,22,37,-.2083,1.0208,.5],[124,176,0,15,37,-.1562,1.0208,.278],[125,191,0,22,37,-.2083,1.0208,.5],[126,231,156,22,17,-.1979,.6042,.5],[176,212,156,19,18,-.1979,.9688,.383],[8211,369,156,25,15,-.2708,.5833,.5],[8212,394,156,37,15,-.2708,.5833,1],[8722,431,156,28,14,-.1875,.5417,.778],[215,251,129,25,25,-.125,.7708,.778],[177,409,99,30,29,-.2187,.9375,.778],[8901,253,156,17,17,-.1979,.5938,.278],[8230,286,156,38,16,-.1979,.3958,1.172],[956,274,38,26,31,-.2083,.75,.5562],[8242,21,129,19,26,-.25,.8333,.275],[8804,422,0,28,32,-.1979,.9167,.778],[8805,450,0,28,32,-.1979,.9167,.778],[8594,40,129,35,26,-.2187,.7917,1],[8592,75,129,35,26,-.2187,.7917,1],[8593,289,0,25,35,-.2604,.9792,.5],[8595,314,0,25,35,-.2604,.9688,.5],[916,176,99,31,30,-.2292,.9688,.833],[931,207,99,28,30,-.2187,.9688,.722],[963,110,129,26,26,-.2292,.7708,.5562],[961,300,38,25,31,-.2292,.7604,.5562],[955,325,38,25,31,-.25,1.0104,.5],[945,276,129,26,25,-.2292,.7604,.6108],[946,213,0,25,37,-.2187,1.0104,.5562],[952,350,38,25,31,-.2292,.9792,.5562],[967,235,99,25,30,-.2604,.7604,.5],[8730,0,0,32,38,-.2083,1.0729,.833],[8734,302,129,35,25,-.2187,.7188,1],[8776,77,156,30,24,-.2187,.7604,.778]],kerning:[],license:{name:`Gridlight Label Atlas`,copyright:`Copyright (c) 2009-2010 Design Science, Inc.
Copyright (c) 2014-2018 Khan Academy`,reservedFontNames:[`KaTeX_SansSerif`,`KaTeX_Main`],source:[{file:`KaTeX_SansSerif-Regular.ttf`,version:`Version 1.1; ttfautohint (v1.8.3)`,sha256:`f36ea897e19f4a2e571d1e900e4e3710e438deb05a842486045ba0a3e616a4ad`},{file:`KaTeX_Main-Regular.ttf`,version:`Version 1.1; ttfautohint (v1.8.3)`,sha256:`d0332f52868370fd83ae7fa46470f90c8f2eab2fcf12bc4f88080b340c95a830`}],package:`katex@0.16.47`,rasterizer:`Chromium 151.0.7922.34 (playwright)`,text:`SIL OPEN FONT LICENSE

Version 1.1 - 26 February 2007

PREAMBLE

The goals of the Open Font License (OFL) are to stimulate worldwide development of collaborative font projects, to support the font creation efforts of academic and linguistic communities, and to provide a free and open framework in which fonts may be shared and improved in partnership with others.

The OFL allows the licensed fonts to be used, studied, modified and redistributed freely as long as they are not sold by themselves. The fonts, including any derivative works, can be bundled, embedded, redistributed and/or sold with any software provided that any reserved names are not used by derivative works. The fonts and derivatives, however, cannot be released under any other type of license. The requirement for fonts to remain under this license does not apply to any document created using the fonts or their derivatives.

DEFINITIONS

"Font Software" refers to the set of files released by the Copyright Holder(s) under this license and clearly marked as such. This may include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the copyright statement(s).

"Original Version" refers to the collection of Font Software components as distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting, or substituting — in part or in whole — any of the components of the Original Version, by changing formats or by porting the Font Software to a new environment.

"Author" refers to any designer, engineer, programmer, technical writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS

Permission is hereby granted, free of charge, to any person obtaining a copy of the Font Software, to use, study, copy, merge, embed, modify, redistribute, and sell modified and unmodified copies of the Font Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components, in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled, redistributed and/or sold with any software, provided that each copy contains the above copyright notice and this license. These can be included either as stand-alone text files, human-readable headers or in the appropriate machine-readable metadata fields within text or binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font Name(s) unless explicit written permission is granted by the corresponding Copyright Holder. This restriction only applies to the primary font name as presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font Software shall not be used to promote, endorse or advertise any Modified Version, except to acknowledge the contribution(s) of the Copyright Holder(s) and the Author(s) or with their explicit written permission.

5) The Font Software, modified or unmodified, in part or in whole, must be distributed entirely under this license, and must not be distributed under any other license. The requirement for fonts to remain under this license does not apply to any document created using the Font Software.

TERMINATION

This license becomes null and void if any of the above conditions are not met.

DISCLAIMER

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.`}},j={tick:{size:9.5,tone:`muted`,weight:.12,priority:0},label:{size:10,tone:`ink`,weight:.15,priority:1},title:{size:10.5,tone:`ink`,weight:.4,priority:3},value:{size:10,tone:`ink`,weight:.35,priority:2}};function de(e,t){return e/Math.max(t,1)*Math.min(1.45,Math.max(.85,(t/265)**.45))}var M=new Map;for(let e of A.glyphs)M.set(e[0],{atlasX:e[1],atlasY:e[2],atlasWidth:e[3],atlasHeight:e[4],quadLeft:e[5],quadTop:e[6],advance:e[7]});var N=new Map;for(let[e,t,n]of A.kerning)N.set(e*65536+t,n);var fe={"·":`⋅`,µ:`μ`,"`":`′`,"²":`^2`,"³":`^3`,"≠":`!=`," ":` `,"\xA0":` `},pe=`?`.codePointAt(0);function me(e){let t=[];for(let n of e){let e=fe[n];for(let r of e??n){let e=r.codePointAt(0);t.push(M.has(e)?e:pe)}}return t}function he(e){let t=me(e),n=0;return t.forEach((e,r)=>{n+=M.get(e).advance,r+1<t.length&&(n+=N.get(e*65536+t[r+1])??0)}),n}function ge(e=`label`){let t=j[e].size;return _e(t)+.4*t}function _e(e){return Math.round(A.capHeight*e*1.25)}function P(e,t=`label`){return he(e)*j[t].size}var F=0,I=1,L=class{values=new Float32Array(1280);count=0;push(e,t,n,r,i,a,o){if((this.count+1)*20>this.values.length){let e=new Float32Array(this.values.length*2);e.set(this.values),this.values=e}let s=this.count*20;this.values.set([e[0],e[1],0,0,...t,...n,r,i,a,0,...o],s),this.count++}};function R(e,t,n){return n===`canvas`?[t[0]*e.width,t[1]*e.height]:e.toPixel(t[0],t[1])}function ve(e){switch(e){case`left`:return[0,.5];case`right`:return[1,.5];case`top`:return[.5,0];case`bottom`:return[.5,1];case`top-left`:return[0,0];case`top-right`:return[1,0];case`bottom-left`:return[0,1];case`bottom-right`:return[1,1];default:return[.5,.5]}}function z(e,t,n){let r=Math.cos(n),i=Math.sin(n);return[e*r+t*i,-e*i+t*r]}function ye(e,t,n){return e.left<t.right+n&&t.left<e.right+n&&e.top<t.bottom+n&&t.top<e.bottom+n}function B(e,t,n,r,i,a){if(!e.text&&e.swatch===void 0)return null;let o=j[e.role??`label`],s=(e.tone??o.tone)===`muted`?i.muted:i.ink,c=o.size*r,l=me(e.text),u=A.capHeight*c,d=.22*c,f=e.swatch===void 0?0:_e(c),p=e.swatch===void 0||!e.text?0:.4*c,m=he(e.text)*c,h=f+p+m,g=Math.max(u+d,f),_=(e.angle??0)*Math.PI/180,[ee,te]=ve(e.anchor??`center`),v=-ee*h,y=-te*g,ne=y+(g-u-d)/2+u,[b,x]=R(n,e.at,e.space);if(!Number.isFinite(b)||!Number.isFinite(x))return null;let re=e.offset??[0,0],S=[b+re[0]*r,x-re[1]*r],C=[z(v,y,_),z(v+h,y,_),z(v,y+g,_),z(v+h,y+g,_)],w=()=>({left:S[0]+Math.min(...C.map(e=>e[0])),right:S[0]+Math.max(...C.map(e=>e[0])),top:S[1]+Math.min(...C.map(e=>e[1])),bottom:S[1]+Math.max(...C.map(e=>e[1]))}),T=1.5*r,E=w(),ie=E.left<T?T-E.left:E.right>n.width-T?n.width-T-E.right:0,D=E.top<T?T-E.top:E.bottom>n.height-T?n.height-T-E.bottom:0;if(Math.abs(ie)>E.right-E.left||Math.abs(D)>E.bottom-E.top||(S=[Math.round(S[0]+ie),Math.round(S[1]+D)],E=w(),E.right-E.left>n.width||E.bottom-E.top>n.height))return null;let ae=[s[0],s[1],s[2],a],oe=o.weight;return{label:e,index:t,priority:e.priority??o.priority,pivot:S,box:E,emit:t=>{let n=v;if(e.swatch!==void 0){let r=typeof e.swatch==`number`?i.categories[(e.swatch%i.categories.length+i.categories.length)%i.categories.length]:e.swatch,o=Math.round(y+(g-f)/2);t.push(S,[Math.round(n),o,f,f],[0,0,0,0],_,I,0,[r[0],r[1],r[2],a]),n+=f+p}let r=Math.round(ne),o=c/A.em;l.forEach((e,i)=>{let a=M.get(e);e!==32&&t.push(S,[n+a.quadLeft*c,r-a.quadTop*c,a.atlasWidth*o,a.atlasHeight*o],[a.atlasX,a.atlasY,a.atlasWidth,a.atlasHeight],_,F,oe,ae),n+=a.advance*c,i+1<l.length&&(n+=(N.get(e*65536+l[i+1])??0)*c)})}}}function V(e,t,n,r,i,a){let o=(e,t)=>t?[e[0]+t[0]*n,e[1]-t[1]*n]:e,s=o(R(t,e.from,e.space),e.fromOffset),c=o(R(t,e.to,e.space),e.toOffset);if(![...s,...c].every(Number.isFinite))return;let l=e.tone??`muted`,u=l===`ink`?r.ink:l===`background`?r.background:r.muted,d=Math.max(1,Math.round((e.width??1)*n)),f=c[0]-s[0],p=c[1]-s[1],m=Math.hypot(f,p);if(m<.5)return;let h=e=>d%2==1?Math.floor(e)+.5:Math.round(e),g=Math.abs(p)<1e-6,_=[Math.abs(f)<1e-6?h(s[0]):s[0],g?h(s[1]):s[1]],ee=Math.atan2(-p,f),te=[u[0],u[1],u[2],i*(e.opacity??1)],v=e.dash?e.dash*n:0,y=v>0?v*2:m;for(let e=0;e<m;e+=y){let t=Math.min(v>0?v:m,m-e);a.push(_,[e,-d/2,t,d],[0,0,0,0],ee,I,0,te)}}function be(e,t,n,r,i=1){let a=new L;for(let o of e.rules??[])V(o,t,n,r,i,a);let o=[],s=[];(e.labels??[]).forEach((e,a)=>{let c=B(e,a,t,n,r,i);c?o.push(c):s.push(e.text)});let c=2*n,l=new Map;for(let e of o){let t=e.label.group;if(!t)continue;let n=l.get(t)??[];n.push(e),l.set(t,n)}let u=new Set;for(let e of l.values())for(let t=1;t<=e.length;t++){let n=e.filter((e,n)=>n%t===0);if(n.every((e,t)=>n.slice(t+1).every(t=>!ye(e.box,t.box,c)))){for(let t of e)n.includes(t)||u.add(t);break}}let d=o.filter(e=>!u.has(e)||(s.push(e.label.text),!1)).sort((e,t)=>t.priority-e.priority||e.index-t.index),f=[],p=[];for(let e of d)f.some(t=>ye(t,e.box,c))?s.push(e.label.text):(f.push(e.box),p.push(e.label.text),e.emit(a));let m=a.count,h=new Float32Array(m*2*20);h.set(a.values.subarray(0,m*20),0),h.set(a.values.subarray(0,m*20),m*20);for(let e=0;e<m*2;e++)h[e*20+15]=e<m?1:2;return{instances:h,instanceCount:m*2,placed:p,dropped:s}}var xe=1.4,H=.82,U=.72,Se=12,W=`
struct TextUniforms {
  // Canvas size in pixels.
  canvas: vec2f,
  // Atlas size in texels.
  atlas: vec2f,
  // Distance field reach in texels, halo width in pixels, halo strength, layer opacity.
  spread: f32,
  halo: f32,
  haloStrength: f32,
  alpha: f32,
  background: vec4f
};

struct TextInstance {
  // Pivot in canvas pixels (y down).
  pivot: vec4f,
  // Quad in the label's frame relative to the pivot: x, y (y down), width, height in pixels.
  box: vec4f,
  // Glyph cell in the atlas: x, y, width, height in texels.
  cell: vec4f,
  // Angle (radians, counter-clockwise on screen), kind (0 glyph, 1 rectangle), weight (pixels),
  // pass (1 halo, 2 ink).
  style: vec4f,
  color: vec4f
};

// DemoLayer always binds its look buffer at binding 0. Text never reads it (it has TextUniforms);
// the declaration only keeps the binding in the shader layout.
@group(0) @binding(0) var<uniform> look: vec4f;
@group(0) @binding(1) var<storage, read> instances: array<TextInstance>;
@group(0) @binding(2) var<storage, read> atlas: array<u32>;
@group(0) @binding(3) var<uniform> textUniforms: TextUniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) instance: u32
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let item = instances[instanceIndex];
  // Rectangles grow by the halo; glyph cells already hold the field's padding.
  let margin = select(0.0, textUniforms.halo + 1.0, item.style.y > 0.5);
  let local = corners[vertexIndex] * (item.box.zw + 2.0 * margin) - margin;
  let frame = item.box.xy + local;
  let cosine = cos(item.style.x);
  let sine = sin(item.style.x);
  let pixel = item.pivot.xy + vec2f(frame.x * cosine + frame.y * sine, -frame.x * sine + frame.y * cosine);
  var output: VertexOutput;
  output.position = vec4f(pixel.x / textUniforms.canvas.x * 2.0 - 1.0, 1.0 - pixel.y / textUniforms.canvas.y * 2.0, 0.0, 1.0);
  output.local = local;
  output.instance = instanceIndex;
  return output;
}

fn readTexel(x: i32, y: i32) -> f32 {
  let index = u32(y) * u32(textUniforms.atlas.x) + u32(x);
  let word = atlas[index >> 2u];
  return f32((word >> ((index & 3u) * 8u)) & 255u) / 255.0;
}

/** Bilinear sample of the field inside one glyph cell; texel centers are at half texels. */
fn sampleField(cell: vec4f, texel: vec2f) -> f32 {
  let position = clamp(texel - 0.5, vec2f(0.0), cell.zw - 1.0);
  let base = floor(position);
  let fraction = position - base;
  let x0 = i32(cell.x + base.x);
  let y0 = i32(cell.y + base.y);
  let x1 = min(x0 + 1, i32(cell.x + cell.z) - 1);
  let y1 = min(y0 + 1, i32(cell.y + cell.w) - 1);
  let top = mix(readTexel(x0, y0), readTexel(x1, y0), fraction.x);
  let bottom = mix(readTexel(x0, y1), readTexel(x1, y1), fraction.x);
  return mix(top, bottom, fraction.y);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let item = instances[input.instance];
  var distance: f32;
  var halo = textUniforms.halo;
  if (item.style.y < 0.5) {
    let pixelsPerTexel = item.box.z / max(item.cell.z, 1.0);
    var field: f32;
    if (pixelsPerTexel < 0.75) {
      // Small text (device pixel ratio 1): one pixel covers more than a texel. Four taps over the
      // pixel average the field, so thin stems do not alias.
      field = 0.25 * (
        sampleField(item.cell, (input.local + vec2f(-0.25, -0.25)) / pixelsPerTexel) +
        sampleField(item.cell, (input.local + vec2f(0.25, -0.25)) / pixelsPerTexel) +
        sampleField(item.cell, (input.local + vec2f(-0.25, 0.25)) / pixelsPerTexel) +
        sampleField(item.cell, (input.local + vec2f(0.25, 0.25)) / pixelsPerTexel)
      );
    } else {
      field = sampleField(item.cell, input.local / pixelsPerTexel);
    }
    // Signed distance to the outline in screen pixels, positive inside, thickened by the weight.
    distance = (field - 0.5) * 2.0 * textUniforms.spread * pixelsPerTexel + item.style.z;
    // The field reaches only so far: keep the halo inside it.
    halo = min(halo, textUniforms.spread * pixelsPerTexel - 1.0);
  } else {
    let size = item.box.zw;
    let inside = min(min(input.local.x, size.x - input.local.x), min(input.local.y, size.y - input.local.y));
    distance = inside + 0.5 * min(min(size.x, size.y), 1.0) - 0.5;
  }
  // Halo pass (style.w 1): the background color around the outline. Ink pass (2): the glyph.
  if (item.style.w < 1.5) {
    let haloAlpha = clamp(distance + halo + 0.5, 0.0, 1.0) * textUniforms.haloStrength * item.color.a;
    if (haloAlpha < 0.002) {
      discard;
    }
    return vec4f(textUniforms.background.rgb, haloAlpha * textUniforms.alpha);
  }
  let inkAlpha = clamp(distance + 0.5, 0.0, 1.0) * item.color.a;
  if (inkAlpha < 0.002) {
    discard;
  }
  return vec4f(item.color.rgb, inkAlpha * textUniforms.alpha);
}
`,G=new WeakMap;function Ce(e,t){let n=G.get(e);if(n)return n.users++,n.buffer;let r=y(e,t);return G.set(e,{buffer:r,users:1}),r}function K(e){let t=G.get(e);t&&(t.users--,t.users<=0&&(t.buffer.destroy(),G.delete(e)))}function we(e,t,n){return[e[0]+(t[0]-e[0])*n,e[1]+(t[1]-e[1])*n,e[2]+(t[2]-e[2])*n]}function Te(e){return{ink:e.ink,muted:we(e.ink,e.background,.33),background:[e.background[0],e.background[1],e.background[2]],categories:e.categories}}var Ee=class extends d{instanceBuffer;uniformBuffer;uniformValues=new Float32Array(Se);placeholder;capacity;text={};options={};layoutKey=``;layout=null;hasAtlas=!1;isDestroyed=!1;constructor(t,n){let r=i(t,5120),a=t.createBuffer({byteLength:48,usage:e.UNIFORM|e.COPY_DST}),o=le(),s=i(t,1);super(t,`run-text`,W,{instances:r,atlas:o?Ce(t,o):s,textUniforms:a},0),this.instanceBuffer=r,this.uniformBuffer=a,this.placeholder=s,this.capacity=256,this.hasAtlas=!!o,o||ue().then(e=>{this.isDestroyed||(this.model.setBindings({atlas:Ce(t,e)}),this.hasAtlas=!0,n?.())})}setText(e,t={}){this.text=e,this.options=t}getLayout(){return this.layout}update(e,t,n={}){let r=this.options.displayWidth??e.width/(globalThis.devicePixelRatio||1),a=de(e.width,r),o=(this.options.alpha??1)*(this.options.context?U:1);this.uniformValues.set([e.width,e.height,A.width,A.height,A.spread,xe*a,H,1,...t.background],0),this.uniformBuffer.write(this.uniformValues);let s=JSON.stringify([this.text,e.width,e.height,e.worldWidth,e.worldHeight,e.origin,e.longitudeScale,r,o,t.ink,t.background,t.categories]);if(s===this.layoutKey)return;this.layoutKey=s;let c=be(this.text,{width:e.width,height:e.height,toPixel:(t,n)=>ee(e,t,n)},a,Te(t),o);this.layout=c,c.instanceCount>this.capacity&&(this.instanceBuffer.destroy(),this.capacity=Math.max(c.instanceCount,this.capacity*2),this.instanceBuffer=i(this.device,this.capacity*20),this.model.setBindings({instances:this.instanceBuffer})),c.instanceCount>0&&this.instanceBuffer.write(c.instances.subarray(0,c.instanceCount*20)),this.model.setInstanceCount(c.instanceCount)}draw(e){this.hasAtlas&&this.layout&&this.layout.instanceCount!==0&&super.draw(e)}destroy(){this.isDestroyed=!0,super.destroy(),this.instanceBuffer.destroy(),this.uniformBuffer.destroy(),this.placeholder.destroy(),this.hasAtlas&&K(this.device)}},De={float32:{components:1,create:e=>new Float32Array(e)},uint32:{components:1,create:e=>new Uint32Array(e)},sint32:{components:1,create:e=>new Int32Array(e)},float32x2:{components:2,create:e=>new Float32Array(e)},float32x3:{components:3,create:e=>new Float32Array(e)},float32x4:{components:4,create:e=>new Float32Array(e)},uint32x2:{components:2,create:e=>new Uint32Array(e)},uint32x3:{components:3,create:e=>new Uint32Array(e)},uint32x4:{components:4,create:e=>new Uint32Array(e)}},Oe=class{buffers=[];resources=[];consumableResourceCount=0;createBuilder(e,t){let n=(e,n,r,i)=>(this.buffers.push(r),{id:e,format:n,length:i,buffer:r,view:m(t,e,r,n,i),write:e=>r.write(e)});return{device:e,graph:t,upload:(t,r,i)=>n(t,r,y(e,i),i.length/De[r].components),allocate:(t,r,a)=>n(t,r,i(e,a*De[r].components),a),own:(e,t)=>(this.resources.push(e),t?.reusable||this.consumableResourceCount++,e)}}destroy(){for(let e of this.resources)e.destroy();this.resources.length=0,this.consumableResourceCount=0;for(let e of this.buffers)e.destroy();this.buffers.length=0}},ke={sequential:0,diverging:1,categorical:2,cyclic:3,grey:0},Ae=[.04,.2,.4,.6,.8,.98].map(e=>[e,e,e]),je=[.07,.14,.95,.88],Me=0x56bc75e2d63100000;function q(e){return e.format===`uint32`?`uint32`:`float32`}function Ne(e,t){let n=t===`uint32`?new Uint32Array(e.buffer,e.byteOffset,e.byteLength>>2):new Float32Array(e.buffer,e.byteOffset,e.byteLength>>2),r=1/0,i=-1/0;for(let e of n)!Number.isFinite(e)||Math.abs(e)>=Me||(e<r&&(r=e),e>i&&(i=e));return r<=i?[r,i]:null}var Pe=45,Fe=140;function Ie(a){let d=a.parameters??[];return Object.assign(async f=>{let{device:m}=f;c(m);let b=o(f.theme),x=n(d,f.parameters),S=f.width/f.height,C={width:S,height:1},w={worldWidth:C.width,worldHeight:C.height,width:f.width,height:f.height};if(a.geographicWindow){let[e,t,n,r]=a.geographicWindow;w.origin=[e,t],w.worldWidth=n-e,w.worldHeight=r-t,w.longitudeScale=Math.cos((r+t)*Math.PI/360)}let E=a.createInputs({seed:f.seed,random:r(f.seed),world:C,aspect:S}),D=async()=>{let e=new Oe,t;try{return{runner:await p.create(m,`run-operation`,n=>{t=a.build(e.createBuilder(m,n),E,x)}),buffers:e,result:t}}catch(t){throw e.destroy(),t}},ae=new Set(a.rebuildOn??[]),oe=()=>d.filter(e=>!O.result.update||ae.has(e.id)).map(e=>x[e.id]).join(`,`),O=await D(),ce=oe(),k=new Map,le=e=>{e.runner.destroy(),e.buffers.destroy()},A=(e,t)=>{let n=a.buildCache??0;if(n<=0||k.has(e)||t.buffers.consumableResourceCount>0)le(t);else for(k.set(e,t);k.size>n;){let[e,t]=k.entries().next().value;k.delete(e),le(t)}},j=new Map,de=()=>{for(let e of j.values()){e.layer.destroy();for(let t of e.owned)t.destroy()}j.clear()},M=new WeakMap,N=new Map,fe=null,pe=()=>fe??=i(m,4),me=[],he=!1,ge=0,_e=0,P=!1,F=!0,I=!1,L=3,R=null,ve=!1,z=0,ye=[],B=a.interaction,V={rest:[.5,.5],target:null,effective:[.5,.5],isGliding:!1,lastTime:0,result:{},pointer:null},be=null,xe={},H,U=null,Se=0,W=!0,G=0,Ce=new Set,K=new Map,we=0,Te=null,De=!1,Me=!1,Ie=!1,Re=e=>{let t=e.buffer,n=K.get(t);return n||(n={format:e.format,data:null},K.set(t,n),Ie=!0),n.data},ze=()=>{if(Ie=!1,K.size===0)return Promise.resolve();if(De)return Me=!0,Te??Promise.resolve();De=!0;let e=we,t=(async()=>{let t=[...K],n=await Promise.all(t.map(async([e,t])=>{try{let n=await e.readAsync();if(P)return null;let r=n.buffer.slice(n.byteOffset,n.byteOffset+n.byteLength);return t.format.startsWith(`uint32`)?new Uint32Array(r):t.format.startsWith(`sint32`)?new Int32Array(r):new Float32Array(r)}catch{return null}}));P||e!==we||(t.forEach(([,e],t)=>{e.data=n[t]}),Y({isPeekRefresh:!0,applyValues:W}),L=Math.max(L,2),$.request())})().catch(()=>void 0).finally(()=>{De=!1,Te===t&&(Te=null),Me&&!P&&(Me=!1,ze())});return Te=t,t},Be=()=>({inputs:E,outputs:O.result.outputs,values:x,world:C,pixelSize:g(w).pixelSize,toUnit:e=>{let[t,n]=ee(w,e[0],e[1]);return[t/w.width,n/w.height]},peek:Re,build:z}),J=new te(m),Ve=!1,He=e=>{let t=e.spotlight===void 0?B?.kind===`probe`?`auto`:null:e.spotlight;J.set(e.glyphs??[],w,t),f.reducedMotion&&J.setSpotlightOn(Ve)},Ue=e=>{Object.assign(x,e);let t=!!O.result.update;Object.keys(e).some(e=>!t||ae.has(e))&&d.length>0&&(I=!0),F=!0,$.request()},We=e=>{if(!e)return null;let t={},n=!1;for(let[r,i]of Object.entries(e))Number.isFinite(i)&&x[r]!==i&&(t[r]=i,n=!0);return n?t:null},Ge=()=>{clearTimeout(H);let e=We(U??void 0);U=null,e&&(Ue(e),f.reportParameters?.(e))},Y=(e={})=>{if(!B||P||(I||R)&&!e.isAfterRebuild)return;let[t,n]=V.effective,[r,i]=s(w,[t*w.width,n*w.height]),a={x:r,y:i,unitX:t,unitY:n},o=B.resolve(a,Be())??{};V.pointer=a,V.result=o;for(let e of Object.keys(o.values??{}))Ce.add(e);for(let e of Object.keys(o.settle??{}))Ce.add(e);if(e.applyValues!==!1){let t=We(o.values);if(t&&e.isPeekRefresh&&++G>3&&(t=null),t){let e=I;Ue(t),I&&!e&&Se++,f.reportParameters?.(t)}clearTimeout(H),We(o.settle)?(U={...o.settle},H=setTimeout(Ge,Fe)):U=null}o.rerun&&!e.isPeekRefresh&&(F=!0);let c=e.applyValues===!1&&Object.entries({...o.values,...o.settle}).some(([e,t])=>Number.isFinite(t)&&x[e]!==t);He(c?{}:o);let l=c?null:o.readout??null;l!==be&&(be=l,f.reportReadout?.(be)),L=Math.max(L,2),Ie&&!F&&ze()},Ke=(e=!1)=>{if(!B)return;J.setSpotlightOn(Ve),V.isGliding=!1;let t=V.target;t&&(t[0]!==V.effective[0]||t[1]!==V.effective[1])&&(V.effective=[t[0],t[1]],Y()),e&&U&&Ge()},qe=e=>{let t=Math.min(100,Math.max(0,e-V.lastTime));V.lastTime=e;let n=V.target;if(!B||(J.fadeSpotlight(Ve,t)&&(L=Math.max(L,1)),!n))return;let[r,i]=V.effective,a=n[0]-r,o=n[1]-i;if(a!==0||o!==0){if(!V.isGliding||Math.hypot(a,o)<.004)V.isGliding=!1,V.effective=[n[0],n[1]];else{let e=1-Math.exp(-t/Pe);V.effective=[r+a*e,i+o*e]}Y()}},Je=(e,t)=>{if(e.range&&e.range!==`auto`)return[e.range[0],e.range[1]];if(!t)return[0,1];N.set(t.buffer,q(t));let n=M.get(t.buffer)??[0,1];if(e.ramp===`diverging`){let e=Math.max(Math.abs(n[0]),Math.abs(n[1]),1e-9);return[-e,e]}return n},X=(e,t,n=t)=>{let r=!!e.context,i=e,a=t?q(t):`float32`,o=i.ramp??(a===`uint32`?`categorical`:`sequential`),s=i.color??[...r?b.muted:b.ink,r?.55:.85],c=[0,1,2,3,4,5].map(e=>{let t=e/5;return[b.background[0]+(b.muted[0]-b.background[0])*t,b.background[1]+(b.muted[1]-b.background[1])*t,b.background[2]+(b.muted[2]-b.background[2])*t]}),l=!!t;return{mode:l&&!i.color?ke[o]:-1,range:Je(i,n),alpha:e.alpha??(r?.9:1),solid:s,ramp:r&&l?c:o===`diverging`?b.diverging:o===`grey`?Ae:void 0,categories:r&&l?[b.muted]:i.categories,logarithmic:i.logarithmic,bands:i.bands,fade:i.fade}},Ye=(e,t)=>{if(e&&typeof e!=`number`)return{buffer:e.buffer,owned:[]};let n=y(m,Uint32Array.of(e??t));return{buffer:n,owned:[n]}},Xe=(e,t)=>{let n=Le(e),r=[e.layer,...n],i=j.get(t),a;if(i&&i.signature.length===r.length&&i.signature.every((e,t)=>e===r[t]))a=i.layer;else{if(i){i.layer.destroy();for(let e of i.owned)e.destroy()}let n=Ze(e);a=n.layer,j.set(t,{signature:r,layer:n.layer,owned:n.owned})}return{...Qe(e,a),clip:e.clip}},Ze=e=>{switch(e.layer){case`raster`:return{layer:new v(m,e.values.buffer,e.columns,e.rows,q(e.values)),owned:[]};case`dots`:return{layer:new _(m,e.positions.buffer,e.values?.buffer??pe(),e.count??e.positions.length,e.values?q(e.values):`float32`),owned:[]};case`segments`:return{layer:new ie(m,e.nodes.buffer,e.edges.buffer,e.values?.buffer??pe(),e.edges.length,e.values?q(e.values):`float32`),owned:[]};case`lines`:{let t=Ye(e.count,e.segments.length),n=e.segments.length;return{layer:new u(m,e.segments.buffer,t.buffer,n),owned:t.owned}}case`paths`:{let t=Ye(e.count,e.ids.length);return{layer:new l(m,e.nodes.buffer,e.ids.buffer,t.buffer,Math.max(e.ids.length-1,1)),owned:t.owned}}case`triangles`:{let t=e.triangles.format===`uint32x3`?e.triangles.length:Math.floor(e.triangles.length/3),n=Ye(e.count,t);return{layer:new ne(m,e.positions.buffer,e.triangles.buffer,n.buffer,t),owned:n.owned}}case`polygons`:return{layer:new T(m,e.positions.buffer,e.indices.buffer,e.owners.buffer,e.values.buffer,e.triangleCount,q(e.values)),owned:[]};case`chart`:{let t=e.values.length;return{layer:new se(m,e.kind,e.values.buffer,e.xValues?.buffer??pe(),t,q(e.values)),owned:[]}}case`text`:return{layer:new Ee(m,()=>{L=3,$.request()}),owned:[]}}},Qe=(e,t)=>{switch(e.layer){case`raster`:{let n=X(e,e.values);return{layer:t,look:{...n,mode:e.color?-1:n.mode,smoothing:e.smoothing,hillshade:e.hillshade,elevation:e.elevation}}}case`dots`:return{layer:t,look:{...X(e,e.values),radius:e.radius??2.4,hillshade:.5,sizeByValue:e.sizeByValue}};case`segments`:return{layer:t,look:{...X(e,e.values),radius:e.radius??1.4}};case`lines`:case`paths`:case`triangles`:return{layer:t,look:{...X(e,void 0),radius:e.radius??1.2,solid:e.color??[...e.context?b.muted:b.ink,e.context?.6:.9]}};case`polygons`:return{layer:t,look:X(e,e.values)};case`chart`:{let n=t,r=e.rectangle??je;return n.setPlot([r[0]*C.width,r[1]*C.height,r[2]*C.width,r[3]*C.height],e.xRange??[0,1]),{layer:t,look:{...X(e,e.kind===`bars`?e.values:void 0,e.values),radius:e.radius??(e.kind===`scatter`?3:1.6)}}}case`text`:return t.setText(e,{displayWidth:f.displayWidth,context:e.context,alpha:e.alpha}),{layer:t,look:{}}}},Z=null,$e=async()=>{if(P)return[];let t=[],n=0;for(let[e,r]of N){if(e.destroyed){N.delete(e);continue}let i=Math.floor(e.byteLength/4)*4;i!==0&&(t.push({buffer:e,format:r,byteOffset:n,byteLength:i}),n+=i)}if(t.length===0)return[];if(n>m.limits.maxBufferSize||t.some(t=>(t.buffer.usage&e.COPY_SRC)===0))return Promise.all(t.map(async e=>{try{let t=await e.buffer.readAsync();return P?[e.buffer,null]:[e.buffer,Ne(t,e.format)]}catch{return[e.buffer,null]}}));(!Z||Z.byteLength<n)&&(Z?.destroy(),Z=null,Z=new re(m,{id:`run-ranges`,byteLength:n,slotCount:1}));let r=Z.tryAcquire();if(!r)return[];let i=!1,a=!1;try{if(P)return[];let e=m.createCommandEncoder({id:`run-ranges-encoder`});try{for(let n of t)e.copyBufferToBuffer({sourceBuffer:n.buffer,destinationBuffer:r.buffer,destinationOffset:n.byteOffset,size:n.byteLength});let a=e.finish();r.markEncoded({byteOffset:0,byteLength:n}),i=!0,m.submit(a)}finally{e.destroy()}a=!0;let o=await r.read();return P?[]:t.map(e=>[e.buffer,Ne(o.subarray(e.byteOffset,e.byteOffset+e.byteLength),e.format)])}finally{if(i&&!a){let e=r.read();r.cancel(),await e.catch(()=>void 0)}else r.cancel()}},et=null,tt=!1,nt=!1,rt=()=>{if(et)return tt=!0,et;let e=(async()=>{try{do tt=!1,await it();while(tt&&!P)}finally{et=null}})();return et=e,e},it=async()=>{let e;try{e=await $e()}catch{return}if(P)return;let t=!1;for(let[n,r]of e){if(!r)continue;let e=M.get(n);e&&e[0]===r[0]&&e[1]===r[1]||(t=!0),M.set(n,r)}t&&(L=3,$.request())},at,ot=null,st=0,ct=null,lt=()=>typeof a.draw==`function`?a.draw({inputs:E,outputs:O.result.outputs,values:x,theme:b,world:C,extra:at,interaction:V.result,pointer:V.pointer}):a.draw,ut=()=>{me=[],ye=[],lt().forEach((e,t)=>{(!he||e.context)&&(ye.push(e.layer),me.push(Xe(e,t)))});for(let e of me)e.layer.update(w,b,e.look);let e=J.getLayers(w,b),t=m.beginRenderPass({clearColor:[...b.background]}),[n,r]=m.getDefaultCanvasContext().getDrawingBufferSize(),i=!1;for(let e of me){if(e.clip&&Number.isFinite(e.clip.x)){let a=Math.min(n,Math.max(0,Math.round(ee(w,e.clip.x,0)[0]/w.width*n))),o=e.clip.side===`left`?0:a,s=e.clip.side===`left`?a:n-a;if(s<=0)continue;t.setParameters({scissorRect:[o,0,s,r]}),i=!0}else i&&=(t.setParameters({scissorRect:[0,0,n,r]}),!1);e.layer.draw(t)}i&&t.setParameters({scissorRect:[0,0,n,r]});for(let n of e)n.draw(t);t.end(),m.submit(),nt&&(nt=!1,rt())},dt=()=>{O.result.update?.(x),O.runner.run(),_e++},ft=async e=>{let n=a.afterRun;if(!n)return;let r=++st,i=new Oe,o;try{let r=i.createBuilder(m,new t(m,{id:`run-after`}));o=await n({device:m,compiled:e.runner.compiled,inputs:E,outputs:e.result.outputs,values:{...x},upload:r.upload,allocate:r.allocate})}catch(e){if(i.destroy(),P||r!==st)return;throw e}P||r!==st?i.destroy():(ot?.destroy(),ot=i,at=o,nt=!0,L=3,$.request())},Q=null,pt=()=>{if(!a.afterRun||P)return Promise.resolve();if(ct)return Q=O,ct;let e=(async()=>{let e=O,t=null;try{for(;e&&!P;){Q=null;try{await ft(e)}catch(e){t={error:e}}if(P)return;e=Q&&Q===O?Q:null}}finally{Q=null,ct=null}if(t)throw t.error})();return ct=e.catch(()=>void 0),e},mt=()=>{R??=(async()=>{I=!1;let e=oe(),t=k.get(e);t&&k.delete(e);let n=t??await D();P?le(n):(de(),N.clear(),K.clear(),we++,z++,st++,Q=null,A(ce,O),O=n,ce=e,t||ge++,F=!0,ve=!0,B&&!I&&Y({applyValues:W,isAfterRebuild:!0}),L=3)})().catch(e=>{P||console.error(`Could not rebuild operation`,e)}).finally(()=>{R=null,P||(I?mt():$.request())})},ht=()=>{P||(I&&mt(),ve&&R||(F&&!R?(F=!1,ve=!1,dt(),L=3,ut(),rt(),ze(),pt().catch(e=>console.error(e))):ut()))},$=new h(ht);if(dt(),F=!1,await pt(),B){let e=typeof B.rest==`function`?B.rest(Be()):B.rest??[.5,.5];V.rest=[e[0],e[1]],V.effective=[e[0],e[1]];let t=d.every(e=>x[e.id]===e.default);W=t,Y({applyValues:t}),xe={...V.result.values,...V.result.settle},F&&(F=!1,dt(),await pt()),await ze(),F&&(F=!1,dt(),await pt())}return lt().some(e=>e.layer===`text`)&&await ue(),ut(),await rt(),{scrubParameter:d.find(e=>e.cardScrub)??d.find(e=>O.result.update&&!ae.has(e.id)&&!e.options),gesture:B?{kind:B.kind,hint:B.hint,rest:[V.rest[0],V.rest[1]]}:void 0,render(e){qe(e),(L>0||F||I)&&(L=Math.max(L-1,0),$.flush())},resize(e,t){w.width=e,w.height=t,L=3,B&&Y({applyValues:!1}),$.request()},setParameters(e){B&&(V.target=null,Ve=!1,W=!1,G=0,clearTimeout(H),U=null),Ue(e),B&&Y({applyValues:!1})},setPointer:B?e=>{e&&(!V.target&&!f.reducedMotion&&(V.isGliding=!0),Ve=!0,W=!0,G=0,V.target=[Math.min(1,Math.max(0,e[0]/w.width)),Math.min(1,Math.max(0,e[1]/w.height))],f.reducedMotion?Ke():L=Math.max(L,1),$.request())}:void 0,resetPointer:B?()=>{V.target=null,V.isGliding=!1,V.effective=[V.rest[0],V.rest[1]],Ve=!1,W=!0,G=0,J.setSpotlightOn(!1),Y(),$.request()}:void 0,destroy(){P=!0,$.cancel(),clearTimeout(H),de(),J.destroy(),fe&&fe.destroy(),ot?.destroy(),Z?.destroy(),le(O);for(let e of k.values())le(e);k.clear()},debug:{setInputsOnly(e){he=e,L=3,$.request()},async settle(){if(P)return;$.cancel();let e=()=>new Promise(e=>requestAnimationFrame(e));Ke(!0);let t=()=>!!(R||I||F||ct||Te||U);for(let n=0;n<400&&t();n++)if(U&&!R&&Ge(),ht(),await e(),P)return;t()&&console.warn(`Operation run did not settle within 400 animation frames; drawing the latest committed state.`),ut(),await rt(),P||(ut(),await e())},getRunCount:()=>_e,getRebuildCount:()=>ge,getLayerKinds:()=>ye,getReadout:()=>be,getRestValues:()=>xe,getPointerRebuildCount:()=>Se,getPointerParameterIds:()=>[...Ce]}}},{parameters:d,caption:a.caption})}function Le(e){switch(e.layer){case`raster`:return[e.values,e.columns,e.rows];case`dots`:return[e.positions,e.values,e.count];case`segments`:return[e.nodes,e.edges,e.values];case`lines`:return[e.segments,e.count];case`paths`:return[e.nodes,e.ids,e.count];case`triangles`:return[e.positions,e.triangles,e.count];case`polygons`:return[e.positions,e.indices,e.owners,e.values,e.triangleCount];case`chart`:return[e.kind,e.values,e.xValues];case`text`:return[]}}export{re as i,ge as n,P as r,Ie as t};