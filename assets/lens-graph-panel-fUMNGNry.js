import{r as e}from"./rolldown-runtime-XKyaHtl9.js";import{t}from"./react-DlmbjZJV.js";import{t as n}from"./gpu-canvas-rTkYWLOu.js";import{s as r}from"./index-DIfTuwM1.js";import{t as i}from"./jsx-runtime-LKYXHABd.js";import{Q as a}from"./dist-BusamGlx.js";import{t as o}from"./model-CeCyfMN-.js";var s=e(t(),1);function c(e){return e.some(e=>!e.bytesKind||e.bytesKind===`unspecified`)?`unspecified`:e.some(e=>e.bytesKind===`unavailable`)?`unavailable`:e.some(e=>e.bytesKind===`estimated`)?`estimated`:e.some(e=>e.bytesKind===`lower-bound`)?`lower-bound`:`logical`}function l(e,t){let n=d(e);switch(t){case`logical`:return n;case`lower-bound`:return`≥${e} B`;case`unavailable`:return`size unavailable`;case`estimated`:return`~${n}`;default:return`${n} (precision unknown)`}}function u(e){return e.inventoryKnown===!1?`inventory unavailable`:`${f(e.count)} buffers`}function d(e){return e<1024?`${e} B`:e<1048576?`${p(e/1024)} KB`:e<1024**3?`${p(e/1024**2)} MB`:`${p(e/1024**3)} GB`}function f(e){return e.toLocaleString(`en-US`)}function p(e){return e>=100?e.toFixed(0):e>=10?e.toFixed(1):e.toFixed(2)}function m(e,t={}){let{nodeWidth:n=168,nodeHeight:r=38,layerGap:i=74,rowGap:a=14,padding:o=20,targetWidth:s=1040}=t,c=new Map,l=new Map;for(let t of e.nodes)c.set(t.id,[]),l.set(t.id,[]);for(let t of e.edges)c.get(t.to)?.push(t.from),l.get(t.from)?.push(t.to);let u=new Map;for(let t of e.nodes){let e=0;for(let n of c.get(t.id)??[])e=Math.max(e,(u.get(n)??0)+1);u.set(t.id,e)}let d=Math.max(0,...u.values())+1,f=Array.from({length:d},()=>[]);for(let t of e.nodes)f[u.get(t.id)??0].push(t.id);let p=new Map,m=()=>f.forEach(e=>e.forEach((e,t)=>p.set(e,t)));m();let h=(e,t)=>{let n=t.get(e)??[];return n.length===0?p.get(e)??0:n.reduce((e,t)=>e+(p.get(t)??0),0)/n.length};for(let e=0;e<4;e++){for(let e=1;e<d;e++)f[e].sort((e,t)=>h(e,c)-h(t,c)),m();for(let e=d-2;e>=0;e--)f[e].sort((e,t)=>h(e,l)-h(t,l)),m()}let g=Math.max(1,Math.floor((s-2*o+i)/(n+i))),_=Math.ceil(d/g),v=Array.from({length:_},(e,t)=>Math.max(1,...f.slice(t*g,(t+1)*g).map(e=>e.length))),y=[],b=o;for(let e of v)y.push(b),b+=e*r+(e-1)*a+40;let x=[],S=new Map;f.forEach((e,t)=>{let s=Math.floor(t/g),c=t%g,l=s%2==0?c:g-1-c,u=(v[s]*r+(v[s]-1)*a-(e.length*r+(e.length-1)*a))/2;e.forEach((e,c)=>{let d={id:e,x:o+l*(n+i),y:y[s]+u+c*(r+a),width:n,height:r,layer:t};x.push(d),S.set(e,d)})});let C=[];for(let t of e.edges){let e=S.get(t.from),n=S.get(t.to);if(!e||!n)continue;let r=n.layer-e.layer,i=Math.floor(e.layer/g)===Math.floor(n.layer/g),a,o,s,c,l,u,d,f;if(i&&n.x>e.x){a=e.x+e.width,o=e.y+e.height/2,s=n.x,c=n.y+n.height/2;let t=Math.max(24,(s-a)*.5);[l,u,d,f]=[a+t,o,s-t,c]}else if(i){a=e.x,o=e.y+e.height/2,s=n.x+n.width,c=n.y+n.height/2;let t=Math.max(24,(a-s)*.5);[l,u,d,f]=[a-t,o,s+t,c]}else{a=e.x+e.width/2,o=e.y+e.height,s=n.x+n.width/2,c=n.y;let t=Math.max(20,(c-o)*.5);[l,u,d,f]=[a,o+t,s,c-t]}C.push({from:t.from,to:t.to,x0:a,y0:o,x1:s,y1:c,controlX0:l,controlY0:u,controlX1:d,controlY1:f,span:r})}let w=Math.min(d,g);return{width:o*2+w*n+(w-1)*i,height:b-40+o,nodes:x,edges:C}}function h(e){return`M${e.x0},${e.y0} C${e.controlX0},${e.controlY0} ${e.controlX1},${e.controlY1} ${e.x1},${e.y1}`}var g=new Map;function _(e){let t=g.get(e);return t||(t=r(e),g.set(e,t)),(0,s.use)(t)}var v=i(),y=188,b=30,x=17,S=26;function C(e,t){return t<=e.first?`unborn`:t<=e.last?`alive`:`dead`}function w(e,t){let n=0;for(let r of e)C(r,t)===`alive`&&(n+=r.bytes);return n}function T(e,t){let n=0;for(let r=1;r<=t;r++)n=Math.max(n,w(e,r));return n}function E(e,t){let n=e.startsWith(`${t}-`)?e.slice(t.length+1):e;return n.length>27?`${n.slice(0,26)}…`:n}function ee({model:e,step:t}){let n=e.lifetimes??[],r=e.nodes.length,i=y+r*b+12,a=S+n.length*x+10,o=e.title,s=y+t*b;return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(`div`,{className:`meta-scroll`,children:(0,v.jsxs)(`svg`,{className:`lifetime-chart`,viewBox:`0 0 ${i} ${a}`,style:{minWidth:Math.min(i,520),maxWidth:Math.max(720,i*1.3)},role:`img`,"aria-label":`Lifetimes of ${n.length} buffers over ${r} steps`,children:[e.nodes.map((e,r)=>(0,v.jsxs)(`g`,{children:[(0,v.jsx)(`title`,{children:`${r+1}. ${e.label}`}),(0,v.jsx)(`rect`,{className:`lifetime-column`,"data-now":r===t-1,x:y+r*b,y:22,width:b,height:n.length*x+6}),(0,v.jsx)(`text`,{className:`lifetime-tick`,x:y+r*b+b/2,y:16,textAnchor:`middle`,children:r+1})]},e.id)),n.map((e,n)=>{let r=C(e,t),i=e.bytesKind===void 0&&e.ownership===`imported`?`unknown`:e.ownership,a=S+n*x,s=y+e.first*b,c=(e.last-e.first+1)*b,u=Math.max(0,Math.min(t-e.first,e.last-e.first+1))*b;return(0,v.jsxs)(`g`,{className:`lifetime-row`,"data-state":r,children:[(0,v.jsx)(`title`,{children:`${e.id}: ${l(e.bytes,e.bytesKind)}, ownership ${i}, steps ${e.first+1} to ${e.last+1}`}),(0,v.jsx)(`text`,{className:`lifetime-name`,x:180,y:a+11,textAnchor:`end`,children:E(e.id,o)}),(0,v.jsx)(`rect`,{className:`lifetime-ghost`,x:s+2,y:a+2,width:c-4,height:11,rx:3}),(0,v.jsx)(`rect`,{className:`lifetime-bar`,"data-ownership":i,x:s+2,y:a+2,width:Math.max(0,u-4),height:11,rx:3})]},e.id)}),(0,v.jsx)(`line`,{className:`lifetime-playhead`,x1:s,x2:s,y1:20,y2:S+n.length*x+2})]})}),(0,v.jsx)(`p`,{className:`meta-scroll-hint`,children:`Scroll sideways to follow every step.`})]})}var D=[[.04,.05,.16],[.18,.12,.42],[.43,.15,.55],[.82,.27,.45],[.98,.55,.3],[.99,.9,.55]];function O(e){let t=Math.min(1,Math.max(0,e))*(D.length-1),n=Math.min(D.length-2,Math.floor(t)),r=t-n,i=D[n],a=D[n+1];return[i[0]+(a[0]-i[0])*r,i[1]+(a[1]-i[1])*r,i[2]+(a[2]-i[2])*r]}function k(e,t=1){let n=Number.parseInt(e.slice(1),16);return[(n>>16&255)/255,(n>>8&255)/255,(n&255)/255,t]}function A(e){return e===`dark`?{background:k(`#0d121b`),accent:k(`#3de0c5`),signal:k(`#ffb454`),ink:k(`#e8eef6`),muted:k(`#93a1b4`)}:{background:k(`#ffffff`),accent:k(`#07745f`),signal:k(`#b4570a`),ink:k(`#0b1220`),muted:k(`#4d5a6b`)}}function j(e){return{blend:!0,blendColorOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:e===`dark`?`one`:`one-minus-src-alpha`,blendAlphaOperation:`add`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`}}var M=`
fn sampleRamp(position: f32) -> vec3f {
  var stops = array<vec3f, ${D.length}>(
    ${D.map(([e,t,n])=>`vec3f(${e}, ${t}, ${n})`).join(`,
    `)}
  );
  let scaled = clamp(position, 0.0, 1.0) * ${D.length-1}.0;
  let index = min(u32(floor(scaled)), ${D.length-2}u);
  return mix(stops[index], stops[index + 1u], scaled - f32(index));
}
`;function N(e,t){return e.createBuffer({byteLength:t*4,usage:a.UNIFORM|a.COPY_DST})}function P(e,t){return e.createBuffer({data:t,usage:a.STORAGE|a.COPY_DST})}function F(e,t){return new o(e,{id:t.id,source:t.source,topology:`triangle-list`,vertexCount:6,instanceCount:Math.max(1,t.instanceCount),bindings:t.bindings,parameters:t.blendMode===`opaque`?{blend:!1}:{...j(t.theme)}})}function I(e){if(e.type!==`webgpu`)throw Error(`WebGPU required`)}var L=2600,te=.08,ne=.12,R=.22,z=.1,B=class{x=0;y=0;presence=0;target=null;reducedMotion;lastSeconds=-1;constructor(e){this.reducedMotion=e}get isActive(){return this.target!==null}set(e){e&&this.presence<.01&&(this.x=e[0],this.y=e[1]),this.target=e}update(e){let t=this.lastSeconds<0?0:e-this.lastSeconds;this.lastSeconds=e;let n=this.reducedMotion||t<=0||t>z,r=+!!this.target;if(n){this.target&&([this.x,this.y]=this.target),this.presence=r;return}if(this.target){let e=1-Math.exp(-t/te);this.x+=(this.target[0]-this.x)*e,this.y+=(this.target[1]-this.y)*e}let i=1-Math.exp(-t/(r>this.presence?ne:R));this.presence+=(r-this.presence)*i,Math.abs(r-this.presence)<.002&&(this.presence=r)}},V=28,H=9,re=[1,1],ie=[.5,1],ae=[0,.3],U=[1.8,1.35],W=[.25,1],G=1400,K=`
struct Look {
  world: vec2f,
  size: vec2f,
  clock: vec4f,
  background: vec4f,
  accent: vec4f,
  signal: vec4f,
  ink: vec4f,
  pointer: vec4f
};
@group(0) @binding(0) var<uniform> look: Look;

fn toClip(position: vec2f) -> vec2f {
  return vec2f(position.x / look.world.x * 2.0 - 1.0, 1.0 - position.y / look.world.y * 2.0);
}

// The wave front moves through schedule progress 0..1 and pauses past the end.
fn waveGlow(progress: f32) -> f32 {
  let distance = (progress - (look.clock.y * 1.3 - 0.1)) / 0.12;
  return exp(-distance * distance);
}
`,q=`
${K}
@group(0) @binding(1) var<storage, read> rects: array<vec4f>;
@group(0) @binding(2) var<storage, read> kinds: array<vec2f>;
@group(0) @binding(3) var<storage, read> nodeFocus: array<vec2f>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) offset: vec2f,
  @location(1) extent: vec2f,
  @location(2) kind: f32,
  @location(3) glow: f32,
  @location(4) alpha: f32
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let rect = rects[instanceIndex];
  let center = rect.xy + rect.zw * 0.5;
  let halo = 22.0;
  let extent = rect.zw * 0.5 + vec2f(halo);
  let corner = corners[vertexIndex];
  var output: VertexOutput;
  output.position = vec4f(toClip(center + corner * extent), 0.0, 1.0);
  output.offset = corner * extent;
  output.extent = rect.zw * 0.5;
  output.kind = kinds[instanceIndex].x;
  let focus = nodeFocus[instanceIndex];
  // Lit nodes glow at least focus.x; dimmed ones (alpha < 1) also lose the schedule wave.
  let wave = waveGlow(kinds[instanceIndex].y);
  output.glow = mix(wave, max(wave, focus.x) * focus.y, look.pointer.z);
  output.alpha = mix(1.0, focus.y, look.pointer.z);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let edge = abs(input.offset) - input.extent + vec2f(6.0);
  let distance = length(max(edge, vec2f(0.0))) + min(max(edge.x, edge.y), 0.0) - 6.0;
  var color = look.accent.rgb;
  if (input.kind > 0.5 && input.kind < 1.5) {
    color = look.signal.rgb;
  } else if (input.kind > 2.5) {
    color = look.ink.rgb;
  } else if (input.kind > 1.5) {
    color = mix(look.accent.rgb, look.signal.rgb, 0.5);
  }
  var strength = 0.16 + 0.84 * input.glow;
  if (input.kind > 2.5) {
    strength = strength * 0.45;
  }
  let halo = exp(-max(distance, 0.0) / 9.0) * strength * 0.55;
  let body = (1.0 - smoothstep(-1.0, 1.0, distance)) * (0.05 + 0.4 * input.glow);
  return vec4f(color, clamp(halo + body, 0.0, 1.0) * input.alpha);
}
`,oe=`
${K}
@group(0) @binding(1) var<storage, read> curves: array<vec4f>;
@group(0) @binding(2) var<storage, read> particles: array<vec4f>;
@group(0) @binding(3) var<storage, read> edgeFocus: array<vec2f>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) offset: vec2f,
  @location(1) brightness: f32,
  @location(2) tint: f32
};

fn bezier(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
  let u = 1.0 - t;
  return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3;
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let particle = particles[instanceIndex];
  let edgeIndex = u32(particle.x);
  let ends = curves[edgeIndex * 3u];
  let controls = curves[edgeIndex * 3u + 1u];
  let schedule = curves[edgeIndex * 3u + 2u];
  let t = fract(particle.y + look.clock.x * particle.z);
  let position = bezier(ends.xy, controls.xy, controls.zw, ends.zw, t);
  let corner = corners[vertexIndex];
  let focus = edgeFocus[edgeIndex];
  let radius = particle.w * (0.8 + 0.5 * sin(t * 3.14159)) * mix(1.0, focus.y, look.pointer.z);
  var output: VertexOutput;
  output.position = vec4f(toClip(position + corner * radius), 0.0, 1.0);
  output.offset = corner;
  output.brightness =
    (0.5 + 0.5 * waveGlow(mix(schedule.x, schedule.y, t))) * mix(1.0, focus.x, look.pointer.z);
  output.tint = schedule.z;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let distance = length(input.offset);
  if (distance > 1.0) {
    discard;
  }
  let core = 1.0 - smoothstep(0.0, 1.0, distance);
  let color = mix(look.accent.rgb, look.signal.rgb, input.tint);
  return vec4f(color, min(core * core * input.brightness, 1.0));
}
`;function J(e,t={}){return async()=>({default:Y(e,t)})}function Y(e,t={}){return({device:n,theme:r,width:i,height:a,reducedMotion:o})=>new ue(n,e,t,r,i,a,o)}function se(e){let t=new Map;for(let n of e){let e=t.get(n.progress)??{x:0,y:0,count:0};e.x+=n.x+n.width/2,e.y+=n.y+n.height/2,e.count++,t.set(n.progress,e)}return[...t.entries()].sort(([e],[t])=>e-t).map(([e,t])=>({x:t.x/t.count,y:t.y/t.count,progress:e}))}function ce(e,t,n){if(e.length===0)return 0;let r=e[0].progress,i=1/0;for(let a=0;a+1<e.length;a++){let o=e[a],s=e[a+1],c=s.x-o.x,l=s.y-o.y,u=c*c+l*l,d=u===0?0:Math.min(1,Math.max(0,((t-o.x)*c+(n-o.y)*l)/u)),f=Math.hypot(t-(o.x+d*c),n-(o.y+d*l));f<i&&(i=f,r=o.progress+d*(s.progress-o.progress))}return r}function le(e,t,n){let r=-1,i=1/0;return e.forEach((e,a)=>{let o=Math.hypot(Math.max(e.x-t,0,t-(e.x+e.width)),Math.max(e.y-n,0,n-(e.y+e.height)));o<i&&(i=o,r=a)}),r}var ue=class{device;graph;colors;look=new Float32Array(V);lookBuffer;buffers=[];nodeModel;particleModel;reducedMotion;interaction;dependencyDepth;pointer;scrubPolyline;nodeFocus;edgeFocus;nodeFocusBuffer;edgeFocusBuffer;hoveredNode=-1;wasActive=!1;phaseOffsetSeconds=0;lastPointerWave=0;width;height;constructor(e,t,n,r,i,a,o){I(e),this.device=e,this.graph=t,this.colors=A(r),this.width=i,this.height=a,this.reducedMotion=o,this.interaction=n.interaction??`none`,this.dependencyDepth=n.dependencyDepth??`direct`,this.pointer=new B(o),this.scrubPolyline=se(t.nodes),this.lookBuffer=N(e,V);let s=new Float32Array(Math.max(1,t.nodes.length)*4),c=new Float32Array(Math.max(1,t.nodes.length)*2);t.nodes.forEach((e,t)=>{s.set([e.x,e.y,e.width,e.height],t*4),c.set([e.kind,e.progress],t*2)});let l=new Float32Array(Math.max(1,t.edges.length)*12),u=[],d=Math.min(G,t.edges.length*5),f=t.edges.reduce((e,t)=>e+.25+t.weight,0)||1;t.edges.forEach((e,t)=>{if(l.set([e.x0,e.y0,e.x1,e.y1],t*12),l.set([e.controlX0,e.controlY0,e.controlX1,e.controlY1],t*12+4),l.set([e.progressStart,e.progressEnd,e.weight>.66?.8:0,0],t*12+8),e.weight<=0)return;let n=Math.max(1,Math.round(d*(.25+e.weight)/f));for(let r=0;r<n;r++){let i=(r/n+t*.61803%1)%1,a=.16+.1*((t*7+r*3)%5)/5;u.push(t,i,a,2.6+3.4*e.weight)}});let p=new Float32Array(Math.max(4,u.length));p.set(u),this.nodeFocus=new Float32Array(Math.max(1,t.nodes.length)*2);for(let e=0;e<this.nodeFocus.length;e+=2)this.nodeFocus.set([0,1],e);this.edgeFocus=new Float32Array(Math.max(1,t.edges.length)*2).fill(1);let m=P(e,s),h=P(e,c),g=P(e,l),_=P(e,p);this.nodeFocusBuffer=P(e,this.nodeFocus),this.edgeFocusBuffer=P(e,this.edgeFocus),this.buffers.push(m,h,g,_,this.nodeFocusBuffer,this.edgeFocusBuffer),this.nodeModel=F(e,{id:`flow-nodes`,source:q,bindings:{look:this.lookBuffer,rects:m,kinds:h,nodeFocus:this.nodeFocusBuffer},instanceCount:t.nodes.length,theme:r}),this.particleModel=F(e,{id:`flow-particles`,source:oe,bindings:{look:this.lookBuffer,curves:g,particles:_,edgeFocus:this.edgeFocusBuffer},instanceCount:u.length/4,theme:r}),t.nodes.length===0&&this.nodeModel.setInstanceCount(0),t.edges.length===0&&this.particleModel.setInstanceCount(0)}setPointer(e){this.pointer.set(e)}writeFocus(e){let{nodes:t,edges:n}=this.graph,r=new Set([e]);if(this.dependencyDepth===`direct`)for(let t of n)t.fromNode===e&&t.toNode!==void 0&&r.add(t.toNode),t.toNode===e&&t.fromNode!==void 0&&r.add(t.fromNode);else{let t=[e];for(let e=0;e<t.length;e++)for(let i of n)i.fromNode===t[e]&&i.toNode!==void 0&&!r.has(i.toNode)&&(r.add(i.toNode),t.push(i.toNode))}t.forEach((t,n)=>{let i=n===e?re:r.has(n)?ie:ae;this.nodeFocus.set(i,n*2)}),n.forEach((t,n)=>{let i=t.fromNode!==void 0&&t.toNode!==void 0&&(this.dependencyDepth===`direct`?t.fromNode===e||t.toNode===e:r.has(t.fromNode)&&r.has(t.toNode));this.edgeFocus.set(i?U:W,n*2)}),this.nodeFocusBuffer.write(this.nodeFocus),this.edgeFocusBuffer.write(this.edgeFocus)}render(e){let t=(this.reducedMotion?L:e)/1e3,n=this.look,r=this.pointer;r.update(e/1e3);let i=r.x*this.graph.width/this.width,a=r.y*this.graph.height/this.height,o=t%H/H;if(this.interaction===`scrub`){if(r.presence>0){let e=ce(this.scrubPolyline,i,a);this.lastPointerWave=(e+.1)/1.3}this.wasActive&&!r.isActive&&!this.reducedMotion&&(this.phaseOffsetSeconds=this.lastPointerWave*H-t%H),this.wasActive=r.isActive;let e=((t+this.phaseOffsetSeconds)%H+H)%H/H;o=e+(this.lastPointerWave-e)*r.presence}if(this.interaction===`dependencies`&&r.presence>0){let e=le(this.graph.nodes,i,a);e!==this.hoveredNode&&(this.hoveredNode=e,this.writeFocus(e))}let s=this.interaction===`dependencies`?r.presence:0;n.set([this.graph.width,this.graph.height,this.width,this.height],0),n.set([t,o,0,0],4),n.set([0,0,s,0],24),n.set(this.colors.background,8),n.set(this.colors.accent,12),n.set(this.colors.signal,16),n.set(this.colors.ink,20),this.lookBuffer.write(n);let c=this.device.beginRenderPass({clearColor:this.colors.background});this.nodeModel.draw(c),this.particleModel.draw(c),c.end(),this.device.submit()}resize(e,t){this.width=e,this.height=t}destroy(){this.nodeModel.destroy(),this.particleModel.destroy(),this.lookBuffer.destroy();for(let e of this.buffers)e.destroy()}},de=.28,fe=.12,pe={nodeWidth:188,nodeHeight:42,layerGap:64,rowGap:14,padding:22,targetWidth:1040},me=620;function he({scenario:e,...t}){return typeof e==`string`?(0,v.jsx)(s.Suspense,{fallback:(0,v.jsxs)(`div`,{className:`meta-panel meta-skeleton`,children:[`loading `,e]}),children:(0,v.jsx)(ge,{scenarioId:e,options:t})}):(0,v.jsx)(Q,{model:e,...t})}function ge({scenarioId:e,options:t}){let n=_(e);return n?(0,v.jsx)(Q,{model:n,...t}):(0,v.jsxs)(`div`,{className:`meta-panel meta-skeleton`,children:[`No recorded run for `,e]})}var X={raw:`read after write`,war:`write after read`,waw:`write after write`};function Z(e){let t=e.hazards??[];return t.includes(`waw`)?`waw`:t.includes(`war`)?`war`:`raw`}function Q({model:e,compact:t=!1,showHazards:r=!1,defaultView:i=`graph`,defaultStep:a,highlightNodeIds:o}){let c=e.nodes.length,d=!!e.lifetimes?.length,[p,g]=(0,s.useState)(d?i:`graph`),[_,y]=(0,s.useState)(Math.min(a??c,c)),[b,x]=(0,s.useState)(!1),S=(0,s.useMemo)(()=>new Map(e.nodes.map((e,t)=>[e.id,t])),[e]);(0,s.useEffect)(()=>{if(!b)return;let e=window.setInterval(()=>{y(e=>e>=c?(x(!1),e):e+1)},650);return()=>window.clearInterval(e)},[b,c]);let C=()=>{if(b)return x(!1);_>=c&&y(0),x(!0)},w=_<c,T=_>0?e.nodes[_-1]:void 0,E=(0,s.useMemo)(()=>m(e,pe),[e]),[D,O]=(0,s.useState)(null),k=(0,s.useMemo)(()=>$(e,E),[e,E]),A=(0,s.useMemo)(()=>J(k),[k]),j=(0,s.useMemo)(()=>new Map(e.nodes.map(e=>[e.id,e])),[e]),M=(0,s.useMemo)(()=>Se(e.edges,5),[e]),N=Math.max(1,...e.edges.map(e=>e.bytes)),P=o?.join(`
`)??``,F=(0,s.useMemo)(()=>P?new Set(P.split(`
`)):void 0,[P]),I=e=>F!==void 0&&!(F.has(e.from)&&F.has(e.to));return(0,v.jsxs)(`figure`,{className:`meta-panel`,"data-hazards":r,"aria-label":`Command graph of ${e.title}`,children:[t?null:(0,v.jsxs)(`div`,{className:`meta-panel__bar`,children:[(0,v.jsx)(`span`,{className:`meta-panel__title`,children:e.title}),(0,v.jsxs)(`ul`,{className:`meta-stats`,children:[(0,v.jsxs)(`li`,{children:[(0,v.jsx)(`strong`,{children:f(e.recordedNodeCount)}),` kernels`,e.collapsed?(0,v.jsxs)(v.Fragment,{children:[` `,`in `,(0,v.jsx)(`strong`,{children:e.nodes.length}),` groups`]}):null]}),(0,v.jsxs)(`li`,{children:[(0,v.jsx)(`strong`,{children:f(e.recordedEdgeCount)}),` dependencies`]}),(0,v.jsxs)(`li`,{children:[(0,v.jsx)(`strong`,{children:l(e.resources.totalBytes,e.resources.totalBytesKind)}),` `,`in `,(0,v.jsx)(`strong`,{children:u(e.resources)})]}),(0,v.jsxs)(`li`,{children:[(0,v.jsx)(`strong`,{children:f(e.totals.workgroups)}),` workgroups`]})]}),(0,v.jsxs)(`span`,{className:`meta-legend`,"aria-hidden":`true`,children:[(0,v.jsxs)(`span`,{children:[(0,v.jsx)(`i`,{"data-kind":`compute`}),`compute`]}),(0,v.jsxs)(`span`,{children:[(0,v.jsx)(`i`,{"data-kind":`copy`}),`copy`]})]})]}),t?null:(0,v.jsxs)(`div`,{className:`meta-controls`,children:[d?(0,v.jsxs)(`div`,{className:`meta-toggle`,role:`group`,"aria-label":`View`,children:[(0,v.jsx)(`button`,{type:`button`,"aria-pressed":p===`graph`,onClick:()=>g(`graph`),children:`Graph`}),(0,v.jsx)(`button`,{type:`button`,"aria-pressed":p===`buffers`,onClick:()=>g(`buffers`),children:`Buffers`})]}):null,(0,v.jsx)(`button`,{type:`button`,className:`meta-play`,onClick:C,children:b?`Pause`:_>=c?`Replay`:`Play`}),(0,v.jsxs)(`label`,{className:`meta-slider meta-slider--wide`,children:[(0,v.jsxs)(`span`,{children:[`Step `,(0,v.jsx)(`b`,{children:_}),` / `,c]}),(0,v.jsx)(`input`,{type:`range`,min:0,max:c,step:1,value:_,onChange:e=>{x(!1),y(Number(e.target.value))},"aria-valuetext":T?`step ${_}: ${T.label}`:`before the first step`})]}),(0,v.jsx)(`span`,{className:`meta-controls__now`,"aria-live":`polite`,children:T?(0,v.jsxs)(v.Fragment,{children:[`ran `,(0,v.jsx)(`b`,{children:T.label})]}):`nothing has run`})]}),r&&!t&&p===`graph`?(0,v.jsx)(`ul`,{className:`meta-hazard-key`,"aria-label":`Dependency kinds`,children:[`raw`,`war`,`waw`].map(t=>(0,v.jsxs)(`li`,{"data-hazard":t,children:[(0,v.jsx)(`i`,{}),X[t],(0,v.jsx)(`span`,{children:f(e.hazards[t])})]},t))}):null,p===`buffers`&&e.lifetimes?(0,v.jsx)(ee,{model:e,step:_}):(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(`div`,{className:`meta-scroll`,children:(0,v.jsxs)(`div`,{className:`meta-stage`,style:{minWidth:Math.min(E.width,me),aspectRatio:`${E.width} / ${E.height}`},children:[w||F?null:(0,v.jsx)(`div`,{className:`meta-stage__canvas`,"aria-hidden":`true`,children:(0,v.jsx)(n,{scene:A,fallback:null,seed:1,label:`Data flowing through ${e.title}`,aspectRatio:E.width/E.height})}),(0,v.jsxs)(`svg`,{className:`meta-stage__svg`,viewBox:`0 0 ${E.width} ${E.height}`,role:`img`,"aria-label":`${e.nodes.length} kernel groups and ${e.edges.length} dependencies`,children:[(0,v.jsx)(`g`,{children:E.edges.map((t,n)=>{let i=e.edges[n],a=w?(S.get(i.to)??0)<_?`done`:(S.get(i.from)??0)<_?`live`:`idle`:void 0,o=D?.kind===`edge`?D.index===n:D?.kind===`node`&&(D.id===i.from||D.id===i.to);return(0,v.jsxs)(`g`,{opacity:I(i)?fe:void 0,children:[(0,v.jsx)(`path`,{className:`lens-edge`,"data-active":o,"data-skip":t.span>1,"data-state":a,"data-hazard":Z(i),d:h(t),strokeWidth:xe(i.bytes,N),style:r&&t.span>1&&!o&&a!==`idle`?{strokeOpacity:.32}:void 0}),(0,v.jsx)(`path`,{className:`lens-edge-hit`,d:h(t),onPointerEnter:()=>O({kind:`edge`,index:n}),onPointerLeave:()=>O(null)})]},`${t.from}>${t.to}`)})}),(0,v.jsx)(`g`,{"aria-hidden":`true`,children:[...M].map(t=>{let n=E.edges[t],r=e.edges[t];return n.x1<=n.x0||n.span>1||I(r)?null:(0,v.jsx)(`text`,{className:`lens-edge-label`,x:(n.x0+n.x1)/2,y:(n.y0+n.y1)/2-4,textAnchor:`middle`,children:l(r.bytes,r.bytesKind)},t)})}),(0,v.jsx)(`g`,{children:E.nodes.map(e=>{let t=j.get(e.id);if(!t)return null;let n=S.get(t.id)??0,r=w?n<_-1?`done`:n===_-1?`running`:`pending`:void 0;return(0,v.jsxs)(`g`,{className:`lens-node`,"data-kind":t.kind,"data-state":r,"data-active":D?.kind===`node`&&D.id===t.id,"data-dimmed":F&&!F.has(t.id)?`true`:void 0,opacity:F&&!F.has(t.id)?de:void 0,transform:`translate(${e.x} ${e.y})`,tabIndex:0,onPointerEnter:()=>O({kind:`node`,id:t.id}),onPointerLeave:()=>O(null),onClick:()=>O(e=>e?.kind===`node`&&e.id===t.id?null:{kind:`node`,id:t.id}),onFocus:()=>O({kind:`node`,id:t.id}),onBlur:()=>O(null),children:[(0,v.jsx)(`title`,{children:t.label}),(0,v.jsx)(`rect`,{width:e.width,height:e.height,rx:7}),(0,v.jsx)(`text`,{className:`lens-node__label`,x:10,y:17,children:_e(t.label,e.width)}),(0,v.jsx)(`text`,{className:`lens-node__meta`,x:10,y:32,children:ve(t)}),t.count>1?(0,v.jsxs)(`text`,{className:`lens-node__badge`,x:e.width-8,y:32,textAnchor:`end`,children:[`x`,t.count]}):null]},t.id)})})]})]})}),(0,v.jsx)(`p`,{className:`meta-scroll-hint`,children:`Scroll sideways; tap a kernel for details.`})]}),(0,v.jsx)(`figcaption`,{className:`meta-detail`,"aria-live":`polite`,children:p===`buffers`&&e.lifetimes?be(e,_):ye(e,D,j)})]})}function _e(e,t){let n=Math.max(8,Math.floor((t-20)/6.4));return e.length>n?`${e.slice(0,n-1)}…`:e}function ve(e){let t=[e.kind];return e.workgroups>0&&t.push(`${f(e.workgroups)} wg`),e.milliseconds!==void 0&&t.push(`${e.milliseconds.toFixed(2)} ms`),t.join(` / `)}function ye(e,t,n){if(t?.kind===`node`){let e=n.get(t.id);if(e)return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(`strong`,{children:e.label}),(0,v.jsxs)(`span`,{children:[(0,v.jsx)(`b`,{children:e.kind}),e.count>1?(0,v.jsxs)(v.Fragment,{children:[` x`,e.count,` unrolled`]}):null]}),e.operators.length?(0,v.jsx)(`span`,{children:e.operators.join(`, `)}):null,(0,v.jsxs)(`span`,{children:[(0,v.jsx)(`b`,{children:f(e.workgroups)}),` workgroups`]}),(0,v.jsxs)(`span`,{children:[`reads `,(0,v.jsx)(`b`,{children:d(e.readBytes)}),`, writes `,(0,v.jsx)(`b`,{children:d(e.writeBytes)})]})]})}if(t?.kind===`edge`){let n=e.edges[t.index];if(n)return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsxs)(`strong`,{children:[n.resources.join(`, `),n.resourceCount>n.resources.length?`, ...`:``]}),(0,v.jsxs)(`span`,{children:[(0,v.jsx)(`b`,{children:l(n.bytes,n.bytesKind)}),` in `,(0,v.jsx)(`b`,{children:n.resourceCount}),` `,n.resourceCount===1?`buffer`:`buffers`]}),(0,v.jsxs)(`span`,{children:[Z(n)===`raw`?`written by one kernel, read by the next`:X[Z(n)],n.hazards&&n.hazards.length>1?` (also ${n.hazards.filter(e=>e!==Z(n)).map(e=>X[e]).join(`, `)})`:null]})]})}return(0,v.jsx)(v.Fragment,{children:(0,v.jsxs)(`span`,{children:[`Hover or tap a kernel or a dependency.`,e.collapsed?` Unrolled loops and repeated passes are folded into one node (xN).`:``]})})}function be(e,t){let n=e.lifetimes??[],r=n.filter(e=>w([e],t)>0).length,i=n.some(e=>e.ownership===`transient`),a=n.some(e=>e.ownership===`unknown`||e.bytesKind===void 0),o=c(n);return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsxs)(`span`,{children:[(0,v.jsx)(`b`,{children:l(w(n,t),o)}),` alive in`,` `,(0,v.jsx)(`b`,{children:r}),` of `,n.length,` buffers`]}),(0,v.jsxs)(`span`,{children:[i?`peak`:`peak logical live`,` `,(0,v.jsx)(`b`,{children:l(T(n,e.nodes.length),o)})]}),(0,v.jsxs)(`span`,{children:[a?`Ownership is unobserved for some buffers; allocation sharing cannot be inferred from these lifetimes.`:i?`Buffers that never overlap in time can share one block of memory.`:`All buffers are imported, so all stay allocated for the whole run; a bar is the logical span between first write and last use.`,e.lifetimesOmitted?` ${e.lifetimesOmitted} smaller buffers not shown.`:``]})]})}function xe(e,t){return 1+2.4*(Math.log10(1+e)/Math.log10(1+t))}function Se(e,t){return new Set(e.map((e,t)=>({bytes:e.bytes,index:t})).filter(e=>e.bytes>=1024).sort((e,t)=>t.bytes-e.bytes).slice(0,t).map(e=>e.index))}function $(e,t){let n=Math.max(1,...e.edges.map(e=>e.bytes)),r=Math.max(1,...t.nodes.map(e=>e.layer)),i=new Map(t.nodes.map(e=>[e.id,e.layer])),a=new Map(e.nodes.map(e=>[e.id,e.kind])),o=new Map(t.nodes.map((e,t)=>[e.id,t]));return{width:t.width,height:t.height,nodes:t.nodes.map(e=>{let t=a.get(e.id);return{x:e.x,y:e.y,width:e.width,height:e.height,kind:t===`copy`?1:t===`compute`?0:3,progress:e.layer/r}}),edges:t.edges.map((t,a)=>({x0:t.x0,y0:t.y0,x1:t.x1,y1:t.y1,controlX0:t.controlX0,controlY0:t.controlY0,controlX1:t.controlX1,controlY1:t.controlY1,weight:t.span>1?0:Math.log10(1+e.edges[a].bytes)/Math.log10(1+n),progressStart:(i.get(t.from)??0)/r,progressEnd:(i.get(t.to)??0)/r,fromNode:o.get(t.from),toNode:o.get(t.to)}))}}export{f as _,L as a,c as b,F as c,A as d,O as f,d as g,m as h,M as i,P as l,T as m,$ as n,B as o,w as p,J as r,I as s,he as t,N as u,l as v,u as y};