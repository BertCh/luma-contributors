import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{r as t}from"./kernel-C3tfQOn4.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{x as r}from"./gpu-contributor-utils-BIwLQON5.js";import{h as i,l as a,o,u as s}from"./raster-band-utils-BcTxc8_Y.js";import{n as c,u as l}from"./terrain-illumination-utils-CoEB4UZd.js";import{n as u,t as d}from"./gpu-terrain-derivatives-DmS0Xo0r.js";import{t as f}from"./define-run-B-brkCe4.js";import{a as p}from"./pointer-overlay-Qh8qflcA.js";import{t as m}from"./horizon-cisgaCRR.js";import{t as h}from"./relief-CcqfaXl5.js";import{i as g,m as _}from"./terrain-Cc9dW1uq.js";import{L as v}from"./entry-preview-D70TdSbK.js";var y={normal:0,multiply:1,screen:2,overlay:3,"soft-light":4,luminosity:5};function b(e,t=new Float32Array(e.length*5)){if(e.length<1||e.length>5)throw Error(`Relief blend supports 1 to 5 layers`);o(`getGPUReliefBlendParameterValues`,t,e.length*5);for(let[n,r]of e.entries()){let e=r.opacity??1;if(![r.minimum,r.maximum,e].every(Number.isFinite))throw Error(`Relief blend settings must be finite`);if(r.maximum<=r.minimum)throw Error(`Relief blend layer maximum must be greater than minimum`);if(e<0||e>1)throw Error(`Relief blend opacity must be in [0, 1]`);let i=y[r.blendMode??`normal`];if(i===void 0)throw Error(`Relief blend blendMode is unknown`);let a=n*5;t[a]=r.minimum,t[a+1]=r.maximum,t[a+2]=+!!r.invert,t[a+3]=i,t[a+4]=e}return t}Object.freeze([{minimum:0,maximum:1,blendMode:`normal`,opacity:1},{minimum:0,maximum:50,invert:!0,blendMode:`luminosity`,opacity:.5},{minimum:68,maximum:93,blendMode:`overlay`,opacity:.5},{minimum:.7,maximum:1,blendMode:`multiply`,opacity:.25}]),Object.freeze([{minimum:0,maximum:1,blendMode:`normal`,opacity:1},{minimum:0,maximum:15,invert:!0,blendMode:`luminosity`,opacity:.5},{minimum:85,maximum:93,blendMode:`overlay`,opacity:.5},{minimum:.9,maximum:1,blendMode:`multiply`,opacity:.25}]);var x=class{id;props;requiredHalo=0;constructor(e){this.id=e.id??`relief-blend`,this.props=e;let{id:t}=this,{output:n}=e,r=s(t,e.grid),o=e.layers.length;if(o<1||o>5)throw Error(`${t} layers must contain 1 to 5 views`);for(let[n,i]of e.layers.entries())l(t,`layer ${n}`,i,`float32`,r);if(!n.blend&&!n.color&&!n.outputValidity)throw Error(`${t} requires at least one output`);l(t,`output.blend`,n.blend,`float32`,r),l(t,`output.color`,n.color,`uint32`,r),l(t,`output.outputValidity`,n.outputValidity,`uint32`,r),i(t,e.parameters,o*5),a(t,[n.blend,n.color,n.outputValidity],[...e.layers,e.parameters])}getCommandNodes(i){e(i,this.id,this.props,{"blend scratch":4*this.props.grid.width*this.props.grid.height});let{id:a,props:o}=this,{output:s}=o,l=o.grid.width*o.grid.height;r(a,i,[...o.layers,o.parameters,s.blend,s.color,s.outputValidity]);let u=s.blend??(s.color||s.outputValidity?t(i,`${a}-blend`,`float32`,l):void 0),d=[...o.layers.map((e,t)=>({name:`layer${t}`,view:e,type:`f32`,access:`read`})),{name:`settings`,view:o.parameters,type:`f32`,access:`read`},{name:`blend`,view:u,type:`f32`,access:`read_write`}],f=o.layers.map((e,t)=>{let n=t*5,r=`normalizeLayer(layer${t}[layer${t}Offset + index], ${n}u)`;return t===0?`allFinite = allFinite && isFiniteValue(layer0[layer0Offset + index]);
  var background = ${r};`:`allFinite = allFinite && isFiniteValue(layer${t}[layer${t}Offset + index]);
  background = renderLayer(${r}, background, ${n}u);`}),p=[n(i,{id:`${a}-blend`,operation:`GPUReliefBlend`,variant:`blend`,bindings:d,invocationCount:l,declarations:`${c}
fn normalizeLayer(value: f32, base: u32) -> f32 {
  let minimum = settings[settingsOffset + base];
  let maximum = settings[settingsOffset + base + 1u];
  let normalized = clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
  return select(normalized, 1.0 - normalized, settings[settingsOffset + base + 2u] > 0.5);
}
fn blendLayer(mode: u32, foreground: f32, background: f32) -> f32 {
  if (mode == 1u) {
    return foreground * background;
  }
  if (mode == 2u) {
    return 1.0 - (1.0 - foreground) * (1.0 - background);
  }
  if (mode == 3u) {
    return select(
      2.0 * background * foreground,
      1.0 - (1.0 - 2.0 * (background - 0.5)) * (1.0 - foreground),
      background > 0.5
    );
  }
  if (mode == 4u) {
    return select(
      2.0 * background * (1.0 - foreground) + sqrt(background) * (2.0 * foreground - 1.0),
      2.0 * background * foreground + background * background * (1.0 - 2.0 * foreground),
      foreground < 0.5
    );
  }
  return foreground;
}
fn renderLayer(foreground: f32, background: f32, base: u32) -> f32 {
  let mode = u32(settings[settingsOffset + base + 3u] + 0.5);
  let opacity = settings[settingsOffset + base + 4u];
  let top = blendLayer(mode, foreground, background);
  return top * opacity + background * (1.0 - opacity);
}`,body:`var allFinite = true;
  ${f.join(`
  `)}
  blend[blendOffset + index] = select(getQuietNaN(index), background, allFinite);`})];if(s.color||s.outputValidity){let e=[{name:`blend`,view:u,type:`f32`,access:`read`}];s.color&&e.push({name:`color`,view:s.color,type:`u32`,access:`read_write`}),s.outputValidity&&e.push({name:`validityValues`,view:s.outputValidity,type:`u32`,access:`read_write`}),p.push(n(i,{id:`${a}-encode`,operation:`GPUReliefBlend`,variant:`encode`,bindings:e,invocationCount:l,declarations:c,body:`let value = blend[blendOffset + index];
  let isValid = isFiniteValue(value);
  let grey = clamp(value, 0.0, 1.0);
  ${s.color?`color[colorOffset + index] = select(0u, pack4x8unorm(vec4<f32>(grey, grey, grey, 1.0)), isValid);`:``}
  ${s.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`}))}return p}},S=[`normal`,`multiply`,`luminosity`,`overlay`],C=f({parameters:v,caption:`Hillshade, slope, openness and sky view layers in; one blended relief image out.`,createInputs:({world:e})=>g(e),build(e,t){let n=_(e,t),{grid:r,band:i,cellCount:a}=n,o=e.allocate(`derivative-parameters`,`float32`,8);o.write(u({zFactor:2},r));let s=e.allocate(`slope`,`float32`,a),c=e.allocate(`hillshade`,`float32`,a);e.graph.add(new d({grid:r,elevation:i,parameters:o.view,output:{slope:s.view,hillshade:c.view}}));let{skyViewFactor:l,positiveOpenness:f}=m(e,n,{directionCount:16,maximumRadius:30}),p=e.allocate(`parameters`,`float32`,20),h=e.allocate(`blend`,`float32`,a);return e.graph.add(new x({grid:r,layers:[c.view,s.view,f.view,l.view],parameters:p.view,output:{blend:h.view}})),{outputs:{blend:h},update:e=>p.write(b([{minimum:0,maximum:1,blendMode:`normal`,opacity:1},{minimum:0,maximum:50,invert:!0,blendMode:S[e.slopeBlendMode],opacity:e.slopeOpacity},{minimum:68,maximum:93,blendMode:`overlay`,opacity:e.opennessOpacity},{minimum:.7,maximum:1,blendMode:`multiply`,opacity:e.skyViewOpacity}]))}},interaction:p(v,{kind:`pad`,hint:`mix slope and openness`,x:`slopeOpacity`,y:`opennessOpacity`,decorate:(e,t)=>({glyphs:[{shape:`crosshair`,at:[t.x,t.y]}],readout:`slope ${e.slopeOpacity.toFixed(2)} · openness ${e.opennessOpacity.toFixed(2)}`})}),draw:({inputs:e,outputs:t})=>[h(e,t.blend)]});export{C as default};