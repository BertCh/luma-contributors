import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,x as i}from"./gpu-contributor-utils-BIwLQON5.js";import{l as a,u as o}from"./raster-band-utils-BcTxc8_Y.js";import{t as s}from"./define-run-B-brkCe4.js";import{i as c,o as l,p as u}from"./terrain-Cc9dW1uq.js";import{T as d}from"./entry-preview-D70TdSbK.js";var f=[-11e3,9e3],p=-12e3,m=`bitcast<f32>(0x7fc00000u | (pixelIndex & 0u))`,h=class{id;props;validRange;constructor(e){this.id=e.id??`terrain-rgb-decode`,this.props=e;let{id:t}=this,{output:n}=e,i=o(t,e.grid);if(e.encoding!==`terrarium`&&e.encoding!==`mapbox`)throw Error(`${t} encoding must be 'terrarium' or 'mapbox'`);let{texture:s,buffer:c}=e.input;if(!!s==!!c)throw Error(`${t} input requires exactly one of texture or buffer`);if(s){if(s.format!==`rgba8unorm`)throw Error(`${t} input texture must be rgba8unorm`);if(s.dimension!==`2d`||s.mipLevelCount!==1||s.arrayLayerCount!==1||s.width!==e.grid.width||s.height!==e.grid.height)throw Error(`${t} input texture must be one 2D mip with the grid extent`);if(e.validity)throw Error(`${t} validity requires a buffer input`)}if(c&&(r(c,[`uint32`],`${t} input buffer`),c.length!==i))throw Error(`${t} input buffer must contain one packed RGBA8 word per pixel`);if(e.validity&&(r(e.validity,[`uint32`],`${t} validity`),e.validity.length!==i))throw Error(`${t} validity must contain one value per pixel`);if(r(n.values,[`float32`],`${t} output.values`),r(n.outputValidity,[`uint32`],`${t} output.outputValidity`),n.values.length!==i||n.outputValidity.length!==i)throw Error(`${t} outputs must contain one value per pixel`);if(e.noDataRGB&&(e.noDataRGB.length!==3||!e.noDataRGB.every(e=>Number.isInteger(e)&&e>=0&&e<=255)))throw Error(`${t} noDataRGB must be three integer bytes`);this.validRange=e.validRange??f;let[l,u]=this.validRange;if(Number.isNaN(l)||Number.isNaN(u)||l>u)throw Error(`${t} validRange must be [minimum, maximum] with minimum <= maximum`);a(t,[n.values,n.outputValidity],[c,e.validity])}getCommandNodes(t){e(t,this.id,this.props);let{id:r,props:a}=this,{output:o}=a;if(i(r,t,[a.input.buffer,a.validity,o.values,o.outputValidity]),a.grid.width*a.grid.height*4>t.device.limits.maxStorageBufferBindingSize)throw Error(`${r} output exceeds the device storage binding limit`);if(a.input.texture){if(a.input.texture.texture.graph!==t)throw Error(`${r} views must belong to the target graph`);return[this.getTextureNode(t,a.input.texture)]}let s=[{name:`inputWords`,view:a.input.buffer,type:`u32`,access:`read`},...a.validity?[{name:`inputValidity`,view:a.validity,type:`u32`,access:`read`}]:[],{name:`outputValues`,view:o.values,type:`f32`,access:`read_write`},{name:`outputValidity`,view:o.outputValidity,type:`u32`,access:`read_write`}];return[n(t,{id:`${r}-decode`,operation:`GPUTerrainRGBDecode`,variant:`${a.encoding}-buffer`,bindings:s,invocationCount:a.grid.width*a.grid.height,declarations:this.getDecodeDeclarations(),body:`let pixelIndex = index;
  let word = inputWords[inputWordsOffset + pixelIndex];
  let red = word & 0xffu;
  let green = (word >> 8u) & 0xffu;
  let blue = (word >> 16u) & 0xffu;
  let alpha = word >> 24u;
  var isValid = isEncodedPixelValid(red, green, blue, alpha);
  ${a.validity?`isValid = isValid && inputValidity[inputValidityOffset + pixelIndex] != 0u;`:``}
  ${this.getDecodeStatements()}`})]}getDecodeDeclarations(){let{props:e,validRange:n}=this,[r,i]=n,a=e.noDataRGB,o=[Number.isFinite(r)?`height >= ${t(r)}`:``,Number.isFinite(i)?`height <= ${t(i)}`:``].filter(Boolean),s=e.encoding===`terrarium`?`f32(code) * 0.00390625 - 32768.0`:`f32(i32(code) - 100000) * ${t(.1)}`;return`const WIDTH: u32 = ${e.grid.width}u;
const HEIGHT: u32 = ${e.grid.height}u;
fn isEncodedPixelValid(red: u32, green: u32, blue: u32, alpha: u32) -> bool {
  ${e.alphaNoData===!1?``:`if (alpha == 0u) { return false; }`}
  ${a?`if (red == ${a[0]}u && green == ${a[1]}u && blue == ${a[2]}u) { return false; }`:``}
  return true;
}
// Integer code N = R * 65536 + G * 256 + B < 2^24, then exactly one float operation sequence.
fn decodeHeight(red: u32, green: u32, blue: u32) -> f32 {
  let code = (red << 16u) | (green << 8u) | blue;
  return ${s};
}
fn isHeightInRange(height: f32) -> bool {
  return ${o.join(` && `)||`true`};
}
fn applySeaClamp(height: f32) -> f32 {
  ${e.clampBathymetry?`if (height < 0.0 && height > ${t(p)}) { return 0.0; }`:``}
  return height;
}`}getDecodeStatements(){return`let height = decodeHeight(red, green, blue);
  isValid = isValid && isHeightInRange(height);
  outputValidity[outputValidityOffset + pixelIndex] = select(0u, 1u, isValid);
  outputValues[outputValuesOffset + pixelIndex] =
    select(${m}, applySeaClamp(height), isValid);`}getTextureNode(e,t){let{id:r,props:i}=this,{output:a}=i;return n(e,{id:`${r}-decode`,operation:`GPUTerrainRGBDecode`,variant:`${i.encoding}-texture`,bindings:[{name:`outputValues`,view:a.values,type:`f32`,access:`read_write`},{name:`outputValidity`,view:a.outputValidity,type:`u32`,access:`read_write`}],textures:[{name:`sourceTexture`,texture:t,sampleType:`float`}],invocationCount:i.grid.width*i.grid.height,declarations:this.getDecodeDeclarations(),body:`let pixelIndex = index;
  let texelCoordinate = vec2<i32>(i32(pixelIndex % WIDTH), i32(pixelIndex / WIDTH));
  // Exact for unorm8: the load is within half a code of k / 255.
  let texel = clamp(textureLoad(sourceTexture, texelCoordinate, 0), vec4<f32>(0.0), vec4<f32>(1.0));
  let bytes = vec4<u32>(round(texel * 255.0));
  let red = bytes.x;
  let green = bytes.y;
  let blue = bytes.z;
  var isValid = isEncodedPixelValid(red, green, blue, bytes.w);
  ${this.getDecodeStatements()}`})}};function g(e,t,n){let r,i,a;if(t===`terrarium`){let t=e+32768;r=Math.floor(t/256),i=Math.floor(t)%256,a=Math.floor((t-Math.floor(t))*256)}else{let t=Math.round((e+1e4)*10);r=Math.floor(t/65536),i=Math.floor(t/256)%256,a=t%256}return(r|i<<8|a<<16|n<<24)>>>0}var _=s({parameters:d,rebuildOn:[`alphaNoData`,`validMaximum`],caption:`An RGBA elevation tile in; heights in meters, with missing data marked, out.`,createInputs:({random:e,world:t})=>{let n=c(t,48),r=Array.from({length:4},()=>({column:Math.floor(e()*(n.columns-10)),row:Math.floor(e()*(n.rows-8)),width:4+Math.floor(e()*6),height:3+Math.floor(e()*5)})),i=e=>{let t=e%n.columns,i=Math.floor(e/n.columns);return r.some(e=>t>=e.column&&t<e.column+e.width&&i>=e.row&&i<e.row+e.height)};return{...n,isHole:i}},build(e,t,n){let r=`terrarium`,i=Uint32Array.from(t.elevation,(e,n)=>t.isHole(n)?g(0,r,0):g(e,r,255)),a=e.upload(`tile`,`uint32`,i),o=e.upload(`stored-red`,`float32`,Float32Array.from(i,e=>e&255)),s=e.allocate(`decoded`,`float32`,i.length),c=e.allocate(`output-validity`,`uint32`,i.length);return e.graph.add(new h({grid:u(t),encoding:r,input:{buffer:a.view},alphaNoData:n.alphaNoData===1,validRange:[-11e3,n.validMaximum],output:{values:s.view,outputValidity:c.view}})),{outputs:{storedRed:o,decoded:s}}},interaction:{kind:`swipe`,hint:`swipe tile and meters`,rest:[.5,.5],resolve:(e,{values:t})=>({curtain:e.x,glyphs:[{shape:`curtain`,x:e.x}],readout:`RGBA tile | metres up to ${t.validMaximum} m`})},draw:({inputs:e,outputs:t,interaction:n,world:r})=>{let i=n.curtain??r.width/2;return[l(e,t.storedRed,{range:[128,138],clip:{side:`left`,x:i}}),l(e,t.decoded,{range:[650,2300],hillshade:.5,elevation:60,clip:{side:`right`,x:i}})]}});export{_ as default};