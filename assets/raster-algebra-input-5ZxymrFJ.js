import{o as e}from"./gpu-command-graph-BdzdmV6u.js";import{E as t,O as n,T as r,f as i,g as a,v as o,x as s,y as c}from"./gpu-contributor-utils-BIwLQON5.js";import{c as l,n as u,r as d,s as f,u as p}from"./raster-band-utils-BcTxc8_Y.js";import{c as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as h,u as g}from"./wgsl-common-DfyXiYB0.js";import{n as _}from"./raster-storage-preflight-CIxFc-0g.js";function v(e,t){let r=new Map;for(let[i,a]of Object.entries(t)){if(!a)continue;let o=Array.isArray(t)?`outputs[${i}]`:i,s=a.buffer.defaultBuffer?n(a.buffer.defaultBuffer):a.buffer,c=r.get(s);if(c!==void 0)throw Error(`${e} outputs must not share buffers: ${c} conflicts with ${o}`);r.set(s,o)}}function y(e){return`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${h}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
${g}
fn isNoDataValue(value: f32) -> bool {
  return isNaNValue(value)${e===void 0?``:` || value == ${m(e)}`};
}`}function b(e,n,r,i,a){if(r&&(t(r,[i],`${e} ${n}`),r.length<a))throw Error(`${e} ${n} must hold at least ${a} rows`)}function x(e,t,n,r,a){i(e,t,n,r,a)}function S(e,t){t!==void 0&&a(e,`noDataValue`,t)}function C(t,n=`output`){let r={};for(let[i,a]of Object.entries(t)){let t=`${n}.${i}`;a instanceof e?r[t]=a:a&&typeof a==`object`&&Object.assign(r,C(a,t))}return r}function w(e,t,n){v(e,t),c(e,t,n)}function T(e,t,n){s(e,t,n)}function E(e,t){return`
fn ${e}(first: u32, count: u32, value: f32, closedRight: bool) -> u32 {
  var low = 0u;
  var high = count;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    let boundary = ${t}[${t}Offset + first + middle];
    let isBelow = select(boundary <= value, boundary < value, closedRight);
    if (isBelow) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`}var D=34028234663852886e22,O=4294967295,k=-2147483648,A=2147483647,j=1,M=3;function N(e){return`storage`in e}function P(e,t){return e?N(e)?{[`${t}.storage.values`]:e.storage.kind===`buffer`?e.storage.values:void 0,[`${t}.validity`]:e.validity}:{[t]:e}:{}}function F(e){return e?N(e)?u(e):[e]:[]}function I(t,n){return t?N(t)?u(t).map(t=>new e(t.buffer,{format:t.format,byteOffset:t.byteOffset,byteStride:t.byteStride,rowByteLength:t.rowByteLength,length:n})):[t]:[]}function L(e,t,n,r){let i=r?r.width*r.height:0;for(let r of n)r&&N(r)&&l(t,e,r,[]);let a=n.flatMap(e=>I(e,i));_(e,t,a,{"canonical band planes":n.some(e=>e&&N(e))?i*Float32Array.BYTES_PER_ELEMENT:0})}function R(e,n,o,s,c){if(r(e,!!o,`${n} must be a float32 view or raster band`),!N(o))return b(e,n,o,`float32`,c??0),o.length;r(e,!!s,`${n} requires grid for a raster band`);let l=p(e,s);r(e,c===void 0||c<=l,`${n} grid must cover rowCount ${c}`);for(let[t,r]of[[`scale`,o.scale??1],[`offset`,o.offset??0]])a(e,`${n}.${t}`,r,-34028234663852886e22,D);if(o.noDataValue!==void 0&&(o.format===`float32`?Number.isNaN(o.noDataValue)||a(e,`${n}.noDataValue`,o.noDataValue,-34028234663852886e22,D):i(e,`${n}.noDataValue`,o.noDataValue,o.format===`uint32`?0:k,o.format===`uint32`?O:A)),f(o,s,`${e} ${n}`),o.storage.kind===`buffer`)t(o.storage.values,[o.format],`${e} ${n}.storage.values`);else{let{view:t,channel:a=0}=o.storage;r(e,t.dimension===`2d`,`${n}.storage.view.dimension must be 2d`);for(let[r,a]of[[`mipLevelCount`,t.mipLevelCount],[`arrayLayerCount`,t.arrayLayerCount],[`texture.samples`,t.texture.samples]])i(e,`${n}.storage.view.${r}`,a,j,j);let s={float32:`float`,uint32:`uint`,sint32:`sint`}[o.format],c=[`r32${s}`,`rgba32${s}`];r(e,c.includes(t.format),`${n}.storage.view.format must be ${c.join(` or `)} for ${n}.format ${o.format}`),i(e,`${n}.storage.channel`,a,0,t.format.startsWith(`rgba`)?M:0)}b(e,`${n}.validity`,o.validity,`uint32`,l);let u=o.storage.kind===`buffer`?o.storage.values.buffer.graph:o.storage.view.texture.graph;return r(e,!o.validity||o.validity.buffer.graph===u,`${n}.validity must belong to the same graph as ${n}.storage`),l}function z(e,t,n,r){if(!N(n))return{values:n,nodes:[]};l(t,e,n,[]);let i=r.width*r.height;o(e.device,t,`band canonicalization`,i*Float32Array.BYTES_PER_ELEMENT,`reduce grid dimensions`);let a=n.validity?e.createDataView(n.validity.buffer,{format:`uint32`,byteOffset:n.validity.byteOffset,length:i}):void 0,s={...n,validity:a};if(n.storage.kind===`buffer`){let{values:t}=n.storage;s={...s,storage:{kind:`buffer`,values:e.createDataView(t.buffer,{format:t.format,byteOffset:t.byteOffset,length:i})}}}let c=d(e,t,s,r,!0);return{values:c.band.storage.values,nodes:c.nodes}}export{N as a,E as c,w as d,x as f,v as g,b as h,F as i,C as l,S as m,I as n,R as o,T as p,z as r,L as s,P as t,y as u};