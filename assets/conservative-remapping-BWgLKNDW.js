import{r as e}from"./kernel-C3tfQOn4.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{T as r,f as i}from"./gpu-contributor-utils-BIwLQON5.js";import{a,l as o,o as s,s as c,t as l}from"./raster-kernel-utils-DcIN7G5I.js";import{t as u}from"./define-run-B-brkCe4.js";import{i as d}from"./inputs-CJKKIRKK.js";import{c as f}from"./pointer-overlay-Qh8qflcA.js";import{aa as p}from"./entry-preview-D70TdSbK.js";import{t as m}from"./bands-CVDIl0aE.js";import{r as h}from"./gestures-C1AoqRjN.js";var g=4096,_=2**32,v=2**31-1;function y(e){let t=Math.floor(e),n=Math.round((e-t)*_);return n>=_&&(t+=1,n=0),{whole:t,fraction:n}}function b(e,t,n,i,a,o,s){let c=o/i,l=(a-n)/i,u=4*2**-52*(Math.abs(a)+Math.abs(n))/i+1e-12;for(let e=1;e<=g;e++){let t=c*e,n=Math.round(t),r=l*e,i=Math.round(r);if(n>=1&&Math.abs(t-n)<=1e-12*t&&Math.abs(r-i)<=e*u+1e-12*Math.abs(r)){if(Math.abs(i)+(s+1)*n<=v)return{kind:`rational`,denominator:e,numerator:n,offsetUnits:i+0};break}}let d=y(c),f=y(l);return r(e,Math.abs(f.whole)+(s+1)*(d.whole+1)<=v,`the destination ${t} extent, measured in source cells from the source origin, must stay below 2^31`),{kind:`fixed`,ratioWhole:d.whole,ratioFraction:d.fraction,offsetWhole:f.whole,offsetFraction:f.fraction,minimumOverlap:Math.max(1e-6*Math.min(1,c),s*2**-31)}}function x(e,n,r){return r.kind===`rational`?`let ${e}LowerUnits = ${r.offsetUnits} + i32(${n}) * ${r.numerator};
let ${e}UpperUnits = ${e}LowerUnits + ${r.numerator};
let ${e}LowerWhole = getFloorQuotient(${e}LowerUnits, ${r.denominator});
let ${e}UpperWhole = getFloorQuotient(${e}UpperUnits, ${r.denominator});
let ${e}LowerFraction = f32(${e}LowerUnits - ${e}LowerWhole * ${r.denominator}) / ${r.denominator}.0;
let ${e}UpperFraction = f32(${e}UpperUnits - ${e}UpperWhole * ${r.denominator}) / ${r.denominator}.0;
let ${e}MinimumOverlap = 0.0;`:`let ${e}Product = getUnsignedProduct(${n}, ${r.ratioFraction}u);
let ${e}LowerUnits = ${e}Product.y + ${r.offsetFraction}u;
let ${e}LowerCarry = select(0u, 1u, ${e}LowerUnits < ${e}Product.y);
let ${e}LowerWhole = ${r.offsetWhole} + i32(${n}) * ${r.ratioWhole} + i32(${e}Product.x + ${e}LowerCarry);
let ${e}UpperUnits = ${e}LowerUnits + ${r.ratioFraction}u;
let ${e}UpperCarry = select(0, 1, ${e}UpperUnits < ${e}LowerUnits);
let ${e}UpperWhole = ${e}LowerWhole + ${r.ratioWhole} + ${e}UpperCarry;
let ${e}LowerFraction = f32(${e}LowerUnits) * ${1/_};
let ${e}UpperFraction = f32(${e}UpperUnits) * ${1/_};
let ${e}MinimumOverlap = ${t(r.minimumOverlap)};`}var S=`
// Floor of numerator / denominator for a positive denominator (WGSL division truncates).
fn getFloorQuotient(numerator: i32, denominator: i32) -> i32 {
  let quotient = numerator / denominator;
  return select(quotient, quotient - 1, numerator - quotient * denominator < 0);
}
// Exact 64-bit product of two u32 values as (high, low).
fn getUnsignedProduct(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowLow = leftLow * rightLow;
  let lowHigh = leftLow * rightHigh;
  let highLow = leftHigh * rightLow;
  let highHigh = leftHigh * rightHigh;
  let middle = (lowLow >> 16u) + (lowHigh & 0xffffu) + (highLow & 0xffffu);
  let low = (lowLow & 0xffffu) | (middle << 16u);
  let high = highHigh + (lowHigh >> 16u) + (highLow >> 16u) + (middle >> 16u);
  return vec2<u32>(high, low);
}
// Length, in source cells, of the part of [lower, upper] inside the unit cell [0, 1]. Edges are
// given relative to the candidate cell as whole cells plus a fraction in [0, 1], so an edge that
// coincides with a cell boundary is exactly 0 or 1 and a cell that is only touched gets 0.
fn getUnitCellOverlap(lowerWhole: i32, lowerFraction: f32, upperWhole: i32, upperFraction: f32, minimumOverlap: f32) -> f32 {
  let lower = max(f32(lowerWhole) + lowerFraction, 0.0);
  let upper = min(f32(upperWhole) + upperFraction, 1.0);
  let overlap = upper - lower;
  return select(0.0, overlap, overlap > minimumOverlap);
}`,C=class{props;id;candidateWidth;candidateHeight;axisPlans;constructor(e){this.props=e,this.id=e.id??`raster-conservative-remapping`;for(let[t,n]of[[`source`,e.source],[`destination`,e.destination]])i(this.id,`${t}.width`,n.width),i(this.id,`${t}.height`,n.height),i(this.id,`${t} cell count`,n.width*n.height),r(this.id,n.origin.every(Number.isFinite)&&n.spacing.every(e=>Number.isFinite(Math.fround(e))&&Math.fround(e)>0),`${t}.origin must be finite and ${t}.spacing must be positive and finite in float32`);if(r(this.id,[`destination-area`,`covered-area`,`extensive`].includes(e.normalization),`normalization must be 'destination-area', 'covered-area' or 'extensive'`),r(this.id,[e.source,e.destination].every(e=>Number.isFinite(Math.fround(e.spacing[0]*e.spacing[1]))&&Math.fround(e.spacing[0]*e.spacing[1])>0),`cell areas (spacing[0] times spacing[1]) must be positive and finite in float32`),r(this.id,e.destination.origin.every((t,n)=>Number.isFinite(Math.fround(t-e.source.origin[n]))),`destination.origin minus source.origin must be finite in float32`),this.axisPlans=[b(this.id,`x`,e.source.origin[0],e.source.spacing[0],e.destination.origin[0],e.destination.spacing[0],e.destination.width),b(this.id,`y`,e.source.origin[1],e.source.spacing[1],e.destination.origin[1],e.destination.spacing[1],e.destination.height)],this.candidateWidth=Math.ceil(e.destination.spacing[0]/e.source.spacing[0])+1,this.candidateHeight=Math.ceil(e.destination.spacing[1]/e.source.spacing[1])+1,r(this.id,this.candidateWidth*this.candidateHeight<=256,`a destination cell may overlap at most 256 source cells (candidate window ${this.candidateWidth} by ${this.candidateHeight}); use a finer destination or coarser source spacing ratio`),r(this.id,e.weightMode===void 0||[`prepare`,`reuse`].includes(e.weightMode),`weightMode must be 'prepare' or 'reuse'`),r(this.id,e.weightMode!==`reuse`||e.preparedWeights!==void 0,`weightMode 'reuse' requires preparedWeights`),e.preparedWeights){let t=e.destination.width*e.destination.height*this.candidateWidth*this.candidateHeight;r(this.id,e.preparedWeights.indices.format===`uint32`&&e.preparedWeights.areas.format===`float32`,`preparedWeights.indices must be uint32 and preparedWeights.areas float32`),r(this.id,e.preparedWeights.indices.length===t&&e.preparedWeights.areas.length===t,`preparedWeights must hold ${t} entries (destination cells times candidate stride)`)}let t=e.source.width*e.source.height,n=e.destination.width*e.destination.height;r(this.id,e.values.length>=t&&(!e.mask||e.mask.length>=t),`values and mask must hold at least one entry per source cell`),r(this.id,Object.values(e.output).every(e=>e.length>=n),`output views must hold at least one entry per destination cell`),i(this.id,`destination cells times candidate stride`,n*this.candidateWidth*this.candidateHeight);let a=[e.values,...e.mask?[e.mask]:[]],o=e.preparedWeights?Object.values(e.preparedWeights):[],s=Object.values(e.output),l=e.weightMode===`reuse`;c(this.id,[...a,...l?o:[]],[...s,...l?[]:o],[...!l&&o.length?[{inputs:[],outputs:o}]:[],{inputs:[...a,...o],outputs:s}])}getCommandNodes(r){let{props:i,id:c}=this,u=[i.values,...i.mask?[i.mask]:[]],d=i.preparedWeights?Object.values(i.preparedWeights):[];s(c,r,[...u,...i.weightMode===`reuse`?d:[]],[...Object.values(i.output),...i.weightMode===`reuse`?[]:d]);let f=i.source,p=i.destination,m=p.width*p.height,h=this.candidateWidth*this.candidateHeight;o(r,c,[],i.preparedWeights?{}:{"prepared indices and areas scratch":m*h*4});let g=i.preparedWeights?.indices??e(r,`${c}-indices`,`uint32`,m*h),_=i.preparedWeights?.areas??e(r,`${c}-weights`,`float32`,m*h),[v,y]=this.axisPlans,b=i.weightMode===`reuse`?void 0:n(r,{id:`${c}-prepare`,operation:`GPURasterConservativeRemapping`,invocationCount:m,bindings:[a(`indices`,g,!0),a(`weights`,_,!0)],declarations:S,body:`let destinationColumn = index % ${p.width}u;
let destinationRow = index / ${p.width}u;
${x(`x`,`destinationColumn`,v)}
${x(`y`,`destinationRow`,y)}
let first = vec2<i32>(xLowerWhole, yLowerWhole);
for (var candidate = 0u; candidate < ${h}u; candidate++) {
  let coordinate = first + vec2<i32>(i32(candidate % ${this.candidateWidth}u), i32(candidate / ${this.candidateWidth}u));
  let slot = index * ${h}u + candidate;
  indices[indicesOffset + slot] = 0xffffffffu; weights[weightsOffset + slot] = 0.0;
  if (any(coordinate < vec2<i32>(0)) || any(coordinate >= vec2<i32>(${f.width}, ${f.height}))) { continue; }
  let overlapX = getUnitCellOverlap(xLowerWhole - coordinate.x, xLowerFraction, xUpperWhole - coordinate.x, xUpperFraction, xMinimumOverlap);
  let overlapY = getUnitCellOverlap(yLowerWhole - coordinate.y, yLowerFraction, yUpperWhole - coordinate.y, yUpperFraction, yMinimumOverlap);
  if (!(overlapX > 0.0 && overlapY > 0.0)) { continue; }
  indices[indicesOffset + slot] = u32(coordinate.y) * ${f.width}u + u32(coordinate.x);
  weights[weightsOffset + slot] = (overlapX * overlapY) * ${t(f.spacing[0]*f.spacing[1])};
}`}),C=n(r,{id:`${c}-apply`,operation:`GPURasterConservativeRemapping`,invocationCount:m,bindings:[a(`values`,i.values),...i.mask?[a(`mask`,i.mask)]:[],a(`indices`,g),a(`weights`,_),a(`result`,i.output.values,!0),a(`coveredArea`,i.output.coveredArea,!0),a(`status`,i.output.status,!0)],declarations:l,body:`var numerator = 0.0; var support = 0.0;
for (var candidate = 0u; candidate < ${h}u; candidate++) {
  let slot = index * ${h}u + candidate; let sourceIndex = indices[indicesOffset + slot];
  if (sourceIndex == 0xffffffffu) { continue; }
  let value = values[valuesOffset + sourceIndex];
  if (!finite(value) ${i.mask?`|| mask[maskOffset + sourceIndex] == 0u`:``}) { continue; }
  let area = weights[weightsOffset + slot]; numerator += value * area; support += area;
}
coveredArea[coveredAreaOffset + index] = support;
var value = missing(); var code = 1u;
if (support > 0.0) {
  value = numerator / ${i.normalization===`covered-area`?`support`:t(i.normalization===`extensive`?f.spacing[0]*f.spacing[1]:p.spacing[0]*p.spacing[1])};
  code = select(2u, 0u, finite(value));
}
result[resultOffset + index] = value; status[statusOffset + index] = code;`});return b?[b,C]:[C]}},w=[`destination-area`,`covered-area`,`extensive`];function T(e,t){let n=t.factor,r=-n*t.shift;return{spacing:n,origin:r,width:Math.ceil((e.columns-r)/n),height:Math.ceil((e.rows-r)/n)}}var E=u({parameters:p,rebuildOn:[`factor`,`normalization`,`shift`],caption:`A fine raster in; the area-weighted values of a coarse grid out.`,createInputs:({world:e})=>d(e,60),build(e,t,n){let r=t.cellSize,i=T(t,n),a=r*i.spacing,o=i.origin*r,s=i.width*i.height,c=e.upload(`source`,`float32`,t.values),l=e.allocate(`values`,`float32`,s),u=e.allocate(`covered-area`,`float32`,s),d=e.allocate(`status`,`uint32`,s);e.graph.add(new C({source:{width:t.columns,height:t.rows,origin:[0,0],spacing:[r,r]},destination:{width:i.width,height:i.height,origin:[o,o],spacing:[a,a]},values:c.view,normalization:w[Math.round(n.normalization)],output:{values:l.view,coveredArea:u.view,status:d.view}}));let f=[],p=t.cellSize;for(let e=0;e<=i.width;e++){let n=(i.origin+e*i.spacing)*p;n>=0&&n<=t.columns*p&&f.push(n,0,n,t.rows*p)}for(let e=0;e<=i.height;e++){let n=(i.origin+e*i.spacing)*p;n>=0&&n<=t.rows*p&&f.push(0,t.rows*p-n,t.columns*p,t.rows*p-n)}return{outputs:{source:c,result:l,coveredArea:u,overlap:e.allocate(`overlap`,`float32`,t.values.length),grid:e.upload(`destination-grid`,`float32x4`,Float32Array.from(f)),destination:i}}},async afterRun({inputs:e,outputs:t,upload:n}){let r=await t.result.buffer.readAsync(),i=new Float32Array(r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)),{origin:a,spacing:o,width:s,height:c}=t.destination,l=new Float32Array(e.values.length);for(let t=0;t<e.rows;t++)for(let n=0;n<e.columns;n++){let r=Math.min(s-1,Math.floor((n+.5-a)/o)),u=Math.min(c-1,Math.floor((t+.5-a)/o));l[t*e.columns+n]=i[u*s+r]}return{display:n(`display`,`float32`,l)}},interaction:{kind:`probe`,hint:`probe a coarse cell`,rest:[.5,.5],resolve:(e,{inputs:t,outputs:n,world:r,peek:i})=>{let{origin:a,spacing:o,width:s,height:c}=n.destination,l=h(t,r,e),u=Math.min(s-1,Math.floor((l.column+.5-a)/o)),d=Math.min(c-1,Math.floor((l.row+.5-a)/o)),p=a+u*o,m=a+d*o,g=new Float32Array(t.values.length).fill(NaN),_=0,v=0;for(let e=Math.max(0,Math.floor(m));e<Math.min(t.rows,Math.ceil(m+o));e++)for(let n=Math.max(0,Math.floor(p));n<Math.min(t.columns,Math.ceil(p+o));n++){let r=Math.max(0,Math.min(n+1,p+o)-Math.max(n,p))*Math.max(0,Math.min(e+1,m+o)-Math.max(e,m));r<=0||(g[e*t.columns+n]=r,v+=r,_++)}n.overlap.write(g);let y=i(n.result),b=y?y[d*s+u]:NaN,x=r.width/t.columns,S=r.height/t.rows;return{glyphs:[{shape:`box`,minimum:[Math.max(0,p)*x,r.height-Math.min(t.rows,m+o)*S],maximum:[Math.min(t.columns,p+o)*x,r.height-Math.max(0,m)*S]}],readout:`dest [${u}, ${d}] | ${_} cells | covered ${Math.round(100*v/(o*o))}% | ${f(b)}`}}},draw:({inputs:e,outputs:t,extra:n,values:r,theme:i})=>[m(e,t.source,{context:!0,range:[0,1]}),...n?[m(e,n.display,{ramp:`sequential`,range:r.normalization===2?[0,r.factor*r.factor]:[0,1],alpha:.85})]:[],{layer:`lines`,context:!0,segments:t.grid,color:[...i.ink,.45]},m(e,t.overlap,{ramp:`sequential`,range:[0,1],fade:!0})]});export{E as default};