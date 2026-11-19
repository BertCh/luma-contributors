import{c as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n}from"./kernel-C3tfQOn4.js";import{T as r,x as i}from"./gpu-contributor-utils-BIwLQON5.js";import{r as a,u as o}from"./wgsl-common-DfyXiYB0.js";import{a as s,i as c,n as l,o as u,t as d}from"./fixed-point-accumulator-Ce6W3HVY.js";import{t as f}from"./gpu-prediction-utils-DjZFSzGW.js";import{n as p,s as m}from"./prediction-covariance-DfKlqgpX.js";var h=8,g=.001;function _(e){return e.model===`exponential`?0:g*e.sill}function v(e,t){return m(e),r(`GPUVecchiaLogLikelihood`,e.model===`exponential`||e.model===`matern-3/2`,`model must be exponential or matern-3/2`),r(`GPUVecchiaLogLikelihood`,(e.nugget??0)>=_(e)*.999999999999,`nugget must be at least ${g} of the sill for ${e.model}`),p({covariance:e},t)}function y(e,r){let{id:i,operation:o,rowCount:f,cellCount:p,terms:m,counters:h,totals:g}=r,_=n(e,`${i}-scale-word`,`uint32`,1),v=n(e,`${i}-words`,`uint32`,p*3),y=u(f);return[...c(e,{id:`${i}-scale`,operation:o,values:m,scale:_}),l(e,{id:`${i}-words-clear`,operation:o,words:v}),t(e,{id:`${i}-accumulate`,operation:o,variant:`accumulate`,bindings:[{name:`terms`,view:m,type:`f32`,access:`read`},{name:`scale`,view:_,type:`u32`,access:`read`},{name:`fixedWords`,view:v,type:`atomic<u32>`,access:`read_write`}],invocationCount:f*p,workgroupSize:64,declarations:`override FIXED_POINT_HEADROOM_BITS: u32;
${d}
${s(`fixedWords`)}`,constants:{FIXED_POINT_HEADROOM_BITS:y},body:`let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], FIXED_POINT_HEADROOM_BITS);
addFixedPointToFixedWords(index % ${p}u, quantizeFixedPoint(terms[termsOffset + index], unitExponent));`}),t(e,{id:`${i}-decode`,operation:o,variant:`decode`,bindings:[{name:`fixedWords`,view:v,type:`u32`,access:`read`},{name:`scale`,view:_,type:`u32`,access:`read`},{name:`counters`,view:h,type:`u32`,access:`read`},{name:`totals`,view:g,type:`f32`,access:`read_write`}],invocationCount:p,workgroupSize:16,declarations:`override FIXED_POINT_HEADROOM_BITS: u32;
${d}
${a}`,constants:{FIXED_POINT_HEADROOM_BITS:y},body:`if (counters[countersOffset] != 0u || counters[countersOffset + 2u] == 0u) {
  totals[totalsOffset + index * 2u] = getNaN();
  totals[totalsOffset + index * 2u + 1u] = getNaN();
  return;
}
let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], FIXED_POINT_HEADROOM_BITS);
let word = fixedWordsOffset + index * 3u;
let exact = vec3<u32>(fixedWords[word], fixedWords[word + 1u], fixedWords[word + 2u]);
let high = getFixedPointFloat(exact, unitExponent);
let remainder = addFixedPoint(exact, negateFixedPoint(quantizeFixedPoint(high, unitExponent)));
totals[totalsOffset + index * 2u] = high;
totals[totalsOffset + index * 2u + 1u] = getFixedPointFloat(remainder, unitExponent);`})]}function b(n,r){let{id:s,operation:c,rowCount:l,observations:u,neighborhoodOffsets:d,parameters:p,output:m}=r;i(s,n,[u,d,p,m.conditionals,m.conditionalStatus,m.siteTerms,m.siteStatus,m.logLikelihood,m.status]);let h=e(Math.log(2*Math.PI));return[t(n,{id:`${s}-clear-counts`,operation:c,variant:`clear-counts`,bindings:[{name:`counters`,view:m.status,type:`u32`,access:`read_write`}],invocationCount:3,workgroupSize:4,body:`counters[countersOffset + index] = 0u;`}),t(n,{id:`${s}-site-terms`,operation:c,variant:`site-terms`,bindings:[{name:`conditionals`,view:m.conditionals,type:`f32`,access:`read`},{name:`conditionalStatus`,view:m.conditionalStatus,type:`u32`,access:`read`},{name:`observations`,view:u,type:`f32`,access:`read`},{name:`offsets`,view:d,type:`u32`,access:`read`},{name:`params`,view:p,type:`f32`,access:`read`},{name:`siteTerms`,view:m.siteTerms,type:`f32`,access:`read_write`},{name:`siteStatus`,view:m.siteStatus,type:`u32`,access:`read_write`},{name:`counters`,view:m.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,workgroupSize:64,declarations:`${f({conditionals:4,observations:4})}
${a}
${o}
const LOG_TWO_PI: f32 = ${h};`,body:`siteTerms[siteTermsOffset + index] = getNaN();
let observation = read_observations(index);
if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0) {
  siteStatus[siteStatusOffset + index] = 1u;
  atomicAdd(&counters[countersOffset + 1u], 1u);
  return;
}
let isMarginal = offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u];
var mean = 0.0;
var variance = 0.0;
if (isMarginal) {
  if (!(params[paramsOffset] > 0.0)) {
    siteStatus[siteStatusOffset + index] = 2u;
    atomicAdd(&counters[countersOffset], 1u);
    return;
  }
  variance = params[paramsOffset];
} else {
  if (conditionalStatus[conditionalStatusOffset + index] != 0u) {
    siteStatus[siteStatusOffset + index] = 2u;
    atomicAdd(&counters[countersOffset], 1u);
    return;
  }
  let conditional = read_conditionals(index);
  mean = conditional.x;
  variance = conditional.y;
}
variance += observation.y;
let residual = observation.x - mean;
let quadratic = residual * residual / variance;
let term = -0.5 * (LOG_TWO_PI + log(variance) + quadratic);
if (!(variance > 0.0) || !isFiniteValue(variance) || !isFiniteValue(term)) {
  siteStatus[siteStatusOffset + index] = 3u;
  atomicAdd(&counters[countersOffset], 1u);
  return;
}
siteTerms[siteTermsOffset + index] = term;
siteStatus[siteStatusOffset + index] = 0u;
atomicAdd(&counters[countersOffset + 2u], 1u);`}),...y(n,{id:s,operation:c,rowCount:l,cellCount:1,terms:m.siteTerms,counters:m.status,totals:m.logLikelihood})]}var x=2;function S(e,t){let n=1/0,r=1/0,i=-1/0,a=-1/0;for(let o=0;o<t;o++)n=Math.min(n,e[o*2]),i=Math.max(i,e[o*2]),r=Math.min(r,e[o*2+1]),a=Math.max(a,e[o*2+1]);let o=Math.max(i-n,Number.MIN_VALUE),s=Math.max(a-r,Number.MIN_VALUE),c=Math.max(1,Math.round(t/x)),l=Math.max(Math.sqrt(o*s/c),Math.max(o,s)/4096),u=Math.floor(o/l)+1,d=Math.floor(s/l)+1;return{minimumX:n,minimumY:r,cellSize:l,columns:u,rows:d,heads:new Int32Array(u*d).fill(-1),next:new Int32Array(t).fill(-1)}}function C(e,t){return Math.min(e.columns-1,Math.floor((t-e.minimumX)/e.cellSize))}function w(e,t){return Math.min(e.rows-1,Math.floor((t-e.minimumY)/e.cellSize))}function T(e,t,n){let r=w(e,t[n*2+1])*e.columns+C(e,t[n*2]);e.next[n]=e.heads[r],e.heads[r]=n}function E(e,t,n,r,i){let a=C(e,t),o=w(e,n),s=Math.max(a,e.columns-1-a,o,e.rows-1-o);for(let t=0;t<=s;t++){let n=Math.max(t-1,0)*e.cellSize;if(n*n>r())break;for(let n=Math.max(o-t,0);n<=Math.min(o+t,e.rows-1);n++){let r=Math.abs(n-o)===t,s=a-t,c=a+t;if(r)for(let t=Math.max(s,0);t<=Math.min(c,e.columns-1);t++)for(let r=e.heads[n*e.columns+t];r>=0;r=e.next[r])i(r);else{if(s>=0)for(let t=e.heads[n*e.columns+s];t>=0;t=e.next[t])i(t);if(c<e.columns)for(let t=e.heads[n*e.columns+c];t>=0;t=e.next[t])i(t)}}}}function D(e){let{positions:t,neighborCount:n,order:r,validity:i,times:a,timeScale:o=0}=e,s=e.rowCount??t.length/2,c=S(t,s),l=new Uint32Array(s),u=new Uint32Array(s*n),d=new Uint32Array(s);for(let e=0;e<s;e++)d[r?r[e]:e]=e;let f=new Uint32Array(n),p=new Float64Array(n);for(let e=0;e<s;e++){let s=r?r[e]:e;if(i&&!i[s])continue;let m=t[s*2],h=t[s*2+1],g=0;E(c,m,h,()=>g<n?1/0:p[n-1],e=>{if(i&&!i[e])return;let r=m-t[e*2],c=h-t[e*2+1],l=a?o*(a[s]-a[e]):0,u=r*r+c*c+l*l,_=g<n?g:n;for(;_>0&&(p[_-1]>u||p[_-1]===u&&d[f[_-1]]>d[e]);)_<n&&(p[_]=p[_-1],f[_]=f[_-1]),_--;_<n&&(p[_]=u,f[_]=e,g<n&&g++)}),l[s]=g;for(let e=0;e<g;e++)u[s*n+e]=f[e];T(c,t,s)}let m=new Uint32Array(s+1);for(let e=0;e<s;e++)m[e+1]=m[e]+l[e];let h=new Uint32Array(m[s]);for(let e=0;e<s;e++)for(let t=0;t<l[e];t++)h[m[e]+t]=u[e*n+t];return{offsets:m,neighbors:h}}var O=class{keys;rows;size=0;constructor(e){this.keys=new Float64Array(e),this.rows=new Uint32Array(e)}isBefore(e,t){return this.keys[e]>this.keys[t]||this.keys[e]===this.keys[t]&&this.rows[e]<this.rows[t]}swap(e,t){let n=this.keys[e],r=this.rows[e];this.keys[e]=this.keys[t],this.rows[e]=this.rows[t],this.keys[t]=n,this.rows[t]=r}push(e,t){let n=this.size++;for(this.keys[n]=e,this.rows[n]=t;n>0;){let e=n-1>>1;if(!this.isBefore(n,e))break;this.swap(n,e),n=e}}pop(){if(this.size--,this.size>0){this.keys[0]=this.keys[this.size],this.rows[0]=this.rows[this.size];let e=0;for(;;){let t=2*e+1;if(t>=this.size)break;let n=t+1,r=n<this.size&&this.isBefore(n,t)?n:t;if(!this.isBefore(r,e))break;this.swap(r,e),e=r}}}};function k(e,t=e.length/2){let n=new Uint32Array(t);if(t===0)return n;let r=0,i=0;for(let n=0;n<t;n++)r+=e[n*2]/t,i+=e[n*2+1]/t;let a=0,o=1/0;for(let n=0;n<t;n++){let t=e[n*2]-r,s=e[n*2+1]-i,c=t*t+s*s;c<o&&(o=c,a=n)}let s=S(e,t),c=new Uint8Array(t);n[0]=a,c[a]=1,T(s,e,a);let l=new O(t);for(let n=0;n<t;n++)if(n!==a){let t=e[n*2]-e[a*2],r=e[n*2+1]-e[a*2+1];l.push(t*t+r*r,n)}let u=t=>{let n=e[t*2],r=e[t*2+1],i=1/0;return E(s,n,r,()=>i,t=>{let a=n-e[t*2],o=r-e[t*2+1];i=Math.min(i,a*a+o*o)}),i},d=1;for(;d<t;){let t=l.rows[0];if(l.pop(),c[t])continue;let r=u(t);l.size===0||r>l.keys[0]||r===l.keys[0]&&t<l.rows[0]?(n[d++]=t,c[t]=1,T(s,e,t)):l.push(r,t)}return n}export{h as a,b as i,k as n,g as o,y as r,v as s,D as t};