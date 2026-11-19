import{r as e}from"./kernel-C3tfQOn4.js";import{a as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{T as r,f as i,v as a}from"./gpu-contributor-utils-BIwLQON5.js";import{a as o,s,t as c}from"./raster-kernel-utils-DcIN7G5I.js";import{t as l}from"./raster-embedding-usable-mask-D4hp3S_i.js";import{b as u,o as d,r as f,s as p,t as m,x as h}from"./embedding-DKQ2dOol.js";import{t as g}from"./define-run-B-brkCe4.js";import{n as _,t as v}from"./inset-CADbSeau.js";import{r as y}from"./insets-BeGZ3fKP.js";import{oi as b}from"./entry-preview-D70TdSbK.js";var x=class{props;id;rangeCount;constructor(e){this.props=e;let t=e.id??`raster-embedding-covariance`;this.id=t;let{pixelCount:n,dimensions:a,output:o}=e;u(t,n,a),this.rangeCount=e.rangeCount??Math.min(256,n),i(t,`rangeCount`,this.rangeCount,1,256),h(t,`vectors`,e.vectors,n,a,e.mask),r(t,o.mean.length===a&&o.covariance.length===a*a&&(!o.usableCount||o.usableCount.length===1),`output.mean must hold dimensions entries, output.covariance dimensions squared and output.usableCount one`),s(t,this.getInputs(),this.getOutputs(),[{inputs:[],outputs:[o.mean,...o.usableCount?[o.usableCount]:[]]},{inputs:[],outputs:[o.covariance,...o.usableCount?[o.usableCount]:[]]}])}getInputs(){let{vectors:e,mask:t}=this.props;return[e,...t?[t]:[]]}getOutputs(){let{mean:e,covariance:t,usableCount:n}=this.props.output;return[e,t,...n?[n]:[]]}getCommandNodes(r){let{id:s,props:u,rangeCount:d}=this,{pixelCount:f,dimensions:p,output:m}=u;t(r,s,{vectors:u.vectors,mask:u.mask,"output.mean":m.mean,"output.covariance":m.covariance,"output.usableCount":m.usableCount});for(let[e,t]of Object.entries({"usable scratch":f,"partialSums scratch":d*p,"partialCounts scratch":d,"partialCovariance scratch":d*p*p,...m.usableCount?{}:{"usableCount scratch":1}})){let n=t*Uint32Array.BYTES_PER_ELEMENT;a(r.device,s,e,n,`reduce pixelCount, dimensions or rangeCount`),i(s,`${e} byteLength`,n,1,r.device.limits.maxBufferSize)}let h=`GPURasterEmbeddingCovariance`,g=Math.ceil(f/d),{usable:_,node:v}=l(r,{id:s,operation:h,pixelCount:f,dimensions:p,vectors:u.vectors,mask:u.mask}),y=e(r,`${s}-partial-sums`,`float32`,d*p),b=e(r,`${s}-partial-counts`,`uint32`,d),x=e(r,`${s}-partial-covariance`,`float32`,d*p*p),S=m.mean,C=m.usableCount??e(r,`${s}-usable-count`,`uint32`,1),w=`let rangeStart = range * ${g}u;
let rangeEnd = min(rangeStart + ${g}u, ${f}u);`;return[v,n(r,{id:`${s}-partial-sums`,operation:h,variant:`partial-sums`,invocationCount:d*p,workgroupSize:64,bindings:[o(`vectors`,u.vectors),o(`usable`,_),o(`partialSums`,y,!0),o(`partialCounts`,b,!0)],body:`let range = index / ${p}u;
let band = index % ${p}u;
${w}
var total = 0.0;
var count = 0u;
for (var pixel = rangeStart; pixel < rangeEnd; pixel++) {
  if (usable[usableOffset + pixel] != 0u) {
    total += vectors[vectorsOffset + pixel * ${p}u + band];
    count++;
  }
}
partialSums[partialSumsOffset + index] = total;
if (band == 0u) { partialCounts[partialCountsOffset + range] = count; }`}),n(r,{id:`${s}-mean`,operation:h,variant:`mean`,invocationCount:p,workgroupSize:64,declarations:c,bindings:[o(`partialSums`,y),o(`partialCounts`,b),o(`mean`,S,!0),o(`usableCount`,C,!0)],body:`var total = 0.0;
var count = 0u;
for (var range = 0u; range < ${d}u; range++) {
  total += partialSums[partialSumsOffset + range * ${p}u + index];
  count += partialCounts[partialCountsOffset + range];
}
mean[meanOffset + index] = select(missing(), total / f32(count), count > 0u);
if (index == 0u) { usableCount[usableCountOffset] = count; }`}),n(r,{id:`${s}-partial-covariance`,operation:h,variant:`partial-covariance`,invocationCount:d*p*p,workgroupSize:64,bindings:[o(`vectors`,u.vectors),o(`usable`,_),o(`mean`,S),o(`partialCovariance`,x,!0)],body:`let range = index / ${p*p}u;
let pair = index % ${p*p}u;
let row = pair / ${p}u;
let column = pair % ${p}u;
${w}
var total = 0.0;
// Only the upper triangle is computed; the reduction mirrors it.
if (column >= row) {
  let rowMean = mean[meanOffset + row];
  let columnMean = mean[meanOffset + column];
  for (var pixel = rangeStart; pixel < rangeEnd; pixel++) {
    if (usable[usableOffset + pixel] != 0u) {
      total += (vectors[vectorsOffset + pixel * ${p}u + row] - rowMean) *
        (vectors[vectorsOffset + pixel * ${p}u + column] - columnMean);
    }
  }
}
partialCovariance[partialCovarianceOffset + index] = total;`}),n(r,{id:`${s}-covariance`,operation:h,variant:`covariance`,invocationCount:p*p,workgroupSize:64,declarations:c,bindings:[o(`partialCovariance`,x),o(`usableCount`,C,!0),o(`covariance`,m.covariance,!0)],body:`let row = min(index / ${p}u, index % ${p}u);
let column = max(index / ${p}u, index % ${p}u);
var total = 0.0;
for (var range = 0u; range < ${d}u; range++) {
  total += partialCovariance[partialCovarianceOffset + range * ${p*p}u + row * ${p}u + column];
}
let count = usableCount[usableCountOffset];
covariance[covarianceOffset + index] = select(missing(), total / f32(count - 1u), count > 1u);`})]}},S=[.32,.55],C=[.66,.16,.96,.84],w=.6,T=g({parameters:b,caption:`Covariance of the vectors inside the box; the panel is the 6 x 6 matrix, bands blue to SWIR2 down and across. Six Sentinel-2 bands stand in for a learned embedding.`,createInputs:({aspect:e,random:t})=>{let n=f(e),r=Float32Array.from({length:n.vectors.length},()=>t()*2-1),i=n.columns/n.rows,a=(C[2]-C[0])*i,o=C[3]-C[1],s=Math.min(a,o)/6,c=C[0]*i+(a-s*6)/2,l=C[3]-(o-s*6)/2,u=Float32Array.from({length:72},(e,t)=>{let n=t>>1;return t&1?l-(Math.floor(n/6)+.5)*s:c+(n%6+.5)*s}),d=Math.floor(S[1]*n.rows)*n.columns+Math.floor(S[0]*n.columns);return{...n,noise:r,matrixPositions:u,center:d,worldWidth:i}},build(e,t){let n=e.allocate(`vectors`,`float32`,t.vectors.length),r=e.allocate(`validity`,`uint32`,t.pixelCount),i=e.allocate(`covariance`,`float32`,36),a=e.allocate(`usable-count`,`uint32`,1);return e.graph.add(new x({pixelCount:t.pixelCount,dimensions:6,vectors:n.view,mask:r.view,output:{mean:e.allocate(`mean`,`float32`,6).view,covariance:i.view,usableCount:a.view}})),{outputs:{covariance:i,usableCount:a,nearInfrared:e.upload(`near-infrared`,`float32`,t.nearInfrared),matrixPositions:e.upload(`matrix-positions`,`float32x2`,t.matrixPositions),backing:v(e,`panel`,y(C,t.worldWidth))},update:e=>{n.write(t.vectors.map((n,r)=>n+e.noise*t.noise[r]));let i=Math.round(e.windowRadius),a=t.center%t.columns,o=Math.floor(t.center/t.columns);r.write(Uint32Array.from({length:t.pixelCount},(e,n)=>+(Math.abs(n%t.columns-a)<=i&&Math.abs(Math.floor(n/t.columns)-o)<=i)))}}},interaction:{kind:`window`,hint:`move the window`,rest:S,resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=Math.round(r.windowRadius),s=w*i.width-(o+.5)/t.rows,{index:c}=d(t,i,Math.min(e.x,s),e.y),l=c!==t.center;t.center=c;let u=a(n.covariance),f=a(n.usableCount)?.[0]??0,h={row:0,column:1,value:NaN};if(u)for(let e=0;e<6;e++)for(let t=e+1;t<6;t++){let n=u[e*6+t];Math.abs(n)<=Math.abs(h.value)||(h={row:e,column:t,value:n})}return{rerun:l,spotlight:null,glyphs:[p(t,i,c,o)],readout:`strongest cov(${m[h.row]}, ${m[h.column]}) ${Number.isFinite(h.value)?h.value.toFixed(3):`-`} · usableCount ${f}`}}},draw:({inputs:e,outputs:t,theme:n})=>[{layer:`raster`,context:!0,values:t.nearInfrared,columns:e.columns,rows:e.rows,ramp:`grey`,smoothing:!0},_(t.backing,n),{layer:`dots`,positions:t.matrixPositions,values:t.covariance,ramp:`diverging`,range:[-.15,.15],radius:16}]});export{T as default};