import{n as e,r as t}from"./trajectory-binding-validation-C89oJIAZ.js";import{r as n}from"./kernel-C3tfQOn4.js";import{s as r,v as i}from"./spatial-context-DwZEd4EZ.js";import{D as a,E as o,S as s,T as c,f as l,g as u,v as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{o as m}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as h}from"./trajectory-common-DFp8lX7h.js";import{d as g,f as _,u as v}from"./trajectory-metrics-kernels-Dj0TkZQS.js";import{r as y,t as b}from"./movement-common-nOvANN43.js";import{r as x}from"./movement-DZVB3aIP.js";var S=`GPUBrownianBridgeUtilization`,C=16384;function w(e,t=new Float32Array(4)){if(t.length<4)throw Error(`${S} parameter target must hold 4 elements`);let[n,r,i,a]=e.extent;for(let t of e.extent)u(S,`extent`,t);return c(S,i>n&&a>r,`extent must have max > min`),t.set([n,r,i,a]),t}var T=class{id;rowCount;subStepCount;props;constructor(e){let n=e.id??`brownian-bridge-utilization`;i(n,e.spatialContext,r);let{positions:u,timestamps:d,trackOffsets:f,gridSize:m,output:g}=e;o(u,[`float32x2`],`${n} positions`),o(d,[`float32`,`uint32x2`],`${n} timestamps`),o(f,[`uint32`],`${n} trackOffsets`),this.rowCount=u.length,l(n,`fix count`,this.rowCount,2),s(n,`fix count`,this.rowCount,C),a(n,`timestamps`,d,this.rowCount),h(n,d,e.temporalContext),c(n,f.length>=2,`trackOffsets must contain at least two rows`),o(e.motionVariances,[`float32`],`${n} motionVariances`),a(n,`motionVariances`,e.motionVariances,this.rowCount),o(e.locationErrors,[`float32`],`${n} locationErrors`),a(n,`locationErrors`,e.locationErrors,this.rowCount),l(n,`gridSize columns`,m[0]),l(n,`gridSize rows`,m[1]),o(e.parameters,[`float32`],`${n} parameters`),c(n,e.parameters.length>=4,`parameters must contain 4 float32 rows`),this.subStepCount=e.subStepCount??15,l(n,`subStepCount`,this.subStepCount,1,256),o(g.density,[`float32`],`${n} output.density`),a(n,`output.density`,g.density,m[0]*m[1]),t(n,e.output),p(n,[g.density],[u,d,f,e.motionVariances,e.locationErrors,e.parameters]),this.id=n,this.props=e}getCommandNodes(t){e(t,this.id,this.props);let{id:r,props:i,rowCount:a,subStepCount:o}=this,{positions:s,timestamps:c,trackOffsets:l,gridSize:u,output:p}=i;f(r,t,[s,c,l,i.motionVariances,i.locationErrors,i.parameters,p.density]),d(t.device,r,`output.density`,u[0]*u[1]*4,`use a smaller grid`);let h=n(t,`${r}-row-starts`,`uint32`,a),x=n(t,`${r}-row-ends`,`uint32`,a),C=n(t,`${r}-row-tracks`,`uint32`,a),w=g(i);return[...y(t,{id:`${r}-row-bounds`,operation:S,trackOffsets:l,rowCount:a,rowStarts:h,rowEnds:x,rowTracks:C}),m(t,{id:`${r}-density`,operation:S,variant:w,invocationCount:u[0]*u[1],bindings:[{name:`positions`,view:s,type:`f32`,access:`read`},...v(i),{name:`rowEnds`,view:x,type:`u32`,access:`read`},{name:`motionVariances`,view:i.motionVariances,type:`f32`,access:`read`},{name:`locationErrors`,view:i.locationErrors,type:`f32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`density`,view:p.density,type:`f32`,access:`read_write`}],declarations:`${_(w)}
const COLUMNS: u32 = ${u[0]}u;
const ROWS: u32 = ${u[1]}u;
const FIX_COUNT: u32 = ${a}u;
const SUB_STEPS: u32 = ${o}u;
const CUTOFF: f32 = 7.0;
const TWO_PI: f32 = 6.283185307179586;
fn fixPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`
  let minimum = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let maximum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let cellSize = (maximum - minimum) / vec2<f32>(f32(COLUMNS), f32(ROWS));
  let center = minimum + (vec2<f32>(f32(index % COLUMNS), f32(index / COLUMNS)) + vec2<f32>(0.5)) * cellSize;
  var weightedDensity = 0.0;
  var totalDuration = 0.0;
  for (var row = 0u; row + 1u < FIX_COUNT; row = row + 1u) {
    let trackEnd = rowEnds[rowEndsOffset + row];
    if (trackEnd == ${b} || row + 1u >= trackEnd) {
      continue;
    }
    let duration = timeDifferenceRows(row + 1u, row);
    if (!(duration > 0.0)) {
      continue;
    }
    totalDuration = totalDuration + duration;
    let pathStart = fixPosition(row);
    let pathEnd = fixPosition(row + 1u);
    let motion = max(motionVariances[motionVariancesOffset + row], 0.0);
    let errorFrom = locationErrors[locationErrorsOffset + row];
    let errorTo = locationErrors[locationErrorsOffset + row + 1u];
    let largestError = max(abs(errorFrom), abs(errorTo));
    let reach = CUTOFF * sqrt(0.25 * duration * motion + largestError * largestError);
    let lower = min(pathStart, pathEnd) - vec2<f32>(reach);
    let upper = max(pathStart, pathEnd) + vec2<f32>(reach);
    if (any(center < lower) || any(center > upper)) {
      continue;
    }
    var segmentDensity = 0.0;
    for (var step = 0u; step < SUB_STEPS; step = step + 1u) {
      let fraction = (f32(step) + 0.5) / f32(SUB_STEPS);
      let mean = pathStart + fraction * (pathEnd - pathStart);
      let variance = duration * fraction * (1.0 - fraction) * motion +
        (1.0 - fraction) * (1.0 - fraction) * errorFrom * errorFrom +
        fraction * fraction * errorTo * errorTo;
      if (variance > 0.0) {
        let offset = center - mean;
        segmentDensity = segmentDensity + exp(-dot(offset, offset) / (2.0 * variance)) / (TWO_PI * variance);
      }
    }
    weightedDensity = weightedDensity + duration * segmentDensity / f32(SUB_STEPS);
  }
  density[densityOffset + index] = select(0.0, weightedDensity / totalDuration, totalDuration > 0.0);`})]}};function E(e){let t=x(e,8),n=0,r=-1;for(let e=0;e<t.trackCount;e++){let i=0;for(let n=t.trackOffsets[e];n<t.trackOffsets[e+1]-1;n++)i+=Math.hypot(t.positions[n*2+2]-t.positions[n*2],t.positions[n*2+3]-t.positions[n*2+1]);i>r&&(n=e,r=i)}let i=t.trackOffsets[n],a=t.trackOffsets[n+1],o=t.positions.slice(i*2,a*2),s={...t,trackCount:1,rowCount:a-i,positions:o,timestamps:t.timestamps.slice(i,a),trackOffsets:Uint32Array.of(0,a-i),segments:t.segments.slice((i-n)*4,(a-n-1)*4),segmentCount:a-i-1},c=Float32Array.from(o,(e,t)=>t%2==1?1-e:e);return{...s,world:e,columns:Math.round(64*e.width),rows:64,mirrored:c}}function D(e,t){let n=e.allocate(`motion-variances`,`float32`,t.rowCount),r=e.allocate(`location-errors`,`float32`,t.rowCount),i=e.allocate(`bridge-parameters`,`float32`,4),a=e.allocate(`density`,`float32`,t.columns*t.rows);return e.graph.add(new T({positions:e.upload(`positions`,`float32x2`,t.mirrored).view,timestamps:e.upload(`timestamps`,`float32`,t.timestamps).view,trackOffsets:e.upload(`offsets`,`uint32`,t.trackOffsets).view,motionVariances:n.view,locationErrors:r.view,gridSize:[t.columns,t.rows],parameters:i.view,subStepCount:15,output:{density:a.view}})),{density:a,segments:e.upload(`segments`,`float32x4`,t.segments),write(e,a){i.write(w({extent:[0,0,t.world.width,t.world.height]})),n.write(new Float32Array(t.rowCount).fill(e)),r.write(new Float32Array(t.rowCount).fill(a))}}}function O(e,t,n){let{columns:r,rows:i,positions:a,timestamps:o,trackOffsets:s,world:c}=e,l=new Float64Array(r*i),u=0,d=n*n;for(let e=0;e<s.length-1;e++)for(let n=s[e];n<s[e+1]-1;n++){let e=o[n+1]-o[n];if(!(e<=0)){u+=e;for(let o=0;o<15;o++){let s=(o+.5)/15,u=e*s*(1-s)*t+((1-s)**2+s**2)*d;if(u<=0)continue;let f=a[n*2]+s*(a[n*2+2]-a[n*2]),p=a[n*2+1]+s*(a[n*2+3]-a[n*2+1]);for(let t=0;t<i;t++){let n=1-(t+.5)/i;for(let i=0;i<r;i++){let a=((i+.5)/r*c.width-f)**2+(n-p)**2;a>49*u||(l[t*r+i]+=e/15*Math.exp(-a/(2*u))/(2*Math.PI*u))}}}}}return Float32Array.from(l,e=>e/u)}export{E as n,O as r,D as t};