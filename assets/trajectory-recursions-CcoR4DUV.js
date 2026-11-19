import{a as e,n as t,r as n}from"./trajectory-binding-validation-C89oJIAZ.js";import{r}from"./kernel-C3tfQOn4.js";import{s as i,v as a}from"./spatial-context-DwZEd4EZ.js";import{D as o,E as s,S as c,T as l,f as u,v as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{o as m,t as h}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as g}from"./trajectory-common-DFp8lX7h.js";import{d as _,f as v,u as y}from"./trajectory-metrics-kernels-Dj0TkZQS.js";import{r as b,t as x}from"./movement-common-nOvANN43.js";import{t as S}from"./define-run-B-brkCe4.js";import{d as C,o as w,s as T}from"./pointer-overlay-Qh8qflcA.js";import{c as E,l as D,r as O,u as k}from"./movement-DZVB3aIP.js";import{n as A,t as j}from"./units-Boef4ppn.js";import{d as M}from"./entry-preview-D70TdSbK.js";var N=`GPUTrajectoryRecursions`,P=65536,F=class{id;rowCount;radiusCount;maximumTrackSegments;props;constructor(t){let r=t.id??`trajectory-recursions`;a(r,t.spatialContext,i);let{positions:d,timestamps:f,trackOffsets:m,radii:h,output:_}=t;s(d,[`float32x2`],`${r} positions`),s(f,[`float32`,`uint32x2`],`${r} timestamps`),s(m,[`uint32`],`${r} trackOffsets`),this.rowCount=d.length,u(r,`fix count`,this.rowCount),o(r,`timestamps`,f,this.rowCount),g(r,f,t.temporalContext),l(r,m.length>=2,`trackOffsets must contain at least two rows`),s(h,[`float32`],`${r} radii`),this.radiusCount=h.length,u(r,`radius count`,this.radiusCount,1,32),this.maximumTrackSegments=t.maximumTrackSegments??4096,u(r,`maximumTrackSegments`,this.maximumTrackSegments),c(r,`maximumTrackSegments`,this.maximumTrackSegments,P),l(r,!!(_.revisits||_.residenceTimes||_.firstPassageTimes),`output must request revisits, residenceTimes or firstPassageTimes`);let v=this.rowCount*this.radiusCount;for(let[e,t,n]of[[`revisits`,_.revisits,`uint32`],[`residenceTimes`,_.residenceTimes,`float32`],[`firstPassageTimes`,_.firstPassageTimes,`float32`],[`firstPassageCensored`,_.firstPassageCensored,`uint32`]])t&&(s(t,[n],`${r} output.${e}`),o(r,`output.${e}`,t,v));l(r,!_.firstPassageCensored||!!_.firstPassageTimes,`output.firstPassageCensored requires output.firstPassageTimes`),this.id=r,this.props=t,n(this.id,this.props.output),e(this.id,[{revisits:_.revisits,residenceTimes:_.residenceTimes},{firstPassageTimes:_.firstPassageTimes,firstPassageCensored:_.firstPassageCensored}]),p(r,this.getOutputViews(),[d,f,m,h])}getOutputViews(){let{output:e}=this.props;return[e.revisits,e.residenceTimes,e.firstPassageTimes,e.firstPassageCensored,e.status?.candidateOverflow]}getCommandNodes(e){t(e,this.id,this.props);let{id:n,props:i,rowCount:a,radiusCount:o,maximumTrackSegments:s}=this,{positions:c,timestamps:l,trackOffsets:u,radii:p,output:g}=i;f(n,e,[c,l,u,p,...this.getOutputViews()]),d(e.device,n,`output.revisits`,a*o*4,`use fewer fixes or radii per graph`);let S=r(e,`${n}-row-starts`,`uint32`,a),C=r(e,`${n}-row-ends`,`uint32`,a),w=r(e,`${n}-row-tracks`,`uint32`,a),T=_(i),E=`${v(T)}
const RADIUS_COUNT: u32 = ${o}u;
const CAP: u32 = ${s}u;
fn fixPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

// Segments [low, high) of the row's track that the row examines.
fn getWindow(row: u32) -> vec2<u32> {
  let start = rowStarts[rowStartsOffset + row];
  let end = rowEnds[rowEndsOffset + row];
  let segmentCount = end - start - 1u;
  if (segmentCount <= CAP) {
    return vec2<u32>(start, start + segmentCount);
  }
  let wanted = select(start, row - CAP / 2u, row >= start + CAP / 2u);
  let low = min(wanted, end - 1u - CAP);
  return vec2<u32>(low, low + CAP);
}

// Parameters of the closed disc inside a segment: (first, last, 1) or (0, 0, 0) when it misses.
fn getInsideInterval(pathStart: vec2<f32>, pathEnd: vec2<f32>, center: vec2<f32>, radius: f32) -> vec3<f32> {
  let step = pathEnd - pathStart;
  let offset = pathStart - center;
  let a = dot(step, step);
  let b = 2.0 * dot(offset, step);
  let c = dot(offset, offset) - radius * radius;
  if (a == 0.0) {
    return select(vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 1.0), c <= 0.0);
  }
  let discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    return vec3<f32>(0.0);
  }
  let root = sqrt(discriminant);
  let first = max((-b - root) / (2.0 * a), 0.0);
  let last = min((-b + root) / (2.0 * a), 1.0);
  return select(vec3<f32>(0.0), vec3<f32>(first, last, 1.0), first <= last);
}`,D={name:`rowStarts`,view:S,type:`u32`,access:`read`},O={name:`rowEnds`,view:C,type:`u32`,access:`read`},k=[...b(e,{id:`${n}-row-bounds`,operation:N,trackOffsets:u,rowCount:a,rowStarts:S,rowEnds:C,rowTracks:w})];return(g.revisits||g.residenceTimes)&&k.push(m(e,{id:`${n}-visits`,operation:N,variant:T,invocationCount:a*o,bindings:[{name:`positions`,view:c,type:`f32`,access:`read`},...y(i),D,O,{name:`radii`,view:p,type:`f32`,access:`read`},...g.revisits?[{name:`revisits`,view:g.revisits,type:`u32`,access:`read_write`}]:[],...g.residenceTimes?[{name:`residenceTimes`,view:g.residenceTimes,type:`f32`,access:`read_write`}]:[]],declarations:E,body:`
  let row = index / RADIUS_COUNT;
  let radius = radii[radiiOffset + index % RADIUS_COUNT];
  var visits = 0u;
  var residence = 0.0;
  if (rowEnds[rowEndsOffset + row] != ${x} && radius > 0.0) {
    let center = fixPosition(row);
    let window = getWindow(row);
    for (var segment = window.x; segment < window.y; segment = segment + 1u) {
      let pathStart = fixPosition(segment);
      let outside = distance(pathStart, center) > radius;
      if (segment == window.x && !outside) {
        visits = visits + 1u;
      }
      let interval = getInsideInterval(pathStart, fixPosition(segment + 1u), center, radius);
      if (interval.z != 0.0) {
        if (outside) {
          visits = visits + 1u;
        }
        let duration = timeDifferenceRows(segment + 1u, segment);
        if (duration > 0.0) {
          residence = residence + (interval.y - interval.x) * duration;
        }
      }
    }
  }
${g.revisits?`  revisits[revisitsOffset + index] = select(0u, visits - 1u, visits > 0u);`:``}
${g.residenceTimes?`  residenceTimes[residenceTimesOffset + index] = residence;`:``}`})),g.firstPassageTimes&&k.push(m(e,{id:`${n}-passage`,operation:N,variant:T,invocationCount:a*o,bindings:[{name:`positions`,view:c,type:`f32`,access:`read`},...y(i),D,O,{name:`radii`,view:p,type:`f32`,access:`read`},{name:`passageTimes`,view:g.firstPassageTimes,type:`f32`,access:`read_write`},...g.firstPassageCensored?[{name:`censored`,view:g.firstPassageCensored,type:`u32`,access:`read_write`}]:[]],declarations:E,body:`
  let row = index / RADIUS_COUNT;
  let radius = radii[radiiOffset + index % RADIUS_COUNT];
  var passage = 0.0;
  var isCensored = 1u;
  let trackEnd = rowEnds[rowEndsOffset + row];
  if (trackEnd != ${x} && radius > 0.0) {
    let center = fixPosition(row);
    let window = getWindow(row);
    let last = min(window.y, trackEnd - 1u);
    var segment = max(row, window.x);
    var found = false;
    for (; segment < last; segment = segment + 1u) {
      let pathEnd = fixPosition(segment + 1u);
      if (distance(pathEnd, center) > radius) {
        // The segment starts inside the disc and ends outside: it leaves at the larger root.
        let interval = getInsideInterval(fixPosition(segment), pathEnd, center, radius);
        passage = timeDifferenceRows(segment, row) + interval.y * timeDifferenceRows(segment + 1u, segment);
        isCensored = 0u;
        found = true;
        break;
      }
    }
    if (!found) {
      passage = timeDifferenceRows(last, row);
    }
  }
  passageTimes[passageTimesOffset + index] = passage;
${g.firstPassageCensored?`  censored[censoredOffset + index] = isCensored;`:`  _ = isCensored;`}`})),g.status?.candidateOverflow&&k.push(h(e,{id:`${n}-candidate-reset`,operation:N,view:g.status.candidateOverflow,type:`u32`,value:`0u`,componentCount:1}),m(e,{id:`${n}-candidate-overflow`,operation:N,variant:`overflow`,invocationCount:a,bindings:[D,O,{name:`candidateOverflow`,view:g.status.candidateOverflow,type:`atomic<u32>`,access:`read_write`}],declarations:`const CAP: u32 = ${s}u;`,body:`
  let end = rowEnds[rowEndsOffset + index];
  if (end != ${x} && end - rowStarts[rowStartsOffset + index] - 1u > CAP) {
    atomicStore(&candidateOverflow[candidateOverflowOffset], 1u);
  }`})),k}},I=6,L=S({parameters:M,caption:`Six Bay vessel tracks. Each fix is colored by the chosen column for the circle of radii[0] around it; the ring is drawn around the busiest fix. Distances and speeds are approximate: the frame stretches x and y differently.`,createInputs:({world:e})=>{let t=O(e,I),n=0,r=-1;for(let e=0;e<t.rowCount;e++){let i=0;for(let n=0;n<t.rowCount;n++)i+=+(Math.hypot(t.positions[e*2]-t.positions[n*2],t.positions[e*2+1]-t.positions[n*2+1])<.06);i>r&&(n=e,r=i)}return{...t,anchor:n}},build(e,t){let n=k(e,t),r=e.allocate(`radii`,`float32`,1),i=e.allocate(`revisits`,`uint32`,t.rowCount),a=e.allocate(`residence-times`,`float32`,t.rowCount),o=e.allocate(`first-passage-times`,`float32`,t.rowCount),s=e.allocate(`censored`,`uint32`,t.rowCount);return e.graph.add(new F({positions:n.positions.view,timestamps:n.timestamps.view,trackOffsets:n.trackOffsets.view,radii:r.view,output:{revisits:i.view,residenceTimes:a.view,firstPassageTimes:o.view,firstPassageCensored:s.view}})),{outputs:{segments:e.upload(`segments`,`float32x4`,t.segments),positions:n.positions,revisits:i,residenceTimes:a,firstPassageTimes:o,censored:s},update:e=>r.write(Float32Array.of(e.radius))}},async afterRun({outputs:e,values:t,upload:n}){let r=Math.round(t.metric);return{shown:n(`shown`,`float32`,r===0?Float32Array.from(await D(e.revisits)):await E(r===1?e.residenceTimes:e.firstPassageTimes))}},interaction:{kind:`reach`,hint:`reach from the anchor`,rest:({toUnit:e,inputs:t})=>e([t.positions[t.anchor*2]+M[0].default,t.positions[t.anchor*2+1]]),resolve(e,{inputs:t,outputs:n,peek:r}){let i=[t.positions[t.anchor*2],t.positions[t.anchor*2+1]],a=T(M,`radius`),o=C(a,w(Math.hypot(e.x-i[0],e.y-i[1]),a.min,a.max)),s=r(n.revisits),c=r(n.residenceTimes),l=r(n.firstPassageTimes),u=r(n.censored);return{values:{radius:o},glyphs:[{shape:`ring`,at:i,radius:o},{shape:`line`,from:i,to:[e.x,e.y],dashed:!0}],readout:s&&c&&l&&u?`radius ${j(o,t.metersPerUnit)} · ${s[t.anchor]} revisits · ${A(c[t.anchor])} inside · leaves in ${A(l[t.anchor])}${u[t.anchor]?`+`:``}`:`radius ${j(o,t.metersPerUnit)}`}}},draw:({inputs:e,outputs:t,extra:n,values:r})=>[{layer:`lines`,context:!0,segments:t.segments,count:e.segmentCount},...n?[{layer:`dots`,positions:t.positions,values:n.shown,radius:8,range:[0,[6,120,60][Math.round(r.metric)]]}]:[]]});export{L as default};