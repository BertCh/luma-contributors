import{a as e,n as t,r as n}from"./trajectory-binding-validation-C89oJIAZ.js";import{r}from"./kernel-C3tfQOn4.js";import{a as i,m as a,v as o,y as s}from"./spatial-context-DwZEd4EZ.js";import{D as c,E as l,S as u,T as d,f,v as p,x as m,y as h}from"./gpu-contributor-utils-BIwLQON5.js";import{o as g,t as _}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as v}from"./geodesic-wgsl-I0CYDKcz.js";import{t as y}from"./movement-common-nOvANN43.js";var b=`GPUTrajectoryLocations`,x=Object.freeze({...i,measure:`distance`});function S(e,t=new Float32Array(4)){if(!Number.isFinite(Math.fround(e.epsilon))||Math.fround(e.epsilon)<=0)throw Error(`Location epsilon must be finite and positive`);return s(`GPUTrajectoryLocations`,e.spatialContext,e.epsilon),t.set([e.epsilon,0,0,0]),t}var C=class{id;props;spatialContext;maximumIterations;capacity;constructor(t){this.id=t.id??`trajectory-locations`,this.props=t;let r=this.id;this.spatialContext=o(r,t.spatialContext,x);let{centroids:i,tracks:a,count:s,parameters:p,output:m}=t;l(i,[`float32x2`],`${r} centroids`),l(a,[`uint32`],`${r} tracks`),l(s,[`uint32`],`${r} count`),l(p,[`float32`],`${r} parameters`),l(m.locationIds,[`uint32`],`${r} output.locationIds`),d(r,s.length>=1,`count must contain one uint32 row`),this.capacity=i.length,c(r,`tracks`,a,this.capacity),c(r,`output.locationIds`,m.locationIds,this.capacity),f(r,`staypoint capacity`,this.capacity),d(r,p.length>=4,`parameters must hold 4 float32 values`),m.centers&&(l(m.centers,[`float32x2`],`${r} output.centers`),c(r,`output.centers`,m.centers,this.capacity)),this.maximumIterations=t.maximumIterations??32,u(r,`maximumIterations`,this.maximumIterations,256),n(this.id,this.props.output),e(this.id,[{converged:m.status.converged,iterationLimitReached:m.status.iterationLimitReached}]),h(r,[m.locationIds,m.status.count,m.status.converged,m.status.iterationLimitReached,m.centers],[i,a,s,p])}getCommandNodes(e){let{props:n,id:i,capacity:o,maximumIterations:s}=this,{centroids:c,tracks:l,count:u,parameters:d,output:f}=n;m(i,e,[c,l,u,d,f.locationIds,f.status.count,f.status.converged,f.status.iterationLimitReached,f.centers]),p(e.device,i,`centroids`,o*8,`use fewer staypoints per graph`),t(e,this.id,this.props);let h=a(this.spatialContext),x=r(e,`${i}-labels-a`,`uint32`,o),S=r(e,`${i}-labels-b`,`uint32`,o),C=r(e,`${i}-group-starts`,`uint32`,o),w=r(e,`${i}-group-ends`,`uint32`,o),T=r(e,`${i}-changed`,`uint32`,1),E=h===void 0?`length(a - b)`:`geodesicHaversineDistance(a, b, ${h.toExponential(10)})`,D=`${h===void 0?``:v}
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(centroids[centroidsOffset + 2u * row], centroids[centroidsOffset + 2u * row + 1u]);
}
fn neighborDistance(a: vec2<f32>, b: vec2<f32>) -> f32 { return ${E}; }
`,O=[_(e,{id:`${i}-init-a`,operation:b,view:x,type:`u32`,value:`index`}),g(e,{id:`${i}-groups`,operation:b,invocationCount:1,bindings:[{name:`tracks`,view:l,type:`u32`,access:`read`},{name:`stayCount`,view:u,type:`u32`,access:`read`},{name:`groupStarts`,view:C,type:`u32`,access:`read_write`},{name:`groupEnds`,view:w,type:`u32`,access:`read_write`}],body:`let total = min(stayCount[stayCountOffset], ${o}u);
  var groupStart = 0u;
  for (var row = 0u; row <= total; row++) {
    if (row == total || (row > 0u && tracks[tracksOffset + row] != tracks[tracksOffset + row - 1u])) {
      for (var member = groupStart; member < row; member++) {
        groupStarts[groupStartsOffset + member] = groupStart;
        groupEnds[groupEndsOffset + member] = row;
      }
      groupStart = row;
    }
  }`})],k=(t,n,r,a)=>g(e,{id:`${i}-hook-${t}`,operation:b,variant:`hook`,invocationCount:o,bindings:[{name:`centroids`,view:c,type:`f32`,access:`read`},{name:`parameters`,view:d,type:`f32`,access:`read`},{name:`stayCount`,view:u,type:`u32`,access:`read`},{name:`groupStarts`,view:C,type:`u32`,access:`read`},{name:`groupEnds`,view:w,type:`u32`,access:`read`},{name:`labelsIn`,view:n,type:`u32`,access:`read`},...a?[{name:`changed`,view:T,type:`atomic<u32>`,access:`read_write`}]:[{name:`labelsOut`,view:r,type:`u32`,access:`read_write`}]],declarations:D,body:`if (index < min(stayCount[stayCountOffset], ${o}u)) {
    let center = point(index);
    var label = labelsIn[labelsInOffset + index];
    let radius = parameters[parametersOffset];
    for (var other = groupStarts[groupStartsOffset + index]; other < groupEnds[groupEndsOffset + index]; other++) {
      if (other != index && neighborDistance(center, point(other)) <= radius) {
        label = min(label, labelsIn[labelsInOffset + other]);
      }
    }
    ${a?`if (label != labelsIn[labelsInOffset + index]) { atomicStore(&changed[changedOffset], 1u); }`:`labelsOut[labelsOutOffset + index] = label;`}
  }${a?``:` else { labelsOut[labelsOutOffset + index] = ${y}; }`}`});for(let t=0;t<s;t++)O.push(k(`${t}`,x,S,!1),g(e,{id:`${i}-compress-${t}`,operation:b,variant:`compress`,invocationCount:o,bindings:[{name:`stayCount`,view:u,type:`u32`,access:`read`},{name:`labelsIn`,view:S,type:`u32`,access:`read`},{name:`labelsOut`,view:x,type:`u32`,access:`read_write`}],body:`if (index < min(stayCount[stayCountOffset], ${o}u)) {
    let label = labelsIn[labelsInOffset + index];
    labelsOut[labelsOutOffset + index] = labelsIn[labelsInOffset + label];
  } else {
    labelsOut[labelsOutOffset + index] = ${y};
  }`}));O.push(_(e,{id:`${i}-changed-reset`,operation:b,view:T,type:`u32`,value:`0u`,componentCount:1}),k(`check`,x,S,!0));let A=r(e,`${i}-id-map`,`uint32`,o);O.push(g(e,{id:`${i}-relabel`,operation:b,invocationCount:1,bindings:[{name:`stayCount`,view:u,type:`u32`,access:`read`},{name:`labels`,view:x,type:`u32`,access:`read`},{name:`idMap`,view:A,type:`u32`,access:`read_write`},{name:`locationCount`,view:f.status.count,type:`u32`,access:`read_write`}],body:`var nextId = 0u;
  for (var row = 0u; row < ${o}u; row++) {
    if (row < min(stayCount[stayCountOffset], ${o}u) && labels[labelsOffset + row] == row) {
      idMap[idMapOffset + row] = nextId;
      nextId++;
    }
  }
  locationCount[locationCountOffset] = nextId;`}),g(e,{id:`${i}-write-ids`,operation:b,invocationCount:o,bindings:[{name:`stayCount`,view:u,type:`u32`,access:`read`},{name:`labels`,view:x,type:`u32`,access:`read`},{name:`idMap`,view:A,type:`u32`,access:`read`},{name:`locationIds`,view:f.locationIds,type:`u32`,access:`read_write`}],body:`locationIds[locationIdsOffset + index] = select(${y}, idMap[idMapOffset + labels[labelsOffset + index]], index < min(stayCount[stayCountOffset], ${o}u));`})),f.centers&&O.push(g(e,{id:`${i}-centers`,operation:b,invocationCount:o,bindings:[{name:`centroids`,view:c,type:`f32`,access:`read`},{name:`stayCount`,view:u,type:`u32`,access:`read`},{name:`labels`,view:x,type:`u32`,access:`read`},{name:`groupEnds`,view:w,type:`u32`,access:`read`},{name:`idMap`,view:A,type:`u32`,access:`read`},{name:`centers`,view:f.centers,type:`f32`,access:`read_write`}],declarations:`fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(centroids[centroidsOffset + 2u * row], centroids[centroidsOffset + 2u * row + 1u]);
}
fn longitudeOffset(delta: f32) -> f32 { return ${h===void 0?`delta`:`delta - 360.0 * round(delta / 360.0)`}; }`,body:`let total = min(stayCount[stayCountOffset], ${o}u);
  if (index < total && labels[labelsOffset + index] == index) {
    let anchor = point(index);
    var offset = vec2<f32>(0.0, 0.0);
    var distinctCount = 0u;
    for (var row = index; row < groupEnds[groupEndsOffset + index]; row++) {
      if (labels[labelsOffset + row] != index) { continue; }
      let position = point(row);
      var isDuplicate = false;
      for (var earlier = index; earlier < row; earlier++) {
        if (labels[labelsOffset + earlier] != index) { continue; }
        let other = point(earlier);
        if (other.x == position.x && other.y == position.y) { isDuplicate = true; break; }
      }
      if (!isDuplicate) {
        offset += vec2<f32>(longitudeOffset(position.x - anchor.x), position.y - anchor.y);
        distinctCount++;
      }
    }
    let center = anchor + offset / f32(max(distinctCount, 1u));
    let locationIndex = idMap[idMapOffset + index];
    centers[centersOffset + 2u * locationIndex] = ${h===void 0?`center.x`:`((center.x + 180.0) - 360.0 * floor((center.x + 180.0) / 360.0)) - 180.0`};
    centers[centersOffset + 2u * locationIndex + 1u] = center.y;
  }`}));let{converged:j,iterationLimitReached:M}=f.status;return(j||M)&&O.push(g(e,{id:`${i}-status`,operation:b,invocationCount:1,bindings:[{name:`changed`,view:T,type:`u32`,access:`read`},...j?[{name:`converged`,view:j,type:`u32`,access:`read_write`}]:[],...M?[{name:`iterationLimitReached`,view:M,type:`u32`,access:`read_write`}]:[]],body:`let didChange = changed[changedOffset] != 0u;
  ${j?`converged[convergedOffset] = select(1u, 0u, didChange);`:``}
  ${M?`iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, didChange);`:``}`})),O}};export{S as n,C as t};