import{a as e,i as t,n,r}from"./trajectory-binding-validation-C89oJIAZ.js";import{r as i}from"./kernel-C3tfQOn4.js";import{t as a}from"./gpu-compaction-B6IzcF0P.js";import{a as o,m as s,v as c,y as l}from"./spatial-context-DwZEd4EZ.js";import{D as u,E as d,S as f,T as p,f as m,m as h,v as g,x as _,y as v}from"./gpu-contributor-utils-BIwLQON5.js";import{n as y,o as b,t as x}from"./wgsl-kernel-nodes-DagJklH2.js";import{o as S}from"./trajectory-common-DFp8lX7h.js";import{d as C,f as w,u as T}from"./trajectory-metrics-kernels-Dj0TkZQS.js";import{a as E,i as D,n as O,o as k,r as A,t as j}from"./movement-common-nOvANN43.js";import{t as M}from"./construction-snapshot-DdkX9Pc0.js";import{t as N}from"./sf-bay-ais-CkxLNyrO.js";import{n as P,t as F}from"./fleet-Cm7PfbI5.js";var I=`GPUTrajectoryStaypoints`,L=Object.freeze({...o,measure:`distance`}),R=1<<24;function z(e,t=new Float32Array(4)){let{distanceThreshold:n,timeThreshold:r,maximumGap:i=0,includeLast:a=!1}=e;if(t.length<4)throw Error(`Staypoint target must hold 4 elements`);if(!Number.isFinite(Math.fround(n))||Math.fround(n)<=0)throw Error(`Staypoint distanceThreshold must be finite and positive`);if(!Number.isFinite(Math.fround(r))||r<0)throw Error(`Staypoint timeThreshold must be finite and non-negative`);if(!Number.isFinite(Math.fround(i))||i<0)throw Error(`Staypoint maximumGap must be finite and non-negative`);return l(`GPUTrajectoryStaypoints`,e.spatialContext,n),t.set([n,r,i,+!!a]),t}var B=class{id;props;spatialContext;maximumSearchRows;constructor(t){this.id=t.id??`trajectory-staypoints`,this.props=t;let n=this.id;this.spatialContext=c(n,t.spatialContext,L);let{output:i,positions:a,timestamps:o,trackOffsets:s,parameters:l}=t;d(a,[`float32x2`],`${n} positions`),d(o,[`float32`,`uint32x2`],`${n} timestamps`),d(s,[`uint32`],`${n} trackOffsets`),d(l,[`float32`],`${n} parameters`),u(n,`timestamps`,o,a.length),p(n,s.length>=2,`trackOffsets must contain at least two rows`),S(n,o,t.temporalContext),p(n,l.length>=4,`parameters must hold 4 float32 values`),this.maximumSearchRows=t.maximumSearchRows??4096,f(n,`maximumSearchRows`,this.maximumSearchRows,R),h(n,i),m(n,`staypoint capacity`,i.ids.length);let g=i.ids.length;for(let[e,t,r]of[[`startRows`,i.startRows,`uint32`],[`endRows`,i.endRows,`uint32`],[`finishRows`,i.finishRows,`uint32`],[`centroids`,i.centroids,`float32x2`],[`durations`,i.durations,`float32`]])t&&(d(t,[r],`${n} output.${e}`),u(n,`output.${e}`,t,g));i.staypointIds&&(d(i.staypointIds,[`uint32`],`${n} output.staypointIds`),u(n,`output.staypointIds`,i.staypointIds,a.length)),r(this.id,this.props.output),e(this.id,[{ids:i.ids,startRows:i.startRows},{endRows:i.endRows,finishRows:i.finishRows},{candidateOverflow:i.status.candidateOverflow},{count:i.status.count,overflow:i.status.overflow,requiredCount:i.status.requiredCount,drawInstanceCount:i.drawInstanceCount}]),v(n,[i.durations],[i.startRows,i.finishRows]),v(n,[i.centroids,i.staypointIds],[i.startRows,i.endRows]),v(n,this.getOutputViews(),this.getInputViews())}getInputViews(){let{positions:e,timestamps:t,trackOffsets:n,parameters:r}=this.props;return[e,t,n,r]}getOutputViews(){let{output:e}=this.props;return[e.ids,e.status.count,e.status.overflow,e.status.requiredCount,e.status.candidateOverflow,e.drawInstanceCount,e.startRows,e.endRows,e.finishRows,e.centroids,e.durations,e.staypointIds]}getCommandNodes(e){let{props:t,id:r}=this,{output:o,positions:c,trackOffsets:l,parameters:u}=t;_(r,e,[...this.getInputViews(),...this.getOutputViews()]);let d=c.length,f=o.ids.length;g(e.device,r,`positions`,d*8,`split the tracks over several graphs`),n(e,this.id,this.props);let p=C(t),m=s(this.spatialContext),h=T(t),v=i(e,`${r}-row-starts`,`uint32`,d),S=i(e,`${r}-row-ends`,`uint32`,d),E=i(e,`${r}-row-tracks`,`uint32`,d),M=i(e,`${r}-next`,`uint32`,d),N=i(e,`${r}-flags`,`uint32`,d),P=i(e,`${r}-reach`,`uint32`,d),F=i(e,`${r}-row-ids`,`uint32`,d),L=i(e,`${r}-stay-flags`,`uint32`,d),R=i(e,`${r}-stay-rows`,`uint32`,d),z=i(e,`${r}-stay-total`,`uint32`,1),B=[...A(e,{id:`${r}-row-bounds`,operation:I,trackOffsets:l,rowCount:d,rowStarts:v,rowEnds:S,rowTracks:E}),b(e,{id:`${r}-search`,operation:I,variant:p,invocationCount:d,bindings:[{name:`positions`,view:c,type:`f32`,access:`read`},...h,{name:`parameters`,view:u,type:`f32`,access:`read`},{name:`rowEnds`,view:S,type:`u32`,access:`read`},{name:`next`,view:M,type:`u32`,access:`read_write`},{name:`flags`,view:N,type:`u32`,access:`read_write`}],declarations:`${w(p)}\n${k(m)}`,body:`let distanceThreshold = parameters[parametersOffset];
  let timeThreshold = parameters[parametersOffset + 1u];
  let maximumGap = parameters[parametersOffset + 2u];
  let includeLast = parameters[parametersOffset + 3u] != 0.0;
  let trackEnd = rowEnds[rowEndsOffset + index];
  var successor = index;
  var state = 0u;
  if (trackEnd != ${j}) {
    let searchEnd = min(trackEnd, index + 1u + ${this.maximumSearchRows}u);
    var decided = false;
    for (var row = index + 1u; row < searchEnd; row++) {
      if (maximumGap > 0.0 && timeDifferenceRows(row, row - 1u) > maximumGap) {
        successor = row;
        decided = true;
        break;
      }
      if (rowDistance(index, row) >= distanceThreshold) {
        successor = row;
        if (timeDifferenceRows(row, index) >= timeThreshold) { state = 1u; }
        decided = true;
        break;
      }
    }
    if (!decided) {
      if (searchEnd == trackEnd) {
        if (includeLast && timeDifferenceRows(trackEnd - 1u, index) >= timeThreshold) { state = 5u; }
      } else {
        successor = searchEnd;
        state = 2u;
        if (timeDifferenceRows(searchEnd, index) >= timeThreshold) { state = 3u; }
      }
    }
  }
  next[nextOffset + index] = successor;
  flags[flagsOffset + index] = state;`}),...O(e,{id:`${r}-chain`,operation:I,trackOffsets:l,next:M,reach:P,rowCount:d}),b(e,{id:`${r}-select`,operation:I,invocationCount:d,bindings:[{name:`reach`,view:P,type:`u32`,access:`read`},{name:`flags`,view:N,type:`u32`,access:`read`},{name:`stayFlags`,view:L,type:`u32`,access:`read_write`}],body:`stayFlags[stayFlagsOffset + index] = select(0u, 1u, reach[reachOffset + index] != 0u && (flags[flagsOffset + index] & 1u) != 0u);`})];return o.status.candidateOverflow&&B.push(x(e,{id:`${r}-candidate-reset`,operation:I,view:o.status.candidateOverflow,type:`u32`,value:`0u`,componentCount:1}),b(e,{id:`${r}-candidate-overflow`,operation:I,invocationCount:d,bindings:[{name:`reach`,view:P,type:`u32`,access:`read`},{name:`flags`,view:N,type:`u32`,access:`read`},{name:`candidateOverflow`,view:o.status.candidateOverflow,type:`atomic<u32>`,access:`read_write`}],body:`if (reach[reachOffset + index] != 0u && (flags[flagsOffset + index] & 2u) != 0u) { atomicStore(&candidateOverflow[candidateOverflowOffset], 1u); }`})),B.push(D(e,{id:`${r}-row-ids`,operation:I,rowIds:F}),...new a({id:`${r}-compact`,input:F,flags:L,output:R,count:z}).getCommandNodes(e),...this.createGatherNodes(e,{rowCount:d,capacity:f,stayRows:R,stayTotal:z,next:M,flags:N,rowEnds:S,rowTracks:E}),y(e,{id:`${r}-publish`,operation:I,requiredCount:z,output:{ids:o.ids,status:{count:o.status.count,overflow:o.status.overflow,requiredCount:o.status.requiredCount}},extraCounts:o.drawInstanceCount?[o.drawInstanceCount]:[]})),B}createGatherNodes(e,t){let{id:n,props:r}=this,{output:a,positions:o}=r,{rowCount:c,capacity:l,stayRows:u,stayTotal:d,next:f,flags:p,rowEnds:m,rowTracks:h}=t,g=a.startRows??i(e,`${n}-start-rows`,`uint32`,l),_=a.endRows??i(e,`${n}-end-rows`,`uint32`,l),v=a.finishRows??i(e,`${n}-finish-rows`,`uint32`,l),y=s(this.spatialContext),S=C(r),E=[b(e,{id:`${n}-gather-ends`,operation:I,invocationCount:l,bindings:[{name:`stayRows`,view:u,type:`u32`,access:`read`},{name:`stayTotal`,view:d,type:`u32`,access:`read`},{name:`next`,view:f,type:`u32`,access:`read`},{name:`flags`,view:p,type:`u32`,access:`read`},{name:`rowEnds`,view:m,type:`u32`,access:`read`},{name:`endRows`,view:_,type:`u32`,access:`read_write`},{name:`finishRows`,view:v,type:`u32`,access:`read_write`}],body:`if (index < stayTotal[stayTotalOffset]) {
    let row = stayRows[stayRowsOffset + index];
    let isTrackEnd = (flags[flagsOffset + row] & 4u) != 0u;
    let trackEnd = rowEnds[rowEndsOffset + row];
    endRows[endRowsOffset + index] = select(next[nextOffset + row], trackEnd, isTrackEnd);
    finishRows[finishRowsOffset + index] = select(next[nextOffset + row], trackEnd - 1u, isTrackEnd);
  } else {
    endRows[endRowsOffset + index] = ${j};
    finishRows[finishRowsOffset + index] = ${j};
  }`}),b(e,{id:`${n}-gather-rows`,operation:I,invocationCount:l,bindings:[{name:`stayRows`,view:u,type:`u32`,access:`read`},{name:`stayTotal`,view:d,type:`u32`,access:`read`},{name:`rowTracks`,view:h,type:`u32`,access:`read`},{name:`ids`,view:a.ids,type:`u32`,access:`read_write`},{name:`startRows`,view:g,type:`u32`,access:`read_write`}],body:`if (index < stayTotal[stayTotalOffset]) {
    let row = stayRows[stayRowsOffset + index];
    ids[idsOffset + index] = rowTracks[rowTracksOffset + row];
    startRows[startRowsOffset + index] = row;
  } else {
    ids[idsOffset + index] = ${j};
    startRows[startRowsOffset + index] = ${j};
  }`})];return a.durations&&E.push(b(e,{id:`${n}-durations`,operation:I,variant:S,invocationCount:l,bindings:[...T(r),{name:`stayTotal`,view:d,type:`u32`,access:`read`},{name:`startRows`,view:g,type:`u32`,access:`read`},{name:`finishRows`,view:v,type:`u32`,access:`read`},{name:`durations`,view:a.durations,type:`f32`,access:`read_write`}],declarations:w(S),body:`durations[durationsOffset + index] = select(0.0, timeDifferenceRows(finishRows[finishRowsOffset + index], startRows[startRowsOffset + index]), index < stayTotal[stayTotalOffset]);`})),a.centroids&&E.push(b(e,{id:`${n}-centroids`,operation:I,invocationCount:l,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},{name:`stayTotal`,view:d,type:`u32`,access:`read`},{name:`startRows`,view:g,type:`u32`,access:`read`},{name:`endRows`,view:_,type:`u32`,access:`read`},{name:`centroids`,view:a.centroids,type:`f32`,access:`read_write`}],declarations:`fn longitudeOffset(delta: f32) -> f32 { return ${y===void 0?`delta`:`delta - 360.0 * round(delta / 360.0)`}; }
fn rowPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`var centroid = vec2<f32>(0.0, 0.0);
  if (index < stayTotal[stayTotalOffset]) {
    let start = startRows[startRowsOffset + index];
    let end = endRows[endRowsOffset + index];
    let anchor = rowPosition(start);
    var offset = vec2<f32>(0.0, 0.0);
    var distinctCount = 0u;
    for (var row = start; row < end; row++) {
      let position = rowPosition(row);
      var isDuplicate = false;
      for (var earlier = start; earlier < row; earlier++) {
        let other = rowPosition(earlier);
        if (other.x == position.x && other.y == position.y) { isDuplicate = true; break; }
      }
      if (!isDuplicate) {
        offset += vec2<f32>(longitudeOffset(position.x - anchor.x), position.y - anchor.y);
        distinctCount++;
      }
    }
    centroid = anchor + offset / f32(max(distinctCount, 1u));
    centroid.x = ${y===void 0?`centroid.x`:`((centroid.x + 180.0) - 360.0 * floor((centroid.x + 180.0) / 360.0)) - 180.0`};
  }
  centroids[centroidsOffset + index * 2u] = centroid.x;
  centroids[centroidsOffset + index * 2u + 1u] = centroid.y;`})),a.staypointIds&&E.push(x(e,{id:`${n}-row-ids-reset`,operation:I,view:a.staypointIds,type:`u32`,value:j}),b(e,{id:`${n}-row-staypoints`,operation:I,invocationCount:l,bindings:[{name:`stayTotal`,view:d,type:`u32`,access:`read`},{name:`startRows`,view:g,type:`u32`,access:`read`},{name:`endRows`,view:_,type:`u32`,access:`read`},{name:`staypointIds`,view:a.staypointIds,type:`u32`,access:`read_write`}],body:`if (index < stayTotal[stayTotalOffset]) {
    for (var row = startRows[startRowsOffset + index]; row < min(endRows[endRowsOffset + index], ${c}u); row++) {
      staypointIds[staypointIdsOffset + row] = index;
    }
  }`})),E}},V=`GPUTrajectorySegments`;function H(e,t=new Float32Array(4)){let n=e.maximumGap??0,{activityMinimumDuration:r}=e;if(!Number.isFinite(Math.fround(n))||n<0)throw Error(`Segment maximumGap must be finite and non-negative`);if(!Number.isFinite(Math.fround(r))||r<0)throw Error(`Segment activityMinimumDuration must be finite and non-negative`);return t.set([n,r,0,0]),t}var U=class{id;props;constructor(t){t=M(t,[`id`,`positions`,`timestamps`,`temporalContext`,`trackOffsets`,`staypointIds`,`staypoints`,`parameters`,`output`]);let n=M(t.output,[`triplegs`,`trips`]),i=e=>{let t=M(e,[`ids`,`status`,`startRows`,`endRows`,`rowTriplegIds`,`finishRows`,`originStaypoints`,`destinationStaypoints`,`triplegTripIds`]);return{...t,status:M(t.status,[`count`,`overflow`,`requiredCount`,`candidateOverflow`])}},a=t.staypoints&&M(t.staypoints,[`tracks`,`startRows`,`finishRows`,`durations`,`status`]);t={...t,output:{triplegs:i(n.triplegs),trips:n.trips&&i(n.trips)},staypoints:a&&{...a,status:M(a.status,[`count`,`overflow`,`candidateOverflow`])}},this.id=t.id??`trajectory-segments`,this.props=t;let o=this.id,{positions:s,timestamps:c,trackOffsets:l,staypointIds:f,parameters:g,output:_}=t;p(o,!_.trips||_.trips.status.candidateOverflow!==void 0,`output.trips.status.candidateOverflow is required because triplegs can be truncated`),p(o,t.staypoints?.status.overflow===void 0&&t.staypoints?.status.candidateOverflow===void 0||_.triplegs.status.candidateOverflow!==void 0,`output.triplegs.status.candidateOverflow is required with supplied staypoint loss`),d(s,[`float32x2`],`${o} positions`),d(c,[`float32`,`uint32x2`],`${o} timestamps`),d(l,[`uint32`],`${o} trackOffsets`),d(f,[`uint32`],`${o} staypointIds`),d(g,[`float32`],`${o} parameters`),u(o,`timestamps`,c,s.length),u(o,`staypointIds`,f,s.length),S(o,c,t.temporalContext),p(o,l.length>=2,`trackOffsets must contain at least two rows`),p(o,g.length>=4,`parameters must hold 4 float32 values`),h(o,_.triplegs),m(o,`tripleg capacity`,_.triplegs.ids.length);for(let[e,t]of[[`startRows`,_.triplegs.startRows],[`endRows`,_.triplegs.endRows]])t&&(d(t,[`uint32`],`${o} output.triplegs.${e}`),u(o,`output.triplegs.${e}`,t,_.triplegs.ids.length));_.triplegs.rowTriplegIds&&(d(_.triplegs.rowTriplegIds,[`uint32`],`${o} output.triplegs.rowTriplegIds`),u(o,`output.triplegs.rowTriplegIds`,_.triplegs.rowTriplegIds,s.length));let{trips:y}=_;if(y){p(o,!!t.staypoints,`staypoints are required for trips`),h(o,y),m(o,`trip capacity`,y.ids.length);for(let[e,t]of[[`startRows`,y.startRows],[`finishRows`,y.finishRows],[`originStaypoints`,y.originStaypoints],[`destinationStaypoints`,y.destinationStaypoints]])t&&(d(t,[`uint32`],`${o} output.trips.${e}`),u(o,`output.trips.${e}`,t,y.ids.length));y.triplegTripIds&&(d(y.triplegTripIds,[`uint32`],`${o} output.trips.triplegTripIds`),u(o,`output.trips.triplegTripIds`,y.triplegTripIds,_.triplegs.ids.length))}r(this.id,this.props.output);for(let[e,n]of Object.entries(t.staypoints?.status??{}))n&&(d(n,[`uint32`],`${o} staypoints.status.${e}`),p(o,n.length>=1,`staypoints.status.${e} must have at least 1 row`));e(o,[{ids:_.triplegs.ids,startRows:_.triplegs.startRows,endRows:_.triplegs.endRows},{..._.triplegs.status},..._.trips&&t.staypoints?[{startRows:_.trips.startRows,finishRows:_.trips.finishRows,originStaypoints:_.trips.originStaypoints,destinationStaypoints:_.trips.destinationStaypoints},{..._.trips.status}]:[]]),v(o,[_.triplegs.rowTriplegIds],[_.triplegs.startRows,_.triplegs.endRows]),_.trips&&t.staypoints&&v(o,[_.trips.triplegTripIds],[_.triplegs.startRows,_.trips.startRows,_.trips.finishRows]),v(o,this.getOutputViews(),this.getInputViews())}getInputViews(){let{props:e}=this;return[e.positions,e.timestamps,e.trackOffsets,e.staypointIds,e.parameters,e.staypoints?.tracks,e.staypoints?.startRows,e.staypoints?.finishRows,e.staypoints?.durations,e.staypoints?.status.count,e.staypoints?.status.overflow,e.staypoints?.status.candidateOverflow]}getOutputViews(){let{triplegs:e,trips:t}=this.props.output;return[e.ids,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount,e.status.candidateOverflow,e.startRows,e.endRows,e.rowTriplegIds,t?.ids,t?.status.count,t?.status.overflow,t?.status.candidateOverflow,t?.status.requiredCount,t?.status.candidateOverflow,t?.startRows,t?.finishRows,t?.originStaypoints,t?.destinationStaypoints,t?.triplegTripIds]}getCommandNodes(e){let{id:r,props:o}=this,{positions:s,trackOffsets:c,staypointIds:l,parameters:u,output:d}=o;_(r,e,[...this.getInputViews(),...this.getOutputViews()]);let f=s.length,p=c.length-1,m=d.triplegs.ids.length,h=[o.staypoints?.status.overflow,o.staypoints?.status.candidateOverflow].filter(e=>!!e);g(e.device,r,`positions`,f*8,`split the tracks over several graphs`),n(e,this.id,this.props),t(e,r,{"track offsets":(p+1)*4,...d.trips&&o.staypoints?{"trip records":d.trips.ids.length*16,"merged trip events":(o.staypoints.startRows.length+m)*4}:{}});let v=C(o),S=w(v),E=i(e,`${r}-tripleg-flags`,`uint32`,f),O=i(e,`${r}-tripleg-ends`,`uint32`,f),k=i(e,`${r}-row-tracks`,`uint32`,f),A=i(e,`${r}-row-ids`,`uint32`,f),M=i(e,`${r}-tripleg-rows`,`uint32`,f),N=i(e,`${r}-tripleg-total`,`uint32`,1),P=d.triplegs.startRows??i(e,`${r}-tripleg-starts`,`uint32`,m),F=d.triplegs.endRows??i(e,`${r}-tripleg-ends-out`,`uint32`,m),I=[b(e,{id:`${r}-reset-tripleg-flags`,operation:V,invocationCount:f,bindings:[{name:`startFlags`,view:E,type:`u32`,access:`read_write`},{name:`rowEnds`,view:O,type:`u32`,access:`read_write`},{name:`rowTracks`,view:k,type:`u32`,access:`read_write`}],body:`startFlags[startFlagsOffset + index] = 0u;
  rowEnds[rowEndsOffset + index] = ${j};
  rowTracks[rowTracksOffset + index] = ${j};`}),b(e,{id:`${r}-tripleg-scan`,operation:V,variant:v,invocationCount:p,bindings:[{name:`positions`,view:s,type:`f32`,access:`read`},...T(o),{name:`trackOffsets`,view:c,type:`u32`,access:`read`},{name:`staypointIds`,view:l,type:`u32`,access:`read`},{name:`parameters`,view:u,type:`f32`,access:`read`},{name:`startFlags`,view:E,type:`u32`,access:`read_write`},{name:`rowEnds`,view:O,type:`u32`,access:`read_write`},{name:`rowTracks`,view:k,type:`u32`,access:`read_write`}],declarations:`${S}
fn isGapStep(row: u32) -> bool {
  let maximumGap = parameters[parametersOffset];
  return maximumGap > 0.0 && timeDifferenceRows(row, row - 1u) > maximumGap;
}
fn isOutsideStaypoint(row: u32) -> bool {
  return staypointIds[staypointIdsOffset + row] == ${j};
}`,body:`let begin = min(trackOffsets[trackOffsetsOffset + index], ${f}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${f}u);
  var row = begin;
  while (row < end) {
    if (!isOutsideStaypoint(row)) { row++; continue; }
    var runEnd = row + 1u;
    while (runEnd < end && isOutsideStaypoint(runEnd) && !isGapStep(runEnd)) { runEnd++; }
    var isValid = runEnd - row >= 2u;
    if (isValid) {
      var moves = false;
      for (var scan = row + 1u; scan < runEnd; scan++) {
        if (positions[positionsOffset + 2u * scan] != positions[positionsOffset + 2u * row] || positions[positionsOffset + 2u * scan + 1u] != positions[positionsOffset + 2u * row + 1u]) { moves = true; break; }
      }
      isValid = moves;
    }
    if (isValid) {
      startFlags[startFlagsOffset + row] = 1u;
      rowEnds[rowEndsOffset + row] = runEnd;
      for (var member = row; member < runEnd; member++) { rowTracks[rowTracksOffset + member] = index; }
    }
    row = runEnd;
  }`}),D(e,{id:`${r}-row-ids`,operation:V,rowIds:A}),...new a({id:`${r}-compact`,input:A,flags:E,output:M,count:N}).getCommandNodes(e),b(e,{id:`${r}-tripleg-table`,operation:V,invocationCount:m,bindings:[{name:`compact`,view:M,type:`u32`,access:`read`},{name:`total`,view:N,type:`u32`,access:`read`},{name:`rowEnds`,view:O,type:`u32`,access:`read`},{name:`rowTracks`,view:k,type:`u32`,access:`read`},{name:`ids`,view:d.triplegs.ids,type:`u32`,access:`read_write`},{name:`starts`,view:P,type:`u32`,access:`read_write`},{name:`ends`,view:F,type:`u32`,access:`read_write`}],body:`if (index < total[totalOffset]) {
    let row = compact[compactOffset + index];
    ids[idsOffset + index] = rowTracks[rowTracksOffset + row];
    starts[startsOffset + index] = row;
    ends[endsOffset + index] = rowEnds[rowEndsOffset + row];
  } else {
    ids[idsOffset + index] = ${j};
    starts[startsOffset + index] = ${j};
    ends[endsOffset + index] = ${j};
  }`}),y(e,{id:`${r}-tripleg-publish`,operation:V,requiredCount:N,output:{...d.triplegs,status:d.triplegs.status},candidateOverflowSources:h})];return d.triplegs.rowTriplegIds&&I.push(x(e,{id:`${r}-row-tripleg-reset`,operation:V,view:d.triplegs.rowTriplegIds,type:`u32`,value:j}),b(e,{id:`${r}-row-triplegs`,operation:V,invocationCount:m,bindings:[{name:`total`,view:N,type:`u32`,access:`read`},{name:`starts`,view:P,type:`u32`,access:`read`},{name:`ends`,view:F,type:`u32`,access:`read`},{name:`rowTriplegIds`,view:d.triplegs.rowTriplegIds,type:`u32`,access:`read_write`}],body:`if (index < min(total[totalOffset], ${m}u)) {
    for (var row = starts[startsOffset + index]; row < ends[endsOffset + index]; row++) {
      rowTriplegIds[rowTriplegIdsOffset + row] = index;
    }
  }`})),d.trips&&o.staypoints&&I.push(...this.createTripNodes(e,{trackCount:p,rowCount:f,triplegTotal:N,triplegStarts:P,triplegEnds:F})),I}createTripNodes(e,t){let{id:n,props:r}=this,a=r.output.trips,o=r.staypoints,{trackCount:s,triplegTotal:c,triplegStarts:l,triplegEnds:u}=t,d=r.output.triplegs.ids,f=o.startRows.length,p=l.length,m=f+p,h=C(r),g=w(h),_=i(e,`${n}-track-stay-offsets`,`uint32`,s+1),v=i(e,`${n}-track-tripleg-offsets`,`uint32`,s+1),y=i(e,`${n}-track-event-counts`,`uint32`,s+1),x=i(e,`${n}-events`,`uint32`,m),S=i(e,`${n}-event-starts`,`uint32`,m),D=i(e,`${n}-event-finishes`,`uint32`,m),O=i(e,`${n}-track-trip-offsets`,`uint32`,s+1),k=i(e,`${n}-track-trip-counts`,`uint32`,s),A=i(e,`${n}-trip-total`,`uint32`,1),M=i(e,`${n}-trip-records`,`uint32`,a.ids.length*4),N=b(e,{id:`${n}-merge-events`,operation:V,invocationCount:s,bindings:[{name:`stayStarts`,view:o.startRows,type:`u32`,access:`read`},{name:`stayDurations`,view:o.durations,type:`f32`,access:`read`},{name:`triplegStarts`,view:l,type:`u32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`trackStayOffsets`,view:_,type:`u32`,access:`read`},{name:`trackTriplegOffsets`,view:v,type:`u32`,access:`read`},{name:`eventIdentifiers`,view:x,type:`u32`,access:`read_write`},{name:`trackEventCounts`,view:y,type:`u32`,access:`read_write`}],body:`let activityThreshold = parameters[parametersOffset + 1u];
  let stayEnd = trackStayOffsets[trackStayOffsetsOffset + index + 1u];
  let triplegEnd = trackTriplegOffsets[trackTriplegOffsetsOffset + index + 1u];
  var stayCursor = trackStayOffsets[trackStayOffsetsOffset + index];
  var triplegCursor = trackTriplegOffsets[trackTriplegOffsetsOffset + index];
  var event = stayCursor + triplegCursor;
  trackEventCounts[trackEventCountsOffset + index] = event;
  while (stayCursor < stayEnd || triplegCursor < triplegEnd) {
    let takeStay = triplegCursor >= triplegEnd || (stayCursor < stayEnd && stayStarts[stayStartsOffset + stayCursor] <= triplegStarts[triplegStartsOffset + triplegCursor]);
    if (takeStay) {
      let isActivity = stayDurations[stayDurationsOffset + stayCursor] > activityThreshold;
      eventIdentifiers[eventIdentifiersOffset + event] = (stayCursor << 2u) | select(0u, 1u, isActivity);
      stayCursor++;
    } else {
      eventIdentifiers[eventIdentifiersOffset + event] = (triplegCursor << 2u) | 2u;
      triplegCursor++;
    }
    event++;
  }
  if (index == ${s-1}u) { trackEventCounts[trackEventCountsOffset + ${s}u] = event; }`}),P=b(e,{id:`${n}-event-rows`,operation:V,invocationCount:m,bindings:[{name:`eventIdentifiers`,view:x,type:`u32`,access:`read`},{name:`stayStarts`,view:o.startRows,type:`u32`,access:`read`},{name:`stayFinishes`,view:o.finishRows,type:`u32`,access:`read`},{name:`triplegStarts`,view:l,type:`u32`,access:`read`},{name:`triplegEnds`,view:u,type:`u32`,access:`read`},{name:`eventCount`,view:y,type:`u32`,access:`read`},{name:`eventStartRows`,view:S,type:`u32`,access:`read_write`},{name:`eventFinishRows`,view:D,type:`u32`,access:`read_write`}],body:`if (index < eventCount[eventCountOffset + ${s}u]) {
    let packed = eventIdentifiers[eventIdentifiersOffset + index];
    let identifier = packed >> 2u;
    if ((packed & 2u) != 0u) {
      eventStartRows[eventStartRowsOffset + index] = triplegStarts[triplegStartsOffset + identifier];
      eventFinishRows[eventFinishRowsOffset + index] = triplegEnds[triplegEndsOffset + identifier] - 1u;
    } else {
      eventStartRows[eventStartRowsOffset + index] = stayStarts[stayStartsOffset + identifier];
      eventFinishRows[eventFinishRowsOffset + index] = stayFinishes[stayFinishesOffset + identifier];
    }
  }`}),F=`${g}
struct TripState { count: u32, activeTripleg: bool, nonActivityCount: u32, firstStart: u32, lastFinish: u32, origin: u32, hasDestination: bool, destination: u32, openDestination: bool }
fn isEventActivity(event: u32) -> bool { return (eventIdentifiers[eventIdentifiersOffset + event] & 1u) != 0u; }
fn isEventTripleg(event: u32) -> bool { return (eventIdentifiers[eventIdentifiersOffset + event] & 2u) != 0u; }
fn eventId(event: u32) -> u32 { return eventIdentifiers[eventIdentifiersOffset + event] >> 2u; }
fn isGapBetween(a: u32, b: u32) -> bool {
  let maximumGap = parameters[parametersOffset];
  return maximumGap > 0.0 && timeDifferenceRows(eventStartRows[eventStartRowsOffset + b], eventFinishRows[eventFinishRowsOffset + a]) > maximumGap;
}`,I=e=>`let firstEvent = trackEventCounts[trackEventCountsOffset + index];
  let lastEvent = trackEventCounts[trackEventCountsOffset + index + 1u];
  var state: TripState;
  state.count = 0u; state.activeTripleg = false; state.nonActivityCount = 0u;
  state.origin = ${j}; state.destination = ${j}; state.hasDestination = false; state.openDestination = false;
  state.firstStart = ${j}; state.lastFinish = ${j};
  ${e?`let baseRow = trackTripOffsets[trackTripOffsetsOffset + index];`:``}
  for (var event = firstEvent; event < lastEvent; event++) {
    let isActivity = isEventActivity(event);
    let previousIsActivity = event > firstEvent && isEventActivity(event - 1u);
    let nextIsActivity = event + 1u < lastEvent && isEventActivity(event + 1u);
    let isGapBefore = event > firstEvent && isGapBetween(event - 1u, event);
    let isLastActivity = isActivity && !nextIsActivity;
    let isFirstOfTrack = event == firstEvent;
    let startsTrip = isFirstOfTrack || isGapBefore || isLastActivity;
    // A destination pending from the previous non-activity event resolves on this event.
    if (state.openDestination) {
      state.openDestination = false;
      if (isActivity && !isGapBefore) { state.destination = eventId(event); }
      else { state.destination = ${j}; }
    }
    if (startsTrip && event > firstEvent) {
      if (state.activeTripleg) {
        ${e?`let row = baseRow + state.count;
        if (row < ${a.ids.length}u) {
          tripRecords[tripRecordsOffset + 4u * row] = state.firstStart;
          tripRecords[tripRecordsOffset + 4u * row + 1u] = state.lastFinish;
          tripRecords[tripRecordsOffset + 4u * row + 2u] = state.origin;
          tripRecords[tripRecordsOffset + 4u * row + 3u] = state.destination;
        }`:``}
        state.count++;
      }
      state.activeTripleg = false;
      state.nonActivityCount = 0u;
      state.destination = ${j};
      state.origin = ${j};
    }
    if (!isActivity) {
      if (state.nonActivityCount == 0u) {
        state.firstStart = eventStartRows[eventStartRowsOffset + event];
        state.origin = select(${j}, eventId(event - 1u), event > firstEvent && previousIsActivity && !isGapBefore);
      }
      state.nonActivityCount++;
      state.lastFinish = eventFinishRows[eventFinishRowsOffset + event];
      if (isEventTripleg(event)) { state.activeTripleg = true; }
      state.openDestination = true;
    }
  }
  if (state.activeTripleg) {
    if (state.openDestination) { state.destination = ${j}; }
    ${e?`let row = baseRow + state.count;
    if (row < ${a.ids.length}u) {
      tripRecords[tripRecordsOffset + 4u * row] = state.firstStart;
      tripRecords[tripRecordsOffset + 4u * row + 1u] = state.lastFinish;
      tripRecords[tripRecordsOffset + 4u * row + 2u] = state.origin;
      tripRecords[tripRecordsOffset + 4u * row + 3u] = state.destination;
    }`:``}
    state.count++;
  }
  ${e?``:`tripCounts[tripCountsOffset + index] = state.count;`}`,L=[...T(r),{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`trackEventCounts`,view:y,type:`u32`,access:`read`},{name:`eventIdentifiers`,view:x,type:`u32`,access:`read`},{name:`eventStartRows`,view:S,type:`u32`,access:`read`},{name:`eventFinishRows`,view:D,type:`u32`,access:`read`}];return[E(e,{id:`${n}-offsets-stays`,operation:V,tracks:o.tracks,count:o.status.count,offsets:_,capacity:f,trackCount:s}),E(e,{id:`${n}-offsets-triplegs`,operation:V,tracks:d,count:c,offsets:v,capacity:p,trackCount:s}),N,P,b(e,{id:`${n}-trip-count`,operation:V,variant:h,invocationCount:s,bindings:[...L,{name:`tripCounts`,view:k,type:`u32`,access:`read_write`}],declarations:F,body:I(!1)}),b(e,{id:`${n}-trip-offsets`,operation:V,invocationCount:1,bindings:[{name:`tripCounts`,view:k,type:`u32`,access:`read`},{name:`trackTripOffsets`,view:O,type:`u32`,access:`read_write`},{name:`tripTotal`,view:A,type:`u32`,access:`read_write`}],body:`var total = 0u;
  for (var track = 0u; track < ${s}u; track++) {
    trackTripOffsets[trackTripOffsetsOffset + track] = total;
    total += tripCounts[tripCountsOffset + track];
  }
  trackTripOffsets[trackTripOffsetsOffset + ${s}u] = total;
  tripTotal[tripTotalOffset] = total;`}),b(e,{id:`${n}-trip-write`,operation:V,variant:h,invocationCount:s,bindings:[...L,{name:`trackTripOffsets`,view:O,type:`u32`,access:`read`},{name:`tripRecords`,view:M,type:`u32`,access:`read_write`}],declarations:F,body:I(!0)}),...this.createTripTableNodes(e,{tripTotal:A,tripRecords:M,trackTripOffsets:O,triplegStarts:l,triplegTotal:c})]}createTripTableNodes(e,t){let{id:n,props:r}=this,a=r.output.trips,{tripTotal:o,tripRecords:s,trackTripOffsets:c,triplegStarts:l,triplegTotal:u}=t,d=r.output.triplegs.ids.length,f=i(e,`${n}-trip-input-loss`,`uint32`,1),p=[r.staypoints?.status.overflow,r.staypoints?.status.candidateOverflow].filter(e=>!!e),m=a.ids.length,h=r.trackOffsets.length-1,g=a.startRows??i(e,`${n}-trip-starts`,`uint32`,m),_=a.finishRows??i(e,`${n}-trip-finishes`,`uint32`,m),v=a.originStaypoints??i(e,`${n}-trip-origins`,`uint32`,m),x=a.destinationStaypoints??i(e,`${n}-trip-destinations`,`uint32`,m),S=[b(e,{id:`${n}-trip-table`,operation:V,invocationCount:m,bindings:[{name:`tripRecords`,view:s,type:`u32`,access:`read`},{name:`tripTotal`,view:o,type:`u32`,access:`read`},{name:`starts`,view:g,type:`u32`,access:`read_write`},{name:`finishes`,view:_,type:`u32`,access:`read_write`},{name:`origins`,view:v,type:`u32`,access:`read_write`},{name:`destinations`,view:x,type:`u32`,access:`read_write`}],body:`if (index < tripTotal[tripTotalOffset]) {
    starts[startsOffset + index] = tripRecords[tripRecordsOffset + 4u * index];
    finishes[finishesOffset + index] = tripRecords[tripRecordsOffset + 4u * index + 1u];
    origins[originsOffset + index] = tripRecords[tripRecordsOffset + 4u * index + 2u];
    destinations[destinationsOffset + index] = tripRecords[tripRecordsOffset + 4u * index + 3u];
  } else {
    starts[startsOffset + index] = ${j};
    finishes[finishesOffset + index] = ${j};
    origins[originsOffset + index] = ${j};
    destinations[destinationsOffset + index] = ${j};
  }`}),b(e,{id:`${n}-trip-ids`,operation:V,invocationCount:m,bindings:[{name:`tripTotal`,view:o,type:`u32`,access:`read`},{name:`trackTripOffsets`,view:c,type:`u32`,access:`read`},{name:`ids`,view:a.ids,type:`u32`,access:`read_write`}],body:`if (index < tripTotal[tripTotalOffset]) {
    var low = 0u;
    var high = ${h}u;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (trackTripOffsets[trackTripOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; }
    }
    ids[idsOffset + index] = low;
  } else {
    ids[idsOffset + index] = ${j};
  }`}),y(e,{id:`${n}-trip-publish`,operation:V,requiredCount:o,output:{...a,status:a.status},candidateOverflowSources:[f,...p]})];return S.unshift(b(e,{id:`${n}-trip-input-loss`,operation:V,invocationCount:1,bindings:[{name:`triplegTotal`,view:u,type:`u32`,access:`read`},{name:`loss`,view:f,type:`u32`,access:`read_write`}],body:`loss[lossOffset] = select(0u, 1u, triplegTotal[triplegTotalOffset] > ${d}u);`})),a.triplegTripIds&&S.push(b(e,{id:`${n}-tripleg-trips`,operation:V,invocationCount:r.output.triplegs.ids.length,bindings:[{name:`triplegStarts`,view:l,type:`u32`,access:`read`},{name:`triplegTotal`,view:u,type:`u32`,access:`read`},{name:`tripTotal`,view:o,type:`u32`,access:`read`},{name:`tripStarts`,view:g,type:`u32`,access:`read`},{name:`tripFinishes`,view:_,type:`u32`,access:`read`},{name:`triplegTripIds`,view:a.triplegTripIds,type:`u32`,access:`read_write`}],body:`var trip = ${j};
  if (index < min(triplegTotal[triplegTotalOffset], ${r.output.triplegs.ids.length}u)) {
    let row = triplegStarts[triplegStartsOffset + index];
    let total = min(tripTotal[tripTotalOffset], ${m}u);
    var low = 0u;
    var high = total;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (tripStarts[tripStartsOffset + middle] <= row) { low = middle + 1u; } else { high = middle; }
    }
    if (low > 0u && tripStarts[tripStartsOffset + low - 1u] <= row && row <= tripFinishes[tripFinishesOffset + low - 1u]) { trip = low - 1u; }
  }
  triplegTripIds[triplegTripIdsOffset + index] = trip;`})),S}};function W(e){let t=[];for(let e of N){let n=[e[0]];for(let r=1;r<e.length;r++){let i=e[r-1];Math.hypot(e[r][0]-i[0],e[r][1]-i[1])>4e3&&(t.push(n),n=[]),n.push(e[r])}t.push(n)}return t.filter(e=>e.length>=14).sort((e,t)=>t.length-e.length).slice(0,e)}function G(e,t){let n=F(e,{trackCount:t,minimumFixes:0,maximumFixes:999}),r=W(t).flat(),i=r.map(e=>e[0]),a=r.map(e=>e[1]),o=Math.max(...i)-Math.min(...i),s=Math.max(...a)-Math.min(...a),c=Math.sqrt(o/(.9*e.width)*(s/.9));return{...n,metersPerUnit:c}}function K(e,t){return{positions:e.upload(`positions`,`float32x2`,t.positions),timestamps:e.upload(`timestamps`,`float32`,t.timestamps),trackOffsets:e.upload(`offsets`,`uint32`,t.trackOffsets),edges:e.upload(`edges`,`uint32x2`,P(t.trackOffsets))}}function q(e,t,n,r){let i={parameters:e.allocate(`staypoint-parameters`,`float32`,4),tracks:e.allocate(`staypoint-tracks`,`uint32`,512),count:e.allocate(`staypoint-count`,`uint32`,1),overflow:e.allocate(`staypoint-overflow`,`uint32`,1),candidateOverflow:e.allocate(`staypoint-candidate-overflow`,`uint32`,1),startRows:e.allocate(`staypoint-start-rows`,`uint32`,512),endRows:e.allocate(`staypoint-end-rows`,`uint32`,512),finishRows:e.allocate(`staypoint-finish-rows`,`uint32`,512),centroids:e.allocate(`staypoint-centroids`,`float32x2`,512),durations:e.allocate(`staypoint-durations`,`float32`,512),rowStaypointIds:e.allocate(`row-staypoint-ids`,`uint32`,t.rowCount)};return e.graph.add(new B({positions:n.positions.view,timestamps:n.timestamps.view,trackOffsets:n.trackOffsets.view,parameters:i.parameters.view,maximumSearchRows:r,output:{ids:i.tracks.view,status:{count:i.count.view,overflow:i.overflow.view,candidateOverflow:i.candidateOverflow.view},startRows:i.startRows.view,endRows:i.endRows.view,finishRows:i.finishRows.view,centroids:i.centroids.view,durations:i.durations.view,staypointIds:i.rowStaypointIds.view}})),i}function J(e,t){e.parameters.write(z({distanceThreshold:t.distanceThreshold,timeThreshold:t.timeThreshold,maximumGap:t.maximumGap??0}))}async function Y(e){let t=await e.buffer.readAsync();return new Uint32Array(t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength))}async function X(e){let t=await e.buffer.readAsync();return new Float32Array(t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength))}function Z(e,t){let n=0,r=e.length-2;for(;n<r;){let i=n+r+1>>1;e[i]<=t?n=i:r=i-1}return n}function Q(e,t,n){let r=[],{positions:i,timestamps:a,trackOffsets:o}=e;for(let e=0;e<o.length-1;e++){let s=o[e+1],c=o[e];for(let o=c+1;o<s;o++){if(Math.hypot(i[o*2]-i[c*2],i[o*2+1]-i[c*2+1])<t)continue;let s=a[o]-a[c];if(s>=n){let t=0,n=0;for(let e=c;e<o;e++)t+=i[e*2],n+=i[e*2+1];r.push({track:e,startRow:c,endRow:o,centroid:[t/(o-c),n/(o-c)],duration:s})}c=o}}return r}function $(e,t){let n=[];for(let r=0;r<t.length-1;r++)for(let i=t[r];i<t[r+1]-1;i++)n.push(e[i*2],e[i*2+1],e[i*2+2],e[i*2+3]);return Float32Array.from(n)}function ee(e,t=e.positions){let n=new Float32Array(e.rowCount);for(let r=0;r<e.trackCount;r++)for(let i=e.trackOffsets[r]+1;i<e.trackOffsets[r+1];i++){let r=e.timestamps[i]-e.timestamps[i-1];n[i]=Math.hypot(t[i*2]-t[i*2-2],t[i*2+1]-t[i*2-1])/Math.max(r,1e-6)}return n}function te(e,t,n,r){let i={parameters:e.allocate(`segment-parameters`,`float32`,4),triplegTracks:e.allocate(`tripleg-tracks`,`uint32`,512),triplegCount:e.allocate(`tripleg-count`,`uint32`,1),triplegOverflow:e.allocate(`tripleg-overflow`,`uint32`,1),triplegStartRows:e.allocate(`tripleg-start-rows`,`uint32`,512),triplegEndRows:e.allocate(`tripleg-end-rows`,`uint32`,512),rowTriplegIds:e.allocate(`row-tripleg-ids`,`uint32`,t.rowCount),tripCount:e.allocate(`trip-count`,`uint32`,1),tripCandidateOverflow:e.allocate(`trip-candidate-overflow`,`uint32`,1),triplegCandidateOverflow:e.allocate(`tripleg-candidate-overflow`,`uint32`,1),triplegTripIds:e.allocate(`tripleg-trip-ids`,`uint32`,512)},a=e.allocate(`trip-tracks`,`uint32`,512),o=i.triplegOverflow,s=e.allocate(`trip-overflow`,`uint32`,1);return e.graph.add(new U({positions:n.positions.view,timestamps:n.timestamps.view,trackOffsets:n.trackOffsets.view,staypointIds:r.rowStaypointIds.view,staypoints:{tracks:r.tracks.view,startRows:r.startRows.view,finishRows:r.finishRows.view,durations:r.durations.view,status:{count:r.count.view,overflow:r.overflow.view,candidateOverflow:r.candidateOverflow.view}},parameters:i.parameters.view,output:{triplegs:{ids:i.triplegTracks.view,status:{count:i.triplegCount.view,overflow:o.view,candidateOverflow:i.triplegCandidateOverflow.view},startRows:i.triplegStartRows.view,endRows:i.triplegEndRows.view,rowTriplegIds:i.rowTriplegIds.view},trips:{ids:a.view,status:{count:i.tripCount.view,overflow:s.view,candidateOverflow:i.tripCandidateOverflow.view},triplegTripIds:i.triplegTripIds.view}}})),i}export{Q as a,X as c,J as d,H as f,$ as i,Y as l,q as n,ee as o,G as r,Z as s,te as t,K as u};