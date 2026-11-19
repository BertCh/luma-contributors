import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";function t(e,t){return{cellCount:e,periodCount:t,columns:12+t,rows:e,left:12}}function n(e,t){return[t.width*e.left/e.columns,0,t.width,1]}function r(e,t,r,i){let a=n(e,t),o=Math.min(e.periodCount-1,Math.max(0,Math.floor((r-a[0])/(a[2]-a[0])*e.periodCount)));return{cell:Math.min(e.cellCount-1,Math.max(0,Math.floor((1-i)/(a[3]-a[1])*e.cellCount))),period:o,isInside:r>=a[0]}}function i(e,t,r,i){let a=n(e,t),o=(a[2]-a[0])/e.periodCount,s=(a[3]-a[1])/e.cellCount;return{minimum:[a[0]+i*o,1-(r+1)*s],maximum:[a[0]+(i+1)*o,1-r*s]}}function a(e,t,n,r){let[i,a]=[t.width*.035,t.width*(e.left/e.columns)*.94],[o,s]=[.14,.86],c=new Float32Array(n*r*2);for(let e=0;e<r;e++)for(let t=0;t<n;t++){let l=e*n+t;c[l*2]=i+(t+.5)/n*(a-i),c[l*2+1]=s-(e+.5)/r*(s-o)}return c}var o=[`counts`,`sums`,`means`];function s(e){return e.allocate(`selection`,`uint32`,4)}var c=`bitcast<f32>(0x7fc00000u | (selection[selectionOffset] & 0u))`;function l(t,n){let{frame:r}=n,i=t.allocate(`heatmap`,`float32`,r.columns*r.rows);return t.graph.add(e(t.graph,{id:`cube-heatmap`,operation:`display-cube-heatmap`,invocationCount:r.columns*r.rows,bindings:[{name:`counts`,view:n.counts.view,type:`u32`,access:`read`},{name:`sums`,view:n.sums.view,type:`f32`,access:`read`},{name:`means`,view:n.means.view,type:`f32`,access:`read`},{name:`selection`,view:n.selection.view,type:`u32`,access:`read`},{name:`raster`,view:i.view,type:`f32`,access:`read_write`}],body:`
  let row = index / ${r.columns}u;
  let column = index % ${r.columns}u;
  var result = ${c};
  if (column >= ${r.left}u) {
    let bin = row * ${r.periodCount}u + (column - ${r.left}u);
    let count = counts[countsOffset + bin];
    if (count > 0u) {
      let statistic = selection[selectionOffset];
      if (statistic == 0u) {
        result = f32(count);
      } else if (statistic == 1u) {
        result = sums[sumsOffset + bin];
      } else {
        result = means[meansOffset + bin];
      }
    }
  }
  raster[rasterOffset + index] = result;`})),i}function u(t,n){let{frame:r}=n,i=t.allocate(`period-slice`,`float32`,r.cellCount);return t.graph.add(e(t.graph,{id:`period-slice`,operation:`display-period-slice`,invocationCount:r.cellCount,bindings:[{name:`counts`,view:n.counts.view,type:`u32`,access:`read`},{name:`sums`,view:n.sums.view,type:`f32`,access:`read`},{name:`means`,view:n.means.view,type:`f32`,access:`read`},{name:`selection`,view:n.selection.view,type:`u32`,access:`read`},{name:`slice`,view:i.view,type:`f32`,access:`read_write`}],body:`
  let bin = index * ${r.periodCount}u + selection[selectionOffset + 1u];
  let count = counts[countsOffset + bin];
  var result = ${c};
  if (count > 0u) {
    let statistic = selection[selectionOffset];
    if (statistic == 0u) {
      result = f32(count);
    } else if (statistic == 1u) {
      result = sums[sumsOffset + bin];
    } else {
      result = means[meansOffset + bin];
    }
  }
  slice[sliceOffset + index] = result;`})),i}function d(e,t){return{cellCount:e,periodCount:t,columns:t+2+e,rows:Math.max(e,t)}}function f(e,t,n,r,i){let a=t.width/e.columns,o=1/e.rows,s=i===`cell-major`?r:e.periodCount+2+n,c=i===`cell-major`?n:r;return{minimum:[s*a,1-(c+1)*o],maximum:[(s+1)*a,1-c*o]}}function p(e,t,n,r){let i=Math.floor(n/t.width*e.columns),a=Math.floor((1-r)*e.rows);if(i<e.periodCount)return a<e.cellCount?{cell:a,period:i}:null;let o=i-e.periodCount-2;return o>=0&&o<e.cellCount&&a<e.periodCount?{cell:o,period:a}:null}function m(t,n){let{frame:r}=n,i=n.cellMajor.format===`uint32`,a=t.allocate(`layouts`,`float32`,r.columns*r.rows),o=(e,t)=>i?`f32(${e}[${e}Offset + ${t}])`:`${e}[${e}Offset + ${t}]`,s=(e,t)=>i?`${e}[${e}Offset + ${t}] == 0u`:`!(${e}[${e}Offset + ${t}] == ${e}[${e}Offset + ${t}])`;return t.graph.add(e(t.graph,{id:`layout-composite`,operation:`display-layout-composite`,invocationCount:r.columns*r.rows,bindings:[{name:`cellMajor`,view:n.cellMajor.view,type:i?`u32`:`f32`,access:`read`},{name:`periodMajor`,view:n.periodMajor.view,type:i?`u32`:`f32`,access:`read`},{name:`raster`,view:a.view,type:`f32`,access:`read_write`}],body:`
  let row = index / ${r.columns}u;
  let column = index % ${r.columns}u;
  var result = bitcast<f32>(0x7fc00000u | (row & 0u));
  if (column < ${r.periodCount}u) {
    if (row < ${r.cellCount}u) {
      let bin = row * ${r.periodCount}u + column;
      if (!(${s(`cellMajor`,`bin`)})) { result = ${o(`cellMajor`,`bin`)}; }
    }
  } else if (column >= ${r.periodCount+2}u) {
    let cell = column - ${r.periodCount+2}u;
    if (row < ${r.periodCount}u) {
      let bin = row * ${r.cellCount}u + cell;
      if (!(${s(`periodMajor`,`bin`)})) { result = ${o(`periodMajor`,`bin`)}; }
    }
  }
  raster[rasterOffset + index] = result;`})),a}export{u as a,i as c,p as d,f,m as i,t as l,l as n,a as o,d as p,s as r,r as s,o as t,n as u};