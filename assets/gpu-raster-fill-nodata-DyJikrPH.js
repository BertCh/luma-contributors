import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{T as t,f as n,s as r}from"./gpu-contributor-utils-BIwLQON5.js";import{t as i}from"./status-aOwAAXek.js";import{a,c as o,g as s,n as c,r as l}from"./raster-band-utils-BcTxc8_Y.js";import{o as u}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as d}from"./raster-kernel-utils-DcIN7G5I.js";import{u as f}from"./wgsl-common-DfyXiYB0.js";import{a as p,i as m,r as h}from"./raster-expansion-utils-AZOLseCG.js";var g=256,_=class{id;props;constructor(e){e={...e,input:a(e.input)},this.id=e.id??`raster-fill-nodata`,this.props=e,s(this.id,e.input,e.grid),t(this.id,e.method===void 0||[`nearest`,`inverse-distance`,`gdal-four-quadrant`].includes(e.method),`method must be 'nearest', 'inverse-distance' or 'gdal-four-quadrant'`),t(this.id,typeof e.output.status==`object`&&e.output.status!==null,`output.status is required (an object holding the optional invalidCount view, the count of cells left invalid)`);let r=p(this.id,e.grid,c(e.input),v(e.output));h(e.output.values,r,this.id,`output.values`),h(e.output.validity,r,this.id,`output.validity`),h(e.output.filled,r,this.id,`output.filled`),i(this.id,e.output.status),e.output.status.invalidCount&&h(e.output.status.invalidCount,1,this.id,`output.status.invalidCount`),n(this.id,`maximumDistance`,e.maximumDistance,1,g)}getCommandNodes(t){let n=this.props,i=n.grid.width*n.grid.height;e(t,this.id,[...c(n.input),...r(n.output)],{"canonical band planes":i*4});let{props:a,id:s}=this,{output:p,grid:h}=a,g=h.width*h.height;o(s,t,a.input,[]),m(s,t,[...c(a.input),...v(p)]);let _=l(t,s,a.input,h,!0);return[..._.nodes,u(t,{id:s,operation:`GPURasterFillNoData`,invocationCount:g,bindings:[{name:`source`,view:_.band.storage.values,type:`f32`,access:`read`},{name:`sourceValidity`,view:_.band.validity,type:`u32`,access:`read`},{name:`values`,view:p.values,type:`f32`,access:`read_write`},{name:`validity`,view:p.validity,type:`u32`,access:`read_write`},{name:`filled`,view:p.filled,type:`u32`,access:`read_write`}],declarations:f,body:`var result = source[sourceOffset + index];
var valid = sourceValidity[sourceValidityOffset + index] != 0u;
let originalValid = valid;
if (!valid) {
  var total = 0.0; var weight = 0.0; var nearestDistance = 3.402823e38;
  let column = i32(index % ${h.width}u); let row = i32(index / ${h.width}u);
  ${a.method===`gdal-four-quadrant`?y(h,a.maximumDistance):`  for (var rowOffset = -${a.maximumDistance}; rowOffset <= ${a.maximumDistance}; rowOffset++) {
    for (var columnOffset = -${a.maximumDistance}; columnOffset <= ${a.maximumDistance}; columnOffset++) {
      let squaredDistance = columnOffset * columnOffset + rowOffset * rowOffset;
      let sampleColumn = column + columnOffset; let sampleRow = row + rowOffset;
      if (squaredDistance == 0 || squaredDistance > ${a.maximumDistance*a.maximumDistance} || sampleColumn < 0 || sampleRow < 0 || sampleColumn >= ${h.width} || sampleRow >= ${h.height}) { continue; }
      let sampleIndex = u32(sampleRow) * ${h.width}u + u32(sampleColumn);
      if (sourceValidity[sourceValidityOffset + sampleIndex] == 0u) { continue; }
      let distance = sqrt(f32(squaredDistance));
      ${a.method===`nearest`?`if (distance < nearestDistance) { nearestDistance = distance; total = source[sourceOffset + sampleIndex]; weight = 1.0; }`:`let contribution = 1.0 / distance; total += source[sourceOffset + sampleIndex] * contribution; weight += contribution;`}
    }
  }`}

  valid = weight > 0.0; result = total / max(weight, 1.0e-30);
}
valid = valid && isFiniteValue(result);
values[valuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), result, valid);
validity[validityOffset + index] = u32(valid);
filled[filledOffset + index] = u32(valid && !originalValid);`}),...p.status.invalidCount?d(t,{id:`${s}-unresolved`,operation:`GPURasterFillNoData`,values:p.validity,count:g,condition:`value == 0u`,output:p.status.invalidCount}):[]]}};function v(e){return[e.values,e.validity,e.filled,...e.status.invalidCount?[e.status.invalidCount]:[]]}function y(e,t){return`var quadrantDistances = array<i32, 4>(${(t+1)**2}, ${(t+1)**2}, ${(t+1)**2}, ${(t+1)**2});
  var quadrantValues: array<f32, 4>;
  for (var step = 0; step <= ${t}; step++) {
    for (var side = 0; side < 2; side++) {
      if (side == 1 && step == 0) { continue; }
      let sampleColumn = select(max(0, column - step), min(${e.width-1}, column + step), side == 1);
      for (var direction = 0; direction < 2; direction++) {
        let quadrant = u32(side * 2 + direction);
        for (var verticalOffset = direction; verticalOffset <= ${t}; verticalOffset++) {
          let sampleRow = select(row - verticalOffset, row + verticalOffset, direction == 1);
          if (sampleRow < 0 || sampleRow >= ${e.height}) { break; }
          let sampleIndex = u32(sampleRow) * ${e.width}u + u32(sampleColumn);
          if (sourceValidity[sourceValidityOffset + sampleIndex] == 0u) { continue; }
          let horizontalOffset = sampleColumn - column;
          let squaredDistance = horizontalOffset * horizontalOffset + verticalOffset * verticalOffset;
          if (squaredDistance > 0 && squaredDistance < quadrantDistances[quadrant]) {
            quadrantDistances[quadrant] = squaredDistance;
            quadrantValues[quadrant] = source[sourceOffset + sampleIndex];
          }
          // Only the closest original donor in this column/direction can win.
          break;
        }
      }
    }
  }
  for (var quadrant = 0u; quadrant < 4u; quadrant++) {
    if (quadrantDistances[quadrant] <= ${t**2}) {
      let contribution = inverseSqrt(f32(quadrantDistances[quadrant]));
      total += quadrantValues[quadrant] * contribution;
      weight += contribution;
    }
  }`}export{_ as t};