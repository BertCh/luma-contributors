import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{n as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,s as i,x as a}from"./gpu-contributor-utils-BIwLQON5.js";import{l as o,u as s}from"./raster-band-utils-BcTxc8_Y.js";import{t as c}from"./raster-grid-utils-D2opjNmg.js";import{a as l,i as u,n as d,t as f}from"./cost-CyJMsucP.js";import{t as p}from"./define-run-B-brkCe4.js";import{_t as m}from"./entry-preview-D70TdSbK.js";import{o as h}from"./pointer-overlay-Qh8qflcA.js";var g=`GPUCostDistancePath`,_=class{id;props;constructor(e){this.id=e.id??`cost-distance-path`,this.props=e;let{id:t}=this,n=s(t,e.grid),{output:i}=e,{path:a}=i,c=i.status?.iterationLimitReached;for(let[n,i]of[[`backLinks`,e.backLinks],[`target`,e.target],[`output.path.ids`,a.ids],[`output.path.count`,a.status.count],[`output.path.overflow`,a.status.overflow],[`output.path.requiredCount`,a.status.requiredCount],[`output.status.iterationLimitReached`,c]])i&&r(i,[`uint32`],`${t} ${n}`);if(e.backLinks.length!==n)throw Error(`${t} backLinks must contain one value per cell`);for(let[n,r]of[[`target`,e.target],[`output.path.count`,a.status.count],[`output.path.overflow`,a.status.overflow],[`output.path.requiredCount`,a.status.requiredCount],[`output.status.iterationLimitReached`,c]])if(r&&r.length!==1)throw Error(`${t} ${n} must contain exactly one row`);if(a.ids.length<1)throw Error(`${t} output.path.ids must have a capacity of at least one`);o(t,[a.ids,a.status.count,a.status.overflow,a.status.requiredCount,c],[e.backLinks,e.target])}getCommandNodes(r){let o=this.props;e(r,this.id,[o.backLinks,o.target,...i(o.output.path),...Object.values(o.output.status??{})],{"path total":4});let{id:s,props:d}=this,{output:f}=d,{path:p}=f,{width:m,height:h}=d.grid,_=f.status?.iterationLimitReached;a(s,r,[d.backLinks,d.target,p.ids,p.status.count,p.status.overflow,p.status.requiredCount,_]);let v=m*h,y=c(r,g,s,`${s}-total`,`uint32`,1);return[n(r,{id:`${s}-walk`,operation:g,variant:`walk`,bindings:[{name:`backLinks`,view:d.backLinks,type:`u32`,access:`read`},{name:`targetCells`,view:d.target,type:`u32`,access:`read`},{name:`ids`,view:p.ids,type:`u32`,access:`read_write`},{name:`total`,view:y,type:`u32`,access:`read_write`},..._?[{name:`iterationLimitReached`,view:_,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const GRID_WIDTH: u32 = ${m}u;
const GRID_HEIGHT: u32 = ${h}u;
const CELL_COUNT: u32 = ${v}u;
const CAPACITY: u32 = ${p.ids.length}u;
const NONE: u32 = ${u}u;
const UNRESOLVED: u32 = ${l}u;

fn getColumnOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  return offsets[direction];
}
fn getRowOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  return offsets[direction];
}
fn getNeighbor(cell: u32, direction: u32) -> u32 {
  let column = i32(cell % GRID_WIDTH) + getColumnOffset(direction);
  let row = i32(cell / GRID_WIDTH) + getRowOffset(direction);
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) { return NONE; }
  return u32(row) * GRID_WIDTH + u32(column);
}`,body:`let targetCell = targetCells[targetCellsOffset];
  var pathLength = 0u;
  var unresolved = false;
  if (targetCell < CELL_COUNT && backLinks[backLinksOffset + targetCell] != NONE) {
    var cell = targetCell;
    // Only a walk that ends at a source (code 0) is a path; every other exit reports none.
    var complete = false;
    for (var walkIndex = 0u; walkIndex < CELL_COUNT; walkIndex++) {
      let code = backLinks[backLinksOffset + cell];
      if (code == UNRESOLVED) { unresolved = true; break; }
      if (pathLength < CAPACITY) { ids[idsOffset + pathLength] = cell; }
      pathLength++;
      if (code == 0u) { complete = true; break; }
      if (code > 128u || countOneBits(code) != 1u) { break; }
      let next = getNeighbor(cell, firstTrailingBit(code));
      if (next == NONE || backLinks[backLinksOffset + next] == NONE) { break; }
      cell = next;
    }
    if (!complete) { pathLength = 0u; }
  }
  total[totalOffset] = pathLength;
  ${_?`iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, unresolved);`:``}`}),t(r,{id:`${s}-publish`,operation:g,requiredCount:y,output:p})]}};function v(e,t,n){let r=Math.min(e.columns-1,Math.floor(e.columns*t/100)),i=Math.min(e.rows-1,Math.floor(e.rows*n/100));return{column:r,row:i,index:i*e.columns+r}}var y=p({parameters:m,caption:`The route follows the cheapest valley and detours through a gap in the wall.`,createInputs:e=>({...d(e),lastTarget:[m[0].default,m[1].default]}),build(e,t){let n=f(e,t),r=e.allocate(`target`,`uint32`,1),i=e.allocate(`path-ids`,`uint32`,t.cellCount),a=e.allocate(`path-count`,`uint32`,1);e.graph.add(new _({grid:t.grid,backLinks:n.backLinks.view,target:r.view,output:{path:{ids:i.view,status:{count:a.view,overflow:e.allocate(`overflow`,`uint32`,1).view}}}}));let o=e.upload(`cell-centers`,`float32x2`,t.cellCenters);return{outputs:{costs:n.costs,sources:n.sources,cellCenters:o,pathIds:i,pathCount:a},update:e=>{n.writeSettings({...e,sourceCount:1}),r.write(Uint32Array.of(v(t,e.targetColumn,e.targetRow).index))}}},interaction:{kind:`place`,hint:`move the target`,rest:[m[0].default/100,m[1].default/100],resolve(e,{inputs:t,outputs:n,peek:r}){let i=h(Math.round(e.unitX*100),0,99),a=h(Math.round(e.unitY*100),0,99);t.isWallCell[v(t,i,a).index]?[i,a]=t.lastTarget:t.lastTarget=[i,a];let o=v(t,i,a),s=r(n.costs),c=s?s[o.index]:NaN;return{values:{targetColumn:i,targetRow:a},glyphs:[{shape:`ring`,at:[t.cellCenters[o.index*2],t.cellCenters[o.index*2+1]],radius:t.cellSize*2.5}],readout:`target [${o.column}, ${o.row}]${Number.isFinite(c)?` · cost ${c.toFixed(1)}`:``}`}}},draw:({inputs:e,outputs:t})=>[{layer:`raster`,context:!0,values:t.costs,columns:e.columns,rows:e.rows,ramp:`sequential`,bands:14},{layer:`dots`,context:!0,positions:t.sources,count:1,radius:5,color:[1,1,1,1]},{layer:`paths`,nodes:t.cellCenters,ids:t.pathIds,count:t.pathCount,radius:5,color:[1,.95,.45,1]}]});export{y as default};