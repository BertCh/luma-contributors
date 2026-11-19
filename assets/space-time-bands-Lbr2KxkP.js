import{n as e}from"./temporal-context-xddVhAJD.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";var n=`0xffffffffu`;function r(e,t,n){if(n.length<2||n.length>17)throw Error(`${e} ${t} needs 2 to 17 edges`);n.forEach((r,i)=>{if(!Number.isFinite(Math.fround(r))||r<0)throw Error(`${e} ${t} edges must be finite and non-negative`);if(i>0&&Math.fround(r)<=Math.fround(n[i-1]))throw Error(`${e} ${t} edges must be strictly ascending as float32`)})}function i(e){let t=String(Math.fround(e));return/[.e]/.test(t)?t:`${t}.0`}function a(e,t){let r=t.slice(1,-1).map((e,t)=>`  if (value > ${i(e)}) { band = ${t+1}u; }`);return`
fn get${e}Band(value: f32) -> u32 {
  if (value < ${i(t[0])} || value > ${i(t[t.length-1])}) {
    return ${n};
  }
  var band = 0u;
${r.join(`
`)}
  return band;
}
`}function o(t,n){return n&&e(t,n,{encodings:[`float32-relative`]})}function s(e,n){return t(e,{id:`${n.id}-entry-mask`,operation:n.operation,variant:`entry-mask`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`entryMask`,view:n.entryMask,type:`u32`,access:`read_write`}],invocationCount:n.rows,declarations:`const ROWS: u32 = ${n.rows}u;`,body:`let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var entry = start; entry < end; entry++) {
    let other = neighbors[neighborsOffset + entry];
    var counted = other < ROWS && other != index;
    if (counted) {
      for (var earlier = start; earlier < entry; earlier++) {
        if (neighbors[neighborsOffset + earlier] == other) {
          counted = false;
          break;
        }
      }
    }
    entryMask[entryMaskOffset + entry] = select(0u, 1u, counted);
  }`})}export{r as a,o as i,s as n,a as r,n as t};