import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as r,x as i}from"./gpu-contributor-utils-BIwLQON5.js";import{c as a,l as o,n as s,r as c,u as l}from"./raster-band-utils-BcTxc8_Y.js";import{t as u}from"./define-run-B-brkCe4.js";import{r as d}from"./text-DYSgVV-n.js";import{n as f,r as p,t as m}from"./class-highlight-WAHL_zPO.js";import{a as h,d as g,i as _,l as v,m as y,n as b,o as x,u as S}from"./terrain-Cc9dW1uq.js";import{D as C}from"./entry-preview-D70TdSbK.js";var w={regular:0,peak:1,pit:2,saddle:3,boundary:4,noData:5},T=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`terrain-critical-points`,this.props=e;let{id:t}=this,{output:n}=e,i=l(t,e.grid);if(e.connectivity!==void 0&&e.connectivity!==8&&e.connectivity!==6)throw Error(`${t} connectivity must be 8 or 6`);if(r(n.classes,[`uint32`],`${t} output.classes`),n.classes.length!==i)throw Error(`${t} output.classes must contain one value per pixel`);if(n.signChanges&&(r(n.signChanges,[`uint32`],`${t} output.signChanges`),n.signChanges.length!==i))throw Error(`${t} output.signChanges must contain one value per pixel`);if(n.counts&&(r(n.counts,[`uint32`],`${t} output.counts`),n.counts.length!==6))throw Error(`${t} output.counts must contain one value per class (6)`);o(t,[n.classes,n.signChanges,n.counts],s(e.elevation))}getCommandNodes(r){e(r,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height});let{id:o,props:s}=this,{output:l}=s,{width:u,height:d}=s.grid;a(o,r,s.elevation,[]),i(o,r,[l.classes,l.signChanges,l.counts]);let f=c(r,o,s.elevation,s.grid,!0),p=[...f.nodes];l.counts&&p.push(n(r,{id:`${o}-clear-counts`,operation:`GPUTerrainCriticalPoints`,view:l.counts,type:`u32`,value:`0u`}));let m=(s.connectivity??8)===8?E:D,h=[{name:`elevationValues`,view:f.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:f.band.validity,type:`u32`,access:`read`},{name:`classValues`,view:l.classes,type:`u32`,access:`read_write`},...l.signChanges?[{name:`signChangeValues`,view:l.signChanges,type:`u32`,access:`read_write`}]:[],...l.counts?[{name:`classCounts`,view:l.counts,type:`atomic<u32>`,access:`read_write`}]:[]],g=m.map(([e,t],n)=>`let side${n} = getSide(column, row, ${e}, ${t}, centerHeight, index);`).join(`
  `),_=m.map((e,t)=>`side${t} == 2u`).join(` || `),v=m.map((e,t)=>`select(0u, 1u, side${t} != side${(t+1)%m.length})`).join(` +
    `),y=m.map((e,t)=>`side${t}`).join(` + `);return p.push(t(r,{id:`${o}-classify`,operation:`GPUTerrainCriticalPoints`,variant:`ring-${m.length}`,bindings:h,invocationCount:u*d,declarations:`const WIDTH: u32 = ${u}u;
const HEIGHT: u32 = ${d}u;
const RING_COUNT: u32 = ${m.length}u;
const REGULAR: u32 = ${w.regular}u;
const PEAK: u32 = ${w.peak}u;
const PIT: u32 = ${w.pit}u;
const SADDLE: u32 = ${w.saddle}u;
const BOUNDARY: u32 = ${w.boundary}u;
const NO_DATA: u32 = ${w.noData}u;
// 1 when the neighbor is higher (simulation of simplicity), 0 when lower, 2 when it is invalid.
// The neighbor is always inside the grid because edge pixels return earlier.
fn getSide(column: u32, row: u32, columnOffset: i32, rowOffset: i32, centerHeight: f32, centerIndex: u32) -> u32 {
  let neighborIndex = u32(i32(row) + rowOffset) * WIDTH + u32(i32(column) + columnOffset);
  if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) { return 2u; }
  let neighborHeight = elevationValues[elevationValuesOffset + neighborIndex];
  return select(0u, 1u, neighborHeight > centerHeight || (neighborHeight == centerHeight && neighborIndex > centerIndex));
}`,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  var classCode = NO_DATA;
  var changeCount = 0u;
  if (elevationValidity[elevationValidityOffset + index] != 0u) {
    if (column == 0u || row == 0u || column == WIDTH - 1u || row == HEIGHT - 1u) {
      classCode = BOUNDARY;
    } else {
      let centerHeight = elevationValues[elevationValuesOffset + index];
      ${g}
      if (${_}) {
        classCode = BOUNDARY;
      } else {
        changeCount = ${v};
        let higherCount = ${y};
        classCode = select(SADDLE, REGULAR, changeCount == 2u);
        if (changeCount == 0u) {
          classCode = select(PEAK, PIT, higherCount == RING_COUNT);
        }
      }
    }
  }
  classValues[classValuesOffset + index] = classCode;
  ${l.signChanges?`signChangeValues[signChangeValuesOffset + index] = changeCount;`:``}
  ${l.counts?`atomicAdd(&classCounts[classCountsOffset + classCode], 1u);`:``}`})),p}},E=[[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1]],D=[[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]],O=[[0,0,0,0],[.9,.2,.15,1],[.15,.4,.95,1],[.95,.75,.1,1],[0,0,0,0],[0,0,0,0]],k=5,A=3,j=[`regular`,`peak`,`pit`,`saddle`,`boundary`,`no data`],M=u({parameters:C,rebuildOn:[`connectivity`],caption:`Noisy elevation in; peaks, pits and saddles per cell, and their counts, out.`,createInputs:({random:e,world:t})=>{let n=_(t,48);return{...n,noise:Float32Array.from(n.elevation,()=>e()),maskState:m()}},build(e,t,n){let{grid:r,band:i,elevation:a,cellCount:o}=y(e,t),s=e.allocate(`classes`,`uint32`,o),c=e.allocate(`counts`,`uint32`,6),l=e.allocate(`sign-changes`,`uint32`,o),u=e.allocate(`highlight`,`float32`,o);return e.graph.add(new T({grid:r,elevation:i,connectivity:n.connectivity===1?6:8,output:{classes:s.view,counts:c.view,signChanges:l.view}})),{outputs:{elevation:a,classes:s,counts:c,signChanges:l,highlight:u},update:e=>a.write(b(t,t.noise,e.noise))}},async afterRun({outputs:e}){let t=await e.counts.buffer.readAsync();return new Uint32Array(t.slice().buffer)},interaction:{kind:`probe`,hint:`hover a peak or pit`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,world:r,peek:i}){let[a,o]=v(t,r,e.x,e.y),s=i(n.classes),c=i(n.counts),l=g(t,r),u=-1;if(s){let e=1/0;for(let n=-5;n<=k;n++)for(let r=-5;r<=k;r++){let i=a+r,c=o+n;if(i<0||c<0||i>=t.columns||c>=t.rows)continue;let l=s[c*t.columns+i],d=Math.hypot(r,n)+(l===3?A:0);l>=1&&l<=3&&d<e&&(e=d,u=c*t.columns+i)}}if(!s)return{spotlight:null,readout:`peaks, pits, saddles`};if(u<0)return p(n.highlight,s,-1,t.maskState),{spotlight:null,glyphs:[{shape:`dot`,at:[e.x,e.y]}],readout:c?`regular · ${c[1]} peaks ${c[2]} pits ${c[3]} saddles`:`regular`};let d=s[u];p(n.highlight,s,d,t.maskState);let f=S(t,r,u%t.columns,Math.floor(u/t.columns)),m=i(n.signChanges),h=1.5*l;return{spotlight:null,glyphs:[{shape:`box`,minimum:[f[0]-h,f[1]-h],maximum:[f[0]+h,f[1]+h]}],readout:`${j[d]}${m?` · ${m[u]} sign changes`:``}${c?` · ${c[d]} cells`:``}`}}},draw:({inputs:e,outputs:t,theme:n,extra:r})=>[h(e,t.elevation),x(e,t.classes,{ramp:`categorical`,categories:O}),f(e,t.highlight,[n.background[0],n.background[1],n.background[2],.6]),{layer:`text`,...d({title:r?`cells per class`:void 0,corner:`bottom-right`,entries:[1,2,3].map(e=>({label:`${j[e]}${r?` ${r[e]}`:``}`,swatch:[O[e][0],O[e][1],O[e][2]]}))})}]});export{M as default};