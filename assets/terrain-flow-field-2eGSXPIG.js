import{r as e}from"./terrain-binding-preflight-laSEObQE.js";import{n as t,r as n}from"./raster-grid-utils-D2opjNmg.js";import{o as r}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as i,x as a,y as o}from"./gpu-contributor-utils-BIwLQON5.js";import{u as s}from"./raster-grid-VB0AM_GY.js";import{c,d as l,h as u,n as d,o as f,r as p,u as m}from"./raster-band-utils-BcTxc8_Y.js";import{n as h,r as g,t as _}from"./gpu-line-integral-convolution-CpssL2pb.js";import{n as v,t as y}from"./gpu-raster-grid-C_izEAl0.js";import{t as b}from"./define-run-B-brkCe4.js";import{i as x}from"./pointer-overlay-Qh8qflcA.js";import{nt as S}from"./entry-preview-D70TdSbK.js";import{d as C,o as w}from"./terrain-Bwc0vRX3.js";var T=`GPUTerrainFlowField`;function E(e,t,r=new Float32Array(8)){return f(`getGPUTerrainFlowFieldParameterValues`,r,8),n(r,t),r[4]=e.wind[0],r[5]=e.wind[1],r[6]=e.verticalExaggeration??1,r[7]=0,r}var D=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`terrain-flow-field`,this.props=e;let{id:t}=this,{output:n}=e,r=m(t,e.grid);if(u(t,e.parameters,8),s(t,e.cellSizeMode),l(t,e.grid,e.cellSizeMode),i(n.velocities,[`float32x2`],`${t} output.velocities`),n.velocities.length!==r)throw Error(`${t} output.velocities must contain one row per cell`);o(t,[n.velocities],[...d(e.elevation),e.parameters])}getCommandNodes(n){e(n,this.id,this.props,{"canonical elevation":4*this.props.grid.width*this.props.grid.height});let{id:i,props:o}=this,{output:s}=o,{width:l,height:u}=o.grid;c(i,n,o.elevation,[]),a(i,n,[o.parameters,s.velocities]);let d=p(n,i,o.elevation,o.grid,!0),f=d.band.storage.values,m={width:l,height:u,cellSizeMode:o.cellSizeMode??`uniform`},h=r(n,{id:`${i}-deflect`,operation:T,variant:`deflect`,bindings:[{name:`elevation`,view:f,type:`f32`,access:`read`},{name:`settings`,view:o.parameters,type:`f32`,access:`read`},{name:`velocities`,view:s.velocities,type:`f32`,access:`read_write`}],invocationCount:l*u,declarations:`${t(m)}
// Direction indices of the grid helpers: 2 = next row, 6 = previous row.
const DIRECTION_NEXT_ROW: u32 = 2u;
const DIRECTION_PREVIOUS_ROW: u32 = 6u;`,body:`let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  let column0 = select(column - 1u, 0u, column == 0u);
  let column1 = min(column + 1u, GRID_WIDTH - 1u);
  let row0 = select(row - 1u, 0u, row == 0u);
  let row1 = min(row + 1u, GRID_HEIGHT - 1u);
  let center = elevation[elevationOffset + index];
  let previousColumn = elevation[elevationOffset + row * GRID_WIDTH + column0];
  let nextColumn = elevation[elevationOffset + row * GRID_WIDTH + column1];
  let previousRow = elevation[elevationOffset + row0 * GRID_WIDTH + column];
  let nextRow = elevation[elevationOffset + row1 * GRID_WIDTH + column];
  let output = 2u * index + velocitiesOffset;
  if (!isFiniteValue(center) || !isFiniteValue(previousColumn) || !isFiniteValue(nextColumn) ||
      !isFiniteValue(previousRow) || !isFiniteValue(nextRow)) {
    velocities[output] = getQuietNaN();
    velocities[output + 1u] = getQuietNaN();
    return;
  }
  let exaggeration = settings[settingsOffset + 6u];
  var gradientX = 0.0;
  if (column1 > column0) {
    let spanX = f32(column1 - column0) * getGroundCellSizeAtRowCoordinate(f32(row) + 0.5).x;
    gradientX = exaggeration * ((nextColumn - previousColumn) / spanX);
  }
  var gradientY = 0.0;
  if (row1 > row0) {
    var spanY = 0.0;
    if (row1 > row) { spanY = spanY + getD8Distance(DIRECTION_NEXT_ROW, row); }
    if (row0 < row) { spanY = spanY + getD8Distance(DIRECTION_PREVIOUS_ROW, row); }
    gradientY = exaggeration * ((nextRow - previousRow) / spanY);
  }
  let wind = vec2<f32>(settings[settingsOffset + 4u], settings[settingsOffset + 5u]);
  let gradient = vec2<f32>(gradientX, gradientY);
  let deflection = dot(wind, gradient) / (1.0 + dot(gradient, gradient));
  let velocity = wind - deflection * gradient;
  velocities[output] = velocity.x;
  velocities[output + 1u] = velocity.y;`});return[...d.nodes,h]}},O=2,k=b({parameters:S,caption:`Wind bends around ridges and speeds over summits; noise is streaked along the vectors.`,createInputs:e=>{let t=w(e),n=v({width:t.columns,height:t.rows,cellSize:t.cellSize,rowDirection:`north`});return{...t,grid:n}},build(e,t){let{columns:n,rows:r,cellCount:i,cellSize:a,grid:o}=t,s=e.upload(`dem`,`float32`,t.elevation),c=e.allocate(`flow-field-parameters`,`float32`,8),l=e.allocate(`velocities`,`float32x2`,i);e.graph.add(new D({grid:o,elevation:y(s.view,o,{id:`dem`}),parameters:c.view,output:{velocities:l.view}}));let u=i*O*O,d=e.allocate(`lic-parameters`,`float32`,12),f=e.allocate(`lic-words`,`uint32`,4),p=e.allocate(`values`,`float32`,u),m=e.allocate(`speeds`,`float32`,u);return e.graph.add(new _({velocities:l.view,fieldWidth:n,fieldHeight:r,grid:v({width:n*O,height:r*O}),stepCount:20,parameters:d.view,uint32Parameters:f.view,output:{values:p.view,speeds:m.view}})),{outputs:{elevation:s,values:p,speeds:m},update:e=>{let t=e.windDirection*Math.PI/180;c.write(E({wind:[Math.cos(t),Math.sin(t)],verticalExaggeration:e.verticalExaggeration},o));let n={fieldExtent:[0,0,a,a],outputExtent:[0,0,a/O,a/O],stepLength:.5};d.write(h(n)),f.write(g({seed:0}))}}},interaction:x(S,{hint:`aim the wind`,angle:`windDirection`,convention:`math`,distance:{parameter:`verticalExaggeration`},readout:e=>`wind ${e.windDirection} deg · verticalExaggeration ${e.verticalExaggeration}`}),draw:({inputs:e,outputs:t})=>[C(e,t.elevation),{layer:`raster`,values:t.speeds,columns:e.columns*O,rows:e.rows*O,ramp:`diverging`,range:[0,2],alpha:.55,smoothing:!0},{layer:`raster`,values:t.values,columns:e.columns*O,rows:e.rows*O,ramp:`sequential`,range:[.2,.8],alpha:.4}]});export{k as default};