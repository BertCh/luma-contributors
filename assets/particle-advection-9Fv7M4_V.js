import{C as e,s as t,x as n,y as r}from"./gpu-contributor-utils-BIwLQON5.js";import{n as i,t as a}from"./philox-random-DfVxWNPn.js";import{n as o}from"./raster-storage-preflight-CIxFc-0g.js";import{d as s,u as c}from"./kernel-C3tfQOn4.js";import{o as l}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as u}from"./vector-field-sampling-KzXxZj8z.js";import{t as d}from"./define-run-B-brkCe4.js";import{o as f}from"./pointer-overlay-Qh8qflcA.js";import{ut as p}from"./entry-preview-D70TdSbK.js";import{t as m}from"./vector-field-DwVuDUyq.js";var h=a.particleAdvectionSpawn.id,g=a.particleAdvectionDrop.id;function _(t,n,r=new Float32Array(12)){if(r.length<12)throw Error(`Particle advection parameter target must hold 12 elements`);let[i,a,o,s]=t.fieldExtent;e(`GPUParticleAdvection`,`fieldExtent cellWidth`,o),e(`GPUParticleAdvection`,`fieldExtent cellHeight`,s);let c=t.spawnBounds??[i,a,i+o*n[0],a+s*n[1]];return r.set([i,a,o,s,t.timeStep,t.speedScale??1,t.dropRate??0,t.minimumSpeed??0,c[0],c[1],c[2],c[3]]),r}function v(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`Particle advection uint32 parameter target must hold 4 elements`);for(let[t,n]of[[`seed`,e.seed],[`frame`,e.frame],[`maximumAge`,e.maximumAge]])if(!Number.isInteger(n)||n<0||n>4294967295)throw Error(`Particle advection ${t} must be a uint32 integer`);return t[0]=e.seed,t[1]=e.frame,t[2]=e.maximumAge,t[3]=+!!e.reset,t}var y=`GPUParticleAdvection`,b=class{id;particleCount;props;constructor(e){let t=e.id??`particle-advection`,{fieldWidth:n,fieldHeight:i,state:a}=e,o=e.output??{},{trails:l}=o;for(let[e,r]of[[`fieldWidth`,n],[`fieldHeight`,i]])if(!Number.isInteger(r)||r<1||r>65535)throw Error(`${t} ${e} must be an integer in [1, 65535]`);if(s(e.velocities,[`float32x2`],`${t} velocities`),e.velocities.length!==n*i)throw Error(`${t} velocities must have fieldWidth * fieldHeight rows`);if(s(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<12)throw Error(`${t} parameters must contain 12 float32 rows`);if(c(e.uint32Parameters,`${t} uint32Parameters`),e.uint32Parameters.length<4)throw Error(`${t} uint32Parameters must contain 4 uint32 rows`);s(a.positions,[`float32x2`],`${t} state.positions`);let u=a.positions.length;if(u<1)throw Error(`${t} needs at least one particle`);c(a.ages,`${t} state.ages`),c(a.generations,`${t} state.generations`),o.previousPositions&&s(o.previousPositions,[`float32x2`],`${t} output.previousPositions`),o.speeds&&s(o.speeds,[`float32`],`${t} output.speeds`);for(let[e,n]of[[`state.ages`,a.ages],[`state.generations`,a.generations],[`output.previousPositions`,o.previousPositions],[`output.speeds`,o.speeds]])if(n&&n.length!==u)throw Error(`${t} ${e} must have one row per particle`);if(l){if(!Number.isInteger(l.length)||l.length<1)throw Error(`${t} output.trails.length must be a positive integer`);if(s(l.positions,[`float32x2`],`${t} output.trails.positions`),l.positions.length!==u*l.length)throw Error(`${t} output.trails.positions must have particleCount * trails.length rows`)}r(t,[a.positions,a.ages,a.generations,o.previousPositions,o.speeds,l?.positions],[e.velocities,e.parameters,e.uint32Parameters]),this.id=t,this.particleCount=u,this.props=e}getCommandNodes(e){let r=this.props;o(e,this.id,[r.velocities,r.parameters,r.uint32Parameters,...Object.values(r.state),...t(r.output??{}),r.output?.trails?.positions]);let{id:a,props:s,particleCount:c}=this,{state:d}=s,f=s.output??{},{trails:p}=f;n(a,e,[s.velocities,s.parameters,s.uint32Parameters,d.positions,d.ages,d.generations,f.previousPositions,f.speeds,p?.positions]);let m=[{name:`velocities`,view:s.velocities,type:`f32`,access:`read`},{name:`parameters`,view:s.parameters,type:`f32`,access:`read`},{name:`words`,view:s.uint32Parameters,type:`u32`,access:`read`},{name:`positions`,view:d.positions,type:`f32`,access:`read_write`},{name:`ages`,view:d.ages,type:`u32`,access:`read_write`},{name:`generations`,view:d.generations,type:`u32`,access:`read_write`}];f.previousPositions&&m.push({name:`previousPositions`,view:f.previousPositions,type:`f32`,access:`read_write`}),f.speeds&&m.push({name:`speeds`,view:f.speeds,type:`f32`,access:`read_write`});let _=[l(e,{id:`${a}-step`,operation:y,variant:`step`,bindings:m,invocationCount:c,declarations:`${i}
${u(`velocities`,s.fieldWidth,s.fieldHeight)}
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getSpeed(sample: vec2<f32>, speedScale: f32) -> f32 { return length(sample) * abs(speedScale); }`,body:`
  let extent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let timeStep = readParameter(4u);
  let speedScale = readParameter(5u);
  let dropRate = readParameter(6u);
  let minimumSpeed = readParameter(7u);
  let spawnMin = vec2<f32>(readParameter(8u), readParameter(9u));
  let spawnMax = vec2<f32>(readParameter(10u), readParameter(11u));
  let seed = words[wordsOffset];
  let frame = words[wordsOffset + 1u];
  let maximumAge = words[wordsOffset + 2u];
  let reset = words[wordsOffset + 3u] != 0u;
  let step = timeStep * speedScale;
  let halfStep = 0.5 * step;
  var position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var previous = position;
  var age = ages[agesOffset + index];
  var generation = generations[generationsOffset + index];
  var respawn = reset;
  var speed = 0.0;
  if (reset) {
    generation = 0u;
  } else {
    let k1 = sampleField(position, extent);
    let middle = position + halfStep * k1.xy;
    let k2 = sampleField(middle, extent);
    let next = position + step * k2.xy;
    let landing = sampleField(next, extent);
    let dropWord = philox4x32(vec4<u32>(index, frame, 0u, 0u), vec2<u32>(seed, ${g}u)).x;
    let expired = maximumAge != 0u && age + 1u >= maximumAge;
    let tooSlow = k2.z != 0.0 && getSpeed(k2.xy, speedScale) < minimumSpeed;
    if (k1.z == 0.0 || k2.z == 0.0 || landing.z == 0.0 || expired || tooSlow ||
        philoxUnitFloat(dropWord) < dropRate) {
      respawn = true;
      generation = generation + 1u;
    } else {
      position = next;
      age = age + 1u;
      speed = getSpeed(landing.xy, speedScale);
    }
  }
  if (respawn) {
    let random = philox4x32(vec4<u32>(index, generation, 0u, 0u), vec2<u32>(seed, ${h}u));
    position = spawnMin + vec2<f32>(philoxUnitFloat(random.x), philoxUnitFloat(random.y)) * (spawnMax - spawnMin);
    previous = position;
    age = 0u;
    if (reset) {
      age = min(u32(floor(philoxUnitFloat(random.z) * f32(maximumAge))), max(maximumAge, 1u) - 1u);
    }
    let landing = sampleField(position, extent);
    speed = select(0.0, getSpeed(landing.xy, speedScale), landing.z != 0.0);
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;
  ages[agesOffset + index] = age;
  generations[generationsOffset + index] = generation;
  ${f.previousPositions?`previousPositions[previousPositionsOffset + 2u * index] = previous.x;
  previousPositions[previousPositionsOffset + 2u * index + 1u] = previous.y;`:`_ = previous;`}
  ${f.speeds?`speeds[speedsOffset + index] = speed;`:`_ = speed;`}`})];return p&&_.push(l(e,{id:`${a}-trails`,operation:y,variant:`trails`,bindings:[{name:`words`,view:s.uint32Parameters,type:`u32`,access:`read`},{name:`positions`,view:d.positions,type:`f32`,access:`read`},{name:`ages`,view:d.ages,type:`u32`,access:`read`},{name:`trails`,view:p.positions,type:`f32`,access:`read_write`}],invocationCount:c,declarations:`const TRAIL_LENGTH: u32 = ${p.length}u;`,body:`
  let frame = words[wordsOffset + 1u];
  let reset = words[wordsOffset + 3u] != 0u;
  let x = positions[positionsOffset + 2u * index];
  let y = positions[positionsOffset + 2u * index + 1u];
  let base = trailsOffset + 2u * index * TRAIL_LENGTH;
  if (reset || ages[agesOffset + index] == 0u) {
    for (var slot = 0u; slot < TRAIL_LENGTH; slot = slot + 1u) {
      trails[base + 2u * slot] = x;
      trails[base + 2u * slot + 1u] = y;
    }
  } else {
    let slot = frame % TRAIL_LENGTH;
    trails[base + 2u * slot] = x;
    trails[base + 2u * slot + 1u] = y;
  }`})),_}},x=1500,S=6,C=.07,w=[.62,.42],T=d({parameters:p,caption:`Particles released in the box, advected along the wind; trails show where they went.`,createInputs:e=>{let t=m(e);return{...t,release:[...w],maximumSpeed:Math.max(...t.operationSpeeds)}},build(e,t){let n=t.columns*t.cellSize,r=e.upload(`velocities`,`float32x2`,t.velocities),i=e.upload(`field-speeds`,`float32`,t.speeds),a=e.allocate(`parameters`,`float32`,12),o=e.allocate(`words`,`uint32`,4),s=e.allocate(`positions`,`float32x2`,x),c=Array.from({length:S},(t,n)=>e.allocate(`frame-${n}-positions`,`float32x2`,x)),u=e.allocate(`speeds`,`float32`,x),d={positions:s.view,ages:e.allocate(`ages`,`uint32`,x).view,generations:e.allocate(`generations`,`uint32`,x).view},p=e.allocate(`reset-words`,`uint32`,4);for(let n=0;n<S;n++)e.graph.add(new b({id:`particle-advection-${n}`,velocities:r.view,fieldWidth:t.columns,fieldHeight:t.rows,parameters:a.view,uint32Parameters:n===0?p.view:o.view,state:d,output:{previousPositions:c[n].view,speeds:n===5?u.view:void 0}}));let m=e.allocate(`jump-limit`,`float32`,1),h=c.map((t,n)=>e.allocate(`trails-${n}`,`float32x4`,x));return e.graph.add({getCommandNodes:e=>h.map((t,n)=>l(e,{id:`particle-trails-${n}`,operation:`particle-trails`,bindings:[{name:`startPoints`,view:c[n].view,type:`f32`,access:`read`},{name:`endPoints`,view:(c[n+1]??s).view,type:`f32`,access:`read`},{name:`limit`,view:m.view,type:`f32`,access:`read`},{name:`trail`,view:t.view,type:`f32`,access:`read_write`}],invocationCount:x,body:`let x0 = startPoints[startPointsOffset + index * 2u];
  let y0 = startPoints[startPointsOffset + index * 2u + 1u];
  let x1 = endPoints[endPointsOffset + index * 2u];
  let y1 = endPoints[endPointsOffset + index * 2u + 1u];
  let isJump = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) > limit[limitOffset] * limit[limitOffset];
  let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
  trail[trailOffset + index * 4u] = select(x0, missing, isJump);
  trail[trailOffset + index * 4u + 1u] = select(y0, missing, isJump);
  trail[trailOffset + index * 4u + 2u] = select(x1, missing, isJump);
  trail[trailOffset + index * 4u + 3u] = select(y1, missing, isJump);`}))}),{outputs:{speedField:i,positions:s,speeds:u,trails:h},update:e=>{let r={fieldExtent:[0,0,t.cellSize,t.cellSize],timeStep:e.timeStep,speedScale:e.speedScale,dropRate:e.dropRate,spawnBounds:[f(t.release[0]-C,0,n),f(t.release[1]-C,0,1),f(t.release[0]+C,0,n),f(t.release[1]+C,0,1)]};a.write(_(r,[t.columns,t.rows])),m.write(Float32Array.of(t.maximumSpeed*e.timeStep*e.speedScale*1.3)),p.write(v({seed:7,frame:0,maximumAge:0,reset:!0})),o.write(v({seed:7,frame:1,maximumAge:0}))}}},interaction:{kind:`place`,hint:`release particles`,rest:({toUnit:e})=>e(w),resolve(e,{inputs:t,values:n,world:r}){let i=f(e.x,C,r.width-C),a=f(e.y,C,r.height-C),o=t.release[0]!==i||t.release[1]!==a;return t.release=[i,a],{rerun:o,glyphs:[{shape:`box`,minimum:[i-C,a-C],maximum:[i+C,a+C],dashed:!0},{shape:`crosshair`,at:[i,a]}],readout:`release (${i.toFixed(2)}, ${a.toFixed(2)}) · timeStep ${n.timeStep}`}}},draw:({inputs:e,outputs:t,values:n})=>[{layer:`raster`,context:!0,values:t.speedField,columns:e.columns,rows:e.rows,smoothing:!0},...t.trails.map(e=>({layer:`lines`,segments:e,radius:.9,color:[1,1,1,.22]})),{layer:`dots`,positions:t.positions,values:t.speeds,ramp:`sequential`,range:[0,e.maximumSpeed*n.speedScale],radius:2.4}]});export{T as default};