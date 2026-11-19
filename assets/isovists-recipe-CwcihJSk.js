import{o as e,u as t}from"./recipe-utils-CICB7qiS.js";import{s as n,u as r}from"./wgsl-common-DfyXiYB0.js";import{E as i,T as a,f as o}from"./gpu-contributor-utils-BIwLQON5.js";import{t as s}from"./classify-viewpoints-BsmQE951.js";import{t as c}from"./urban-landscape-recipe-validation-DTjY7ODf.js";import{a as l,i as u,n as d,r as f,t as p}from"./urban-ray-utils-BSeNtKp8.js";import{t as m}from"./define-run-B-brkCe4.js";import{n as h,t as g}from"./kernel-BXyKsp4B.js";import{a as _,r as v,t as y}from"./urban-D7A1ccno.js";import{Xo as b}from"./entry-preview-D70TdSbK.js";var x=4096,S=65536,C=.01;function w(e,t,n,r){o(e,`rayCount`,t,3,S),a(e,Number.isSafeInteger(n)&&n>0&&n<=16777216,`segments must have between 1 and ${p} rows`),a(e,r===`uniform`||r===`vertices`,`sampling must be 'uniform' or 'vertices'`);let i=t+(r===`vertices`?6*n:0);return a(e,r!==`vertices`||i<=4096,`rayCount plus 6 rays per obstacle segment must be at most ${x} when sampling is 'vertices'`),i}function T(o,p){let m=p.id??`isovists`;i(p.positions,[`float32x2`],`${m} positions`),i(p.segments,[`float32x4`],`${m} segments`);let h=p.maximumDistance,g=p.sampling===`vertices`,_=p.rayCount,v=p.angularEpsilon??1e-5,y=w(m,_,p.segments.length,p.sampling??`uniform`),b=p.tolerance??1e-6,x=p.positions.length,S=p.segments.length;a(m,Number.isFinite(v)&&v>0&&v<C,`angularEpsilon must be a finite number in (0, ${C})`),a(m,x>0,`positions must have at least 1 row`),a(m,Number.isFinite(h)&&h>0,`maximumDistance must be a positive finite number`),a(m,Number.isFinite(b)&&b>0,`tolerance must be a positive finite number`);let T=p.output,E=x*y,D={positions:p.positions,segments:p.segments};c(o,m,D,{endpoints:{view:T?.endpoints,format:`float32x2`,length:E},distances:{view:T?.distances,format:`float32`,length:E},segments:{view:T?.segments,format:`uint32`,length:E},validity:{view:T?.validity,format:`uint32`,length:E},measures:{view:T?.measures,format:`float32x4`,length:x},angles:{view:T?.angles,format:`float32`,length:E},viewpointClassifications:{view:T?.viewpointClassifications,format:`uint32`,length:x}},[{reads:D,writes:{viewpointClassifications:T?.viewpointClassifications}},{reads:D,writes:{angles:T?.angles}},{reads:{...D,angles:T?.angles},writes:{distances:T?.distances,segments:T?.segments}},{reads:{positions:p.positions,angles:T?.angles,viewpointClassifications:T?.viewpointClassifications},writes:{endpoints:T?.endpoints,distances:T?.distances,segments:T?.segments,validity:T?.validity}},{reads:{positions:p.positions,endpoints:T?.endpoints,validity:T?.validity},writes:{measures:T?.measures}}]);let O=t(o,`${m}-angles`,`float32`,E,T?.angles),k=t(o,`${m}-endpoints`,`float32x2`,E,T?.endpoints),A=t(o,`${m}-distances`,`float32`,E,T?.distances),j=t(o,`${m}-segments`,`uint32`,E,T?.segments),M=t(o,`${m}-validity`,`uint32`,E,T?.validity),N=t(o,`${m}-measures`,`float32x4`,x,T?.measures),P=t(o,`${m}-viewpoint-classifications`,`uint32`,x,T?.viewpointClassifications),F=l(p.obstacleIndex,S)===`bvh`?d(o,{id:`${m}-obstacles`,operation:m,segments:p.segments,tolerance:b}):void 0,I=s(o,{id:m,positions:p.positions,segments:p.segments,tolerance:b,obstacleBVH:F,region:p.region,output:P});return o.add(e(o,{id:`${m}-angular-events`,operation:m,invocationCount:x,workgroupSize:2,bindings:[{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`segments`,view:p.segments,type:`f32`,access:`read`},{name:`angles`,view:O,type:`f32`,access:`read_write`}],declarations:n,body:g?`let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
for(var ray=0u;ray<${_}u;ray++){
  angles[anglesOffset+index*${y}u+ray]=6.283185307179586*f32(ray)/${_}.0;
}
for(var endpoint=0u;endpoint<${p.segments.length*2}u;endpoint++){
  let point=vec2<f32>(segments[segmentsOffset+2u*endpoint],segments[segmentsOffset+2u*endpoint+1u])-origin;
  // The second argument is a coordinate difference, which can be -0.0 (contract section 11).
  let center=guardedAtan2(point.y,point.x);
  for(var side=0u;side<3u;side++){
    var angle=center+(f32(side)-1.0)*${v};
    angle=angle-floor(angle/6.283185307179586)*6.283185307179586;
    angles[anglesOffset+index*${y}u+${_}u+3u*endpoint+side]=angle;
  }
}
for(var position=1u;position<${y}u;position++){
  let value=angles[anglesOffset+index*${y}u+position];
  var insertion=position;
  loop{
    if(insertion==0u||angles[anglesOffset+index*${y}u+insertion-1u]<=value){break;}
    angles[anglesOffset+index*${y}u+insertion]=angles[anglesOffset+index*${y}u+insertion-1u];
    insertion--;
  }
  angles[anglesOffset+index*${y}u+insertion]=value;
}`:`for(var ray=0u;ray<${y}u;ray++){
  angles[anglesOffset+index*${y}u+ray]=6.283185307179586*f32(ray)/${y}.0;
}`})),o.add(e(o,{id:`${m}-cast`,operation:m,invocationCount:E,declarations:u(S,b,F),bindings:[{name:`angles`,view:O,type:`f32`,access:`read`},{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`segments`,view:p.segments,type:`f32`,access:`read`},...F?f(F):[],{name:`distances`,view:A,type:`f32`,access:`read_write`},{name:`hits`,view:j,type:`u32`,access:`read_write`}],body:`let viewpoint=index/${y}u;
let angle=angles[anglesOffset+index];
let direction=vec2<f32>(cos(angle),sin(angle));
let origin=vec2<f32>(positions[positionsOffset+2u*viewpoint],positions[positionsOffset+2u*viewpoint+1u]);
let hit=castRay(origin,direction,${h});
distances[distancesOffset+index]=hit.x;
hits[hitsOffset+index]=select(0xffffffffu,u32(max(hit.y,0.0)),hit.y>=0.0);`})),o.add(e(o,{id:`${m}-cast-validity`,operation:m,invocationCount:E,declarations:r,bindings:[{name:`classifications`,view:I,type:`u32`,access:`read`},{name:`angles`,view:O,type:`f32`,access:`read`},{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`endpoints`,view:k,type:`f32`,access:`read_write`},{name:`distances`,view:A,type:`f32`,access:`read_write`},{name:`hits`,view:j,type:`u32`,access:`read_write`},{name:`validity`,view:M,type:`u32`,access:`read_write`}],body:`let viewpoint=index/${y}u;
let angle=angles[anglesOffset+index];
let direction=vec2<f32>(cos(angle),sin(angle));
let origin=vec2<f32>(positions[positionsOffset+2u*viewpoint],positions[positionsOffset+2u*viewpoint+1u]);
let distanceValue=distances[distancesOffset+index];
let hitRow=hits[hitsOffset+index];
let valid=classifications[classificationsOffset+viewpoint]==0u&&isFiniteValue(origin.x)&&isFiniteValue(origin.y)&&distanceValue>${b};
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
let endpoint=origin+direction*distanceValue;
endpoints[endpointsOffset+2u*index]=select(missing,endpoint.x,valid);
endpoints[endpointsOffset+2u*index+1u]=select(missing,endpoint.y,valid);
distances[distancesOffset+index]=select(missing,distanceValue,valid);
hits[hitsOffset+index]=select(0xffffffffu,hitRow,hitRow!=0xffffffffu&&valid);
validity[validityOffset+index]=u32(valid);`})),o.add(e(o,{id:`${m}-measures`,operation:m,invocationCount:x,bindings:[{name:`positions`,view:p.positions,type:`f32`,access:`read`},{name:`endpoints`,view:k,type:`f32`,access:`read`},{name:`validity`,view:M,type:`u32`,access:`read`},{name:`output`,view:N,type:`f32`,access:`read_write`}],body:`let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
var area=0.0;
var perimeter=0.0;
var maximum=0.0;
var valid=true;
for(var ray=0u;ray<${y}u;ray++){
  let first=index*${y}u+ray;
  let second=index*${y}u+(ray+1u)%${y}u;
  valid=valid&&validity[validityOffset+first]!=0u;
  let start=vec2<f32>(endpoints[endpointsOffset+2u*first],endpoints[endpointsOffset+2u*first+1u])-origin;
  let end=vec2<f32>(endpoints[endpointsOffset+2u*second],endpoints[endpointsOffset+2u*second+1u])-origin;
  area+=0.5*(start.x*end.y-start.y*end.x);
  perimeter+=distance(start,end);
  maximum=max(maximum,length(start));
}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
output[outputOffset+4u*index]=select(missing,area,valid);
output[outputOffset+4u*index+1u]=select(missing,perimeter,valid);
output[outputOffset+4u*index+2u]=select(missing,maximum,valid);
output[outputOffset+4u*index+3u]=f32(valid);`})),{outputs:{endpoints:k,distances:A,segments:j,validity:M,measures:N,angles:O,viewpointClassifications:I},provenance:{positions:p.positions,obstacles:p.segments,rayCount:y,baseRayCount:_,maximumDistance:h,angularSpacing:2*Math.PI/_,sampling:p.sampling??`uniform`,angularEpsilon:v,tolerance:b,obstacleIndex:F?`bvh`:`brute-force`}}}var E=2,D=m({parameters:b,caption:`Visible area from two viewpoints, one under the pointer: rays hit a wall at the first building wall.`,createInputs:({world:e})=>{let t=y(e,24),n=_(t,E);return{layout:t,viewpoints:n,initial:[n[0],n[1]]}},build(e,t,n){let r=v(t.layout,n.fill),i=n.rayCount,a=e.upload(`viewpoints`,`float32x2`,t.viewpoints),o=e.upload(`walls`,`float32x4`,r.segments),s=T(e.graph,{positions:a.view,segments:o.view,rayCount:i,maximumDistance:n.maximumDistance,sampling:`uniform`}),c=E*i,l=e.allocate(`ray-nodes`,`float32x2`,c);g(e,{id:`isovist-publish-pass`,invocationCount:c,bindings:[{name:`endpoints`,buffer:h(s.outputs.endpoints),access:`read`},{name:`nodes`,buffer:l,access:`read_write`}],body:`
  let point = endpoints[endpointsOffset + index];
  nodes[nodesOffset + index] = select(vec2f(-9.0), point, point.x == point.x && point.y == point.y);`});let u=e.upload(`ray-edges`,`uint32x2`,Uint32Array.from({length:c*2},(e,t)=>{let n=Math.floor(t/2),r=Math.floor(n/i),a=n%i;return r*i+(t%2==0?a:(a+1)%i)})),d=e.upload(`ray-owners`,`uint32`,Uint32Array.from({length:c},(e,t)=>Math.floor(t/i))),f=e.allocate(`rays`,`float32x4`,c);return g(e,{id:`isovist-ray-pass`,invocationCount:c,bindings:[{name:`viewpoints`,buffer:a,access:`read`},{name:`nodes`,buffer:l,access:`read`},{name:`rays`,buffer:f,access:`read_write`}],body:`
  let end = nodes[nodesOffset + index];
  rays[raysOffset + index] = select(vec4f(-9.0), vec4f(viewpoints[viewpointsOffset + index / ${i}u], end), end.x > -8.0);`}),{outputs:{walls:e.upload(`wall-segments`,`float32x4`,r.segments),nodes:l,edges:u,owners:d,rays:f,positions:a},update:()=>a.write(t.viewpoints)}},rebuildOn:[`maximumDistance`,`rayCount`,`fill`],interaction:{kind:`place`,hint:`move the viewpoint`,rest:({inputs:e,toUnit:t})=>t(e.initial),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){t.viewpoints[0]=e.x,t.viewpoints[1]=e.y;let a=i(n.nodes),o=0;if(a)for(let t=0;t<r.rayCount;t++){let n=Math.hypot(a[t*2]-e.x,a[t*2+1]-e.y);a[t*2]>-8&&n<r.maximumDistance-.001&&o++}return{rerun:!0,glyphs:[{shape:`ring`,at:[e.x,e.y],radius:r.maximumDistance,dashed:!0},{shape:`crosshair`,at:[e.x,e.y]}],readout:`maximumDistance ${r.maximumDistance.toFixed(2)}${a?` · ${o} of ${r.rayCount} rays stop`:``}`}}},draw:({outputs:e})=>[{layer:`lines`,context:!0,segments:e.walls,radius:2.5},{layer:`segments`,nodes:e.nodes,edges:e.edges,values:e.owners,radius:2},{layer:`lines`,segments:e.rays,alpha:.2,radius:.8},{layer:`dots`,positions:e.positions,values:void 0,radius:6,color:[1,1,1,1]}]});export{D as default};