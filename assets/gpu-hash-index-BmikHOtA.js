import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{c as t,i as n,o as r,r as i,t as a,u as o}from"./kernel-C3tfQOn4.js";import{n as s,r as c}from"./gpu-command-dispatch-metadata-Bpi5I1Ez.js";var l=256,u=4,d=4294967295,f=d,p=class{id;keys;values;firstValue;tableKeys;tableValues;statistics;maxProbeCount;stats;updatePolicy=`rebuild`;constructor(e){this.id=e.id??`gpu-hash-index`,this.keys=e.keys,this.values=e.values,this.firstValue=e.firstValue??0,this.tableKeys=e.tableKeys,this.tableValues=e.tableValues,this.statistics=e.statistics,this.maxProbeCount=e.maxProbeCount??this.tableKeys.length;for(let[e,t]of[[this.keys,`keys`],...this.values?[[this.values,`values`]]:[],[this.tableKeys,`tableKeys`],[this.tableValues,`tableValues`],[this.statistics,`statistics`]])for(let n of c(e))o(n,`${this.id} ${t}`);if(!C(this.tableKeys.length))throw Error(`${this.id} table capacity must be a positive power of two`);if(this.tableKeys.length>d)throw Error(`${this.id} table capacity must fit in uint32`);if(this.tableValues.length!==this.tableKeys.length)throw Error(`${this.id} table key and value capacities must match`);if(this.statistics.length<6)throw Error(`${this.id} statistics must contain six uint32 rows`);if(this.values&&this.values.length!==this.keys.length)throw Error(`${this.id} keys and values lengths must match`);if(this.values&&e.firstValue!==void 0)throw Error(`${this.id} values and firstValue are mutually exclusive`);if(!Number.isSafeInteger(this.firstValue)||this.firstValue<0||this.firstValue>d||this.keys.length>0&&this.firstValue+this.keys.length-1>d)throw Error(`${this.id} generated values must fit in uint32`);x(this.id,this.maxProbeCount,this.tableKeys.length,this.keys.length),S(this.id,[this.keys,...this.values?[this.values]:[]],[this.tableKeys,this.tableValues,this.statistics]),this.stats=Object.freeze({capacity:this.tableKeys.length,maxProbeCount:this.maxProbeCount,tableByteLength:this.tableKeys.length*8,statisticsByteLength:24,outputByteLength:this.tableKeys.length*8+24})}getCommandNodes(e){let t=[];if([this.keys,...this.values?[this.values]:[],this.tableKeys,this.tableValues,this.statistics].flatMap(c).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);let n=this.values?s(e,[this.keys,this.values]):s(e,[this.keys]),r=this.firstValue,i=[];for(let[e,t]of n)i.push({keys:e,values:t,firstValue:r}),r+=e.length;return t.push(...m(e,this,i)),t}};function m(e,t,n){let r=[],a=i(e,`${t.id}-source-rows`,`uint32`,t.tableKeys.length);r.push(...g(e,t,a));let o=0;for(let[i,s]of n.entries()){if(s.keys.length>0){let n={...s,id:`${t.id}-batch-${i}`,firstSourceRow:o};r.push(..._(e,t,a,n)),r.push(...v(e,t,a,n))}o+=s.keys.length}return r}var h=class{id;index;keys;values;found;probes;statistics;maxProbeCount;constructor(e){this.id=e.id??`gpu-hash-index-query`,this.index=e.index,this.keys=e.keys,this.values=e.values,this.found=e.found,this.probes=e.probes,this.statistics=e.statistics,this.maxProbeCount=e.maxProbeCount??this.index.maxProbeCount;for(let[e,t]of[[this.index.tableKeys,`index.tableKeys`],[this.index.tableValues,`index.tableValues`],[this.keys,`keys`],[this.values,`values`],[this.found,`found`],[this.probes,`probes`],[this.statistics,`statistics`]])for(let n of c(e))o(n,`${this.id} ${t}`);if(this.index.tableKeys.length!==this.index.tableValues.length)throw Error(`${this.id} index table key and value capacities must match`);if(!C(this.index.tableKeys.length))throw Error(`${this.id} index capacity must be a positive power of two`);if(this.values.length!==this.keys.length||this.found.length!==this.keys.length||this.probes.length!==this.keys.length)throw Error(`${this.id} query outputs must match key length`);if(this.statistics.length<4)throw Error(`${this.id} statistics must contain four uint32 rows`);x(this.id,this.maxProbeCount,this.index.tableKeys.length,this.keys.length),S(this.id,[this.index.tableKeys,this.index.tableValues,this.keys],[this.values,this.found,this.probes,this.statistics])}getCommandNodes(e){let t=[];if([this.index.tableKeys,this.index.tableValues,this.keys,this.values,this.found,this.probes,this.statistics].flatMap(c).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);t.push(...y(e,this));let n=s(e,[this.keys,this.values,this.found,this.probes]);for(let[r,[i,a,o,s]]of n.entries())t.push(...b(e,{...this,id:`${this.id}-span-${r}`,keys:i,values:a,found:o,probes:s}));return t}};function g(e,n,r){let i=[],a=w(Math.max(n.tableKeys.length,6),e.device.limits.maxComputeWorkgroupsPerDimension),o=`
const CAPACITY: u32 = ${n.tableKeys.length}u;
const DISPATCH_X: u32 = ${a.x}u;
const DISPATCH_Y: u32 = ${a.y}u;
const TABLE_KEYS_OFFSET: u32 = ${t(n.tableKeys)}u;
const TABLE_VALUES_OFFSET: u32 = ${t(n.tableValues)}u;
const SOURCE_ROWS_OFFSET: u32 = ${t(r)}u;
const STATISTICS_OFFSET: u32 = ${t(n.statistics)}u;
@group(0) @binding(0) var<storage, read_write> tableKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> tableValues: array<u32>;
@group(0) @binding(2) var<storage, read_write> sourceRows: array<u32>;
@group(0) @binding(3) var<storage, read_write> statistics: array<u32>;
@compute @workgroup_size(${l}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let elementIndex = workgroupIndex * ${l}u + localIndex;
  if (elementIndex < CAPACITY) {
    tableKeys[TABLE_KEYS_OFFSET + elementIndex] = ${f}u;
    tableValues[TABLE_VALUES_OFFSET + elementIndex] = ${f}u;
    sourceRows[SOURCE_ROWS_OFFSET + elementIndex] = ${f}u;
  }
  if (elementIndex < 6u) {
    statistics[STATISTICS_OFFSET + elementIndex] = 0u;
  }
}`;return i.push(...T(e,{id:`${n.id}-initialize`,source:o,resources:[{buffer:n.tableKeys,usage:`storage-write`},{buffer:n.tableValues,usage:`storage-write`},{buffer:r,usage:`storage-write`},{buffer:n.statistics,usage:`storage-write`}],bindings:{tableKeys:n.tableKeys,tableValues:n.tableValues,sourceRows:r,statistics:n.statistics},dispatchSize:a})),i}function _(e,n,r,i){let a=[],o=w(i.keys.length,e.device.limits.maxComputeWorkgroupsPerDimension),s=i.validity?`@group(0) @binding(1) var<storage, read> inputValidity: array<u32>;`:``,c=i.validity?2:1,d=i.validity?t(i.validity):0,p=i.validity?`if (inputValidity[${d}u + inputIndex] == 0u) { return; }`:``,m=`
const ELEMENT_COUNT: u32 = ${i.keys.length}u;
const CAPACITY_MASK: u32 = ${n.tableKeys.length-1}u;
const MAX_PROBES: u32 = ${n.maxProbeCount}u;
const DISPATCH_X: u32 = ${o.x}u;
const DISPATCH_Y: u32 = ${o.y}u;
const KEYS_OFFSET: u32 = ${t(i.keys)}u;
const TABLE_KEYS_OFFSET: u32 = ${t(n.tableKeys)}u;
const SOURCE_ROWS_OFFSET: u32 = ${t(r)}u;
const STATISTICS_OFFSET: u32 = ${t(n.statistics)}u;
const FIRST_SOURCE_ROW: u32 = ${i.firstSourceRow}u;
@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
${s}
@group(0) @binding(${c}) var<storage, read_write> tableKeys: array<atomic<u32>>;
@group(0) @binding(${c+1}) var<storage, read_write> sourceRows: array<atomic<u32>>;
@group(0) @binding(${c+2}) var<storage, read_write> statistics: array<atomic<u32>>;

fn hashKey(key: u32) -> u32 {
  var value = key;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

@compute @workgroup_size(${l}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let inputIndex = workgroupIndex * ${l}u + localIndex;
  if (inputIndex >= ELEMENT_COUNT) { return; }
  ${p}
  let key = inputKeys[KEYS_OFFSET + inputIndex];
  if (key == ${f}u) {
    atomicAdd(&statistics[STATISTICS_OFFSET + 3u], 1u);
    return;
  }
  let start = hashKey(key) & CAPACITY_MASK;
  var probes = 0u;
  var inserted = false;
  var duplicate = false;
  for (var probe = 0u; probe < MAX_PROBES; probe++) {
    probes = probe + 1u;
    let slot = (start + probe) & CAPACITY_MASK;
    var result = atomicCompareExchangeWeak(
      &tableKeys[TABLE_KEYS_OFFSET + slot], ${f}u, key
    );
    // Bound spurious weak-CAS retries without probing onward and risking a duplicate slot.
    var exchangeAttempt = 1u;
    while (
      !result.exchanged &&
      result.old_value == ${f}u &&
      exchangeAttempt < ${u}u
    ) {
      result = atomicCompareExchangeWeak(
        &tableKeys[TABLE_KEYS_OFFSET + slot], ${f}u, key
      );
      exchangeAttempt++;
    }
    if (!result.exchanged && result.old_value == ${f}u) {
      break;
    }
    if (result.exchanged || result.old_value == key) {
      atomicMin(&sourceRows[SOURCE_ROWS_OFFSET + slot], FIRST_SOURCE_ROW + inputIndex);
      inserted = result.exchanged;
      duplicate = !result.exchanged;
      break;
    }
  }
  atomicAdd(&statistics[STATISTICS_OFFSET + 4u], probes);
  atomicMax(&statistics[STATISTICS_OFFSET + 5u], probes);
  if (inserted) {
    atomicAdd(&statistics[STATISTICS_OFFSET], 1u);
  } else if (duplicate) {
    atomicAdd(&statistics[STATISTICS_OFFSET + 1u], 1u);
  } else {
    atomicAdd(&statistics[STATISTICS_OFFSET + 2u], 1u);
  }
}`;return a.push(...T(e,{id:`${i.id}-build`,source:m,resources:[{buffer:i.keys,usage:`storage-read`},...i.validity?[{buffer:i.validity,usage:`storage-read`}]:[],{buffer:n.tableKeys,usage:`storage-read-write`},{buffer:r,usage:`storage-read-write`},{buffer:n.statistics,usage:`storage-read-write`}],bindings:{inputKeys:i.keys,...i.validity?{inputValidity:i.validity}:{},tableKeys:n.tableKeys,sourceRows:r,statistics:n.statistics},dispatchSize:o})),a}function v(e,n,r,i){let a=[],o=w(n.tableKeys.length,e.device.limits.maxComputeWorkgroupsPerDimension),s=!!(i.values&&i.keys.length>0),c=s?`@group(0) @binding(3) var<storage, read> inputValues: array<u32>;`:``,u=s?`inputValues[${t(i.values)}u + localSourceRow]`:`${i.firstValue}u + localSourceRow`,d=`
const CAPACITY: u32 = ${n.tableKeys.length}u;
const FIRST_SOURCE_ROW: u32 = ${i.firstSourceRow}u;
const SOURCE_ROW_COUNT: u32 = ${i.keys.length}u;
const DISPATCH_X: u32 = ${o.x}u;
const DISPATCH_Y: u32 = ${o.y}u;
const TABLE_KEYS_OFFSET: u32 = ${t(n.tableKeys)}u;
const TABLE_VALUES_OFFSET: u32 = ${t(n.tableValues)}u;
const SOURCE_ROWS_OFFSET: u32 = ${t(r)}u;
@group(0) @binding(0) var<storage, read> tableKeys: array<u32>;
@group(0) @binding(1) var<storage, read> sourceRows: array<u32>;
@group(0) @binding(2) var<storage, read_write> tableValues: array<u32>;
${c}
@compute @workgroup_size(${l}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let slot = workgroupIndex * ${l}u + localIndex;
  if (slot >= CAPACITY || tableKeys[TABLE_KEYS_OFFSET + slot] == ${f}u) { return; }
  let sourceRow = sourceRows[SOURCE_ROWS_OFFSET + slot];
  if (sourceRow < FIRST_SOURCE_ROW || sourceRow - FIRST_SOURCE_ROW >= SOURCE_ROW_COUNT) { return; }
  let localSourceRow = sourceRow - FIRST_SOURCE_ROW;
  tableValues[TABLE_VALUES_OFFSET + slot] = ${u};
}`;return a.push(...T(e,{id:`${i.id}-finalize`,source:d,resources:[{buffer:n.tableKeys,usage:`storage-read`},{buffer:r,usage:`storage-read`},{buffer:n.tableValues,usage:`storage-write`},...s?[{buffer:i.values,usage:`storage-read`}]:[]],bindings:{tableKeys:n.tableKeys,sourceRows:r,tableValues:n.tableValues,...s?{inputValues:i.values}:{}},dispatchSize:o})),a}function y(e,n){let r=[],i=`
const STATISTICS_OFFSET: u32 = ${t(n.statistics)}u;
@group(0) @binding(0) var<storage, read_write> statistics: array<u32>;
@compute @workgroup_size(${l}) fn main(
  @builtin(global_invocation_id) globalId: vec3u
) {
  if (globalId.x < 4u) {
    statistics[STATISTICS_OFFSET + globalId.x] = 0u;
  }
}`;return r.push(...T(e,{id:`${n.id}-initialize`,source:i,resources:[{buffer:n.statistics,usage:`storage-write`}],bindings:{statistics:n.statistics},dispatchSize:{x:1,y:1,z:1}})),r}function b(e,n){let r=[],i=w(n.keys.length,e.device.limits.maxComputeWorkgroupsPerDimension),a=`
const ELEMENT_COUNT: u32 = ${n.keys.length}u;
const CAPACITY_MASK: u32 = ${n.index.tableKeys.length-1}u;
const MAX_PROBES: u32 = ${n.maxProbeCount}u;
const DISPATCH_X: u32 = ${i.x}u;
const DISPATCH_Y: u32 = ${i.y}u;
const KEYS_OFFSET: u32 = ${t(n.keys)}u;
const TABLE_KEYS_OFFSET: u32 = ${t(n.index.tableKeys)}u;
const TABLE_VALUES_OFFSET: u32 = ${t(n.index.tableValues)}u;
const VALUES_OFFSET: u32 = ${t(n.values)}u;
const FOUND_OFFSET: u32 = ${t(n.found)}u;
const PROBES_OFFSET: u32 = ${t(n.probes)}u;
const STATISTICS_OFFSET: u32 = ${t(n.statistics)}u;
@group(0) @binding(0) var<storage, read> queryKeys: array<u32>;
@group(0) @binding(1) var<storage, read> tableKeys: array<u32>;
@group(0) @binding(2) var<storage, read> tableValues: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputFound: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputProbes: array<u32>;
@group(0) @binding(6) var<storage, read_write> statistics: array<atomic<u32>>;

fn hashKey(key: u32) -> u32 {
  var value = key;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

@compute @workgroup_size(${l}) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localIndex: u32
) {
  let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  let queryIndex = workgroupIndex * ${l}u + localIndex;
  if (queryIndex >= ELEMENT_COUNT) { return; }
  let key = queryKeys[KEYS_OFFSET + queryIndex];
  var value = ${f}u;
  var found = false;
  var probes = 0u;
  if (key != ${f}u) {
    let start = hashKey(key) & CAPACITY_MASK;
    for (var probe = 0u; probe < MAX_PROBES; probe++) {
      probes = probe + 1u;
      let slot = (start + probe) & CAPACITY_MASK;
      let storedKey = tableKeys[TABLE_KEYS_OFFSET + slot];
      if (storedKey == key) {
        value = tableValues[TABLE_VALUES_OFFSET + slot];
        found = true;
        break;
      }
      if (storedKey == ${f}u) { break; }
    }
  }
  outputValues[VALUES_OFFSET + queryIndex] = value;
  outputFound[FOUND_OFFSET + queryIndex] = select(0u, 1u, found);
  outputProbes[PROBES_OFFSET + queryIndex] = probes;
  atomicAdd(&statistics[STATISTICS_OFFSET + select(1u, 0u, found)], 1u);
  atomicAdd(&statistics[STATISTICS_OFFSET + 2u], probes);
  atomicMax(&statistics[STATISTICS_OFFSET + 3u], probes);
}`;return r.push(...T(e,{id:`${n.id}-lookup`,source:a,resources:[{buffer:n.keys,usage:`storage-read`},{buffer:n.index.tableKeys,usage:`storage-read`},{buffer:n.index.tableValues,usage:`storage-read`},{buffer:n.values,usage:`storage-write`},{buffer:n.found,usage:`storage-write`},{buffer:n.probes,usage:`storage-write`},{buffer:n.statistics,usage:`storage-read-write`}],bindings:{queryKeys:n.keys,tableKeys:n.index.tableKeys,tableValues:n.index.tableValues,outputValues:n.values,outputFound:n.found,outputProbes:n.probes,statistics:n.statistics},dispatchSize:i})),r}function x(e,t,n,r){if(!Number.isSafeInteger(t)||t<1||t>n)throw Error(`${e} maxProbeCount must be an integer from one through capacity`);if(r*t>d)throw Error(`${e} aggregate probe count must fit in uint32 statistics`)}function S(e,t,r){let i=t.flatMap(c),a=r.flatMap(c);for(let t of i)for(let r of a)if(n(t,r))throw Error(`${e} input and output views must not overlap`);for(let t=0;t<a.length;t++)for(let r=t+1;r<a.length;r++)if(n(a[t],a[r]))throw Error(`${e} output views must not overlap`)}function C(e){return Number.isSafeInteger(e)&&e>0&&Number.isInteger(Math.log2(e))}function w(e,t){let n=Math.floor(t),r=Math.max(1,Math.ceil(e/l)),i=Math.min(r,n),a=Math.min(Math.ceil(r/i),n),o=Math.ceil(r/i/a);if(o>n)throw Error(`GPUHashIndex work exceeds the device 3D dispatch limit`);return{x:i,y:a,z:o}}function T(t,n){let i=[];return i.push(e({id:n.id,resources:n.resources,compile:({device:e})=>{let t=new a(e,{id:n.id,source:n.source,shaderLayout:{bindings:Object.keys(n.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,t]of Object.entries(n.bindings))a[e]=r(t,i);t.dispatch(e,{bindings:a,x:n.dispatchSize.x,y:n.dispatchSize.y,z:n.dispatchSize.z})},destroy:()=>t.destroy()}}})),i}export{h as n,p as t};