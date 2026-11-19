import{t as e}from"./graph-chunk-groups-BB8ACpQO.js";var t=`var<private> CAPACITY: u32;
var<private> WEIGHTS_ROW_START: u32;`;function n(n,r){return e(n,{...r,parameterValues:[r.weightsRowStart??r.rowBase??0],declarations:`${t}\n${r.declarations??``}`,body:`CAPACITY = arrayLength(&neighbors) - neighborsOffset;
  WEIGHTS_ROW_START = getChunkParameter(36u);
  { let index = index + getChunkParameter(0u);
  ${r.body}
  }`})}export{n as t};