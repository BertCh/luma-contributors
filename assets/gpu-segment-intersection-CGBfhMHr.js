import{t as e}from"./ports-xmdKgL07.js";import{r as t,u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as a}from"./scan-total-C4QddMmi.js";import{n as o,t as s}from"./construction-snapshot-DdkX9Pc0.js";import{n as c,r as l,t as u}from"./spatial-join-construction-snapshot-B4G0PxaB.js";import{E as d,S as f,T as p,l as m,w as h}from"./gpu-contributor-utils-BIwLQON5.js";import{a as g,i as _,n as v,r as y,t as b}from"./segment-table-BeVxjLqW.js";import{t as x}from"./segment-bvh-CDgckfAv.js";import{p as S}from"./wgsl-common-DfyXiYB0.js";import{v as C}from"./spatial-context-DwZEd4EZ.js";import{a as w}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as T}from"./segment-storage-validation-BUxpxgMF.js";var E=`GPUSegmentIntersection`,D=1024,O=class{id;props;spatialContext;isSelfMode;leftSlotCount;rightSlotCount;pairCapacity;leafCapacity;spatialSort;constructor(t){let r=s(t,[`id`,`spatialContext`]);this.id=r.id??`segment-intersection`;let i=r.spatialContext&&o(r.spatialContext);this.spatialContext=C(this.id,i);let a=s(t,[`left`,`right`,`sameFeatureOnly`,`spatialSort`,`leafCapacity`,`output`]),f=e(this.id,`left`,a.left),g=a.right&&e(this.id,`right`,a.right),_=s(a.output,[`pairs`,`status`,`kinds`,`points`,`endPoints`,`leftFeatures`,`rightFeatures`,`leftRings`,`rightRings`]);t=Object.freeze({...r,...a,spatialContext:i,left:u(f),...g?{right:u(g)}:{},output:Object.freeze({..._,pairs:c(_.pairs),..._.status?{status:l(_.status)}:{}})}),this.props=t;let{id:y}=this;this.isSelfMode=!t.right;for(let[e,r]of[[`left`,t.left],[`right`,t.right]]){if(!r)continue;d(r.positions,[`float32x2`],`${y} ${e}.positions`);let t=r.kind===`lines`?[[`lineOffsets`,r.lineOffsets]]:[[`featureOffsets`,r.featureOffsets],[`polygonOffsets`,r.polygonOffsets],[`ringOffsets`,r.ringOffsets]];for(let[r,i]of t)if(n(i,`${y} ${e}.${r}`),i.length<1)throw Error(`${y} ${e}.${r} requires a terminal entry`);if(r.positions.length<1)throw Error(`${y} ${e} must contain at least one vertex`)}if(t.sameFeatureOnly&&!this.isSelfMode)throw Error(`${y} sameFeatureOnly requires self mode (omit right)`);this.leftSlotCount=v(t.left),this.rightSlotCount=v(t.right??t.left),this.spatialSort=t.spatialSort??this.rightSlotCount>=D,this.leafCapacity=t.leafCapacity??m(this.rightSlotCount),h(y,`leafCapacity`,this.leafCapacity);let{pairs:b}=t.output;if(p(y,!!(b.leftRows&&b.rightRows)&&!(`leftIds`in b)&&!(`rightIds`in b),`pairs.leftRows and pairs.rightRows are required; leftIds and rightIds are unsupported application-ID names`),p(y,this.leafCapacity>=this.rightSlotCount||b.status.candidateOverflow!==void 0,`output.pairs.status.candidateOverflow is required when leafCapacity is below the right segment slot count`),n(b.leftRows,`${y} pairs.leftRows`),n(b.rightRows,`${y} pairs.rightRows`),this.pairCapacity=b.leftRows.length,this.pairCapacity<1||b.rightRows.length!==this.pairCapacity)throw Error(`${y} pairs.leftRows and pairs.rightRows must have equal nonzero length`);for(let[e,r]of[[`count`,b.status.count],[`overflow`,b.status.overflow],[`candidateOverflow`,b.status.candidateOverflow],[`requiredCount`,b.status.requiredCount],[`status.uncertainCount`,t.output.status?.uncertainCount]])if(r&&(n(r,`${y} ${e}`),r.length<1))throw Error(`${y} ${e} must contain one uint32 row`);for(let[e,r]of[[`kinds`,t.output.kinds],[`leftFeatures`,t.output.leftFeatures],[`rightFeatures`,t.output.rightFeatures],[`leftRings`,t.output.leftRings],[`rightRings`,t.output.rightRings]])if(r&&(n(r,`${y} ${e}`),r.length!==this.pairCapacity))throw Error(`${y} ${e} length must equal the pair capacity`);for(let[e,n]of[[`points`,t.output.points],[`endPoints`,t.output.endPoints]])if(n&&(d(n,[`float32x2`],`${y} ${e}`),n.length!==this.pairCapacity))throw Error(`${y} ${e} length must equal the pair capacity`)}getInputViews(){let e=[];for(let t of[this.props.left,this.props.right])t&&(e.push(t.positions),t.kind===`lines`?e.push(t.lineOffsets):e.push(t.featureOffsets,t.polygonOffsets,t.ringOffsets));return e}getOutputViews(){let{pairs:e,status:t,...n}=this.props.output,r=t?.uncertainCount;return[e.leftRows,e.rightRows,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount,r,n.kinds,n.points,n.endPoints,n.leftFeatures,n.rightFeatures,n.leftRings,n.rightRings]}getCommandNodes(e){let{id:n,props:o,isSelfMode:s,leftSlotCount:c,rightSlotCount:l,pairCapacity:u}=this,{pairs:d}=o.output;w(e,n,Object.fromEntries([...this.getInputViews(),...this.getOutputViews()].map((e,t)=>[String(t),e])));for(let[t,r]of[[`left`,o.left],[`right`,o.right]])r&&f(n,`${t}.positions rows`,r.positions.length,_(e.device));T(e,n,c,l,u,this.leafCapacity,this.spatialSort);let p=[],m=o.right??o.left,h=t(e,`${n}-right-table`,`uint32`,l*8),v=t(e,`${n}-bounds-minima`,`float32x2`,l),C=t(e,`${n}-bounds-maxima`,`float32x2`,l);p.push(b(e,{id:`${n}-right-table`,operation:E,geometry:m,table:h,minima:v,maxima:C}));let D=h;s||(D=t(e,`${n}-left-table`,`uint32`,c*8),p.push(b(e,{id:`${n}-left-table`,operation:E,geometry:o.left,table:D})));let{bvh:O,nodes:k}=x(e,{id:n,operation:E,minima:v,maxima:C,leafCapacity:this.leafCapacity,spatialSort:this.spatialSort});p.push(...k);let A=t(e,`${n}-state`,`uint32`,4),j=t(e,`${n}-counts`,`uint32`,c),M=t(e,`${n}-offsets`,`uint32`,c);p.push(i(e,{id:`${n}-clear`,operation:E,variant:`clear`,bindings:[{name:`state`,view:A,type:`u32`,access:`read_write`}],invocationCount:4,body:`state[stateOffset + index] = 0u;`}));let N=`${y}
${g(`left`)}
${g(`right`)}
const SELF: bool = ${s};
const SAME_FEATURE: bool = ${!!o.sameFeatureOnly};
const RIGHT_COUNT: u32 = ${l}u;
const INTERNAL_NODE_COUNT: u32 = ${O.internalNodeCount}u;
const PAIR_CAPACITY: u32 = ${u}u;
fn pairHit(left: u32, right: u32, withGeometry: bool) -> SegmentHit {
  var none = SegmentHit(KIND_NONE, vec2f(0.0), vec2f(0.0), 0u);
  // The ID test needs no table read, so it goes first: half of all self-mode candidates fail it.
  if (SELF && right <= left) { return none; }
  if (!leftValid(left) || !rightValid(right)) { return none; }
  if (SAME_FEATURE && leftFeature(left) != rightFeature(right)) { return none; }
  var hit = classifySegments(leftStart(left), leftEnd(left), rightStart(right), rightEnd(right), withGeometry);
  if (SELF && (hit.kind == KIND_TOUCH || hit.kind == KIND_COLLINEAR_TOUCH) &&
      (leftSuccessor(left) == right || rightSuccessor(right) == left)) {
    hit.kind = KIND_NONE;
  }
  return hit;
}
${S}`,P=[{name:`leftTable`,view:D,type:`u32`,access:`read`},{name:`rightTable`,view:h,type:`u32`,access:`read`},{name:`nodeMinima`,view:O.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:O.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:O.leafIds,type:`u32`,access:`read`}],F=s&&this.spatialSort&&this.leafCapacity>=c,I=(e,t,n)=>`${F?`let leftRow = leafIds[leafIdsOffset + index];
  if (leftRow >= RIGHT_COUNT) {
    return;
  }`:`let leftRow = index;`}
  ${t}
  if (leftValid(leftRow)) {
    let start = leftStart(leftRow);
    let end = leftEnd(leftRow);
    let queryMinimum = min(start, end);
    let queryMaximum = max(start, end);
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          let hit = pairHit(leftRow, rightRow, false);
          if (hit.kind != KIND_NONE) {
            ${e}
          }
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${n}`;p.push(i(e,{id:`${n}-count`,operation:E,variant:`count`,bindings:[...P,{name:`counts`,view:j,type:`u32`,access:`read_write`}],invocationCount:c,declarations:N,body:I(`found = found + 1u;`,`var found = 0u;`,`counts[countsOffset + leftRow] = found;`)})),p.push(...new r({id:`${n}-scan`,input:j,output:M,mode:`exclusive`}).getCommandNodes(e)),p.push(i(e,{id:`${n}-total`,operation:E,variant:`total`,bindings:[{name:`counts`,view:j,type:`u32`,access:`read`},{name:`offsets`,view:M,type:`u32`,access:`read`},{name:`state`,view:A,type:`u32`,access:`read_write`}],invocationCount:1,body:`state[stateOffset] = offsets[offsetsOffset + ${c-1}u] + counts[countsOffset + ${c-1}u];`})),p.push(a(e,{id:n,operation:E,counts:j,offsets:M,rowCount:c,total:A}));let{kinds:L,points:R,endPoints:z}=o.output,B=o.output.status?.uncertainCount,V=L||R||z||B?t(e,`${n}-hit-codes`,`uint32`,u):void 0,H=[...P,{name:`offsets`,view:M,type:`u32`,access:`read`},{name:`pairRight`,view:d.rightRows,type:`u32`,access:`read_write`}];if(V&&H.push({name:`hitCodes`,view:V,type:`u32`,access:`read_write`}),p.push(i(e,{id:`${n}-write`,operation:E,variant:`write`,bindings:H,invocationCount:c,declarations:N,body:I(`let slot = offsets[offsetsOffset + leftRow] + found;
            if (slot < PAIR_CAPACITY) {
              pairRight[pairRightOffset + slot] = rightRow;
              ${V?`hitCodes[hitCodesOffset + slot] = hit.kind | (hit.selector << 8u);`:``}
            }
            found = found + 1u;`,`var found = 0u;`,``)})),p.push(i(e,{id:`${n}-expand-left`,operation:E,variant:`expand-left`,bindings:[{name:`offsets`,view:M,type:`u32`,access:`read`},{name:`state`,view:A,type:`u32`,access:`read`},{name:`pairLeft`,view:d.leftRows,type:`u32`,access:`read_write`}],invocationCount:u,declarations:`const PAIR_CAPACITY: u32 = ${u}u;
const LEFT_COUNT: u32 = ${c}u;`,body:`if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  var low = 0u;
  var high = LEFT_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
  }
  pairLeft[pairLeftOffset + index] = max(low, 1u) - 1u;`})),this.spatialSort){let t=[{name:`counts`,view:j,type:`u32`,access:`read`},{name:`offsets`,view:M,type:`u32`,access:`read`},{name:`pairRight`,view:d.rightRows,type:`u32`,access:`read_write`}];V&&t.push({name:`hitCodes`,view:V,type:`u32`,access:`read_write`}),p.push(i(e,{id:`${n}-sort-hits`,operation:E,variant:`sort-hits`,bindings:t,invocationCount:c,declarations:`const PAIR_CAPACITY: u32 = ${u}u;
const GAPS = array<u32, 8>(701u, 301u, 132u, 57u, 23u, 10u, 4u, 1u);`,body:`let first = offsets[offsetsOffset + index];
  let last = min(first + counts[countsOffset + index], PAIR_CAPACITY);
  if (last <= first + 1u) { return; }
  // Shell sort with Ciura gaps; ranges are short unless one segment crosses very many others.
  for (var gapIndex = 0u; gapIndex < 8u; gapIndex++) {
    let gap = GAPS[gapIndex];
    for (var slot = first + gap; slot < last; slot++) {
      let value = pairRight[pairRightOffset + slot];
      ${V?`let valueCode = hitCodes[hitCodesOffset + slot];`:``}
      var hole = slot;
      loop {
        if (hole < first + gap) { break; }
        let earlier = pairRight[pairRightOffset + hole - gap];
        if (earlier <= value) { break; }
        pairRight[pairRightOffset + hole] = earlier;
        ${V?`hitCodes[hitCodesOffset + hole] = hitCodes[hitCodesOffset + hole - gap];`:``}
        hole = hole - gap;
      }
      pairRight[pairRightOffset + hole] = value;
      ${V?`hitCodes[hitCodesOffset + hole] = valueCode;`:``}
    }
  }`}))}if(V&&(L||B)){let t=[{name:`hitCodes`,view:V,type:`u32`,access:`read`},{name:`state`,view:A,type:`atomic<u32>`,access:`read_write`}];L&&t.push({name:`kinds`,view:L,type:`u32`,access:`read_write`}),p.push(i(e,{id:`${n}-kinds`,operation:E,variant:`kinds`,bindings:t,invocationCount:u,declarations:`${y}
const PAIR_CAPACITY: u32 = ${u}u;`,body:`if (index >= min(atomicLoad(&state[stateOffset]), PAIR_CAPACITY)) { return; }
  let kind = hitCodes[hitCodesOffset + index] & 0xffu;
  if (kind == KIND_UNCERTAIN) { atomicAdd(&state[stateOffset + 1u], 1u); }
  ${L?`kinds[kindsOffset + index] = kind;`:``}`}))}if(V&&(R||z)){let t=[{name:`pairLeft`,view:d.leftRows,type:`u32`,access:`read`},{name:`pairRight`,view:d.rightRows,type:`u32`,access:`read`},{name:`leftTable`,view:D,type:`u32`,access:`read`},{name:`rightTable`,view:h,type:`u32`,access:`read`},{name:`hitCodes`,view:V,type:`u32`,access:`read`},{name:`state`,view:A,type:`u32`,access:`read`}];R&&t.push({name:`points`,view:R,type:`f32`,access:`read_write`}),z&&t.push({name:`endPoints`,view:z,type:`f32`,access:`read_write`}),p.push(i(e,{id:`${n}-points`,operation:E,variant:`points`,bindings:t,invocationCount:u,declarations:`${y}
${g(`left`)}
${g(`right`)}
const PAIR_CAPACITY: u32 = ${u}u;`,body:`if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  let code = hitCodes[hitCodesOffset + index];
  let left = pairLeft[pairLeftOffset + index];
  let right = pairRight[pairRightOffset + index];
  let hit = getSegmentHitGeometry(code & 0xffu, code >> 8u, leftStart(left), leftEnd(left), rightStart(right), rightEnd(right));
  ${R?`points[pointsOffset + index * 2u] = hit.point.x; points[pointsOffset + index * 2u + 1u] = hit.point.y;`:``}
  ${z?`endPoints[endPointsOffset + index * 2u] = hit.endPoint.x; endPoints[endPointsOffset + index * 2u + 1u] = hit.endPoint.y;`:``}`}))}for(let[t,r,a,s,c]of[[`left`,d.leftRows,D,o.output.leftFeatures,o.output.leftRings],[`right`,d.rightRows,h,o.output.rightFeatures,o.output.rightRings]]){if(!s&&!c)continue;let o=[{name:`ids`,view:r,type:`u32`,access:`read`},{name:`${t}Table`,view:a,type:`u32`,access:`read`},{name:`state`,view:A,type:`u32`,access:`read`}];s&&o.push({name:`featureColumn`,view:s,type:`u32`,access:`read_write`}),c&&o.push({name:`ringColumn`,view:c,type:`u32`,access:`read_write`}),p.push(i(e,{id:`${n}-${t}-columns`,operation:E,variant:`${t}-columns`,bindings:o,invocationCount:u,declarations:`const SEGMENT_NONE: u32 = 0xffffffffu;
const PAIR_CAPACITY: u32 = ${u}u;
${g(t)}`,body:`if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  let row = ids[idsOffset + index];
  ${s?`featureColumn[featureColumnOffset + index] = ${t}Feature(row);`:``}
  ${c?`ringColumn[ringColumnOffset + index] = ${t}Ring(row);`:``}`}))}let U=[{name:`state`,view:A,type:`u32`,access:`read`}];d.status.candidateOverflow&&U.push({name:`bvhOverflow`,view:O.overflow,type:`u32`,access:`read`});let W=[[`count`,d.status.count],[`overflow`,d.status.overflow],[`candidateOverflow`,d.status.candidateOverflow],[`requiredCount`,d.status.requiredCount],[`uncertainCount`,B]];for(let[e,t]of W)t&&U.push({name:e,view:t,type:`u32`,access:`read_write`});let G=e=>W.some(([t,n])=>t===e&&n);return p.push(i(e,{id:`${n}-finalize`,operation:E,variant:`finalize`,bindings:U,invocationCount:1,declarations:`const PAIR_CAPACITY: u32 = ${u}u;`,body:`let total = state[stateOffset];
  let overflowed = total > PAIR_CAPACITY;
  ${G(`count`)?`count[countOffset] = min(total, PAIR_CAPACITY);`:``}
  ${G(`overflow`)?`overflow[overflowOffset] = select(0u, 1u, overflowed);`:``}
  ${G(`candidateOverflow`)?`candidateOverflow[candidateOverflowOffset] = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u);`:``}
  ${G(`requiredCount`)?`requiredCount[requiredCountOffset] = total;`:``}
  ${G(`uncertainCount`)?`uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];`:``}`})),p}};export{O as t};