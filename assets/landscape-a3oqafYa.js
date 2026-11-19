import{v as e}from"./gpu-contributor-utils-BIwLQON5.js";import{t}from"./spatial-analysis-packed-kernel-CV3ZCYPQ.js";import{u as n}from"./recipe-utils-CICB7qiS.js";import{n as r}from"./land-cover-B1Nqbb4T.js";function i(r,i){let a=i.width*i.height,o=i.patchClasses.length;e(r.device,i.id,`patch parents`,o*4,`reduce the patch count`),e(r.device,i.id,`reconciled patches`,a*4,`reduce the raster size`);let s=n(r,`${i.id}-patch-parents`,`uint32`,o,i.output?.parents),c=n(r,`${i.id}-reconciled-patches`,`uint32`,a,i.output?.patches);r.add(t(r,{id:`${i.id}-patch-parents-init`,operation:i.id,invocationCount:o,bindings:[{name:`parents`,view:s,type:`u32`,access:`read_write`}],body:`parents[parentsOffset+index]=index;`}));let l=`
fn getRoot(owner:u32)->u32 {
  var root=owner;
  loop {
    let parent=atomicLoad(&parents[parentsOffset+root]);
    if(parent==root){return root;}
    let grandparent=atomicLoad(&parents[parentsOffset+parent]);
    atomicMin(&parents[parentsOffset+root],grandparent);
    root=parent;
  }
}`;return r.add(t(r,{id:`${i.id}-reconcile`,operation:i.id,invocationCount:a,declarations:l,bindings:[{name:`patches`,view:i.patches,type:`u32`,access:`read`},{name:`classes`,view:i.patchClasses,type:`u32`,access:`read`},{name:`parents`,view:s,type:`atomic<u32>`,access:`read_write`}],body:`let owner=patches[patchesOffset+index];
if(owner>=${o}u){return;}
let category=classes[classesOffset+owner];
if(category==${i.backgroundClass??4294967295}u||category>=${i.classCount??4294967295}u){return;}
let column=i32(index%${i.width}u);let row=i32(index/${i.width}u);
let directions=array<vec2<i32>,4>(vec2<i32>(1,0),vec2<i32>(0,1),vec2<i32>(1,1),vec2<i32>(-1,1));
for(var direction=0u;direction<${i.connectivity===8?4:2}u;direction++){
  let adjacent=vec2<i32>(column,row)+directions[direction];
  if(any(adjacent < vec2<i32>(0))||adjacent.x>=${i.width}||adjacent.y>=${i.height}){continue;}
  let neighbor=patches[patchesOffset+u32(adjacent.y)*${i.width}u+u32(adjacent.x)];
  if(neighbor>=${o}u||classes[classesOffset+neighbor]!=category){continue;}
  loop {
    let first=getRoot(owner);let second=getRoot(neighbor);
    if(first==second){break;}
    let higher=max(first,second);let lower=min(first,second);
    let linked=atomicCompareExchangeWeak(&parents[parentsOffset+higher],higher,lower);
    if(linked.exchanged){break;}
  }
}`})),r.add(t(r,{id:`${i.id}-patch-roots`,operation:i.id,invocationCount:o,declarations:l,bindings:[{name:`parents`,view:s,type:`atomic<u32>`,access:`read_write`}],body:`let root=getRoot(index);atomicStore(&parents[parentsOffset+index],root);`})),r.add(t(r,{id:`${i.id}-relabel`,operation:i.id,invocationCount:a,bindings:[{name:`patches`,view:i.patches,type:`u32`,access:`read`},{name:`parents`,view:s,type:`u32`,access:`read`},{name:`output`,view:c,type:`u32`,access:`read_write`}],body:`let owner=patches[patchesOffset+index];var label=0xffffffffu;if(owner<${o}u){label=parents[parentsOffset+owner];}output[outputOffset+index]=label;`})),{patches:c,parents:s}}function a(e,t,n,i=4){let a=[50,80,10,30,60],o=new Uint32Array(e*t);for(let s=0;s<o.length;s++){let c=Math.floor(s%e/n)*n,l=Math.floor(Math.floor(s/e)/n)*n,u=r((c+n/2)/e,.5+(t/2-l-n/2)/e),d=a.indexOf(u);o[s]=Math.min(i-1,d<0?a.length:d)}let s=new Uint32Array(e*t).fill(4294967295),c=[];for(let n=0;n<o.length;n++){if(s[n]!==4294967295)continue;let r=c.length;c.push(o[n]);let i=[n];for(s[n]=r;i.length>0;){let a=i.pop(),c=a%e,l=Math.floor(a/e);for(let[a,u]of[[1,0],[-1,0],[0,1],[0,-1]]){let d=c+a,f=l+u;if(d<0||f<0||d>=e||f>=t)continue;let p=f*e+d;s[p]===4294967295&&o[p]===o[n]&&(s[p]=r,i.push(p))}}}return{columns:e,rows:t,classCount:i,cellSizeMeters:4e3/e,classes:o,patches:s,patchClasses:Uint32Array.from(c),patchCount:c.length}}export{i as n,a as t};