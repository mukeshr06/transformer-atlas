export const add=(a,b)=>a.map((x,i)=>x+b[i]);
export const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export const mul=(a,s)=>a.map(x=>x*s);
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const length=a=>Math.hypot(...a);
export const normalize=a=>mul(a,1/(length(a)||1));
export const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
export const clamp=(v,min=0,max=1)=>Math.max(min,Math.min(max,v));
export function multiply(a,b){const r=new Float32Array(16);for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)r[col*4+row]+=a[k*4+row]*b[col*4+k];return r;}
export function transform(m,p){const v=[...p,1];return Array.from({length:4},(_,r)=>v.reduce((s,x,c)=>s+m[c*4+r]*x,0));}
export function lookAt(eye,target){const z=normalize(sub(eye,target)),x=normalize(cross([0,1,0],z)),y=cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);}
export function perspective(fov,aspect,near=.1,far=200){const f=1/Math.tan(fov/2),nf=1/(near-far);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]);}
export function ortho(width,height,near=.1,far=200){return new Float32Array([2/width,0,0,0,0,2/height,0,0,0,0,-2/(far-near),0,0,0,-(far+near)/(far-near),1]);}
export function modelMatrix(pos,size,dir=null){if(!dir)return [size[0],0,0,0,0,size[1],0,0,0,0,size[2],0,...pos,1];const z=normalize(dir),x=normalize(cross(Math.abs(z[1])>.95?[1,0,0]:[0,1,0],z)),y=cross(z,x);return [...mul(x,size[0]),0,...mul(y,size[1]),0,...mul(z,size[2]),0,...pos,1];}
export function bezier(a,b,bend=0,depth=0){if(!bend&&!depth)return[a,b];const c1=[a[0]+(b[0]-a[0])*.32,a[1]+bend,a[2]+depth],c2=[a[0]+(b[0]-a[0])*.68,b[1]+bend,b[2]+depth];return Array.from({length:27},(_,i)=>{const t=i/26,u=1-t;return a.map((v,j)=>u*u*u*v+3*u*u*t*c1[j]+3*u*t*t*c2[j]+t*t*t*b[j]);});}
export function along(points,t){t=((t%1)+1)%1;const p=t*(points.length-1),i=Math.floor(p);return mix(points[i],points[Math.min(i+1,points.length-1)],p-i);}
export function rayBox(origin,dir,pos,size){let near=-Infinity,far=Infinity;for(let i=0;i<3;i++){const min=pos[i]-size[i]/2,max=pos[i]+size[i]/2;if(Math.abs(dir[i])<1e-8){if(origin[i]<min||origin[i]>max)return null;continue;}const a=(min-origin[i])/dir[i],b=(max-origin[i])/dir[i];near=Math.max(near,Math.min(a,b));far=Math.min(far,Math.max(a,b));if(near>far)return null;}return far<0?null:Math.max(0,near);}
export function projectPoint(matrix,pos,w,h,top=0){const p=transform(matrix,pos);return{x:(p[0]/p[3]*.5+.5)*w,y:top+(.5-p[1]/p[3]*.5)*h,z:p[2]/p[3],w:p[3]};}
