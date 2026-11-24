// Frozen scikit-learn preprocessing and tree inference. No training happens here.
export function engineer(input){
 const d={...input};let s=String(d.cpu_model).toUpperCase(),m=s.match(/I[3579]-(\d{4,5})([A-Z]+\d*)?/);
 let gen='Unknown',suffix='Unknown';
 if(m){let n=m[1];gen='Intel gen '+((n.length===5||[10,11,12,13,14].includes(+n.slice(0,2)))?n.slice(0,2):n[0]);suffix=m[2]||'Unknown';}
 else if((m=s.match(/RYZEN\s+[3579]\s+(\d{4})([A-Z]+\d*)?/))){gen='AMD '+m[1][0]+'000 series';suffix=m[2]||'Unknown';}
 else if((m=s.match(/M[1234]/))){gen='Apple '+m[0];suffix='Apple';}
 d.cpu_generation_or_series=gen;d.cpu_suffix=suffix;
 s=String(d.gpu).toUpperCase();let brand='Unknown',family='Unknown';
 if(s==='UNKNOWN'){}
 else if(s.includes('RTX')){brand='NVIDIA';family='RTX';}
 else if(s.includes('GTX')){brand='NVIDIA';family='GTX';}
 else if(s.includes('QUADRO')||/\bP\s*\d{3,4}/.test(s)){brand='NVIDIA';family='Quadro';}
 else if(/\bMX\s*\d/.test(s)){brand='NVIDIA';family='MX';}
 else if(s.includes('RADEON')||s.includes('AMD')||/\bRX\s*\d/.test(s)){brand='AMD';family='Radeon';}
 else if(s.includes('INTEL')||s.includes('IRIS')||s.includes('UHD')){brand='Intel';family='Intel integrated';}
 else if(s.includes('APPLE')){brand='Apple';family='Apple integrated';}
 d.gpu_brand=brand;d.gpu_family=family;return d;
}
export function prepare(model,input){
 const d=engineer(input),x=new Float64Array(model.dimension);
 model.numeric.forEach((key,i)=>{const n=d[key]==null||d[key]===''?model.median[i]:Number(d[key]);x[i]=(n-model.mean[i])/model.scale[i];});
 for(const field of model.categories){const v=d[field.name]??field.fallback;const index=field.mapping[String(v)];if(index!==undefined)x[index]=1;}
 return x;
}
export function predict(model,input){
 let x=prepare(model,input),margin;
 if(model.kind==='linear'){margin=model.intercept;for(let i=0;i<x.length;i++)margin+=x[i]*model.coef[i];return Math.exp(margin);}
 x=Float32Array.from(x);
 if(model.kind==='forest'){
  margin=0;for(const t of model.trees){let n=0;while(t.left[n]!==-1)n=x[t.feature[n]]<=t.threshold[n]?t.left[n]:t.right[n];margin+=t.value[n];}
  return Math.exp(margin/model.trees.length);
 }
 margin=Math.fround(model.base);
 for(const t of model.trees){let n=0;while(t.left[n]!==-1){const v=x[t.feature[n]];n=Number.isNaN(v)?(t.default[n]?t.left[n]:t.right[n]):(v<Math.fround(t.threshold[n])?t.left[n]:t.right[n]);}margin=Math.fround(margin+Math.fround(t.threshold[n]));}
 return Math.fround(Math.exp(margin));
}
