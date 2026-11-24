import fs from 'node:fs';
import assert from 'node:assert/strict';
import {predict,engineer} from '../dist/engine.mjs';
const models=Object.fromEntries(['linear','ridge','forest','xgboost'].map(n=>[n,JSON.parse(fs.readFileSync(new URL('../dist/data/'+n+'.json',import.meta.url)))]));
const cases=JSON.parse(fs.readFileSync(new URL('parity_cases.json',import.meta.url)));
const max={};for(const [name,model] of Object.entries(models)){
 max[name]=0;for(const c of cases){const p=predict(model,c.input),diff=Math.abs(p-c.expected[name]);max[name]=Math.max(max[name],diff);assert.ok(diff<0.02,`${name}: discrepancy ${diff}`);}
}
assert.equal(engineer({cpu_model:'i5-1235U',gpu:'RTX 3060'}).cpu_generation_or_series,'Intel gen 12');
console.log(JSON.stringify({verified_predictions:cases.length*4,max_absolute_difference:max,tolerance:0.02}));
