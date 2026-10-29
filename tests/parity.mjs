import fs from 'node:fs';
import assert from 'node:assert/strict';
import {predict,validate} from '../dist/assets/inference.mjs';
const models=JSON.parse(fs.readFileSync(new URL('../dist/assets/models.json',import.meta.url)));
const probes=JSON.parse(fs.readFileSync(new URL('../artifacts/parity_probes.json',import.meta.url)));
const audit=JSON.parse(fs.readFileSync(new URL('../dist/assets/audit.json',import.meta.url)));
let max=0,n=0;
for(const p of probes) {
  const m=models.find(m=>m.name===p.name);
  p.inputs.forEach((x,i)=>{const error=Math.abs(predict(m,x)-p.expected[i]);max=Math.max(max,error);n++;assert.ok(error<1e-9,`${m.name}: ${error}`);});
}
assert.ok(validate({...probes[0].inputs[0],BMI:999},audit.data_dictionary,models[0].features).length>0);
assert.ok(validate({...probes[0].inputs[0],BMI:''},audit.data_dictionary,models[0].features).length>0);
assert.equal(validate(probes[0].inputs[0],audit.data_dictionary,models[0].features).length,0);
console.log(`PASS: ${n} Python/JavaScript probability comparisons, max error ${max}. Input rejection checks passed.`);
