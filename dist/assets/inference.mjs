// Exact portable inference for the fitted sklearn estimators. No health inputs leave this browser.
export function sigmoid(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }
export function walk(tree, x) {
  let i=0;
  while(tree.left[i]!==-1) i = Math.fround(x[tree.feature[i]]) <= tree.threshold[i] ? tree.left[i] : tree.right[i];
  return tree.value[i];
}
export function predict(model, profile) {
  const x=model.features.map(k=>Number(profile[k]));
  if(x.some(v=>!Number.isFinite(v))) throw new Error('Every required input must be a finite number.');
  let score;
  if(model.kind==='logistic') score=model.intercept+x.reduce((s,v,i)=>s+(v-model.mean[i])/model.scale[i]*model.coef[i],0);
  else if(model.kind==='boosting') score=model.initial+model.rate*model.trees.reduce((s,t)=>s+walk(t,x),0);
  else score=model.trees.reduce((s,t)=>s+walk(t,x),0)/model.trees.length;
  return sigmoid(-(model.calibration.a*score+model.calibration.b));
}
export function validate(profile, dictionary, required) {
  const errors=[];
  for(const f of dictionary.filter(d=>required.includes(d.key))) {
    const v=profile[f.key];
    if(v===''||v===null||v===undefined||!Number.isInteger(Number(v))||Number(v)<f.min||Number(v)>f.max)
      errors.push(`${f.label}: use a whole number from ${f.min} to ${f.max}.`);
  }
  return errors;
}
