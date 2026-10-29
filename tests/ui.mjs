// DOM-level functional checks. This is not a browser screenshot or layout test.
import {JSDOM} from 'jsdom';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=new URL('../dist/',import.meta.url);
const html=await fs.readFile(new URL('index.html',root),'utf8');
const dom=new JSDOM(html,{url:'https://glucoscope.test/'});
for(const k of ['window','document','location','FormData'])globalThis[k]=dom.window[k];
window.scrollTo=()=>{};
globalThis.fetch=async path=>({ok:true,json:async()=>JSON.parse(await fs.readFile(new URL(path,root),'utf8'))});
await import('../dist/assets/app.mjs');
assert.match(document.querySelector('main').textContent,/Diabetes indicator research/);
const navigate=hash=>{window.history.replaceState(null,'',hash);window.dispatchEvent(new window.HashChangeEvent('hashchange'))};
for(const hash of ['explorer','compare','capacity','data','learn']){navigate('#'+hash);assert.ok(document.querySelector('main h1'));assert.ok(!document.querySelector('.errorbox'),hash);}
navigate('#explorer');
assert.equal(document.querySelectorAll('.field').length,21);
document.querySelector('#new-profile').click();
assert.equal(document.querySelectorAll('.field').length,21);
assert.equal(document.querySelector('[name=HighBP]').value,'');
assert.equal(document.querySelector('[name=BMI]').value,'');
assert.equal(document.querySelector('#profile-progress').value,0);
assert.equal(document.querySelector('.score'),null);
document.querySelector('#profile-form').dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));
assert.match(document.querySelector('#form-errors').textContent,/whole number/);
document.querySelector('[data-example="2"]').click();
assert.equal(document.querySelector('#profile-progress').value,21);
const oldScore=document.querySelector('.score').textContent;
document.querySelector('[name=BMI]').value='20';
document.querySelector('[name=BMI]').dispatchEvent(new window.Event('input',{bubbles:true}));
assert.equal(document.querySelector('.score'),null);
assert.match(document.querySelector('#profile-result').textContent,/Answers changed/);
document.querySelector('#profile-form').dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));
assert.notEqual(document.querySelector('.score').textContent,oldScore);
document.querySelector('#explorer-model').value='Short-form logistic';
document.querySelector('#explorer-model').dispatchEvent(new window.Event('change'));
assert.equal(document.querySelectorAll('.field').length,8);
document.querySelector('[name=BMI]').value='999';
document.querySelector('#profile-form').dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true}));
assert.match(document.querySelector('#form-errors').textContent,/whole number/);
navigate('#compare');
for(const name of ['Logistic regression','Decision tree','Random forest','Gradient boosting','Short-form logistic']){
 document.querySelector('#comparison-model').value=name;document.querySelector('#comparison-model').dispatchEvent(new window.Event('change'));
 assert.equal(document.querySelectorAll('.cell').length,4);
}
navigate('#capacity');document.querySelector('#capacity-range').value='100';document.querySelector('#capacity-range').dispatchEvent(new window.Event('input'));assert.match(document.querySelector('#capacity-results').textContent,/7,069/);
navigate('#data');document.querySelector('#data-feature').value='BMI';document.querySelector('#data-feature').dispatchEvent(new window.Event('change'));assert.ok(document.querySelectorAll('#feature-distribution tbody tr').length>20);
navigate('#learn');assert.equal(document.querySelectorAll('details').length,10);
document.querySelector('#menu-toggle').click();
assert.equal(document.querySelector('#menu-toggle').getAttribute('aria-expanded'),'true');
navigate('#learn');
assert.equal(document.querySelector('#menu-toggle').getAttribute('aria-expanded'),'false');
// Check all local downloadable links and primary assets exist.
for(const a of document.querySelectorAll('[download]'))await fs.access(new URL(a.getAttribute('href'),root));
console.log('PASS: six routes, three profiles, model switching, form validation, prediction update, capacity, dataset explorer, lessons, downloads. DOM checks only; visual browser QA unavailable.');
