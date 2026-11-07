const $ = id => document.getElementById(id);
const COLORS = {Positive:'var(--pos)', Negative:'var(--neg)', Neutral:'var(--neu)'};
const ORDER = ['Positive','Neutral','Negative'];
const KEY = 'toneproof.history.v1';
let history = [];
try { history = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { history = []; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(history.slice(0, 12))); } catch (e) {} };
const pct = x => Math.round(x * 1000) / 10 + '%';

function renderHistory() {
  const list = $('history'); list.innerHTML = '';
  history.slice(0, 8).forEach(h => {
    const li = document.createElement('li'), b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = '<span class="dot"></span><span class="txt"></span><span class="pc"></span>';
    b.querySelector('.dot').style.background = COLORS[h.label];
    b.querySelector('.txt').textContent = h.text;
    b.querySelector('.pc').textContent = h.label + ' ' + Math.round(h.conf * 100) + '%';
    b.addEventListener('click', () => { $('input').value = h.text; updateCount(); analyze(); });
    li.appendChild(b); list.appendChild(li);
  });
  $('hist-empty').hidden = history.length > 0;
  $('clear').hidden = history.length === 0;
}

function updateCount() { $('count').textContent = $('input').value.length + ' / 1000'; }

function showError(msg) { const e = $('error'); e.textContent = msg || ''; e.hidden = !msg; }

function render(data) {
  document.documentElement.style.setProperty('--accent', COLORS[data.label]);
  $('empty').hidden = true; $('result').hidden = false;
  $('label').textContent = data.label;
  $('conf').textContent = pct(data.confidence);
  const probs = $('probs'); probs.innerHTML = '';
  ORDER.forEach(c => {
    const row = document.createElement('div'); row.className = 'prob';
    row.innerHTML = '<span></span><div class="track"><div class="fill"></div></div><span class="pct"></span>';
    row.children[0].textContent = c;
    row.querySelector('.fill').style.background = COLORS[c];
    row.querySelector('.pct').textContent = pct(data.probabilities[c]);
    probs.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => row.querySelector('.fill').style.width = data.probabilities[c] * 100 + '%'));
  });
  // marked-up text: scale influence to the strongest word
  const max = Math.max(0.0001, ...data.tokens.filter(t => !t.space).map(t => t.w));
  const p = $('marked'); p.innerHTML = ''; p.classList.remove('reveal'); void p.offsetWidth;
  let n = 0;
  data.tokens.forEach(t => {
    if (t.space || t.w <= max * 0.18) { p.appendChild(document.createTextNode(t.t)); return; }
    const m = document.createElement('mark');
    m.style.setProperty('--i', Math.min(1, t.w / max).toFixed(2));
    m.style.setProperty('--n', n++);
    m.title = 'Pushed the verdict by ' + (t.w * 100).toFixed(1) + ' points';
    m.textContent = t.t; p.appendChild(m);
  });
  p.classList.add('reveal');
  $('proof').hidden = false;
}

async function analyze() {
  const text = $('input').value.trim();
  showError('');
  if (!text) { showError('Enter some text to analyze.'); return; }
  const btn = $('go'); btn.disabled = true; btn.textContent = 'Analyzing…';
  try {
    const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const data = await r.json();
    if (!r.ok) { showError(data.error || 'Something went wrong. Try again.'); return; }
    render(data);
    history = [{text, label:data.label, conf:data.confidence}, ...history.filter(h => h.text !== text)];
    save(); renderHistory();
  } catch (e) {
    showError('Could not reach the server. Check that the app is running and try again.');
  } finally { btn.disabled = false; btn.textContent = 'Analyze text'; }
}

async function loadMetrics() {
  let m; try { const r = await fetch('/api/metrics'); if (!r.ok) return; m = await r.json(); } catch (e) { return; }
  const stats = [
    [pct(m.accuracy), 'Test accuracy on ' + m.test_samples.toLocaleString() + ' held-out tweets', 'var(--pos)'],
    [pct(m.original_notebook_accuracy_reported), 'Accuracy of the first notebook model', 'var(--neu)'],
    [pct(m.f1_weighted), 'Weighted F1 score', 'var(--pos)'],
    [pct(m.strict_group_split_accuracy), 'Accuracy when no tweet family is shared with training', 'var(--neg)'],
  ];
  $('stats').innerHTML = stats.map(s => `<div class="stat" style="--c:${s[2]}"><b>${s[0]}</b><span>${s[1]}</span></div>`).join('');
  $('perclass').querySelector('tbody').innerHTML = m.labels.map(l => {
    const c = m.per_class[l]; return `<tr><td>${l}</td><td>${c.precision.toFixed(3)}</td><td>${c.recall.toFixed(3)}</td><td>${c['f1-score'].toFixed(3)}</td></tr>`; }).join('');
  const cm = $('cm'); let html = '<span class="h"></span>' + m.labels.map(l => `<span class="h">${l}</span>`).join('');
  m.confusion_matrix.forEach((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0);
    html += `<span class="h">${m.labels[i]}</span>` + row.map((v, j) => `<span class="c ${i === j ? 'diag' : ''}" style="--v:${(v / tot).toFixed(2)}">${v.toLocaleString()}</span>`).join('');
  });
  cm.innerHTML = html;
  $('honest').innerHTML = `<h3>Read the accuracy carefully</h3><p>This dataset contains many reworded copies of the same tweet. A random split puts some copies in training and some in testing, which makes the score look high (${pct(m.accuracy)}). When whole tweet families are kept out of training, accuracy is ${pct(m.strict_group_split_accuracy)}. That second number is a better guide to how the model handles brand-new text.</p>`;
}

$('go').addEventListener('click', analyze);
$('input').addEventListener('input', updateCount);
$('input').addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze(); });
document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => { $('input').value = c.dataset.ex; updateCount(); analyze(); }));
$('clear').addEventListener('click', () => { history = []; save(); renderHistory(); });
$('input').value = $('input').value; updateCount(); renderHistory(); loadMetrics(); analyze();
