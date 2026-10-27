const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'frontend', 'index.html'), 'utf8');
const appScript = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const demoData = JSON.parse(fs.readFileSync(path.join(root, 'frontend', 'paysim_network_sample.json'), 'utf8'));
const networkData = demoData;
const labelledPaySimData = JSON.parse(fs.readFileSync(path.join(root, 'frontend', 'paysim_labelled_examples.json'), 'utf8'));

class Element {
  constructor(tag = 'div', id = '') {
    this.tagName = tag.toUpperCase(); this.id = id; this.children = []; this.listeners = {};
    this.attributes = {}; this.dataset = {}; this.style = {}; this.hidden = false; this.disabled = false;
    this.value = ''; this.open = false; this.parent = null; this._text = ''; this._className = '';
    this.classList = {
      add: (...names) => names.forEach(name => this.classList.toggle(name, true)),
      remove: (...names) => names.forEach(name => this.classList.toggle(name, false)),
      contains: name => this._className.split(/\s+/).includes(name),
      toggle: (name, force) => {
        const names = new Set(this._className.split(/\s+/).filter(Boolean));
        const enabled = force === undefined ? !names.has(name) : force;
        if (enabled) names.add(name); else names.delete(name);
        this._className = [...names].join(' '); return enabled;
      },
    };
  }
  set className(value) { this._className = value; }
  get className() { return this._className; }
  set textContent(value) { this.children = []; this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  set innerHTML(value) { this.textContent = value.replace(/<[^>]*>/g, ''); }
  append(...nodes) { for (const node of nodes) { if (node == null) continue; node.parent = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === 'class') this.className = String(value);
    if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(value);
  }
  getAttribute(name) { return this.attributes[name] ?? null; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  fire(name, event = {}) { for (const fn of this.listeners[name] || []) fn({ target: this, currentTarget: this, preventDefault() { }, stopPropagation() { }, ...event }); }
  focus() { }
  showModal() { this.open = true; }
  close() { this.open = false; }
  reportValidity() { return true; }
  reset() { }
  querySelectorAll(selector) { return walk(this).filter(node => matches(node, selector)); }
}

function walk(rootNode) { return rootNode.children.flatMap(node => [node, ...walk(node)]); }
function matches(node, selector) {
  if (selector === '.demo-node') return node.classList.contains('demo-node');
  if (selector === '.demo-edge') return node.classList.contains('demo-edge');
  if (selector === '#demoTransactionRows tr') return node.tagName === 'TR' && hasAncestor(node, 'demoTransactionRows');
  if (selector === '.view') return node.classList.contains('view');
  return false;
}
function hasAncestor(node, id) { while (node) { if (node.id === id) return true; node = node.parent; } return false; }
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
function response(body, status = 200) { return { ok: status >= 200 && status < 300, status, json: async () => body }; }
const tick = () => new Promise(resolve => setImmediate(resolve));

function createApp() {
  const created = [];
  const byId = new Map();
  const get = (id) => { if (!byId.has(id)) { const node = new Element('div', id); byId.set(id, node); created.push(node); } return byId.get(id); };
  for (const match of html.matchAll(/<([a-z0-9]+)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) {
    const node = get(match[2]); node.tagName = match[1].toUpperCase();
    if (/\shidden(?:\s|>|$)/.test(match[0])) node.hidden = true;
    const ariaHidden = match[0].match(/aria-hidden="([^"]+)"/); if (ariaHidden) node.setAttribute('aria-hidden', ariaHidden[1]);
  }
  const transactionForm = get('transactionForm');
  transactionForm.elements = {};
  for (const match of html.matchAll(/<(?:input|select)\b[^>]*\bid="([^"]+)"[^>]*\bname="([^"]+)"[^>]*>/gi)) {
    const field = get(match[1]);
    field.name = match[2];
    transactionForm.elements[match[2]] = field;
    const value = match[0].match(/\bvalue="([^"]*)"/);
    if (value && !field.value) field.value = value[1];
  }
  const views = ['overview', 'transaction', 'graph', 'url', 'history', 'research'].map(name => {
    const node = get(`view-${name}`); node.classList.add('view'); node.hidden = name !== 'overview'; return node;
  });
  const nav = ['overview', 'transaction', 'graph', 'url', 'history', 'research'].map(name => {
    const node = new Element('button'); node.dataset.view = name; created.push(node); return node;
  });
  const document = {
    getElementById: get,
    createElement: tag => { const node = new Element(tag); created.push(node); return node; },
    createElementNS: (_ns, tag) => { const node = new Element(tag); created.push(node); return node; },
    addEventListener() { },
    querySelectorAll(selector) {
      if (selector === '.drawer-nav button[data-view]') return nav;
      if (selector === '.view') return views;
      if (selector === '[data-go]') return [];
      return created.filter(node => matches(node, selector));
    },
  };
  const historyRequests = [];
  const predictionRequests = [];
  const pending = [];
  let failNextHistory = false;
  const localStorage = { value: 'token-A', getItem() { return this.value; }, setItem(_k, v) { this.value = v; }, removeItem() { this.value = null; } };
  const fetch = async (url, options = {}) => {
    if (url === '/static/paysim_network_sample.json') return response(networkData);
    if (url === '/static/paysim_labelled_examples.json') return response(labelledPaySimData);
    if (url === '/api/health') return response({ model_loaded: true });
    if (url === '/api/auth/me') return response({ username: options.headers?.Authorization?.replace('Bearer ', '') === 'token-B' ? 'account_b' : 'account_a' });
    if (url === '/api/history') {
      const auth = options.headers?.Authorization || '';
      historyRequests.push(auth);
      if (failNextHistory) { failNextHistory = false; throw new Error('network unavailable'); }
      const request = deferred(); pending.push(request); return request.promise;
    }
    if (url === '/api/predict') {
      predictionRequests.push({ url, options });
      return response({ fraud_probability: 0.96, is_fraud: true, threshold: 0.95 });
    }
    if (url === '/api/predict/gnn') {
      predictionRequests.push({ url, options });
      return response({ decision: 'NOT FRAUD', is_fraud: false, raw_score: 0.4, threshold: 0.9023 });
    }
    return response({ detail: 'unexpected route' }, 404);
  };
  const context = {
    document, localStorage, fetch, console, Math, Date, Map, Set, Object, Number, String, Array, JSON,
    FormData: class { constructor(form) { this.form = form; } entries() { return Object.entries(this.form?.elements || {}).map(([name, field]) => [name, field.value]); } },
    window: { setTimeout(fn) { fn(); }, scrollTo() { } }, requestAnimationFrame(fn) { fn(); },
  };
  const testExport = `\n;globalThis.__app={showView,loadHistory,authChanged,renderHistoryState,renderDemoGraph,inspectDemoAccount,showDemoTransaction,clearDemoSelection,renderTransactionResult,renderGNNResult,renderUrlAnalysis,renderQrScanResult,historyAssessmentName,historyResultLabel,setDemoError,setSessionForTest:(nextToken,nextUser)=>{token=nextToken;username=nextUser;authReady=true;if(nextToken)localStorage.setItem(tokenKey,nextToken);else localStorage.removeItem(tokenKey);authChanged();}};`;
  vm.runInNewContext(appScript + testExport, context, { filename: 'frontend/index.html' });
  return { app: context.__app, get, pending, historyRequests, predictionRequests, failHistory() { failNextHistory = true; }, created, document };
}

test('history moves through loading, records, empty, and error states; switching accounts clears and guards stale results', async () => {
  const h = createApp(); await tick();
  h.app.showView('history');
  assert.match(h.get('historyMessage').textContent, /Loading your saved assessments/);
  assert.equal(h.pending.length, 1);
  h.pending[0].resolve(response({ assessments: [{ created_at: '2026-10-01T10:00:00Z', kind: 'transaction', decision: 'Potential fraud', score: 0.4, threshold: 0.95, component_version: 'xgb-test' }] }));
  await tick();
  assert.equal(h.get('historyMessage').hidden, true);
  assert.equal(h.get('historyTableWrap').hidden, false);
  assert.equal(h.get('historyRows').children.length, 1);
  assert.match(h.get('historyRows').textContent, /Primary XGBoost model/);
  assert.match(h.get('historyRows').textContent, /Flagged for review/);
  assert.match(h.get('historyRows').textContent, /40\.00%/);
  assert.doesNotMatch(h.get('historyRows').textContent, /xgb-test|threshold/);

  h.app.loadHistory();
  const stale = h.pending[1];
  h.app.setSessionForTest('token-B', 'account_b');
  assert.equal(h.get('historyRows').children.length, 0, 'old account rows clear before the new request returns');
  assert.equal(h.get('historyTableWrap').hidden, true);
  const current = h.pending[2];
  stale.resolve(response({ assessments: [{ created_at: '2026-10-02T10:00:00Z', kind: 'url', decision: 'STALE-A', score: null, threshold: null, component_version: 'old' }] }));
  await tick();
  assert.equal(h.get('historyRows').children.length, 0, 'late prior-account response is ignored');
  current.resolve(response({ assessments: [{ created_at: '2026-10-03T10:00:00Z', kind: 'url', decision: 'high', score: null, threshold: null, component_version: 'url-v1' }] }));
  await tick();
  assert.match(h.get('historyRows').textContent, /Warning signs found/);
  assert.match(h.get('historyRows').textContent, /No model score/);
  assert.doesNotMatch(h.get('historyRows').textContent, /STALE-A/);

  h.failHistory(); h.app.loadHistory();
  assert.match(h.get('historyMessage').textContent, /Loading your saved assessments/);
  await tick();
  assert.match(h.get('historyMessage').textContent, /network unavailable/);
  assert.equal(h.get('historyTableWrap').hidden, true);

  h.app.loadHistory(); h.pending[3].resolve(response({ assessments: [] })); await tick();
  assert.match(h.get('historyMessage').textContent, /No saved assessments yet/);
  assert.equal(h.get('historyTableWrap').hidden, true);

  const before = h.historyRequests.length;
  h.app.setSessionForTest(null, null);
  assert.match(h.get('historyMessage').textContent, /Sign in to view your history/);
  assert.equal(h.get('historyRows').children.length, 0);
  assert.equal(h.historyRequests.length, before, 'signed-out state does not request private history');
});

test('URL and QR results use plain-language labels and keep plain QR text separate from URL indicators', () => {
  const h = createApp();
  const review = { risk_level: 'review', hostname: 'bit.ly', indicators: [{ severity: 'review', message: 'The hostname belongs to a known URL-shortening service.' }], explanation: 'Local indicator check only.' };
  h.app.renderUrlAnalysis(review, 'https://bit.ly/demo', 'Decoded and checked locally.');
  assert.match(h.get('urlResult').textContent, /Decoded QR content/);
  assert.match(h.get('urlResult').textContent, /Warning signs to review/);
  assert.match(h.get('urlResult').textContent, /known URL-shortening service/);
  assert.doesNotMatch(h.get('urlResult').textContent, /Indicator level: review/);

  h.app.renderUrlAnalysis({ risk_level: 'no-listed-indicators', hostname: 'example.test', indicators: [], explanation: 'No guarantees.' });
  assert.match(h.get('urlResult').textContent, /No listed warning signs detected/);
  assert.match(h.get('urlResult').textContent, /does not mean the URL is safe/);

  h.app.renderQrScanResult({ content_type: 'text', decoded_content: 'Payment reference 42', message: 'This QR contains text, not a web URL.' });
  assert.match(h.get('urlResult').textContent, /QR contains plain text/);
  assert.match(h.get('urlResult').textContent, /Payment reference 42/);
  assert.doesNotMatch(h.get('urlResult').textContent, /Warning signs|Indicator level/);

  h.app.renderQrScanResult({
    detected_format: 'upi_payment_uri',
    content_type: 'upi_payment_uri',
    decoded_content: 'upi://pay?pa=alice%40upi&am=125.50&cu=INR',
    upi_fields: [
      { key: 'pa', label: 'Payee VPA (unverified)', value: 'alice@upi' },
      { key: 'am', label: 'Amount supplied', value: '125.50' },
      { key: 'cu', label: 'Currency supplied', value: 'INR' },
    ],
    risk_assessment: {
      status: 'not_assessed', level: 'not_assessed',
      label: 'UPI payment request recognized; payment legitimacy not assessed',
      warning_signs: [], explanation: 'Recipient identity is not verified.',
    },
    message: 'No payment was initiated.',
  });
  assert.match(h.get('urlResult').textContent, /UPI payment URI decoded/);
  assert.match(h.get('urlResult').textContent, /Risk assessment: UPI payment request recognized/);
  assert.match(h.get('urlResult').textContent, /Payee VPA \(unverified\): alice@upi/);
  assert.match(h.get('urlResult').textContent, /No payment was initiated/);
  assert.doesNotMatch(h.get('urlResult').textContent, /fraudulent|guaranteed safe/i);
});

test('transaction result presents decision-based risk labels and keeps raw uncalibrated output in collapsed details', () => {
  const h = createApp();
  const low = Object.freeze({ fraud_probability: 0.0000002, is_fraud: false });
  h.app.renderTransactionResult({ ...low, threshold: 0.95 });
  const result = h.get('transactionResult');
  assert.match(result.children[0].textContent, /Primary model · XGBoost/);
  assert.match(result.children[1].textContent, /NOT FRAUD/);
  assert.match(result.children[2].textContent, /did not flag this record/);
  assert.match(result.children[2].textContent, /does not establish that it is fraud-free/);
  assert.doesNotMatch(result.children.slice(0, 3).map(node => node.textContent).join(' '), /0\.0000002|0\.0000200%/);
  const details = result.children[3];
  assert.equal(details.tagName, 'DETAILS');
  assert.equal(details.open, false, 'technical values stay collapsed by default');
  assert.match(details.textContent, /Model details/);
  assert.match(details.textContent, /Raw XGBoost model output/);
  assert.match(details.textContent, /2e-7/);
  assert.match(details.textContent, /Saved decision threshold/);
  assert.match(details.textContent, /0\.95/);
  assert.match(details.textContent, /Not calibrated as a real-world fraud probability/);
  assert.match(details.textContent, /Not calibrated as a real-world fraud probability/);
  assert.equal(walk(result).some(node => node.classList.contains('meter') || node.classList.contains('threshold')), false);
  assert.equal(low.fraud_probability, 0.0000002, 'display formatting leaves the raw API score unchanged');

  h.app.renderTransactionResult({ fraud_probability: 0.96, is_fraud: true, threshold: 0.95 });
  assert.match(h.get('transactionResult').children[1].textContent, /FRAUD/);
  assert.match(h.get('transactionResult').children[2].textContent, /flagged this record/);
  assert.doesNotMatch(h.get('transactionResult').children.slice(0, 3).map(node => node.textContent).join(' '), /96\.00%|0\.96/);
});

test('experimental GNN has a separate result panel and leaves XGBoost result handling intact', () => {
  const h = createApp();
  h.app.renderTransactionResult({ fraud_probability: 0.96, is_fraud: true, threshold: 0.95 });
  const xgboostResult = h.get('transactionResult');
  h.app.renderGNNResult({ decision: 'NOT FRAUD', raw_score: 0.4, threshold: 0.9023 });

  assert.match(xgboostResult.textContent, /Raw XGBoost model output/);
  assert.match(xgboostResult.textContent, /FRAUD/);
  assert.match(h.get('gnnResult').textContent, /Experimental node-feature-only GNN/);
  assert.match(h.get('gnnResult').textContent, /NOT FRAUD/);
  assert.match(h.get('gnnResult').textContent, /Raw GNN model score/);
  assert.match(h.get('gnnResult').textContent, /Not calibrated as a real-world fraud probability/);
  assert.match(h.app.historyAssessmentName({ kind: 'gnn_transaction' }), /Experimental node-only GNN/);
  assert.match(appScript, /api\('\/api\/predict'/, 'primary XGBoost request path remains present');
  assert.match(appScript, /api\('\/api\/predict\/gnn'/, 'GNN request uses a separate path');
});

test('labelled PaySim examples stay out of prediction payloads and show sample-level comparisons', async () => {
  const h = createApp(); await tick();
  const examples = labelledPaySimData.examples;
  assert.equal(h.get('paysimExample').children.length, 8);
  assert.equal(examples.filter(item => item.ground_truth === 1).length, 4);
  assert.equal(examples.filter(item => item.ground_truth === 0).length, 4);

  h.get('paysimExample').value = examples[0].id;
  h.get('loadPaySimExample').fire('click');
  assert.match(h.get('paysimGroundTruth').textContent, /Known dataset label \(isFraud\): NOT FRAUD/);
  assert.equal(Number(h.get('transactionForm').elements.amount.value), 272660.61);

  h.get('transactionForm').fire('submit');
  await tick();
  assert.equal(h.predictionRequests[0].url, '/api/predict');
  const xgbPayload = JSON.parse(h.predictionRequests[0].options.body);
  assert.equal(xgbPayload.amount, 272660.61);
  assert.equal('ground_truth' in xgbPayload, false);
  assert.equal('isFraud' in xgbPayload, false);
  assert.match(h.get('paysimModelComparison').textContent, /XGBoost: predicted FRAUD\. Mismatch for this sample/);

  h.get('gnnSubmitButton').fire('click');
  await tick();
  assert.equal(h.predictionRequests[1].url, '/api/predict/gnn');
  const gnnPayload = JSON.parse(h.predictionRequests[1].options.body);
  assert.equal(gnnPayload.amount, 272660.61);
  assert.equal('ground_truth' in gnnPayload, false);
  assert.equal('isFraud' in gnnPayload, false);
  assert.match(h.get('paysimModelComparison').textContent, /Experimental node-only GNN: predicted NOT FRAUD\. Correct for this sample/);
  assert.match(h.get('paysimModelComparison').textContent, /Each result below describes this example only, not model accuracy/);

  h.get('transactionForm').elements.amount.value = '1';
  h.get('transactionForm').fire('input', { target: h.get('transactionForm').elements.amount });
  assert.match(h.get('paysimExampleStatus').textContent, /no longer applies/);
  assert.match(h.get('paysimModelComparison').textContent, /Run XGBoost and the experimental node-only GNN independently/);
});

test('PaySim sample graph shows genuine sample account counts and relationship inspector uses the selected sample', async () => {
  const h = createApp(); await tick();
  const nodes = h.document.querySelectorAll('.demo-node');
  assert.equal(nodes.length, 4, `${h.get('demoGraphStatus').textContent} ${h.get('demoDataStatus').textContent}`);
  assert.match(h.get('demoGraphStatus').textContent, /3 selected transactions/);
  const recipient = nodes.find(node => node.getAttribute('data-account-id') === 'C613935961');
  recipient.fire('click');
  assert.match(h.get('demoAccountDetails').textContent, /Connected accounts: 3/);
  assert.match(h.get('demoAccountDetails').textContent, /Incoming \(money received\): 3 transactions/);
  assert.match(h.get('demoAccountDetails').textContent, /Outgoing \(money sent\): 0 transactions/);
  assert.equal(h.document.querySelectorAll('#demoTransactionRows tr').length, 3);
  h.get('paysimGraphSource').value = 'C2127461691';
  h.get('paysimGraphTarget').value = 'C613935961';
  h.get('paysimGraphForm').fire('submit');
  assert.match(h.get('paysimGraphResult').textContent, /Transfers from C2127461691 to C613935961: 1/);
  assert.match(h.get('paysimGraphResult').textContent, /Reverse transfers: 0/);
  assert.match(h.get('paysimGraphResult').textContent, /3 selected test-split transactions only/);
});

test('demo graph exposes a useful load-error state and rejects an invalid account selection clearly', async () => {
  const loaded = createApp(); await tick();
  loaded.app.setDemoError('demo file unavailable');
  assert.match(loaded.get('demoGraphStatus').textContent, /demo file unavailable/);
  assert.match(loaded.get('demoAccountDetails').textContent, /PaySim graph unavailable/);
  loaded.app.renderDemoGraph({ accounts: [], scenarios: [] });
  assert.match(loaded.get('demoAccountDetails').textContent, /No sample accounts/);
  assert.equal(loaded.get('demoTransactionEmpty').hidden, false);
  assert.match(loaded.get('demoTransactionEmpty').textContent, /No PaySim sample transaction rows/);
  loaded.app.inspectDemoAccount('NO-SUCH-ACCOUNT', demoData);
  assert.match(loaded.get('demoGraphStatus').textContent, /selected demo account is unavailable/);
  assert.equal(loaded.document.querySelectorAll('.demo-node').some(node => node.getAttribute('aria-pressed') === 'true'), false);

  const isolated = { accounts: [{ id: 'ISOLATED', name: 'Isolated demo account', x: 90, y: 90 }], scenarios: [] };
  loaded.app.renderDemoGraph(isolated);
  loaded.app.inspectDemoAccount('ISOLATED', isolated);
  assert.match(loaded.get('demoAccountDetails').textContent, /Connected accounts: 0/);
  assert.match(loaded.get('demoAccountDetails').textContent, /Incoming \(money received\): 0 transactions · total amount 0/);
  assert.match(loaded.get('demoAccountDetails').textContent, /Outgoing \(money sent\): 0 transactions · total amount 0/);
  assert.match(loaded.get('demoAccountDetails').textContent, /No connected sample transactions/);
});

