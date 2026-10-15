// node --test web/test_api.js   (no dependencies; fetch is mocked)
const test = require('node:test');
const assert = require('node:assert');
const { Readable } = require('node:stream');

process.env.SUPABASE_URL = 'https://example.supabase.co/';
process.env.SUPABASE_KEY = 'sb_publishable_test';
const state = require('./api/state');
const ingest = require('./api/ingest');

const GOOD = 'A'.repeat(43);
function mkRes() { const r = { headers: {}, statusCode: 0, body: '' }; r.setHeader = (k, v) => { r.headers[k.toLowerCase()] = v; }; r.end = (b) => { r.body = b || ''; r.done = true; }; return r; }
function mkReq(method, headers, body) { const r = Readable.from(body === undefined ? [] : [Buffer.from(body)]); r.method = method; r.headers = headers; return r; }
let calls;
function mockFetch(status, text) { calls = []; global.fetch = async (url, opt) => { calls.push({ url, opt }); return { status, text: async () => text }; }; }

test('state: no key or malformed key is rejected without touching the database', async () => {
  mockFetch(200, '{}');
  for (const h of [{}, { 'x-dash-key': 'short' }, { 'x-dash-key': 'a b'.repeat(10) }, { 'x-dash-key': "x'; drop table".padEnd(30, 'y') }]) {
    const res = mkRes(); await state(mkReq('GET', h), res); assert.equal(res.statusCode, 401);
  }
  assert.equal(calls.length, 0);
});

test('state: passes the key to the database function and returns the snapshot; no-store and noindex headers', async () => {
  mockFetch(200, '{"meta":{"mode":"REPLAY"},"published_age_s":3}');
  const res = mkRes(); await state(mkReq('GET', { 'x-dash-key': GOOD }), res);
  assert.equal(res.statusCode, 200); assert.equal(res.body, '{"meta":{"mode":"REPLAY"},"published_age_s":3}');
  assert.equal(res.headers['cache-control'], 'no-store'); assert.match(res.headers['x-robots-tag'], /noindex/);
  assert.equal(calls[0].url, 'https://example.supabase.co/rest/v1/rpc/paper_get');
  assert.deepEqual(JSON.parse(calls[0].opt.body), { p_token: GOOD });
  assert.equal(calls[0].opt.headers.apikey, 'sb_publishable_test'); assert.equal(calls[0].opt.headers.Authorization, undefined);
});

test('state: a database refusal becomes 401 and nothing from the upstream body leaks', async () => {
  for (const s of [401, 403]) { mockFetch(s, '{"message":"forbidden","hint":"secret detail"}'); const res = mkRes(); await state(mkReq('GET', { 'x-dash-key': GOOD }), res); assert.equal(res.statusCode, 401); assert.ok(!res.body.includes('secret')); }
  mockFetch(500, 'boom'); const r2 = mkRes(); await state(mkReq('GET', { 'x-dash-key': GOOD }), r2); assert.equal(r2.statusCode, 502); assert.ok(!r2.body.includes('boom'));
});

test('state: other methods are refused', async () => { const res = mkRes(); await state(mkReq('POST', { 'x-dash-key': GOOD }), res); assert.equal(res.statusCode, 405); });

test('ingest: needs a bearer token, a JSON object that looks like a snapshot, and POST', async () => {
  mockFetch(204, '');
  let res = mkRes(); await ingest(mkReq('POST', {}, '{}'), res); assert.equal(res.statusCode, 401);
  res = mkRes(); await ingest(mkReq('POST', { authorization: 'Bearer short' }, '{}'), res); assert.equal(res.statusCode, 401);
  res = mkRes(); await ingest(mkReq('POST', { authorization: 'Bearer ' + GOOD }, 'not json'), res); assert.equal(res.statusCode, 400);
  res = mkRes(); await ingest(mkReq('POST', { authorization: 'Bearer ' + GOOD }, '[1,2]'), res); assert.equal(res.statusCode, 400);
  res = mkRes(); await ingest(mkReq('POST', { authorization: 'Bearer ' + GOOD }, '{"hello":1}'), res); assert.equal(res.statusCode, 400);
  res = mkRes(); await ingest(mkReq('GET', { authorization: 'Bearer ' + GOOD }), res); assert.equal(res.statusCode, 405);
  assert.equal(calls.length, 0);
});

test('ingest: a valid snapshot reaches paper_put with the token; a refusal is 401', async () => {
  mockFetch(204, ''); const snap = { meta: { mode: 'LIVE' }, farm: { net: 1 } };
  let res = mkRes(); await ingest(mkReq('POST', { authorization: 'Bearer ' + GOOD }, JSON.stringify(snap)), res);
  assert.equal(res.statusCode, 200); assert.equal(calls[0].url, 'https://example.supabase.co/rest/v1/rpc/paper_put');
  assert.deepEqual(JSON.parse(calls[0].opt.body), { p_token: GOOD, p_payload: snap });
  mockFetch(403, '{"message":"forbidden"}'); res = mkRes(); await ingest(mkReq('POST', { authorization: 'Bearer ' + GOOD }, JSON.stringify(snap)), res); assert.equal(res.statusCode, 401);
});

test('ingest: a body over the limit is refused', async () => {
  mockFetch(204, ''); const big = JSON.stringify({ meta: {}, farm: {}, pad: 'x'.repeat(4_100_000) });
  const res = mkRes(); await ingest(mkReq('POST', { authorization: 'Bearer ' + GOOD }, big), res); assert.equal(res.statusCode, 400); assert.equal(calls.length, 0);
});

test('legacy JWT-style keys are also sent as a bearer', async () => {
  delete require.cache[require.resolve('./api/_supabase')]; delete require.cache[require.resolve('./api/state')];
  process.env.SUPABASE_KEY = 'eyJhbGciOi.test.key'; const st2 = require('./api/state'); mockFetch(200, '{}');
  const res = mkRes(); await st2(mkReq('GET', { 'x-dash-key': GOOD }), res); assert.equal(calls[0].opt.headers.Authorization, 'Bearer eyJhbGciOi.test.key');
});
