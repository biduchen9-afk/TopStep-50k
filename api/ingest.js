const { baseHeaders, send, rpc, TOKEN_RE } = require('./_supabase');

const MAX = 4_000_000;

function readBody(req) {
  return new Promise((resolve, reject) => {
    const parts = []; let n = 0;
    req.on('data', (c) => { n += c.length; if (n > MAX) { reject(new Error('too large')); req.destroy(); } else parts.push(c); });
    req.on('end', () => resolve(Buffer.concat(parts).toString('utf8')));
    req.on('error', reject);
  });
}

module.exports = async (req, res) => {
  baseHeaders(res);
  if (req.method !== 'POST') return send(res, 405, { error: 'method not allowed' });
  const m = /^Bearer (\S+)$/.exec(String(req.headers.authorization || ''));
  if (!m || !TOKEN_RE.test(m[1])) return send(res, 401, { error: 'unauthorized' });
  let payload;
  try { payload = JSON.parse(await readBody(req)); } catch (e) { return send(res, 400, { error: 'bad body' }); }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload) || !payload.meta || !payload.farm) return send(res, 400, { error: 'not a harness snapshot' });
  const r = await rpc('paper_put', { p_token: m[1], p_payload: payload });
  if (r.status >= 200 && r.status < 300) return send(res, 200, { ok: true });
  if (r.status === 401 || r.status === 403) return send(res, 401, { error: 'unauthorized' });
  return send(res, 502, { error: 'upstream' });
};
