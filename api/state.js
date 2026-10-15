const { baseHeaders, send, rpc, TOKEN_RE } = require('./_supabase');

module.exports = async (req, res) => {
  baseHeaders(res);
  if (req.method !== 'GET') return send(res, 405, { error: 'method not allowed' });
  const key = String(req.headers['x-dash-key'] || '');
  if (!TOKEN_RE.test(key)) return send(res, 401, { error: 'unauthorized' });
  const r = await rpc('paper_get', { p_token: key });
  if (r.status === 200) {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/json');
    return res.end(r.body);
  }
  if (r.status === 401 || r.status === 403) return send(res, 401, { error: 'unauthorized' });
  return send(res, 502, { error: 'upstream' });
};
