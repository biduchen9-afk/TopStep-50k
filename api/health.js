// Deployment check: reports whether the two environment variables are set and whether the database answers. Nothing secret is returned or needed.
const { baseHeaders, send, rpc } = require('./_supabase');

module.exports = async (req, res) => {
  baseHeaders(res);
  const env = { SUPABASE_URL: Boolean(process.env.SUPABASE_URL), SUPABASE_KEY: Boolean(process.env.SUPABASE_KEY) };
  if (!env.SUPABASE_URL || !env.SUPABASE_KEY) return send(res, 200, { ok: false, env, db: 'not configured' });
  // A call with a token that cannot be valid: a reachable, correctly installed database answers "forbidden" (403); anything else is a problem.
  const r = await rpc('paper_get', { p_token: 'health-check-not-a-real-token' });
  const db = r.status === 403 || r.status === 401 ? 'reachable, functions installed' : r.status === 404 ? 'reachable, schema not installed' : r.status === 502 ? 'unreachable' : 'unexpected status ' + r.status;
  return send(res, 200, { ok: db.startsWith('reachable, functions'), env, db });
};
