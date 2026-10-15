// Shared helpers for the two functions. The only secrets in this app are the Supabase URL and its PUBLISHABLE key (meant to be public); access to the data is gated by two random tokens
// that live nowhere on the server: the database stores only their SHA-256 hashes and checks them inside paper_get / paper_put.
const URL_ = (process.env.SUPABASE_URL || '').replace(/\/+$/, '');
const KEY = process.env.SUPABASE_KEY || '';

function baseHeaders(res) {
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Robots-Tag', 'noindex, nofollow');
  res.setHeader('X-Content-Type-Options', 'nosniff');
}

function send(res, code, obj) {
  res.statusCode = code;
  res.setHeader('Content-Type', 'application/json');
  res.end(obj === undefined ? '' : JSON.stringify(obj));
}

async function rpc(name, args) {
  if (!URL_ || !KEY) return { status: 500, body: null };
  const headers = { 'Content-Type': 'application/json', apikey: KEY };
  if (KEY.startsWith('eyJ')) headers.Authorization = 'Bearer ' + KEY;       // legacy JWT keys need it, the new sb_publishable_ keys must not carry it
  let r;
  try {
    r = await fetch(`${URL_}/rest/v1/rpc/${name}`, { method: 'POST', headers, body: JSON.stringify(args), signal: AbortSignal.timeout(8000) });
  } catch (e) {
    return { status: 502, body: null };
  }
  const text = await r.text();
  return { status: r.status, body: text };
}

// A token is accepted only if it looks like one we generate (urlsafe, 20-128 chars); the database does the real check.
const TOKEN_RE = /^[A-Za-z0-9_-]{20,128}$/;

module.exports = { baseHeaders, send, rpc, TOKEN_RE };
