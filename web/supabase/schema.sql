-- Schema for the remote viewer. Everything lives in a schema that the REST API does not expose; the only things the public (anon) key can reach are the two functions below,
-- and both refuse unless the caller presents a token whose SHA-256 hash is in paper_private.tokens. Tokens are random 32-byte strings generated outside the database.
create extension if not exists pgcrypto with schema extensions;
create schema if not exists paper_private;

create table if not exists paper_private.tokens (name text primary key check (name in ('read', 'ingest')), hash bytea not null);
create table if not exists paper_private.snapshot (id smallint primary key check (id = 1), updated_at timestamptz not null default now(), payload jsonb not null);
alter table paper_private.tokens enable row level security;
alter table paper_private.snapshot enable row level security;
revoke all on schema paper_private from public, anon, authenticated;
revoke all on all tables in schema paper_private from public, anon, authenticated;

create or replace function public.paper_get(p_token text) returns jsonb
language plpgsql security definer set search_path = '' as $$
declare r record;
begin
  if p_token is null or not exists (select 1 from paper_private.tokens t where t.name = 'read' and t.hash = extensions.digest(p_token, 'sha256')) then
    raise exception 'forbidden' using errcode = '28000';
  end if;
  select s.updated_at, s.payload into r from paper_private.snapshot s where s.id = 1;
  if not found then return jsonb_build_object('waiting', true); end if;
  return r.payload || jsonb_build_object('published_at', r.updated_at, 'published_age_s', extract(epoch from (now() - r.updated_at)));
end $$;

create or replace function public.paper_put(p_token text, p_payload jsonb) returns void
language plpgsql security definer set search_path = '' as $$
begin
  if p_token is null or not exists (select 1 from paper_private.tokens t where t.name = 'ingest' and t.hash = extensions.digest(p_token, 'sha256')) then
    raise exception 'forbidden' using errcode = '28000';
  end if;
  insert into paper_private.snapshot (id, updated_at, payload) values (1, now(), p_payload)
  on conflict (id) do update set updated_at = now(), payload = excluded.payload;
end $$;

revoke all on function public.paper_get(text) from public, anon, authenticated;
revoke all on function public.paper_put(text, jsonb) from public, anon, authenticated;
grant execute on function public.paper_get(text) to anon;
grant execute on function public.paper_put(text, jsonb) to anon;

-- To set or rotate a token (run with the SHA-256 hex of the new token, never the token itself):
--   insert into paper_private.tokens (name, hash) values ('read', decode('<sha256 hex>', 'hex')) on conflict (name) do update set hash = excluded.hash;
