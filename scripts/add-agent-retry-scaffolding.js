#!/usr/bin/env node
// add-agent-retry-scaffolding.js — codemod that applies issue #17 default scaffolding.
//
// For each workflow .js passed on the argv (or all top-level workflow files when
// run with --all), this:
//   1. Injects the canonical agentRetry/expect/guard helper block (from
//      _meta/scaffolding/agent-retry.js) once, right after the inlined arg_guard
//      (or after the `export const meta` block as a fallback).
//   2. Wraps EVERY global `agent(...)` call site as
//        await agentRetry(() => agent(...), { retries: 5 })
//      so a transient API 429 / agent-skip (which returns null) no longer crashes
//      the run when the structured result is dereferenced.
//
// The scanner is string / template-literal / comment aware, so `agent(` tokens
// that appear inside prompt template literals (which routinely contain example
// code with parens) or comments are never wrapped, and the matching close paren
// is found correctly even across multi-line template prompts.
//
// Idempotent: re-running leaves already-wrapped files unchanged.
//
// `--refresh` mode: instead of wrapping calls, REPLACE the existing inlined
// helper block (text between the BEGIN/END agent-retry sentinels) with the
// CURRENT canonical version from _meta/scaffolding/agent-retry.js. Use this after
// editing the canonical helper to propagate the change to every workflow that
// already inlined an older copy. Files lacking the sentinels (hand-edited or
// never injected by this codemod) are skipped with a warning — never corrupted.
//
// Validation is the caller's job: run `node --check` on every output file and the
// fidelity check. This script prints per-file wrap counts and exits non-zero if a
// file's transform looked malformed (unbalanced parens after wrap).

const fs = require('fs')
const path = require('path')

const REPO = path.resolve(__dirname, '..')
const HELPER_PATH = path.join(REPO, '_meta', 'scaffolding', 'agent-retry.js')

// Extract just the three function definitions from the canonical snippet (between
// the START/END sentinels we add here), so we inline exactly the implementations.
function readHelperBlock() {
  const raw = fs.readFileSync(HELPER_PATH, 'utf8')
  const start = raw.indexOf('async function agentRetry(')
  const end = raw.lastIndexOf('function guard(')
  if (start === -1 || end === -1) throw new Error('could not locate helper fns in ' + HELPER_PATH)
  const guardEnd = raw.indexOf('}', raw.indexOf('return obj[field]', end))
  const block = raw.slice(start, guardEnd + 1)
  return '// --- BEGIN inlined agent-retry scaffolding (from _meta/scaffolding/agent-retry.js) ---\n'
    + block.replace(/\n$/, '')
    + '\n// --- END inlined agent-retry scaffolding ---'
}

// One-pass lexer that classifies every char as "code" (true) or inside a
// string / template / comment / regex (false). Template ${...} interpolations
// are code (parens/braces count), ending at their matching `}`. Regex literals
// are lexed (keyword-aware `/` disambiguation) so their internal quotes / parens
// / char-classes do not corrupt string and paren tracking.
const REGEX_KEYWORDS = new Set([
  'return', 'typeof', 'instanceof', 'in', 'of', 'do', 'else', 'void', 'delete',
  'new', 'throw', 'case', 'yield', 'await', 'if', 'while', 'for', 'switch',
  'and', 'or', 'not', 'as', 'else',
])
function buildCodeMask(src) {
  const n = src.length
  const mask = new Uint8Array(n) // 0 = non-code, 1 = code
  let i = 0
  const stack = ['code']
  const top = () => { const e = stack[stack.length - 1]; return typeof e === 'string' ? e : e.state }
  let prevSig = ''      // last non-whitespace code char
  let identBuf = ''     // current identifier run
  let lastIdent = ''    // last completed identifier token
  const isIdent = (c) => /[A-Za-z0-9_$]/.test(c)
  while (i < n) {
    const st = top()
    const c = src[i], c2 = src[i + 1]
    if (st === 'code' || st === 'interp') {
      if (c === '/' && c2 === '/') { stack.push('line'); i += 2; continue }
      if (c === '/' && c2 === '*') { stack.push('block'); i += 2; continue }
      if (c === '/' && regexAllowed()) { stack.push('regex'); mask[i] = 0; i++; continue }
      if (c === "'") { commitIdent(); stack.push('sq'); mask[i] = 0; i++; continue }
      if (c === '"') { commitIdent(); stack.push('dq'); mask[i] = 0; i++; continue }
      if (c === '`') { commitIdent(); stack.push('tpl'); mask[i] = 0; i++; continue }
      if (st === 'interp') {
        const e = stack[stack.length - 1]
        if (c === '{') { e.depth++; note(c); mask[i] = 1; i++; continue }
        if (c === '}') { if (e.depth === 0) { stack.pop(); mask[i] = 0; i++; continue } e.depth--; note(c); mask[i] = 1; i++; continue }
      }
      note(c); mask[i] = 1; i++
    } else if (st === 'line') {
      mask[i] = 0
      if (c === '\n') stack.pop()
      i++
    } else if (st === 'block') {
      mask[i] = 0
      if (c === '*' && c2 === '/') { stack.pop(); i += 2; continue }
      i++
    } else if (st === 'sq' || st === 'dq') {
      mask[i] = 0
      const q = st === 'sq' ? "'" : '"'
      if (c === '\\') { i += 2; continue }
      if (c === q) { stack.pop(); prevSig = q; identBuf = ''; lastIdent = '' }
      i++
    } else if (st === 'tpl') {
      mask[i] = 0
      if (c === '\\') { i += 2; continue }
      if (c === '`') { stack.pop(); prevSig = '`'; identBuf = ''; lastIdent = ''; i++; continue }
      if (c === '$' && c2 === '{') { stack.push({ state: 'interp', depth: 0 }); i += 2; continue }
      i++
    } else if (st === 'regex') {
      mask[i] = 0
      if (c === '\\') { i += 2; continue }
      if (c === '[') { stack.push('regexClass'); i++; continue }
      if (c === '/') {
        stack.pop(); i++
        // consume flags
        while (i < n && /[A-Za-z]/.test(src[i])) { mask[i] = 1; prevSig = src[i]; i++ }
        prevSig = '/' // a following `/` after `/re/` flags is division
        identBuf = ''; lastIdent = ''
        continue
      }
      i++
    } else if (st === 'regexClass') {
      mask[i] = 0
      if (c === '\\') { i += 2; continue }
      if (c === ']') { stack.pop(); i++; continue }
      i++
    }
  }
  function note(c) {
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') return
    if (isIdent(c)) { identBuf += c }
    else { commitIdent(); prevSig = c }
  }
  function commitIdent() {
    if (identBuf) { lastIdent = identBuf; prevSig = identBuf[identBuf.length - 1]; identBuf = '' }
  }
  function regexAllowed() {
    commitIdent()
    if (isIdent(prevSig)) return REGEX_KEYWORDS.has(lastIdent)
    // `/` after these is division (value-end context), not a regex start
    if (/[)\]}"'`]/.test(prevSig)) return false
    if (/[0-9]/.test(prevSig)) return false
    if (prevSig === '.') return false
    return true // after = ( , : [ ; { ! & | ? < > + - * % ^ ~ etc., or at start
  }
  return mask
}

const isIdentChar = (c) => /[A-Za-z0-9_$]/.test(c)

// Find every global `agent(` call site that is not already wrapped, plus its
// matching close paren. Returns [{callStart, openParen, closeParen}] sorted asc.
function findAgentSites(src, mask) {
  const n = src.length
  const sites = []
  // First, a paren-matching pass over code regions to map open->close.
  const openStack = []
  const match = new Map()
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue
    const c = src[i]
    if (c === '(') openStack.push(i)
    else if (c === ')') { const o = openStack.pop(); if (o !== undefined) match.set(o, i) }
  }
  // Detect call sites.
  for (let i = 0; i < n; i++) {
    if (src[i] !== 'a') continue
    if (src.substr(i, 5) !== 'agent') continue
    if (src[i + 5] !== '(') continue
    if (!mask[i]) continue // inside string/comment
    const prev = src[i - 1]
    if (isIdentChar(prev) || prev === '.') continue // myagent( / .agent( / _agent(
    const openParen = i + 5
    const closeParen = match.get(openParen)
    if (closeParen === undefined) continue
    // already wrapped? look back ~40 code chars for `agentRetry(()=>`
    const look = src.slice(Math.max(0, i - 40), i)
    if (/agentRetry\s*\(\s*\(\s*\)\s*=>\s*$/.test(look)) continue
    sites.push({ callStart: i, openParen, closeParen })
  }
  return sites
}

function injectHelper(src, helperBlock) {
  if (src.includes('function agentRetry(')) return src // already present
  const marker = '// --- END inlined arg_guard ---'
  let idx = src.indexOf(marker)
  if (idx !== -1) {
    idx += marker.length
    return src.slice(0, idx) + '\n\n' + helperBlock + src.slice(idx)
  }
  // Fallback: after the `export const meta = { ... }` block (brace match on code).
  const metaKw = src.search(/export\s+const\s+meta\s*=\s*\{/)
  if (metaKw !== -1) {
    const mask = buildCodeMask(src)
    let depth = 0, j = metaKw, started = false
    for (; j < src.length; j++) {
      if (!mask[j]) continue
      if (src[j] === '{') { depth++; started = true }
      else if (src[j] === '}') { depth--; if (started && depth === 0) break }
    }
    const after = src.indexOf('\n', j) + 1
    return src.slice(0, after) + '\n' + helperBlock + '\n' + src.slice(after)
  }
  // Last resort: top of file (after leading comment).
  return helperBlock + '\n\n' + src
}

function transformFile(file, helperBlock) {
  let src = fs.readFileSync(file, 'utf8')
  const beforeHas = src.includes('function agentRetry(')
  const mask = buildCodeMask(src)
  const sites = findAgentSites(src, mask)
  if (!beforeHas && sites.length === 0) {
    // Still inject the helper so the scaffolding is present (default scaffolding).
    src = injectHelper(src, helperBlock)
    fs.writeFileSync(file, src)
    return { injected: !beforeHas, wrapped: 0, note: 'helper only (no agent() calls found in code)' }
  }
  // Apply wraps from last to first (indices stable on original src).
  let out = src
  for (let k = sites.length - 1; k >= 0; k--) {
    const s = sites[k]
    out = out.slice(0, s.closeParen + 1) + ', { retries: 5 })' + out.slice(s.closeParen + 1)
    out = out.slice(0, s.callStart) + 'agentRetry(() => ' + out.slice(s.callStart)
  }
  out = injectHelper(out, helperBlock)
  fs.writeFileSync(file, out)
  return { injected: !beforeHas, wrapped: sites.length }
}

// Splice the inlined helper block: replace the text from the BEGIN sentinel
// through the END sentinel (inclusive) with the current canonical helperBlock.
// Returns { out, refreshed }. refreshed=false (out unchanged) when the file has
// no sentinel-bounded block — caller should skip+warn rather than corrupt a
// hand-edited region.
const BEGIN_SENTINEL = '// --- BEGIN inlined agent-retry scaffolding'
const END_SENTINEL = '// --- END inlined agent-retry scaffolding ---'
function refreshHelper(src, helperBlock) {
  const begin = src.indexOf(BEGIN_SENTINEL)
  if (begin === -1) return { out: src, refreshed: false }
  const end = src.indexOf(END_SENTINEL, begin)
  if (end === -1) return { out: src, refreshed: false } // malformed — bail, do not write
  const spliceEnd = end + END_SENTINEL.length
  return { out: src.slice(0, begin) + helperBlock + src.slice(spliceEnd), refreshed: true }
}

function refreshFile(file, helperBlock) {
  const src = fs.readFileSync(file, 'utf8')
  const { out, refreshed } = refreshHelper(src, helperBlock)
  if (!refreshed) return { refreshed: false, note: 'no sentinel-bounded block found — skipped' }
  if (out === src) return { refreshed: true, note: 'already current' }
  fs.writeFileSync(file, out)
  return { refreshed: true }
}

// --- --allow-null mode -------------------------------------------------------
// With the fail-safe default (throw on terminal null), an agentRetry whose result
// the workflow INTENTIONALLY treats as nullable must opt out via { allowNull: true }
// or the round aborts instead of degrading. This migrates such call sites: for
// every DIRECT assignment `X = await agentRetry(...)` whose variable X is later
// used in a null-tolerant way (`X && X.field`, `X?.field`, `if (X …)` / `if (!X …)`),
// add `allowNull: true` to that call's opts. Results consumed inside `parallel()`
// thunks need NO migration — parallel() turns a thrown thunk into a null slot, so
// the existing `(x && x.field) || fallback` degrade already works.
function buildPairMaps(src, mask) {
  const stack = []
  const openToClose = new Map()
  const closeToOpen = new Map()
  for (let i = 0; i < src.length; i++) {
    if (!mask[i]) continue
    const c = src[i]
    if (c === '(' || c === '{') stack.push({ ch: c, i })
    else if (c === ')' || c === '}') {
      const o = stack.pop()
      if (o && ((c === ')' && o.ch === '(') || (c === '}' && o.ch === '{'))) {
        openToClose.set(o.i, i); closeToOpen.set(i, o.i)
      }
    }
  }
  return { openToClose, closeToOpen }
}

// True iff `varName` is referenced in a null-tolerant way in CODE (mask=1) at or
// after `afterIndex`. Scanning only code regions avoids false positives from a
// variable name that happens to appear in a comment or a prompt-string example —
// a required variable dereferenced directly (`X.field`) must NOT be opted out just
// because a nearby comment/string mentions `X?.` or `if (X)`.
function usedNullTolerant(src, mask, afterIndex, varName) {
  const re = new RegExp(
    `\\b${varName}\\b\\s*&&\\s*\\b${varName}\\b`
    + `|\\b${varName}\\b\\?\\.`
    + `|\\bif\\s*\\(\\s*!?\\s*\\b${varName}\\b\\s*[)&|]`,
    'g',
  )
  let m
  while ((m = re.exec(src)) !== null) {
    if (m.index <= afterIndex) continue          // only usages AFTER the assignment/call
    if (mask[m.index]) return true               // the reference is in code, not comment/string
  }
  return false
}

function findAgentRetryAssignSites(src, mask, maps) {
  const sites = []
  for (let i = 0; i < src.length; i++) {
    if (src.substr(i, 10) !== 'agentRetry') continue
    if (src[i + 10] !== '(') continue
    if (/[A-Za-z0-9_$]/.test(src[i - 1] || '') || src[i - 1] === '.') continue
    if (!mask[i]) continue
    const openParen = i + 10
    const closeParen = maps.openToClose.get(openParen)
    if (closeParen === undefined) continue
    // Only DIRECT assignments: 'X = await agentRetry(' or 'X = agentRetry('.
    const look = src.slice(Math.max(0, i - 80), i)
    const m = look.match(/([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:await\s+)?$/)
    if (!m) continue // inside `() =>` / returned / etc — not a named local; skip
    const varName = m[1]
    // The opts object is the last {...} before closeParen.
    let j = closeParen - 1
    while (j > openParen && /\s/.test(src[j])) j--
    if (src[j] !== '}') continue // no opts object literal — can't inject safely
    const optsCloseBrace = j
    const optsOpenBrace = maps.closeToOpen.get(optsCloseBrace)
    if (optsOpenBrace === undefined) continue
    sites.push({ callStart: i, varName, optsOpenBrace, optsCloseBrace, closeParen })
  }
  return sites
}

function migrateAllowNull(file) {
  let src = fs.readFileSync(file, 'utf8')
  const mask = buildCodeMask(src)
  const maps = buildPairMaps(src, mask)
  const sites = findAgentRetryAssignSites(src, mask, maps)
  const migrated = []
  for (let k = sites.length - 1; k >= 0; k--) { // last→first: indices stay valid
    const s = sites[k]
    if (!usedNullTolerant(src, mask, s.closeParen, s.varName)) continue
    const optsSpan = src.slice(s.optsOpenBrace, s.optsCloseBrace + 1)
    if (/\ballowNull\b/.test(optsSpan)) continue // idempotent
    // Trim back over trailing whitespace inside the opts so we insert right after
    // the last property → `{ retries: 5, allowNull: true }` (not `5 , …`).
    let insAt = s.optsCloseBrace
    while (insAt > s.optsOpenBrace + 1 && /\s/.test(src[insAt - 1])) insAt--
    const contentPresent = insAt > s.optsOpenBrace + 1
    const middle = contentPresent ? ', allowNull: true ' : 'allowNull: true '
    src = src.slice(0, insAt) + middle + src.slice(s.optsCloseBrace)
    migrated.push(s.varName)
  }
  if (migrated.length > 0) fs.writeFileSync(file, src)
  return { migrated }
}

function listAllWorkflows() {
  // Only genuine workflow directories: top-level capitalized/alphabetic method
  // dirs. Skip underscore-prefixed substrate/meta dirs AND `scripts` (which holds
  // repo tooling like this codemod, not runnable workflows).
  const skip = new Set(['_tools', '_meta', '_templates', '_substrate', 'scripts', 'node_modules', 'badges', 'docs', 'outputs', 'paper', 'ppt'])
  const out = []
  for (const entry of fs.readdirSync(REPO)) {
    if (skip.has(entry) || entry.startsWith('.')) continue
    const dir = path.join(REPO, entry)
    if (!fs.statSync(dir).isDirectory()) continue
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith('.js')) continue
      out.push(path.join(dir, f))
    }
  }
  // Only files that actually call agent(
  return out.filter((f) => {
    const s = fs.readFileSync(f, 'utf8')
    return /\bagent\s*\(/.test(s)
  })
}

function main() {
  const argv = process.argv.slice(2)
  const refresh = argv.includes('--refresh')
  const allowNull = argv.includes('--allow-null')
  const files = argv.includes('--all') ? listAllWorkflows() : argv.filter((a) => !a.startsWith('--'))
  if (files.length === 0) {
    console.error('usage: add-agent-retry-scaffolding.js [--all [--refresh | --allow-null] | file1.js ...]')
    process.exit(2)
  }
  const helperBlock = readHelperBlock()
  let totalWrapped = 0
  let totalRefreshed = 0
  let totalMigrated = 0
  for (const f of files) {
    const rel = path.relative(REPO, f)
    if (refresh) {
      const res = refreshFile(f, helperBlock)
      if (res.refreshed) totalRefreshed++
      console.log(`${rel}: refreshed=${res.refreshed}${res.note ? ' (' + res.note + ')' : ''}`)
    } else if (allowNull) {
      const res = migrateAllowNull(f)
      totalMigrated += res.migrated.length
      console.log(`${rel}: migrated=${res.migrated.length}${res.migrated.length ? ' (' + res.migrated.join(', ') + ')' : ''}`)
    } else {
      const res = transformFile(f, helperBlock)
      totalWrapped += res.wrapped
      console.log(`${rel}: injected=${res.injected} wrapped=${res.wrapped}${res.note ? ' (' + res.note + ')' : ''}`)
    }
  }
  if (refresh) console.log(`TOTAL refreshed ${totalRefreshed}/${files.length} file(s).`)
  else if (allowNull) console.log(`TOTAL migrated ${totalMigrated} allowNull site(s) across ${files.length} file(s).`)
  else console.log(`TOTAL wrapped ${totalWrapped} agent() call(s) across ${files.length} file(s).`)
}

// Exported for reuse by scripts/check-agent-retry-guards.js (the enforcement
// linter) and its node:test wrapper. Run as a script (`node add-agent-retry-scaffolding.js`)
// to apply transforms; require it for the analysis primitives.
module.exports = {
  buildCodeMask,
  findAgentSites,
  buildPairMaps,
  findAgentRetryAssignSites,
  usedNullTolerant,
  listAllWorkflows,
}

if (require.main === module) main()
