#!/usr/bin/env python3
"""_substrate/backends/_gen_flag_parser.py — generate substrate run.sh flag parser.

Issue #25: each substrate's run.sh CLI flag contract is declared in a sibling
`flags.yaml` (single source of truth). This script emits the `while/case` parser
block from flags.yaml and (re)writes it into run.sh between sentinel comments:

    # BEGIN AUTO-GENERATED FLAG PARSER ...
    # flags.yaml sha256=<hex>
    # DO NOT EDIT BETWEEN SENTINELS ...
    while [ $# -gt 0 ]; do
      case "$1" in
        --artifact) ARTIFACT="${2:-}"; shift 2 ;;
        ...
        *) err_envelope "unknown arg: $1" 3 ;;
      esac
    done
    # END AUTO-GENERATED FLAG PARSER

The required-arg checks and any substrate-specific post-parse logic live OUTSIDE
the sentinels and remain hand-authored. The generated parser only assigns flags
to shell vars (or `shift 2` for parity/ignored flags with no dest).

Usage:
    python3 _substrate/backends/_gen_flag_parser.py --emit <backend>     # print block
    python3 _substrate/backends/_gen_flag_parser.py --write <backend>    # rewrite run.sh
    python3 _substrate/backends/_gen_flag_parser.py --check <backend>    # exit 0 if in sync

PyYAML is NOT required: flags.yaml uses a constrained block subset parsed here.
"""
import sys
import os
import hashlib
import re

BACKENDS_DIR = os.path.dirname(os.path.abspath(__file__))

# --- minimal block-YAML parser for the flags.yaml subset ---------------------
# Handles: `accepted_flags:`, `  - name: x` (list item), `    key: scalar`,
# `    key: null`, `    aliases:` followed by `    - item` lines. Scalars are
# raw strings (caller interprets null/true/false). No flow style, no nesting
# beyond one list-of-maps. Author flags.yaml in this exact style.
def parse_flags_yaml_robust(text):
    """Parse the constrained flags.yaml block subset into a list of flag dicts."""
    flags = []
    cur = None
    pending_list_key = None
    for raw in text.splitlines():
        line = raw.rstrip('\n')
        if not line.strip() or line.strip().startswith('#'):
            continue
        m_item = re.match(r'^\s*-\s+name:\s*(\S.*)$', line)
        if m_item:
            if cur is not None:
                flags.append(cur)
            cur = {'name': m_item.group(1).strip(), 'aliases': [], 'dest': None,
                   'type': None, 'required': False, 'default': None, 'source_arg': None}
            pending_list_key = None
            continue
        m_listval = re.match(r'^\s+-\s+(\S.*)$', line)
        if m_listval and cur is not None and pending_list_key is not None:
            cur[pending_list_key].append(m_listval.group(1).strip())
            continue
        if cur is None:
            continue
        m_kv = re.match(r'^\s+([a-z_]+):\s*(.*)$', line)
        if not m_kv:
            continue
        key = m_kv.group(1)
        val = m_kv.group(2).strip()
        pending_list_key = None
        if val == '':
            if key == 'aliases':
                cur['aliases'] = []
                pending_list_key = 'aliases'
            continue
        if val in ('null', '~'):
            cur[key] = None
        elif val == 'true':
            cur[key] = True
        elif val == 'false':
            cur[key] = False
        else:
            cur[key] = val
    if cur is not None:
        flags.append(cur)
    return flags


def load_flags(backend):
    path = os.path.join(BACKENDS_DIR, backend, 'flags.yaml')
    with open(path, 'r') as f:
        text = f.read()
    flags = parse_flags_yaml_robust(text)
    if not flags:
        raise SystemExit(f'{backend}: no flags parsed from {path} (check flags.yaml format)')
    return flags, text


def emit_block(backend, flags, flags_text):
    sha = hashlib.sha256(flags_text.encode('utf-8')).hexdigest()[:16]
    # build case entries
    entries = []
    for fl in flags:
        names = [fl['name']] + list(fl.get('aliases') or [])
        pattern = '|'.join('--' + n for n in names)
        dest = fl.get('dest')
        if dest:
            entries.append((pattern, f'{dest}="${{2:-}}"; shift 2'))
        else:
            entries.append((pattern, 'shift 2'))
    # align patterns for readability
    width = max(len(p) for p, _ in entries)
    lines = []
    lines.append('# BEGIN AUTO-GENERATED FLAG PARSER — regenerate from flags.yaml via _substrate/backends/_gen_flag_parser.py')
    lines.append(f'# flags.yaml sha256={sha}')
    lines.append('# DO NOT EDIT BETWEEN SENTINELS — edit flags.yaml and re-run: python3 _substrate/backends/_gen_flag_parser.py --write ' + backend)
    lines.append('while [ $# -gt 0 ]; do')
    lines.append('  case "$1" in')
    for p, body in entries:
        lines.append(f'    {p}){" " * (width - len(p) + 1)}{body} ;;')
    lines.append('    *) err_envelope "unknown arg: $1" 3 ;;')
    lines.append('  esac')
    lines.append('done')
    lines.append('# END AUTO-GENERATED FLAG PARSER')
    return '\n'.join(lines) + '\n'


BEGIN = '# BEGIN AUTO-GENERATED FLAG PARSER'
END = '# END AUTO-GENERATED FLAG PARSER'


def replace_or_insert(run_sh, block):
    """Replace existing sentinel block (or the hand-authored while...done loop) with block."""
    src = open(run_sh, 'r').read()
    if BEGIN in src and END in src:
        pre = src[:src.index(BEGIN)]
        post = src[src.index(END) + len(END):]
        # preserve trailing newline after END sentinel
        if post.startswith('\n'):
            post = post[1:]
        return pre + block + post
    # initial refactor: replace the hand-authored `while [ $# -gt 0 ]; do ... done` block
    m = re.search(r'( *)while \[ \$# -gt 0 \]; do\n.*?\n\1done\n', src, re.DOTALL)
    if not m:
        raise SystemExit(f'{run_sh}: cannot find while...done parser block to replace')
    return src[:m.start()] + block + src[m.end():]


def main(argv):
    if len(argv) != 2 or not argv[0].startswith('--'):
        raise SystemExit('usage: _gen_flag_parser.py --emit|--write|--check <backend>')
    mode, backend = argv
    flags, text = load_flags(backend)
    block = emit_block(backend, flags, text)
    run_sh = os.path.join(BACKENDS_DIR, backend, 'run.sh')
    if mode == '--emit':
        sys.stdout.write(block)
        return 0
    if mode == '--write':
        new = replace_or_insert(run_sh, block)
        open(run_sh, 'w').write(new)
        print(f'{backend}: run.sh parser regenerated')
        return 0
    if mode == '--check':
        src = open(run_sh, 'r').read()
        if BEGIN not in src or END not in src:
            print(f'{backend}: run.sh missing AUTO-GENERATED FLAG PARSER sentinels')
            return 1
        pre = src[src.index(BEGIN):src.index(END) + len(END)]
        if pre != block.rstrip('\n'):
            print(f'{backend}: run.sh parser out of sync with flags.yaml — run: python3 _substrate/backends/_gen_flag_parser.py --write {backend}')
            return 1
        print(f'{backend}: in sync')
        return 0
    raise SystemExit(f'unknown mode {mode}')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
