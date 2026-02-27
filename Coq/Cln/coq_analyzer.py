#!/usr/bin/env python3
"""
coq_analyzer.py — Coq Dependency Analyzer

Parses .v files from a Coq project, builds a full dependency graph,
and generates an interactive HTML page with:
  - Clickable dependency trees for every theorem/lemma/definition
  - Reverse dependency view ("what breaks if I change this?")
  - Admitted/axiom audit trail (trust chain)
  - Proof debt dashboard with stats
  - Unused definition detection
  - Search and filter by kind, status, file
  - Color coding:
        green=proved, yellow=admitted, blue=definition,
        purple=axiom/parameter, red=depends-on-admitted
  - JSON export

Usage:
    python coq_analyzer.py /path/to/coq/project [-o output.html] [--json]
"""

import os
import re
import json
import sys
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from html import escape
import textwrap


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  DATA STRUCTURES
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@dataclass
class CoqItem:
    name: str
    qualified_name: str       # Module.Section.name
    kind: str                 # lemma, theorem, definition, fixpoint, inductive, axiom, etc.
    statement: str            # type signature / statement (no proof body)
    status: str               # proved, admitted, defined, assumed, aborted
    file: str                 # relative path
    line: int                 # 1-based line number
    dependencies: list = field(default_factory=list)
    dependents: list = field(default_factory=list)
    tainted: bool = False     # depends (transitively) on an Admitted or Axiom


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  COQ PARSER
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def strip_comments(text: str) -> str:
    """Remove nested Coq comments (* ... *), preserving line structure."""
    result = []
    i = 0
    depth = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] == '(' and text[i + 1] == '*':
            depth += 1
            i += 2
        elif i + 1 < n and text[i] == '*' and text[i + 1] == ')':
            depth = max(0, depth - 1)
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            result.append('\n' if text[i] == '\n' else ' ')
            i += 1
    return ''.join(result)


def split_sentences(text: str) -> List[Tuple[str, int]]:
    """
    Split Coq text into (sentence, line_number) pairs.
    A sentence ends with '.' followed by whitespace or EOF,
    but NOT inside strings or qualified names like Nat.add.
    """
    sentences = []
    current = []
    current_start_line = 1
    line = 1
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        ch = text[i]

        if ch == '\n':
            line += 1

        if ch == '"':
            in_string = not in_string
            current.append(ch)
            i += 1
            continue

        if in_string:
            current.append(ch)
            i += 1
            continue

        if ch == '.' and (i + 1 >= n or text[i + 1] in ' \t\n\r'):
            # Check it's not a number like 0.5 (rare in Coq but possible)
            current.append('.')
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append((sentence, current_start_line))
            current = []
            current_start_line = line + (1 if i + 1 < n and text[i + 1] == '\n' else 0)
            i += 1
        else:
            if not current:
                current_start_line = line
            current.append(ch)
            i += 1

    remaining = ''.join(current).strip()
    if remaining:
        sentences.append((remaining, current_start_line))

    return sentences


# Coq command patterns
PROVABLE_KINDS = {
    'Lemma', 'Theorem', 'Corollary', 'Proposition', 'Fact',
    'Remark', 'Example', 'Property',
}

DEFINITION_KINDS = {
    'Definition', 'Fixpoint', 'CoFixpoint', 'Let', 'Function',
    'Program Definition', 'Program Fixpoint',
}

TYPE_KINDS = {
    'Inductive', 'CoInductive', 'Record', 'Structure',
    'Class', 'Variant',
}

ASSUMPTION_KINDS = {
    'Axiom', 'Parameter', 'Hypothesis', 'Variable',
    'Conjecture', 'Context', 'Declare Assumption',
}

INSTANCE_KINDS = {
    'Instance', 'Global Instance', 'Local Instance',
    'Program Instance',
}

ALL_KEYWORD_KINDS = (
    PROVABLE_KINDS | DEFINITION_KINDS | TYPE_KINDS |
    ASSUMPTION_KINDS | INSTANCE_KINDS | {'Ltac', 'Notation', 'Tactic Notation'}
)

# Big regex to match the start of Coq vernacular commands
_kind_alts = '|'.join(
    re.escape(k) for k in sorted(ALL_KEYWORD_KINDS, key=lambda x: -len(x))
)
# Match: <Kind> <name> with possible attributes prefix #[...]
COMMAND_RE = re.compile(
    r'^(?:\#\[[^\]]*\]\s*)?'          # optional attributes like #[global]
    r'(?:' + _kind_alts + r')'        # the keyword
    r'\s+'
    r'(\w[\w\']*)'                     # the name (capture group 1)
    r'([\s\S]*)',                       # the rest (capture group 2)
    re.MULTILINE
)

# Separate regex to extract just the kind keyword
KIND_RE = re.compile(
    r'^(?:\#\[[^\]]*\]\s*)?'
    r'(' + _kind_alts + r')'
    r'\s+',
    re.MULTILINE
)

# Proof terminators
PROOF_START_RE = re.compile(r'^Proof\b', re.MULTILINE)
PROOF_END_KEYWORDS = {'Qed', 'Defined', 'Admitted', 'Abort'}


def extract_statement(rest: str, kind: str) -> str:
    """Extract the type statement from the rest of a command, stripping proof body."""
    # For provable things, statement is everything before `:=` or `Proof` or just the `:` part
    # For definitions, statement is everything (type + body)

    if kind in PROVABLE_KINDS or kind in INSTANCE_KINDS:
        # Statement ends at first occurrence of := or Proof
        # The type is the part after the name up to the first '.'
        # Actually, for "Lemma foo (x : nat) : P x." the whole thing is the statement
        return rest.strip()
    else:
        return rest.strip()


def parse_coq_file(filepath: str, base_dir: str) -> Tuple[List[CoqItem], List[str]]:
    """
    Parse a single .v file and return (items, imports).
    """
    rel_path = os.path.relpath(filepath, base_dir)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_text = f.read()

    text = strip_comments(raw_text)
    sentences = split_sentences(text)

    items = []
    imports = []
    module_stack = []    # track Module / Section for qualified names

    # We need to handle proof blocks:
    # After a provable command, sentences until Qed/Admitted/Defined/Abort are proof
    in_proof = False
    current_item: Optional[CoqItem] = None

    for sentence, line_no in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue

        # Track module/section scope
        m_open = re.match(r'^(?:Module|Section)\s+(\w+)', stripped)
        if m_open:
            module_stack.append(m_open.group(1))
            continue

        m_close = re.match(r'^End\s+(\w+)', stripped)
        if m_close:
            if module_stack and module_stack[-1] == m_close.group(1):
                module_stack.pop()
            continue

        # Track imports
        m_import = re.match(
            r'^(?:From\s+\S+\s+)?Require\s+(?:Import|Export)\s+(.*)',
            stripped, re.DOTALL
        )
        if m_import:
            import_text = m_import.group(1).rstrip('.')
            for mod in re.split(r'\s+', import_text):
                mod = mod.strip().rstrip('.')
                if mod:
                    imports.append(mod)
            continue

        # Check if this is a proof terminator
        first_word = stripped.rstrip('.').strip()

        if first_word in PROOF_END_KEYWORDS:
            if current_item:
                if first_word == 'Admitted':
                    current_item.status = 'admitted'
                elif first_word == 'Abort':
                    current_item.status = 'aborted'
                elif first_word == 'Defined':
                    current_item.status = 'defined'
                else:
                    current_item.status = 'proved'
                current_item = None
            in_proof = False
            continue

        # Skip proof internals
        if re.match(r'^Proof\s*\.$|^Proof\s+(using|with)\b', stripped):
            in_proof = True
            continue

        if in_proof:
            continue

        # Try to parse a vernacular command
        m_kind = KIND_RE.match(stripped)
        if not m_kind:
            continue

        kind = m_kind.group(1)
        m_cmd = COMMAND_RE.match(stripped)
        if not m_cmd:
            continue

        name = m_cmd.group(1)
        rest = m_cmd.group(2)

        # Build qualified name
        prefix = '.'.join(module_stack)
        qualified = f"{prefix}.{name}" if prefix else name

        statement = extract_statement(rest, kind)

        # Determine initial status
        if kind in ASSUMPTION_KINDS:
            status = 'assumed'
        elif kind in PROVABLE_KINDS or kind in INSTANCE_KINDS:
            # Will be updated when we find Qed/Admitted/etc.
            status = 'proving'  # temporary
        elif ':=' in rest:
            status = 'defined'
        else:
            status = 'defined'

        item = CoqItem(
            name=name,
            qualified_name=qualified,
            kind=kind.lower(),
            statement=f"{kind} {name} {statement}",
            status=status,
            file=rel_path,
            line=line_no,
        )
        items.append(item)

        # If this is a provable item without an inline := body, expect proof
        if kind in PROVABLE_KINDS and ':=' not in rest:
            current_item = item
            in_proof = False  # will become true when we see "Proof."
        elif kind in INSTANCE_KINDS and ':=' not in rest:
            current_item = item
            in_proof = False

    # Fix any items still marked as 'proving' (might have inline proofs or be malformed)
    for item in items:
        if item.status == 'proving':
            item.status = 'proved'  # assume proved if we never found terminator

    return items, imports


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  DEPENDENCY RESOLUTION
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def build_identifier_set(items: List[CoqItem]) -> Dict[str, CoqItem]:
    """Build mapping from name -> CoqItem. Handles both bare and qualified names."""
    name_map: Dict[str, CoqItem] = {}
    for item in items:
        # Prefer qualified name, but also register bare name
        # (if conflict, last one wins for bare name — qualified is unique)
        name_map[item.qualified_name] = item
        if item.name not in name_map:
            name_map[item.name] = item
    return name_map


def tokenize_for_deps(text: str) -> Set[str]:
    """Extract potential identifier tokens from a Coq statement."""
    # Remove string literals
    text = re.sub(r'"[^"]*"', '', text)
    # Split on non-identifier characters (keep . for qualified names)
    tokens = re.findall(r'\b[A-Za-z_][\w\']*(?:\.[A-Za-z_][\w\']*)*\b', text)
    return set(tokens)


def resolve_dependencies(items: List[CoqItem], name_map: Dict[str, CoqItem]):
    """
    For each item, scan its statement for references to other known items.
    Also scans sub-tokens of qualified names.
    """
    all_names = set(name_map.keys())

    for item in items:
        tokens = tokenize_for_deps(item.statement)
        deps = set()

        for token in tokens:
            # Try the full token
            if token in all_names and token != item.name and token != item.qualified_name:
                target = name_map[token]
                if target.qualified_name != item.qualified_name:
                    deps.add(target.qualified_name)

            # Try the last component of qualified names like Foo.bar -> bar
            if '.' in token:
                parts = token.split('.')
                for part in parts:
                    if part in all_names and part != item.name:
                        target = name_map[part]
                        if target.qualified_name != item.qualified_name:
                            deps.add(target.qualified_name)

        item.dependencies = sorted(deps)

    # Build reverse dependency (dependents) lists
    dep_map = defaultdict(set)
    for item in items:
        for dep in item.dependencies:
            dep_map[dep].add(item.qualified_name)

    for item in items:
        item.dependents = sorted(dep_map.get(item.qualified_name, set()))


def compute_taint(items: List[CoqItem], name_map: Dict[str, CoqItem]):
    """
    Mark items as 'tainted' if they transitively depend on anything
    that is Admitted or Assumed (axiom/parameter).
    """
    tainted_names: Set[str] = set()

    # Seed: all admitted and assumed items are tainted
    for item in items:
        if item.status in ('admitted', 'assumed'):
            tainted_names.add(item.qualified_name)
            item.tainted = True

    # Propagate forward: BFS through dependents
    queue = list(tainted_names)
    visited = set(queue)

    while queue:
        current_name = queue.pop(0)
        current = name_map.get(current_name)
        if not current:
            continue
        for dep_name in current.dependents:
            if dep_name not in visited:
                visited.add(dep_name)
                dep_item = name_map.get(dep_name)
                if dep_item:
                    dep_item.tainted = True
                    queue.append(dep_name)


def find_unused(items: List[CoqItem]) -> List[CoqItem]:
    """Find items that have zero dependents (nothing uses them)."""
    return [item for item in items if not item.dependents]


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  ANALYSIS PIPELINE
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def analyze_project(root_dir: str) -> Tuple[List[CoqItem], Dict]:
    """Main analysis pipeline."""
    root = Path(root_dir).resolve()
    all_items: List[CoqItem] = []
    all_imports: Dict[str, List[str]] = {}  # file -> [imports]

    # Collect all .v files
    v_files = sorted(root.rglob('*.v'))
    if not v_files:
        print(f"No .v files found in {root_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(v_files)} Coq file(s)...", file=sys.stderr)

    for vf in v_files:
        try:
            items, imports = parse_coq_file(str(vf), str(root))
            all_items.extend(items)
            rel = os.path.relpath(str(vf), str(root))
            all_imports[rel] = imports
        except Exception as e:
            print(f"  Warning: failed to parse {vf}: {e}", file=sys.stderr)

    print(f"Found {len(all_items)} items. Resolving dependencies...", file=sys.stderr)

    # Build name map and resolve
    name_map = build_identifier_set(all_items)
    resolve_dependencies(all_items, name_map)
    compute_taint(all_items, name_map)

    # Compute stats
    stats = {
        'total_items': len(all_items),
        'proved': sum(1 for i in all_items if i.status == 'proved'),
        'admitted': sum(1 for i in all_items if i.status == 'admitted'),
        'defined': sum(1 for i in all_items if i.status == 'defined'),
        'assumed': sum(1 for i in all_items if i.status == 'assumed'),
        'aborted': sum(1 for i in all_items if i.status == 'aborted'),
        'tainted': sum(1 for i in all_items if i.tainted),
        'unused': sum(1 for i in all_items if not i.dependents),
        'files': len(v_files),
        'by_kind': {},
        'imports': all_imports,
    }
    for item in all_items:
        stats['by_kind'][item.kind] = stats['by_kind'].get(item.kind, 0) + 1

    print(f"  {stats['proved']} proved, {stats['admitted']} admitted, "
          f"{stats['assumed']} assumed, {stats['tainted']} tainted",
          file=sys.stderr)

    return all_items, stats


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  HTML GENERATION
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def generate_html(items: List[CoqItem], stats: Dict, project_name: str) -> str:
    """Generate a self-contained interactive HTML page."""

    # Serialize items to JSON for the JS frontend
    items_json = []
    for item in items:
        items_json.append({
            'name': item.name,
            'qname': item.qualified_name,
            'kind': item.kind,
            'statement': item.statement,
            'status': item.status,
            'file': item.file,
            'line': item.line,
            'deps': item.dependencies,
            'rdeps': item.dependents,
            'tainted': item.tainted,
        })

    data_json = json.dumps({
        'items': items_json,
        'stats': stats,
        'project': project_name,
    }, indent=None, ensure_ascii=False)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Coq Analyzer — {escape(project_name)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
  --bg: #f8f9fa; --sidebar-bg: #1e1e2e; --sidebar-fg: #cdd6f4;
  --card: #fff; --border: #dee2e6; --text: #212529; --muted: #6c757d;
  --green: #d4edda; --green-border: #28a745;
  --yellow: #fff3cd; --yellow-border: #ffc107;
  --blue: #d1ecf1; --blue-border: #17a2b8;
  --purple: #e2d9f3; --purple-border: #6f42c1;
  --red: #f8d7da; --red-border: #dc3545;
  --gray: #e9ecef; --gray-border: #6c757d;
  --orange: #ffe0b2; --orange-border: #ff9800;
}}
body {{ display: flex; height: 100vh; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: var(--text); background: var(--bg); font-size: 14px; }}

/* ── Sidebar ── */
#sidebar {{ width: 300px; min-width: 260px; background: var(--sidebar-bg); color: var(--sidebar-fg); display: flex; flex-direction: column; overflow: hidden; }}
#sidebar h1 {{ padding: 16px; font-size: 16px; border-bottom: 1px solid #313244; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
#sidebar h1 span {{ color: #89b4fa; }}
#search-box {{ margin: 8px 12px; padding: 8px 10px; border-radius: 6px; border: 1px solid #45475a; background: #313244; color: var(--sidebar-fg); font-size: 13px; outline: none; }}
#search-box:focus {{ border-color: #89b4fa; }}
#filters {{ padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid #313244; }}
.filter-btn {{ padding: 3px 8px; border-radius: 12px; border: 1px solid #45475a; background: transparent; color: var(--sidebar-fg); font-size: 11px; cursor: pointer; }}
.filter-btn.active {{ background: #89b4fa; color: #1e1e2e; border-color: #89b4fa; }}
.filter-btn:hover {{ border-color: #89b4fa; }}
#item-list {{ flex: 1; overflow-y: auto; padding: 4px 0; }}
.item-entry {{ padding: 6px 12px; cursor: pointer; display: flex; align-items: center; gap: 6px; border-left: 3px solid transparent; font-size: 13px; }}
.item-entry:hover {{ background: #313244; }}
.item-entry.selected {{ background: #313244; border-left-color: #89b4fa; }}
.item-badge {{ font-size: 9px; padding: 1px 5px; border-radius: 8px; font-weight: 600; text-transform: uppercase; white-space: nowrap; }}
.badge-proved {{ background: #a6e3a1; color: #1e1e2e; }}
.badge-admitted {{ background: #f9e2af; color: #1e1e2e; }}
.badge-defined {{ background: #89dceb; color: #1e1e2e; }}
.badge-assumed {{ background: #cba6f7; color: #1e1e2e; }}
.badge-aborted {{ background: #f38ba8; color: #1e1e2e; }}
.item-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.item-kind {{ color: #585b70; font-size: 11px; margin-left: auto; white-space: nowrap; }}
.taint-marker {{ color: #f38ba8; font-weight: bold; margin-right: 2px; title: "depends on admitted/axiom"; }}

/* ── Main area ── */
#main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#dashboard {{ padding: 12px 20px; background: var(--card); border-bottom: 1px solid var(--border); display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
.stat-card {{ padding: 8px 14px; border-radius: 8px; text-align: center; min-width: 80px; }}
.stat-card .num {{ font-size: 22px; font-weight: 700; }}
.stat-card .label {{ font-size: 11px; color: var(--muted); }}
.stat-green {{ background: var(--green); }}
.stat-yellow {{ background: var(--yellow); }}
.stat-blue {{ background: var(--blue); }}
.stat-purple {{ background: var(--purple); }}
.stat-red {{ background: var(--red); }}
.stat-gray {{ background: var(--gray); }}

#content-area {{ flex: 1; display: flex; overflow: hidden; }}
#tree-panel {{ flex: 1; overflow-y: auto; padding: 20px; }}
#detail-panel {{ width: 400px; min-width: 300px; background: var(--card); border-left: 1px solid var(--border); overflow-y: auto; padding: 16px; display: none; }}
#detail-panel.visible {{ display: block; }}
#detail-panel h2 {{ font-size: 15px; margin-bottom: 8px; }}
#detail-panel .detail-meta {{ font-size: 12px; color: var(--muted); margin-bottom: 12px; }}
#detail-panel pre {{ background: #f1f3f5; border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 12px; white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono', 'Fira Code', monospace; margin-bottom: 12px; overflow-x: auto; }}
#detail-panel h3 {{ font-size: 13px; color: var(--muted); margin: 12px 0 6px; }}
.dep-link {{ color: #1971c2; cursor: pointer; text-decoration: none; font-size: 13px; display: block; padding: 2px 0; }}
.dep-link:hover {{ text-decoration: underline; }}
.dep-link .dep-badge {{ font-size: 9px; margin-left: 4px; }}

/* ── Tree view ── */
.tree-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }}
.tree-header h2 {{ font-size: 17px; }}
.view-toggle {{ padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); cursor: pointer; font-size: 12px; }}
.view-toggle.active {{ background: #1971c2; color: #fff; border-color: #1971c2; }}
.tree-node {{ margin-left: 20px; border-left: 2px solid #dee2e6; padding-left: 12px; margin-top: 4px; }}
.tree-root {{ margin-left: 0; border-left: none; padding-left: 0; }}
.tree-item {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 6px; margin: 2px 0; cursor: pointer; font-size: 13px; max-width: 100%; }}
.tree-item:hover {{ filter: brightness(0.95); }}
.tree-item .expand-icon {{ font-size: 10px; width: 14px; color: var(--muted); }}
.tree-item.status-proved {{ background: var(--green); border: 1px solid var(--green-border); }}
.tree-item.status-admitted {{ background: var(--yellow); border: 1px solid var(--yellow-border); }}
.tree-item.status-defined {{ background: var(--blue); border: 1px solid var(--blue-border); }}
.tree-item.status-assumed {{ background: var(--purple); border: 1px solid var(--purple-border); }}
.tree-item.status-aborted {{ background: var(--red); border: 1px solid var(--red-border); }}
.tree-item.tainted {{ box-shadow: inset 0 0 0 2px var(--red-border); }}
.tree-item .item-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.tree-item .kind-label {{ font-size: 10px; color: var(--muted); }}

#placeholder {{ color: var(--muted); text-align: center; margin-top: 80px; font-size: 15px; }}

/* ── Tab bar ── */
.tab-bar {{ display: flex; gap: 2px; margin-bottom: 12px; }}
.tab {{ padding: 6px 14px; cursor: pointer; border-radius: 6px 6px 0 0; background: var(--gray); font-size: 12px; }}
.tab.active {{ background: var(--card); font-weight: 600; box-shadow: 0 -1px 0 var(--border); }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #adb5bd; border-radius: 4px; }}
#sidebar ::-webkit-scrollbar-thumb {{ background: #45475a; }}

/* ── Export button ── */
.export-btn {{ margin-left: auto; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); cursor: pointer; font-size: 12px; }}
.export-btn:hover {{ background: var(--gray); }}
</style>
</head>
<body>

<div id="sidebar">
  <h1><span>◆</span> {escape(project_name)}</h1>
  <input id="search-box" type="text" placeholder="Search items..." autocomplete="off">
  <div id="filters"></div>
  <div id="item-list"></div>
</div>

<div id="main">
  <div id="dashboard"></div>
  <div id="content-area">
    <div id="tree-panel">
      <div id="placeholder">Select an item from the sidebar to explore its dependency tree.</div>
    </div>
    <div id="detail-panel"></div>
  </div>
</div>

<script>
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  DATA (injected by Python)
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
const DATA = {data_json};

const itemMap = {{}};
DATA.items.forEach(it => {{ itemMap[it.qname] = it; }});

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  STATE
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
let selectedItem = null;
let viewMode = 'deps'; // 'deps' or 'rdeps'
let activeFilters = new Set();
let searchQuery = '';
let expandedNodes = new Set();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  DASHBOARD
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function renderDashboard() {{
  const s = DATA.stats;
  const dash = document.getElementById('dashboard');
  dash.innerHTML = `
    <div class="stat-card stat-gray"><div class="num">${{s.total_items}}</div><div class="label">Total</div></div>
    <div class="stat-card stat-green"><div class="num">${{s.proved}}</div><div class="label">Proved</div></div>
    <div class="stat-card stat-yellow"><div class="num">${{s.admitted}}</div><div class="label">Admitted</div></div>
    <div class="stat-card stat-blue"><div class="num">${{s.defined}}</div><div class="label">Defined</div></div>
    <div class="stat-card stat-purple"><div class="num">${{s.assumed}}</div><div class="label">Axioms</div></div>
    <div class="stat-card stat-red"><div class="num">${{s.tainted}}</div><div class="label">Tainted</div></div>
    <div class="stat-card stat-gray"><div class="num">${{s.unused}}</div><div class="label">Unused</div></div>
    <button class="export-btn" onclick="exportJSON()">Export JSON</button>
  `;
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  FILTERS
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function renderFilters() {{
  const kinds = [...new Set(DATA.items.map(i => i.kind))].sort();
  const statuses = [...new Set(DATA.items.map(i => i.status))].sort();
  const allFilters = [
    ...statuses.map(s => ({{ key: 'status:' + s, label: s }})),
    {{ key: 'tainted', label: '⚠ tainted' }},
    {{ key: 'unused', label: 'unused' }},
  ];
  const el = document.getElementById('filters');
  el.innerHTML = allFilters.map(f =>
    `<button class="filter-btn" data-filter="${{f.key}}">${{f.label}}</button>`
  ).join('');
  el.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const key = btn.dataset.filter;
      if (activeFilters.has(key)) {{ activeFilters.delete(key); btn.classList.remove('active'); }}
      else {{ activeFilters.add(key); btn.classList.add('active'); }}
      renderItemList();
    }});
  }});
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  ITEM LIST (sidebar)
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function filterItems() {{
  return DATA.items.filter(item => {{
    // Search
    if (searchQuery && !item.name.toLowerCase().includes(searchQuery) &&
        !item.qname.toLowerCase().includes(searchQuery) &&
        !item.file.toLowerCase().includes(searchQuery)) return false;
    // Filters (OR within same category, AND across categories)
    if (activeFilters.size > 0) {{
      let pass = true;
      const statusFilters = [...activeFilters].filter(f => f.startsWith('status:'));
      const special = [...activeFilters].filter(f => !f.startsWith('status:'));
      if (statusFilters.length > 0) {{
        pass = statusFilters.some(f => 'status:' + item.status === f);
      }}
      if (special.includes('tainted') && !item.tainted) pass = false;
      if (special.includes('unused') && item.rdeps.length > 0) pass = false;
      return pass;
    }}
    return true;
  }});
}}

function renderItemList() {{
  const filtered = filterItems();
  const el = document.getElementById('item-list');
  el.innerHTML = filtered.map(item => `
    <div class="item-entry${{selectedItem && selectedItem.qname === item.qname ? ' selected' : ''}}"
         data-qname="${{item.qname}}">
      ${{item.tainted ? '<span class="taint-marker" title="Depends on admitted/axiom">⚠</span>' : ''}}
      <span class="item-badge badge-${{item.status}}">${{item.status.slice(0,3)}}</span>
      <span class="item-name">${{item.name}}</span>
      <span class="item-kind">${{item.kind}}</span>
    </div>
  `).join('');
  el.querySelectorAll('.item-entry').forEach(entry => {{
    entry.addEventListener('click', () => selectItem(entry.dataset.qname));
  }});
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  TREE RENDERING
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function selectItem(qname) {{
  selectedItem = itemMap[qname];
  expandedNodes.clear();
  expandedNodes.add(qname);
  renderItemList();
  renderTree();
  renderDetail();
}}

function renderTree() {{
  const panel = document.getElementById('tree-panel');
  if (!selectedItem) {{
    panel.innerHTML = '<div id="placeholder">Select an item from the sidebar.</div>';
    return;
  }}
  const item = selectedItem;
  const depLabel = viewMode === 'deps' ? 'Dependencies' : 'Reverse Dependencies';
  panel.innerHTML = `
    <div class="tree-header">
      <h2>${{item.name}} — ${{depLabel}}</h2>
      <button class="view-toggle ${{viewMode === 'deps' ? 'active' : ''}}" onclick="switchView('deps')">Uses ↓</button>
      <button class="view-toggle ${{viewMode === 'rdeps' ? 'active' : ''}}" onclick="switchView('rdeps')">Used by ↑</button>
    </div>
    <div class="tree-root">${{renderTreeNode(item.qname, new Set(), true)}}</div>
  `;
  panel.querySelectorAll('.tree-item').forEach(el => {{
    el.addEventListener('click', (e) => {{
      e.stopPropagation();
      const qn = el.dataset.qname;
      const action = el.dataset.action;
      if (action === 'toggle') {{
        if (expandedNodes.has(qn)) expandedNodes.delete(qn);
        else expandedNodes.add(qn);
        renderTree();
      }} else {{
        // clicking name opens detail
        renderDetailFor(qn);
      }}
    }});
  }});
}}

function renderTreeNode(qname, visited, isRoot) {{
  const item = itemMap[qname];
  if (!item) return `<div class="tree-item status-defined"><span class="item-label">${{qname}} (external)</span></div>`;

  const children = viewMode === 'deps' ? item.deps : item.rdeps;
  const hasChildren = children.length > 0;
  const isExpanded = expandedNodes.has(qname);
  const isCycle = visited.has(qname);
  const taintClass = item.tainted ? ' tainted' : '';

  let expandIcon = '';
  if (hasChildren && !isCycle) {{
    expandIcon = isExpanded ? '▼' : '▶';
  }}

  let html = `<div class="tree-item status-${{item.status}}${{taintClass}}" data-qname="${{qname}}" data-action="${{hasChildren && !isCycle ? 'toggle' : 'detail'}}">
    <span class="expand-icon">${{expandIcon}}</span>
    <span class="item-label">${{item.name}}</span>
    <span class="kind-label">${{item.kind}}</span>
  </div>`;

  if (isExpanded && hasChildren && !isCycle) {{
    const newVisited = new Set(visited);
    newVisited.add(qname);
    const childHtml = children.map(cqn => renderTreeNode(cqn, newVisited, false)).join('');
    html += `<div class="tree-node">${{childHtml}}</div>`;
  }}

  return html;
}}

function switchView(mode) {{
  viewMode = mode;
  expandedNodes.clear();
  if (selectedItem) expandedNodes.add(selectedItem.qname);
  renderTree();
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  DETAIL PANEL
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function renderDetail() {{
  if (!selectedItem) {{ document.getElementById('detail-panel').classList.remove('visible'); return; }}
  renderDetailFor(selectedItem.qname);
}}

function renderDetailFor(qname) {{
  const item = itemMap[qname];
  const panel = document.getElementById('detail-panel');
  if (!item) {{ panel.classList.remove('visible'); return; }}
  panel.classList.add('visible');

  const depsHtml = item.deps.length > 0
    ? item.deps.map(d => {{
        const di = itemMap[d];
        const badge = di ? `<span class="dep-badge badge-${{di.status}}">${{di.status.slice(0,3)}}</span>` : '';
        return `<a class="dep-link" data-qname="${{d}}">${{di ? di.name : d}} ${{badge}}</a>`;
      }}).join('')
    : '<span style="color:var(--muted);font-size:12px">None</span>';

  const rdepsHtml = item.rdeps.length > 0
    ? item.rdeps.map(d => {{
        const di = itemMap[d];
        const badge = di ? `<span class="dep-badge badge-${{di.status}}">${{di.status.slice(0,3)}}</span>` : '';
        return `<a class="dep-link" data-qname="${{d}}">${{di ? di.name : d}} ${{badge}}</a>`;
      }}).join('')
    : '<span style="color:var(--muted);font-size:12px">None (leaf / unused)</span>';

  panel.innerHTML = `
    <h2>${{item.name}}</h2>
    <div class="detail-meta">
      ${{item.kind}} · ${{item.status}}${{item.tainted ? ' · <span style="color:var(--red-border)">⚠ tainted</span>' : ''}}
      <br>${{item.file}}:${{item.line}}
    </div>
    <pre>${{escapeHtml(item.statement)}}</pre>
    <h3>Dependencies (${{item.deps.length}})</h3>
    ${{depsHtml}}
    <h3>Used by (${{item.rdeps.length}})</h3>
    ${{rdepsHtml}}
  `;

  panel.querySelectorAll('.dep-link').forEach(a => {{
    a.addEventListener('click', () => {{
      const qn = a.dataset.qname;
      selectItem(qn);
    }});
  }});
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  EXPORT
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
function exportJSON() {{
  const blob = new Blob([JSON.stringify(DATA, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '{escape(project_name)}_deps.json';
  a.click();
}}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  SEARCH
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
document.getElementById('search-box').addEventListener('input', (e) => {{
  searchQuery = e.target.value.toLowerCase();
  renderItemList();
}});

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  KEYBOARD SHORTCUTS
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
document.addEventListener('keydown', (e) => {{
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {{
    e.preventDefault();
    document.getElementById('search-box').focus();
  }}
  if (e.key === 'Escape') {{
    document.getElementById('search-box').blur();
    document.getElementById('search-box').value = '';
    searchQuery = '';
    renderItemList();
  }}
}});

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  INIT
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
renderDashboard();
renderFilters();
renderItemList();
</script>
</body>
</html>
"""
    return html


def export_json(items: List[CoqItem], stats: Dict) -> str:
    data = {
        'stats': stats,
        'items': [
            {
                'name': i.name,
                'qualified_name': i.qualified_name,
                'kind': i.kind,
                'statement': i.statement,
                'status': i.status,
                'file': i.file,
                'line': i.line,
                'dependencies': i.dependencies,
                'dependents': i.dependents,
                'tainted': i.tainted,
            }
            for i in items
        ]
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description='Coq Dependency Analyzer — generates interactive HTML dependency explorer'
    )
    parser.add_argument('project_dir', help='Root directory of the Coq project')
    parser.add_argument('-o', '--output', default='coq_deps.html',
                        help='Output HTML file (default: coq_deps.html)')
    parser.add_argument('--json', action='store_true',
                        help='Also export raw JSON data')
    parser.add_argument('--name', default=None,
                        help='Project name (default: directory name)')

    args = parser.parse_args()

    project_dir = args.project_dir
    if not os.path.isdir(project_dir):
        print(f"Error: {project_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    project_name = args.name or os.path.basename(os.path.abspath(project_dir))

    items, stats = analyze_project(project_dir)

    # Generate HTML
    html = generate_html(items, stats, project_name)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML written to {args.output}", file=sys.stderr)

    # Optional JSON export
    if args.json:
        json_path = args.output.rsplit('.', 1)[0] + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(export_json(items, stats))
        print(f"JSON written to {json_path}", file=sys.stderr)

    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()

"""
python coq_analyzer.py /path/to/your/cln/project -o cln_deps.html --json
    # then open cln_deps.html in a browser
    
python coq_analyzer.py E:\GLogic\Coq\Cln -o cln_deps.html --json
"""