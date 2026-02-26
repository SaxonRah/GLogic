#!/usr/bin/env python3
"""
coq_glob_analyzer.py — Coq Dependency Analyzer (using .glob files)

Uses Coq's compiler-generated .glob files for precise dependency tracking,
combined with .v file parsing for statement text and proof status.

Requires: A compiled Coq project (run `make` or `coq_makefile` first so
.glob files exist alongside .v files).

Generates an interactive HTML page with:
  - Precise dependency trees from compiler data
  - Reverse dependency view ("what breaks if I change this?")
  - Admitted/axiom taint propagation
  - Proof debt dashboard
  - Unused definition detection
  - File-level dependency graph
  - External (stdlib/library) dependency tracking
  - Search, filter, keyboard shortcuts
  - JSON export

Usage:
    python coq_glob_analyzer.py /path/to/compiled/project [-o output.html] [--json]
    python coq_glob_analyzer.py /path/to/project --glob-dir _build/default [-o out.html]
"""

import os
import re
import json
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from html import escape


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  DATA STRUCTURES
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@dataclass
class GlobDef:
    """A definition entry from a .glob file."""
    name: str
    raw_name: str            # name as it appears in glob (may have :N suffix)
    kind: str                # prf, def, ind, ax, etc.
    byte_start: int
    byte_end: int
    file: str                # relative .v path
    logical_path: str        # Coq logical module path


@dataclass
class GlobRef:
    """A reference entry from a .glob file."""
    lib_path: str            # e.g. Cln.Cln_Full, Coq.Init.Datatypes
    name: str                # short name (e.g. bQ)
    raw_name: str            # raw name from glob
    kind: str                # def, var, ind, not, lib, etc.
    byte_start: int
    byte_end: int


@dataclass
class CoqItem:
    name: str
    qualified_name: str       # LogicalPath.name
    kind: str                 # human-readable kind
    kind_code: str            # raw glob kind code
    statement: str            # type signature (no proof body)
    status: str               # proved, admitted, defined, assumed, aborted
    file: str                 # relative .v path
    line: int                 # 1-based line number
    byte_start: int
    byte_end: int
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    ext_dependencies: List[str] = field(default_factory=list)
    tainted: bool = False
    taint_sources: List[str] = field(default_factory=list)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  GLOB KIND MAPPING
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

GLOB_KIND_DISPLAY = {
    'thm': 'theorem', 'lem': 'lemma', 'def': 'definition',
    'ax': 'axiom', 'ind': 'inductive', 'constr': 'constructor',
    'rec': 'fixpoint', 'corec': 'cofixpoint', 'not': 'notation',
    'sec': 'section', 'var': 'variable', 'inst': 'instance',
    'class': 'class', 'proj': 'projection', 'meth': 'method',
    'modtype': 'module type', 'mod': 'module', 'syndef': 'abbreviation',
    'scheme': 'scheme', 'prf': 'proof', 'binder': 'binder',
    'lib': 'library', 'prop': 'proposition', 'coe': 'coercion',
    'ex': 'example', 'morph': 'morphism',
}

# Kinds that are "provable" — they end with Qed/Admitted/Defined
PROVABLE_GLOB_KINDS = {'thm', 'lem', 'prf', 'prop', 'ex', 'morph'}

# Kinds that are assumptions
ASSUMED_GLOB_KINDS = {'ax'}

# Kinds we track in the dependency graph (these become CoqItems)
TRACKED_GLOB_KINDS = {
    'thm', 'lem', 'def', 'ax', 'ind', 'constr', 'rec', 'corec',
    'inst', 'class', 'proj', 'meth', 'prop', 'ex', 'morph',
    'scheme', 'syndef', 'prf',
}

# Kinds to skip entirely as definitions (but still track as references)
SKIP_DEF_KINDS = {'not', 'sec', 'mod', 'modtype', 'lib', 'binder', 'var'}


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  .GLOB FILE PARSER
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Actual format observed:
#   DIGEST <hex>
#   F<logical_path>
#   <kind> <start>:<end> <> <name>                    -- definition
#   R<start>:<end> <lib_path> <> <name> <kind>        -- reference
#   binder <start>:<end> <> <name>                    -- binder (skip)

# Definition regex:  kind start:end <> name
DEF_RE = re.compile(
    r'^(\w+)\s+(\d+):(\d+)\s+<>\s+(.+)$'
)

# Reference regex:  Rstart:end lib <> name kind
REF_RE = re.compile(
    r'^R(\d+):(\d+)\s+(\S+)\s+<>\s+(.+)\s+(\w+)$'
)


def parse_glob_file(glob_path: str, base_dir: str) -> Tuple[str, List[Tuple[GlobDef, List[GlobRef]]]]:
    """
    Parse a .glob file.
    Returns: (logical_path, [(def, [refs_in_scope]), ...])
    """
    rel_v_path = os.path.relpath(
        glob_path.replace('.glob', '.v'), base_dir
    )

    with open(glob_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    logical_path = ''
    current_def: Optional[GlobDef] = None
    current_refs: List[GlobRef] = []
    def_ref_pairs: List[Tuple[GlobDef, List[GlobRef]]] = []

    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue

        # DIGEST line
        if line.startswith('DIGEST'):
            continue

        # File/module declaration: F<logical_path>
        if line.startswith('F'):
            logical_path = line[1:].strip()
            continue

        # Reference line: R<start>:<end> <lib> <> <name> <kind>
        if line.startswith('R'):
            m = REF_RE.match(line)
            if m:
                raw_name = m.group(4).strip()
                # Clean the name: strip scope/notation markers
                name = raw_name
                # Some names are like ::Q_scope:x_'*'_x — these are notations, skip
                if name.startswith('::') or name.startswith("'"):
                    continue

                ref = GlobRef(
                    lib_path=m.group(3),
                    name=clean_name(name),
                    raw_name=raw_name,
                    kind=m.group(5),
                    byte_start=int(m.group(1)),
                    byte_end=int(m.group(2)),
                )
                current_refs.append(ref)
            continue

        # Definition line: <kind> <start>:<end> <> <name>
        m = DEF_RE.match(line)
        if m:
            kind = m.group(1)
            byte_start = int(m.group(2))
            byte_end = int(m.group(3))
            raw_name = m.group(4).strip()

            # Save previous def's refs
            if current_def is not None:
                def_ref_pairs.append((current_def, current_refs))
                current_refs = []

            gdef = GlobDef(
                name=clean_name(raw_name),
                raw_name=raw_name,
                kind=kind,
                byte_start=byte_start,
                byte_end=byte_end,
                file=rel_v_path,
                logical_path=logical_path,
            )
            current_def = gdef
            continue

    # Don't forget the last def
    if current_def is not None:
        def_ref_pairs.append((current_def, current_refs))

    return logical_path, def_ref_pairs


def clean_name(name: str) -> str:
    """Clean a glob name: strip binder suffixes like :1, :2, etc."""
    # Binder names look like "a:1", "b:2" — strip the :N suffix
    m = re.match(r'^(.+):(\d+)$', name)
    if m:
        return m.group(1)
    return name


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  .V FILE HELPERS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def strip_comments_preserve_positions(text: str) -> str:
    """Remove nested Coq comments, replacing with spaces to preserve byte positions."""
    result = list(text)
    i = 0
    depth = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] == '(' and text[i + 1] == '*':
            depth += 1
            result[i] = ' '
            result[i + 1] = ' '
            i += 2
        elif i + 1 < n and text[i] == '*' and text[i + 1] == ')':
            depth = max(0, depth - 1)
            result[i] = ' '
            result[i + 1] = ' '
            i += 2
        elif depth > 0:
            if text[i] != '\n':
                result[i] = ' '
            i += 1
        else:
            i += 1
    return ''.join(result)


def build_byte_to_line_map(raw_text: str) -> List[int]:
    """
    Build byte_offset -> 1-based line number mapping.
    Uses the raw bytes of the UTF-8 encoding.
    """
    raw_bytes = raw_text.encode('utf-8')
    line_map = []
    line = 1
    for b in raw_bytes:
        line_map.append(line)
        if b == ord('\n'):
            line += 1
    line_map.append(line)  # sentinel
    return line_map


def extract_statement_at(raw_bytes: bytes, byte_start: int, byte_end: int, kind_code: str) -> str:
    """
    Extract the Coq statement starting near byte_start.
    Scans backwards to find the command keyword, then forward to the
    terminating '.' (for the statement only, not the proof).
    """
    n = len(raw_bytes)

    # The glob byte_start typically points to the name. Scan backwards to
    # find the keyword (Lemma, Theorem, Definition, etc.)
    search_start = max(0, byte_start - 300)
    prefix = raw_bytes[search_start:byte_start].decode('utf-8', errors='replace')

    keywords = [
        'Theorem', 'Lemma', 'Corollary', 'Proposition', 'Fact',
        'Remark', 'Example', 'Property',
        'Definition', 'Fixpoint', 'CoFixpoint', 'Let', 'Function',
        'Inductive', 'CoInductive', 'Record', 'Structure', 'Class', 'Variant',
        'Axiom', 'Parameter', 'Hypothesis', 'Variable', 'Conjecture', 'Context',
        'Instance', 'Global Instance', 'Local Instance',
        'Program Definition', 'Program Fixpoint', 'Program Lemma',
    ]

    kw_byte = byte_start  # fallback
    for kw in sorted(keywords, key=len, reverse=True):
        idx = prefix.rfind(kw)
        if idx >= 0:
            # Convert char offset in prefix to byte offset
            candidate = search_start + len(prefix[:idx].encode('utf-8'))
            if candidate < kw_byte:
                kw_byte = candidate
            break  # take the closest keyword

    # Scan forward from byte_start to find the terminating '.'
    i = byte_start
    in_string = False
    paren_depth = 0

    while i < n:
        ch = raw_bytes[i:i+1]
        if ch == b'"':
            in_string = not in_string
        elif not in_string:
            if ch == b'(' and i + 1 < n and raw_bytes[i+1:i+2] == b'*':
                # skip comment
                depth = 1
                i += 2
                while i < n and depth > 0:
                    if raw_bytes[i:i+1] == b'(' and i + 1 < n and raw_bytes[i+1:i+2] == b'*':
                        depth += 1; i += 2
                    elif raw_bytes[i:i+1] == b'*' and i + 1 < n and raw_bytes[i+1:i+2] == b')':
                        depth -= 1; i += 2
                    else:
                        i += 1
                continue
            if ch == b'.':
                # Check if followed by whitespace or EOF
                if i + 1 >= n or raw_bytes[i+1:i+2] in (b' ', b'\t', b'\n', b'\r'):
                    i += 1
                    break
        i += 1

    stmt_bytes = raw_bytes[kw_byte:i]
    statement = stmt_bytes.decode('utf-8', errors='replace').strip()

    # Clean up whitespace
    statement = re.sub(r'\s+', ' ', statement)

    # Truncate very long statements
    if len(statement) > 2000:
        statement = statement[:2000] + ' ...'

    return statement


def extract_proof_status(raw_bytes: bytes, byte_start: int) -> str:
    """
    Determine proof status by scanning forward from the end of the
    statement to find Qed/Admitted/Defined/Abort.
    """
    n = len(raw_bytes)

    # First skip to end of the defining command (first '.' + whitespace)
    i = byte_start
    in_string = False
    while i < n:
        ch = raw_bytes[i:i+1]
        if ch == b'"':
            in_string = not in_string
        elif not in_string and ch == b'.':
            if i + 1 >= n or raw_bytes[i+1:i+2] in (b' ', b'\t', b'\n', b'\r'):
                i += 1
                break
        i += 1

    # Now scan sentences looking for proof terminators
    max_scan = min(i + 500000, n)
    while i < max_scan:
        # Skip whitespace
        while i < max_scan and raw_bytes[i:i+1] in (b' ', b'\t', b'\n', b'\r'):
            i += 1
        if i >= max_scan:
            break

        # Read next sentence
        sent_start = i
        in_str = False
        comment_depth = 0
        while i < max_scan:
            ch = raw_bytes[i:i+1]
            if ch == b'"' and comment_depth == 0:
                in_str = not in_str
            elif not in_str:
                if ch == b'(' and i + 1 < max_scan and raw_bytes[i+1:i+2] == b'*':
                    comment_depth += 1; i += 2; continue
                elif ch == b'*' and i + 1 < max_scan and raw_bytes[i+1:i+2] == b')' and comment_depth > 0:
                    comment_depth -= 1; i += 2; continue
                elif comment_depth == 0 and ch == b'.':
                    if i + 1 >= max_scan or raw_bytes[i+1:i+2] in (b' ', b'\t', b'\n', b'\r'):
                        i += 1
                        break
            i += 1

        sentence = raw_bytes[sent_start:i].decode('utf-8', errors='replace').strip()
        word = sentence.rstrip('.')

        if word == 'Qed':
            return 'proved'
        elif word == 'Admitted':
            return 'admitted'
        elif word == 'Defined':
            return 'defined'
        elif word == 'Abort':
            return 'aborted'

        # If we hit another top-level command, this wasn't a proof block
        if re.match(
            r'^(?:Theorem|Lemma|Corollary|Definition|Fixpoint|Inductive|'
            r'CoInductive|Record|Structure|Class|Axiom|Parameter|'
            r'Hypothesis|Variable|Instance|Module|Section|End|'
            r'From|Require|Import|Export|Notation|Ltac|Set|Unset)\b',
            word
        ):
            return 'defined'

    return 'defined'


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  PROJECT SCANNER & FILE FINDER
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def find_glob_files(root_dir: str, glob_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """Find pairs of (.glob, .v) files."""
    root = Path(root_dir).resolve()
    pairs = []

    if glob_dir:
        gdir = Path(glob_dir).resolve()
        for gf in sorted(gdir.rglob('*.glob')):
            vf = gf.with_suffix('.v')
            if vf.exists():
                pairs.append((str(gf), str(vf)))
            else:
                rel = gf.relative_to(gdir)
                vf2 = root / rel.with_suffix('.v')
                if vf2.exists():
                    pairs.append((str(gf), str(vf2)))
        if pairs:
            return pairs

    for vf in sorted(root.rglob('*.v')):
        gf = vf.with_suffix('.glob')
        if gf.exists():
            pairs.append((str(gf), str(vf)))

    if pairs:
        return pairs

    for build_dir in ['_build/default', '_build', '.coq-native']:
        bd = root / build_dir
        if bd.is_dir():
            for gf in sorted(bd.rglob('*.glob')):
                rel = gf.relative_to(bd)
                vf = root / rel.with_suffix('.v')
                if vf.exists():
                    pairs.append((str(gf), str(vf)))
            if pairs:
                return pairs

    return pairs


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  ANALYSIS PIPELINE
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def analyze_project(root_dir: str, glob_dir: Optional[str] = None) -> Tuple[List['CoqItem'], Dict]:
    root = Path(root_dir).resolve()
    pairs = find_glob_files(root_dir, glob_dir)

    if not pairs:
        print("Error: No .glob files found. Make sure the project is compiled.", file=sys.stderr)
        v_count = len(list(root.rglob('*.v')))
        if v_count > 0:
            print(f"  ({v_count} .v files found but no matching .glob files)", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} compiled file(s) with .glob data.", file=sys.stderr)

    all_items: List[CoqItem] = []
    name_map: Dict[str, CoqItem] = {}
    logical_to_file: Dict[str, str] = {}
    file_deps: Dict[str, Set[str]] = defaultdict(set)

    # Also count .v files that don't have .glob (for stats)
    all_v_files = sorted(root.rglob('*.v'))

    for glob_path, v_path in pairs:
        rel_v = os.path.relpath(v_path, str(root))

        # Read raw .v bytes
        with open(v_path, 'rb') as f:
            raw_bytes = f.read()

        # Also read as text for line map
        raw_text = raw_bytes.decode('utf-8', errors='replace')
        line_map = build_byte_to_line_map(raw_text)

        # Parse .glob
        try:
            logical_path, def_ref_pairs = parse_glob_file(glob_path, str(root))
        except Exception as e:
            print(f"  Warning: failed to parse {glob_path}: {e}", file=sys.stderr)
            continue

        logical_to_file[logical_path] = rel_v

        # Debug: report what we found
        tracked = [(d, r) for d, r in def_ref_pairs if d.kind not in SKIP_DEF_KINDS]
        print(f"  {rel_v}: {len(def_ref_pairs)} defs total, {len(tracked)} tracked", file=sys.stderr)

        for gdef, refs in def_ref_pairs:
            if gdef.kind in SKIP_DEF_KINDS:
                continue

            kind_display = GLOB_KIND_DISPLAY.get(gdef.kind, gdef.kind)
            qualified = f"{logical_path}.{gdef.name}" if logical_path else gdef.name

            # Skip items with weird names (notation artifacts etc.)
            if gdef.name.startswith('::') or gdef.name.startswith("'"):
                continue

            # Extract statement from .v
            try:
                statement = extract_statement_at(raw_bytes, gdef.byte_start, gdef.byte_end, gdef.kind)
            except Exception:
                statement = f"{kind_display} {gdef.name}"

            # Determine status
            if gdef.kind in ASSUMED_GLOB_KINDS:
                status = 'assumed'
            elif gdef.kind in PROVABLE_GLOB_KINDS:
                try:
                    status = extract_proof_status(raw_bytes, gdef.byte_start)
                except Exception:
                    status = 'proved'
            else:
                status = 'defined'

            # Line number
            line_no = 1
            if gdef.byte_start < len(line_map):
                line_no = line_map[gdef.byte_start]

            # Collect dependencies from references in this definition's scope
            dep_qnames = []
            ext_deps = []
            seen_deps = set()

            for ref in refs:
                # Skip notations, binders, variables, libs as dep targets
                if ref.kind in ('not', 'var', 'binder', 'lib'):
                    continue

                # Skip self-references
                ref_qname = f"{ref.lib_path}.{ref.name}"
                if ref_qname == qualified:
                    continue
                if ref_qname in seen_deps:
                    continue
                seen_deps.add(ref_qname)

                dep_qnames.append(ref_qname)

                # Track file-level deps
                if ref.lib_path and ref.lib_path != logical_path:
                    file_deps[rel_v].add(ref.lib_path)

            item = CoqItem(
                name=gdef.name,
                qualified_name=qualified,
                kind=kind_display,
                kind_code=gdef.kind,
                statement=statement,
                status=status,
                file=rel_v,
                line=line_no,
                byte_start=gdef.byte_start,
                byte_end=gdef.byte_end,
                dependencies=dep_qnames,
            )

            all_items.append(item)
            name_map[qualified] = item
            # Also register short name for resolution
            if gdef.name not in name_map:
                name_map[gdef.name] = item

    print(f"Parsed {len(all_items)} tracked items from .glob files.", file=sys.stderr)

    # ── Resolve dependencies ──
    project_modules = set(logical_to_file.keys())

    for item in all_items:
        resolved = []
        ext = []
        seen = set()

        for dep_qname in item.dependencies:
            # Direct lookup
            if dep_qname in name_map:
                target = name_map[dep_qname]
                if target.qualified_name not in seen and target.qualified_name != item.qualified_name:
                    seen.add(target.qualified_name)
                    resolved.append(target.qualified_name)
                continue

            # Try short name
            parts = dep_qname.rsplit('.', 1)
            short = parts[-1] if len(parts) > 1 else dep_qname
            lib_path = parts[0] if len(parts) > 1 else ''

            if short in name_map:
                target = name_map[short]
                if target.qualified_name not in seen and target.qualified_name != item.qualified_name:
                    seen.add(target.qualified_name)
                    resolved.append(target.qualified_name)
                continue

            # Is it from our project?
            is_project = any(dep_qname.startswith(m + '.') or lib_path == m for m in project_modules)
            if is_project:
                if dep_qname not in seen:
                    seen.add(dep_qname)
                    resolved.append(dep_qname)
            else:
                ext.append(dep_qname)

        item.dependencies = resolved
        item.ext_dependencies = ext

    # ── Build reverse deps ──
    rdep_map: Dict[str, Set[str]] = defaultdict(set)
    for item in all_items:
        for dep in item.dependencies:
            rdep_map[dep].add(item.qualified_name)
    for item in all_items:
        item.dependents = sorted(rdep_map.get(item.qualified_name, set()))

    # ── Taint propagation ──
    compute_taint(all_items, name_map)

    # ── File-level deps ──
    file_level_deps = {}
    for src, dep_mods in file_deps.items():
        file_level_deps[src] = sorted({
            logical_to_file[m]
            for m in dep_mods
            if m in logical_to_file and logical_to_file[m] != src
        })

    # ── Stats ──
    stats = {
        'total_items': len(all_items),
        'proved': sum(1 for i in all_items if i.status == 'proved'),
        'admitted': sum(1 for i in all_items if i.status == 'admitted'),
        'defined': sum(1 for i in all_items if i.status == 'defined'),
        'assumed': sum(1 for i in all_items if i.status == 'assumed'),
        'aborted': sum(1 for i in all_items if i.status == 'aborted'),
        'tainted': sum(1 for i in all_items if i.tainted),
        'unused': sum(1 for i in all_items if not i.dependents),
        'files': len(pairs),
        'total_v_files': len(all_v_files),
        'by_kind': {},
        'file_deps': file_level_deps,
        'module_map': dict(logical_to_file),
    }
    for item in all_items:
        stats['by_kind'][item.kind] = stats['by_kind'].get(item.kind, 0) + 1

    # Blast radius for admitted items
    admitted_items = [i for i in all_items if i.status == 'admitted']
    stats['admitted_blast'] = {}
    for ai in admitted_items:
        stats['admitted_blast'][ai.qualified_name] = compute_blast_radius(ai.qualified_name, name_map)

    print(f"  {stats['proved']} proved, {stats['admitted']} admitted, "
          f"{stats['assumed']} assumed, {stats['tainted']} tainted, "
          f"{stats['unused']} unused", file=sys.stderr)

    return all_items, stats


def compute_taint(items: List[CoqItem], name_map: Dict[str, CoqItem]):
    seeds = set()
    for item in items:
        if item.status in ('admitted', 'assumed'):
            item.tainted = True
            item.taint_sources = [item.qualified_name]
            seeds.add(item.qualified_name)

    queue = list(seeds)
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
                    dep_item.taint_sources = list(set(
                        dep_item.taint_sources + current.taint_sources
                    ))
                    queue.append(dep_name)


def compute_blast_radius(qname: str, name_map: Dict[str, CoqItem]) -> int:
    visited = set()
    queue = [qname]
    while queue:
        cur = queue.pop(0)
        ci = name_map.get(cur)
        if not ci:
            continue
        for d in ci.dependents:
            if d not in visited:
                visited.add(d)
                queue.append(d)
    return len(visited)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  HTML GENERATION
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def generate_html(items: List[CoqItem], stats: Dict, project_name: str) -> str:
    items_json = []
    for item in items:
        items_json.append({
            'name': item.name,
            'qname': item.qualified_name,
            'kind': item.kind,
            'kindCode': item.kind_code,
            'statement': item.statement,
            'status': item.status,
            'file': item.file,
            'line': item.line,
            'deps': item.dependencies,
            'rdeps': item.dependents,
            'extDeps': item.ext_dependencies,
            'tainted': item.tainted,
            'taintSources': item.taint_sources,
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
<title>Coq Glob Analyzer — {escape(project_name)}</title>
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
#sidebar {{ width: 300px; min-width: 260px; background: var(--sidebar-bg); color: var(--sidebar-fg); display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }}
#sidebar h1 {{ padding: 16px; font-size: 15px; border-bottom: 1px solid #313244; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
#sidebar h1 span {{ color: #89b4fa; }}
#sidebar h1 small {{ font-size: 11px; color: #585b70; display: block; margin-top: 2px; }}
#search-box {{ margin: 8px 12px; padding: 8px 10px; border-radius: 6px; border: 1px solid #45475a; background: #313244; color: var(--sidebar-fg); font-size: 13px; outline: none; }}
#search-box:focus {{ border-color: #89b4fa; }}
#filters {{ padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid #313244; }}
.fbtn {{ padding: 3px 8px; border-radius: 12px; border: 1px solid #45475a; background: transparent; color: var(--sidebar-fg); font-size: 11px; cursor: pointer; user-select: none; }}
.fbtn.active {{ background: #89b4fa; color: #1e1e2e; border-color: #89b4fa; }}
.fbtn:hover {{ border-color: #89b4fa; }}
#item-count {{ padding: 4px 12px; font-size: 11px; color: #585b70; border-bottom: 1px solid #313244; }}
#item-list {{ flex: 1; overflow-y: auto; padding: 4px 0; }}
.ie {{ padding: 6px 12px; cursor: pointer; display: flex; align-items: center; gap: 6px; border-left: 3px solid transparent; font-size: 13px; }}
.ie:hover {{ background: #313244; }}
.ie.sel {{ background: #313244; border-left-color: #89b4fa; }}
.badge {{ font-size: 9px; padding: 1px 5px; border-radius: 8px; font-weight: 600; text-transform: uppercase; white-space: nowrap; }}
.b-proved {{ background: #a6e3a1; color: #1e1e2e; }}
.b-admitted {{ background: #f9e2af; color: #1e1e2e; }}
.b-defined {{ background: #89dceb; color: #1e1e2e; }}
.b-assumed {{ background: #cba6f7; color: #1e1e2e; }}
.b-aborted {{ background: #f38ba8; color: #1e1e2e; }}
.ie-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.ie-kind {{ color: #585b70; font-size: 11px; margin-left: auto; white-space: nowrap; }}
.tmark {{ color: #f38ba8; font-weight: bold; margin-right: 2px; }}
#main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#dashboard {{ padding: 12px 20px; background: var(--card); border-bottom: 1px solid var(--border); display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
.sc {{ padding: 8px 14px; border-radius: 8px; text-align: center; min-width: 72px; }}
.sc .num {{ font-size: 20px; font-weight: 700; }}
.sc .lbl {{ font-size: 11px; color: var(--muted); }}
.sc-green {{ background: var(--green); }} .sc-yellow {{ background: var(--yellow); }}
.sc-blue {{ background: var(--blue); }} .sc-purple {{ background: var(--purple); }}
.sc-red {{ background: var(--red); }} .sc-gray {{ background: var(--gray); }}
.sc-orange {{ background: var(--orange); }}
#tab-bar {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); background: var(--card); padding: 0 20px; }}
.tab {{ padding: 8px 16px; cursor: pointer; font-size: 13px; border-bottom: 2px solid transparent; color: var(--muted); }}
.tab:hover {{ color: var(--text); }}
.tab.active {{ color: #1971c2; border-bottom-color: #1971c2; font-weight: 600; }}
#content-area {{ flex: 1; display: flex; overflow: hidden; }}
.panel {{ flex: 1; overflow-y: auto; padding: 20px; display: none; }}
.panel.visible {{ display: block; }}
#detail-panel {{ width: 400px; min-width: 300px; background: var(--card); border-left: 1px solid var(--border); overflow-y: auto; padding: 16px; display: none; flex-shrink: 0; }}
#detail-panel.visible {{ display: block; }}
#detail-panel h2 {{ font-size: 15px; margin-bottom: 8px; word-break: break-all; }}
#detail-panel .meta {{ font-size: 12px; color: var(--muted); margin-bottom: 12px; }}
#detail-panel pre {{ background: #f1f3f5; border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 12px; white-space: pre-wrap; word-break: break-word; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; margin-bottom: 12px; overflow-x: auto; max-height: 300px; }}
#detail-panel h3 {{ font-size: 13px; color: var(--muted); margin: 14px 0 6px; }}
.dep-link {{ color: #1971c2; cursor: pointer; text-decoration: none; font-size: 13px; display: block; padding: 2px 0; }}
.dep-link:hover {{ text-decoration: underline; }}
.dep-link .db {{ font-size: 9px; margin-left: 4px; }}
.ext-dep {{ color: var(--muted); font-size: 12px; display: block; padding: 1px 0; }}
.tree-hdr {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
.tree-hdr h2 {{ font-size: 17px; }}
.vt {{ padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); cursor: pointer; font-size: 12px; }}
.vt.active {{ background: #1971c2; color: #fff; border-color: #1971c2; }}
.tree-node {{ margin-left: 22px; border-left: 2px solid #dee2e6; padding-left: 12px; margin-top: 2px; }}
.tree-root {{ margin-left: 0; border-left: none; padding-left: 0; }}
.ti {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 6px; margin: 2px 0; cursor: pointer; font-size: 13px; max-width: 100%; }}
.ti:hover {{ filter: brightness(0.93); }}
.ti .ei {{ font-size: 10px; width: 14px; color: var(--muted); }}
.ti.s-proved {{ background: var(--green); border: 1px solid var(--green-border); }}
.ti.s-admitted {{ background: var(--yellow); border: 1px solid var(--yellow-border); }}
.ti.s-defined {{ background: var(--blue); border: 1px solid var(--blue-border); }}
.ti.s-assumed {{ background: var(--purple); border: 1px solid var(--purple-border); }}
.ti.s-aborted {{ background: var(--red); border: 1px solid var(--red-border); }}
.ti.taint {{ box-shadow: inset 0 0 0 2px var(--red-border); }}
.ti .il {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.ti .kl {{ font-size: 10px; color: var(--muted); }}
.ti-cycle {{ opacity: 0.5; font-style: italic; }}
.ti-external {{ background: var(--gray); border: 1px solid var(--gray-border); opacity: 0.7; }}
#placeholder {{ color: var(--muted); text-align: center; margin-top: 80px; font-size: 15px; }}
#placeholder kbd {{ background: #e9ecef; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
.file-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }}
.file-card h4 {{ font-size: 14px; margin-bottom: 6px; }}
.file-card .fc-deps {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.file-card .fc-stats {{ font-size: 12px; margin-top: 4px; display: flex; gap: 8px; flex-wrap: wrap; }}
.file-card .fc-stat {{ padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
.blast-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.blast-table th {{ text-align: left; padding: 6px 10px; border-bottom: 2px solid var(--border); font-weight: 600; }}
.blast-table td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); }}
.blast-table tr:hover {{ background: #f1f3f5; }}
.blast-bar {{ height: 8px; background: var(--red-border); border-radius: 4px; min-width: 2px; }}
.exp-btn {{ padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); cursor: pointer; font-size: 12px; margin-left: auto; }}
.exp-btn:hover {{ background: var(--gray); }}
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #adb5bd; border-radius: 4px; }}
#sidebar ::-webkit-scrollbar-thumb {{ background: #45475a; }}
</style>
</head>
<body>
<div id="sidebar">
  <h1><span>◆</span> {escape(project_name)}<small>.glob-based analysis</small></h1>
  <input id="search-box" type="text" placeholder="Search... (press /)" autocomplete="off" spellcheck="false">
  <div id="filters"></div>
  <div id="item-count"></div>
  <div id="item-list"></div>
</div>
<div id="main">
  <div id="dashboard"></div>
  <div id="tab-bar">
    <div class="tab active" data-tab="tree">Dependency Tree</div>
    <div class="tab" data-tab="files">File Graph</div>
    <div class="tab" data-tab="blast">Proof Debt</div>
  </div>
  <div id="content-area">
    <div id="tree-panel" class="panel visible"></div>
    <div id="files-panel" class="panel"></div>
    <div id="blast-panel" class="panel"></div>
    <div id="detail-panel"></div>
  </div>
</div>
<script>
const DATA = {data_json};
const IM = {{}};
DATA.items.forEach(it => {{ IM[it.qname] = it; }});
let sel = null, vm = 'deps', filters = new Set(), sq = '', expanded = new Set();
const h = t => {{ const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }};
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

function renderDash() {{
  const s = DATA.stats;
  $('#dashboard').innerHTML = `
    <div class="sc sc-gray"><div class="num">${{s.total_items}}</div><div class="lbl">Total</div></div>
    <div class="sc sc-green"><div class="num">${{s.proved}}</div><div class="lbl">Proved</div></div>
    <div class="sc sc-yellow"><div class="num">${{s.admitted}}</div><div class="lbl">Admitted</div></div>
    <div class="sc sc-blue"><div class="num">${{s.defined}}</div><div class="lbl">Defined</div></div>
    <div class="sc sc-purple"><div class="num">${{s.assumed}}</div><div class="lbl">Axioms</div></div>
    <div class="sc sc-red"><div class="num">${{s.tainted}}</div><div class="lbl">Tainted</div></div>
    <div class="sc sc-orange"><div class="num">${{s.unused}}</div><div class="lbl">Unused</div></div>
    <button class="exp-btn" onclick="expJSON()">Export JSON</button>`;
}}

function renderFilters() {{
  const statuses = [...new Set(DATA.items.map(i => i.status))].sort();
  const all = [...statuses.map(s => ({{key:'st:'+s,label:s}})),{{key:'tainted',label:'⚠ tainted'}},{{key:'unused',label:'unused'}}];
  $('#filters').innerHTML = all.map(f => `<button class="fbtn" data-f="${{f.key}}">${{f.label}}</button>`).join('');
  $$('.fbtn').forEach(b => b.addEventListener('click', () => {{
    const k = b.dataset.f;
    if (filters.has(k)) {{ filters.delete(k); b.classList.remove('active'); }}
    else {{ filters.add(k); b.classList.add('active'); }}
    renderList();
  }}));
}}

function filt() {{
  return DATA.items.filter(it => {{
    if (sq && !it.name.toLowerCase().includes(sq) && !it.qname.toLowerCase().includes(sq) && !it.file.toLowerCase().includes(sq)) return false;
    if (filters.size > 0) {{
      const stf = [...filters].filter(f => f.startsWith('st:')), sp = [...filters].filter(f => !f.startsWith('st:'));
      let pass = true;
      if (stf.length > 0) pass = stf.some(f => 'st:'+it.status === f);
      if (sp.includes('tainted') && !it.tainted) pass = false;
      if (sp.includes('unused') && it.rdeps.length > 0) pass = false;
      return pass;
    }}
    return true;
  }});
}}

function renderList() {{
  const items = filt();
  $('#item-count').textContent = items.length + ' of ' + DATA.items.length + ' items';
  const el = $('#item-list');
  el.innerHTML = items.map(it => `<div class="ie${{sel&&sel.qname===it.qname?' sel':''}}" data-q="${{it.qname}}">${{it.tainted?'<span class="tmark" title="Tainted">⚠</span>':''}}<span class="badge b-${{it.status}}">${{it.status.slice(0,3)}}</span><span class="ie-name">${{h(it.name)}}</span><span class="ie-kind">${{it.kind}}</span></div>`).join('');
  el.querySelectorAll('.ie').forEach(e => e.addEventListener('click', () => selItem(e.dataset.q)));
}}

$$('.tab').forEach(t => t.addEventListener('click', () => {{
  $$('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $$('.panel').forEach(p => p.classList.remove('visible'));
  $(`#${{t.dataset.tab}}-panel`).classList.add('visible');
  if (t.dataset.tab === 'files') renderFiles();
  if (t.dataset.tab === 'blast') renderBlast();
}}));

function selItem(qn) {{
  sel = IM[qn]; expanded.clear(); expanded.add(qn);
  renderList(); renderTree(); renderDetail();
  $$('.tab').forEach(x => x.classList.remove('active'));
  $$('.tab')[0].classList.add('active');
  $$('.panel').forEach(p => p.classList.remove('visible'));
  $('#tree-panel').classList.add('visible');
}}

function renderTree() {{
  const p = $('#tree-panel');
  if (!sel) {{ p.innerHTML = '<div id="placeholder">Select an item from the sidebar.<br><br><kbd>/</kbd> search &nbsp;|&nbsp; <kbd>Esc</kbd> clear</div>'; return; }}
  const lbl = vm==='deps' ? 'Dependencies (uses ↓)' : 'Reverse Dependencies (used by ↑)';
  p.innerHTML = `<div class="tree-hdr"><h2>${{h(sel.name)}}</h2><span style="color:var(--muted);font-size:12px">${{lbl}}</span>
    <button class="vt ${{vm==='deps'?'active':''}}" onclick="swView('deps')">Uses ↓</button>
    <button class="vt ${{vm==='rdeps'?'active':''}}" onclick="swView('rdeps')">Used by ↑</button>
    <button class="vt" onclick="expandAll()">Expand all</button>
    <button class="vt" onclick="collapseAll()">Collapse</button></div>
    <div class="tree-root">${{treeNode(sel.qname,new Set(),0)}}</div>`;
  bindTree(p);
}}

function treeNode(qn, visited, depth) {{
  const it = IM[qn];
  if (!it) {{
    const short = qn.includes('.')?qn.split('.').pop():qn;
    return `<div class="ti ti-external"><span class="ei"></span><span class="il">${{h(short)}}</span><span class="kl">external</span></div>`;
  }}
  const children = vm==='deps'?it.deps:it.rdeps;
  const hasCh = children.length>0, isCycle = visited.has(qn), isExp = expanded.has(qn);
  const tc = it.tainted?' taint':'';
  const icon = hasCh&&!isCycle?(isExp?'▼':'▶'):'';
  let out = `<div class="ti s-${{it.status}}${{tc}}${{isCycle?' ti-cycle':''}}" data-q="${{qn}}" data-act="${{hasCh&&!isCycle?'toggle':'detail'}}"><span class="ei">${{icon}}</span><span class="il">${{h(it.name)}}</span><span class="kl">${{it.kind}}${{isCycle?' (cycle)':''}}</span></div>`;
  if (isExp && hasCh && !isCycle && depth < 50) {{
    const nv = new Set(visited); nv.add(qn);
    out += `<div class="tree-node">${{children.map(c=>treeNode(c,nv,depth+1)).join('')}}</div>`;
  }}
  if (isExp && vm==='deps' && it.extDeps && it.extDeps.length>0 && !isCycle) {{
    const extHtml = it.extDeps.slice(0,20).map(e => {{
      const short = e.split('.').slice(-2).join('.');
      return `<div class="ti ti-external"><span class="ei"></span><span class="il">${{h(short)}}</span><span class="kl">ext</span></div>`;
    }}).join('');
    const more = it.extDeps.length>20?`<div class="ti ti-external"><span class="il">... +${{it.extDeps.length-20}} more</span></div>`:'';
    out += `<div class="tree-node">${{extHtml}}${{more}}</div>`;
  }}
  return out;
}}

function bindTree(container) {{
  container.querySelectorAll('.ti[data-q]').forEach(el => {{
    el.addEventListener('click', e => {{
      e.stopPropagation();
      const qn = el.dataset.q, act = el.dataset.act;
      if (act==='toggle') {{ if(expanded.has(qn)) expanded.delete(qn); else expanded.add(qn); renderTree(); }}
      else {{ renderDetailFor(qn); }}
    }});
  }});
}}

function swView(m) {{ vm=m; expanded.clear(); if(sel) expanded.add(sel.qname); renderTree(); }}
function expandAll() {{ if(!sel) return; expandRec(sel.qname,new Set(),0); renderTree(); }}
function expandRec(qn,vis,d) {{
  if(d>8||vis.has(qn)) return; expanded.add(qn); vis.add(qn);
  const it=IM[qn]; if(!it) return;
  (vm==='deps'?it.deps:it.rdeps).forEach(c=>expandRec(c,vis,d+1));
}}
function collapseAll() {{ expanded.clear(); if(sel) expanded.add(sel.qname); renderTree(); }}

function renderDetail() {{ if(!sel){{$('#detail-panel').classList.remove('visible');return;}} renderDetailFor(sel.qname); }}

function renderDetailFor(qn) {{
  const it=IM[qn], p=$('#detail-panel');
  if(!it){{p.classList.remove('visible');return;}}
  p.classList.add('visible');
  const depH = it.deps.length>0 ? it.deps.map(d=>{{const di=IM[d];return di?`<a class="dep-link" data-q="${{d}}">${{h(di.name)}} <span class="db badge b-${{di.status}}">${{di.status.slice(0,3)}}</span></a>`:`<span class="ext-dep">${{d}}</span>`;}}).join('') : '<span style="color:var(--muted);font-size:12px">None</span>';
  const rdepH = it.rdeps.length>0 ? it.rdeps.map(d=>{{const di=IM[d];return di?`<a class="dep-link" data-q="${{d}}">${{h(di.name)}} <span class="db badge b-${{di.status}}">${{di.status.slice(0,3)}}</span></a>`:`<span class="ext-dep">${{d}}</span>`;}}).join('') : '<span style="color:var(--muted);font-size:12px">None</span>';
  const extH = it.extDeps.length>0 ? `<h3>External Deps (${{it.extDeps.length}})</h3>`+it.extDeps.slice(0,30).map(e=>`<span class="ext-dep">${{h(e)}}</span>`).join('')+(it.extDeps.length>30?`<span class="ext-dep">... +${{it.extDeps.length-30}} more</span>`:'') : '';
  const taintH = it.tainted&&it.taintSources.length>0 ? `<h3 style="color:var(--red-border)">⚠ Taint Sources</h3>`+it.taintSources.map(t=>{{const ti=IM[t];return ti?`<a class="dep-link" data-q="${{t}}" style="color:var(--red-border)">${{h(ti.name)}} (${{ti.status}})</a>`:`<span class="ext-dep" style="color:var(--red-border)">${{t}}</span>`;}}).join('') : '';
  p.innerHTML = `<h2>${{h(it.name)}}</h2><div class="meta">${{it.kind}} · ${{it.status}}${{it.tainted?' · <span style="color:var(--red-border)">⚠ tainted</span>':''}}<br>${{h(it.file)}}:${{it.line}}<br><span style="font-size:11px;color:#adb5bd">${{h(it.qname)}}</span></div><pre>${{h(it.statement)}}</pre><h3>Dependencies (${{it.deps.length}})</h3>${{depH}}<h3>Used by (${{it.rdeps.length}})</h3>${{rdepH}}${{extH}}${{taintH}}`;
  p.querySelectorAll('.dep-link').forEach(a=>a.addEventListener('click',()=>selItem(a.dataset.q)));
}}

function renderFiles() {{
  const fd = DATA.stats.file_deps||{{}};
  const allFiles = new Set(Object.keys(fd));
  Object.values(fd).forEach(deps=>deps.forEach(d=>allFiles.add(d)));
  const fileStats = {{}};
  DATA.items.forEach(it=>{{
    if(!fileStats[it.file]) fileStats[it.file]={{total:0,proved:0,admitted:0,defined:0,assumed:0,tainted:0}};
    fileStats[it.file].total++;
    fileStats[it.file][it.status]=(fileStats[it.file][it.status]||0)+1;
    if(it.tainted) fileStats[it.file].tainted++;
  }});
  const rfd={{}};
  Object.entries(fd).forEach(([f,deps])=>deps.forEach(d=>{{if(!rfd[d])rfd[d]=[];rfd[d].push(f);}}));
  $('#files-panel').innerHTML = `<h2 style="margin-bottom:16px">File Dependencies (${{allFiles.size}} files)</h2>`+
    [...allFiles].sort().map(f=>{{
      const deps=fd[f]||[], rdeps=rfd[f]||[], st=fileStats[f]||{{total:0}};
      return `<div class="file-card"><h4>${{h(f)}}</h4><div class="fc-stats">
        ${{st.proved?`<span class="fc-stat" style="background:var(--green)">${{st.proved}} proved</span>`:''}}
        ${{st.admitted?`<span class="fc-stat" style="background:var(--yellow)">${{st.admitted}} admitted</span>`:''}}
        ${{st.tainted?`<span class="fc-stat" style="background:var(--red)">${{st.tainted}} tainted</span>`:''}}
        <span class="fc-stat" style="background:var(--gray)">${{st.total||0}} total</span></div>
        <div class="fc-deps">${{deps.length>0?'→ '+deps.map(d=>`<span style="color:#1971c2">${{h(d)}}</span>`).join(', '):'→ No project imports'}}</div>
        ${{rdeps.length>0?`<div class="fc-deps">← ${{rdeps.map(d=>`<span style="color:#6f42c1">${{h(d)}}</span>`).join(', ')}}</div>`:''}}</div>`;
    }}).join('');
}}

function renderBlast() {{
  const blast=DATA.stats.admitted_blast||{{}};
  const entries=Object.entries(blast).sort((a,b)=>b[1]-a[1]);
  const maxB=entries.length>0?entries[0][1]:1;
  const admitted=DATA.items.filter(i=>i.status==='admitted');
  const assumed=DATA.items.filter(i=>i.status==='assumed');
  let html=`<h2 style="margin-bottom:16px">Proof Debt Dashboard</h2>
    <div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap">
    <div class="sc sc-yellow"><div class="num">${{admitted.length}}</div><div class="lbl">Admitted</div></div>
    <div class="sc sc-purple"><div class="num">${{assumed.length}}</div><div class="lbl">Axioms</div></div>
    <div class="sc sc-red"><div class="num">${{DATA.stats.tainted}}</div><div class="lbl">Tainted</div></div></div>`;
  if(entries.length>0) {{
    html+=`<h3 style="margin-bottom:10px">Admitted Items by Blast Radius</h3>
      <p style="font-size:12px;color:var(--muted);margin-bottom:10px">Blast radius = items transitively depending on this admitted lemma.</p>
      <table class="blast-table"><tr><th>Name</th><th>File</th><th>Blast</th><th></th></tr>
      ${{entries.map(([qn,count])=>{{const it=IM[qn];const pct=Math.max(2,(count/Math.max(maxB,1))*100);return it?`<tr style="cursor:pointer" onclick="selItem('${{qn}}')"><td><strong>${{h(it.name)}}</strong></td><td style="color:var(--muted);font-size:12px">${{h(it.file)}}</td><td>${{count}}</td><td style="width:200px"><div class="blast-bar" style="width:${{pct}}%"></div></td></tr>`:'';}}).join('')}}</table>`;
  }} else html+='<p style="color:var(--muted);margin-top:20px">No admitted items — all proof obligations discharged!</p>';
  if(assumed.length>0) {{
    html+=`<h3 style="margin:20px 0 10px">Axioms & Parameters</h3><table class="blast-table"><tr><th>Name</th><th>Kind</th><th>File</th></tr>
      ${{assumed.map(it=>`<tr style="cursor:pointer" onclick="selItem('${{it.qname}}')"><td>${{h(it.name)}}</td><td>${{it.kind}}</td><td style="color:var(--muted);font-size:12px">${{h(it.file)}}</td></tr>`).join('')}}</table>`;
  }}
  $('#blast-panel').innerHTML=html;
}}

function expJSON() {{
  const b=new Blob([JSON.stringify(DATA,null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download=DATA.project+'_glob_deps.json';a.click();
}}

$('#search-box').addEventListener('input', e=>{{sq=e.target.value.toLowerCase();renderList();}});
document.addEventListener('keydown', e=>{{
  if(e.key==='/'&&document.activeElement.tagName!=='INPUT'){{e.preventDefault();$('#search-box').focus();}}
  if(e.key==='Escape'){{$('#search-box').blur();$('#search-box').value='';sq='';renderList();}}
}});

renderDash(); renderFilters(); renderList();
$('#tree-panel').innerHTML='<div id="placeholder">Select an item from the sidebar.<br><br><kbd>/</kbd> search &nbsp;|&nbsp; <kbd>Esc</kbd> clear</div>';
</script>
</body>
</html>
"""
    return html


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  JSON EXPORT
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def export_json(items: List[CoqItem], stats: Dict) -> str:
    return json.dumps({
        'stats': stats,
        'items': [{
            'name': i.name, 'qualified_name': i.qualified_name,
            'kind': i.kind, 'kind_code': i.kind_code,
            'statement': i.statement, 'status': i.status,
            'file': i.file, 'line': i.line,
            'dependencies': i.dependencies,
            'dependents': i.dependents,
            'external_dependencies': i.ext_dependencies,
            'tainted': i.tainted,
            'taint_sources': i.taint_sources,
        } for i in items]
    }, indent=2, ensure_ascii=False)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  CLI
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main():
    parser = argparse.ArgumentParser(
        description='Coq Dependency Analyzer using .glob files'
    )
    parser.add_argument('project_dir', help='Root directory of the Coq project')
    parser.add_argument('-o', '--output', default='coq_glob_deps.html',
                        help='Output HTML file (default: coq_glob_deps.html)')
    parser.add_argument('--glob-dir', default=None,
                        help='Directory containing .glob files if separate')
    parser.add_argument('--json', action='store_true',
                        help='Also export raw JSON')
    parser.add_argument('--name', default=None,
                        help='Project name (default: directory name)')

    args = parser.parse_args()
    if not os.path.isdir(args.project_dir):
        print(f"Error: {args.project_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    project_name = args.name or os.path.basename(os.path.abspath(args.project_dir))
    items, stats = analyze_project(args.project_dir, args.glob_dir)

    html = generate_html(items, stats, project_name)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML written to {args.output}", file=sys.stderr)

    if args.json:
        json_path = args.output.rsplit('.', 1)[0] + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(export_json(items, stats))
        print(f"JSON written to {json_path}", file=sys.stderr)

    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()

"""
# First, compile your project so .glob files exist
cd /path/to/cln_project
make

# Then run the analyzer
python coq_glob_analyzer.py . -o cln_analysis.html --json

# If .glob files are in a build directory:
python coq_glob_analyzer.py . --glob-dir _build/default -o cln_analysis.html
"""