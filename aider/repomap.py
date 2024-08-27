import colorsys
import math
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path

from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm

from aider.dump import dump
from aider.utils import Spinner

import configparser
from neo4j import GraphDatabase

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser  # noqa: E402

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


class RepoMap:
    CACHE_VERSION = 3
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
        refresh="auto",
        config_file="config.ini",
    ):
        self.io = io
        self.verbose = verbose
        self.refresh = refresh

        if not root:
            root = os.getcwd()
        self.root = root

        self.load_neo4j_config(config_file)
        self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri,
                                                 auth=(self.neo4j_user,
                                                       self.neo4j_password))

        self.load_tags_cache()
        self.cache_threshold = 0.95

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        self.main_model = main_model

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        self.map_processing_time = 0
        self.last_map = None

    def __del__(self):
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

    def load_neo4j_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.neo4j_uri = config.get('neo4j', 'uri')
        self.neo4j_user = config.get('neo4j', 'username')
        self.neo4j_password = config.get('neo4j', 'password')

    def token_count(self, text):
        len_text = len(text)
        if len_text < 200:
            return self.main_model.token_count(text)

        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        step = num_lines // 100 or 1
        lines = lines[::step]
        sample_text = "".join(lines)
        sample_tokens = self.main_model.token_count(sample_text)
        est_tokens = sample_tokens / len(sample_text) * len_text
        return est_tokens

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                max_map_tokens * self.map_mul_no_files,
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        if self.verbose:
            num_tokens = self.token_count(files_listing)
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        if not path.exists():
            self.cache_missing = True
        self.TAGS_CACHE = Cache(path)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found error: {fname}")

    def get_tags_from_cache(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        if cache_key in self.TAGS_CACHE and self.TAGS_CACHE[cache_key][
                "mtime"] == file_mtime:
            return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache
        self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
        self.save_tags_cache()
        return data

    def get_tags_from_neo4j(self, fname, rel_fname):
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        stored_tags = self.get_tags_from_neo4j(rel_fname)

        if stored_tags and stored_tags[0]['mtime'] == file_mtime:
            return [Tag(**tag) for tag in stored_tags]

        # If tags are not in Neo4j or outdated, generate new tags
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update Neo4j with new tags
        self.store_tags_in_neo4j(rel_fname, file_mtime, data)

        return data

    def get_tags(self, fname, rel_fname):
        return self.get_tags_from_neo4j(fname, rel_fname)

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        language = get_language(lang)
        parser = get_parser(lang)

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(self,
                        chat_fnames,
                        other_fnames,
                        mentioned_fnames,
                        mentioned_idents,
                        progress=None):
        import networkx as nx

        G = self.get_function_call_graph_from_neo4j()

        personalization = self.get_personalization(chat_fnames, other_fnames,
                                                   mentioned_fnames)

        if personalization:
            pers_args = dict(personalization=personalization,
                             dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            return []

        ranked_definitions = self.get_ranked_definitions(G, ranked)
        ranked_tags = self.get_ranked_tags_from_definitions(
            ranked_definitions, chat_fnames)

        return ranked_tags

    def get_function_call_graph_from_neo4j(self):
        import networkx as nx

        G = nx.MultiDiGraph()

        with self.neo4j_driver.session() as session:
            result = session.run(
                "MATCH (caller:File)-[:HAS_TAG]->(callerTag:Tag)-[:CALLS]->(calleeTag:Tag)<-[:HAS_TAG]-(callee:File) "
                "RETURN caller.rel_fname as caller_fname, callee.rel_fname as callee_fname, "
                "callerTag.name as caller_name, calleeTag.name as callee_name, "
                "callerTag.kind as caller_kind, calleeTag.kind as callee_kind")

            for record in result:
                G.add_edge(
                    record['caller_fname'],
                    record['callee_fname'],
                    weight=
                    1,  # You may want to adjust this based on your requirements
                    ident=record['callee_name'])

        return G

    def get_personalization(self, chat_fnames, other_fnames, mentioned_fnames):
        personalization = dict()
        fnames = set(chat_fnames).union(set(other_fnames))
        personalize = 100 / len(fnames)

        for fname in fnames:
            rel_fname = self.get_rel_fname(fname)
            if fname in chat_fnames or rel_fname in mentioned_fnames:
                personalization[rel_fname] = personalize

        return personalization

    def get_ranked_definitions(self, G, ranked):
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            src_rank = ranked[src]
            total_weight = sum(
                data["weight"]
                for _src, _dst, data in G.out_edges(src, data=True))
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        return sorted(ranked_definitions.items(),
                      reverse=True,
                      key=lambda x: x[1])

    def get_ranked_tags_from_definitions(self, ranked_definitions,
                                         chat_fnames):
        ranked_tags = []
        chat_rel_fnames = set(
            self.get_rel_fname(fname) for fname in chat_fnames)

        with self.neo4j_driver.session() as session:
            for (fname, ident), rank in ranked_definitions:
                if fname in chat_rel_fnames:
                    continue
                result = session.run(
                    "MATCH (f:File {rel_fname: $fname})-[:HAS_TAG]->(t:Tag {name: $ident}) "
                    "RETURN f.rel_fname as rel_fname, t.name as name, t.kind as kind, t.line as line",
                    fname=fname,
                    ident=ident)
                for record in result:
                    ranked_tags.append(Tag(**record))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # Create a cache key
        cache_key = (
            tuple(sorted(chat_fnames)) if chat_fnames else None,
            tuple(sorted(other_fnames)) if other_fnames else None,
            max_map_tokens,
        )

        if not force_refresh:
            if self.refresh == "manual" and self.last_map:
                return self.last_map

            if self.refresh == "always":
                use_cache = False
            elif self.refresh == "files":
                use_cache = True
            elif self.refresh == "auto":
                use_cache = self.map_processing_time > 1.0

            # Check if the result is in the cache
            if use_cache and cache_key in self.map_cache:
                return self.map_cache[cache_key]

        # If not in cache or force_refresh is True, generate the map
        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(chat_fnames, other_fnames,
                                                   max_map_tokens,
                                                   mentioned_fnames,
                                                   mentioned_idents)
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        # Store the result in the cache
        self.map_cache[cache_key] = result
        self.last_map = result

        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        spin = Spinner("Updating repo map")

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        spin.step()

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = set(
            self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(max_map_tokens // 25, num_tags)
        while lower_bound <= upper_bound:
            # dump(lower_bound, middle, upper_bound)

            spin.step()

            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            if (num_tokens <= max_map_tokens
                    and num_tokens > best_tree_tokens) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        spin.end()
        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        mtime = self.get_mtime(abs_fname)
        key = (rel_fname, tuple(sorted(lois)), mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        if (rel_fname not in self.tree_context_cache
                or self.tree_context_cache[rel_fname]["mtime"] != mtime):
            code = self.io.read_text(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                # header_max=30,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {
                "context": context,
                "mtime": mtime
            }

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None, )
        for tag in sorted(tags) + [dummy_tag]:
            this_rel_fname = tag[0]
            if this_rel_fname in chat_rel_fnames:
                continue

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_scm_fname(lang):
    # Load the tags queries
    try:
        return resources.files(__package__).joinpath(
            "queries", f"tree-sitter-{lang}-tags.scm")
    # for any error, return None
    except:
        # convert string f"queries/tree-sitter-{lang}-tags.scm" to a path
        import pathlib
        return pathlib.Path(f"queries/tree-sitter-{lang}-tags.scm")


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = """
| Language | File extension | Repo map | Linter |
|:--------:|:--------------:|:--------:|:------:|
"""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())

    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if Path(fn).exists() else ""
        linter_support = "✓"
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} | {linter_support:^6} |\n"

    res += "\n"

    return res


if __name__ == "__main__":
    from aider.io import InputOutput
    from aider import models
    fnames = sys.argv[1:]

    chat_fnames = []
    other_fnames = []
    for fname in sys.argv[1:]:
        if Path(fname).is_dir():
            chat_fnames += find_src_files(fname)
        else:
            chat_fnames.append(fname)

    rm = RepoMap(root=".",
                 io=InputOutput(),
                 main_model=models.Model("gpt-4-1106-preview"))
    repo_map = rm.get_ranked_tags_map(chat_fnames, other_fnames)

    # dump(len(repo_map))
    print(repo_map)
