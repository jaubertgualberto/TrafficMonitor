#!/usr/bin/env python3
import subprocess
from pathlib import Path

# 1) obtém lista de arquivos versionados
files = subprocess.run(
    ["git", "ls-files"], stdout=subprocess.PIPE, text=True
).stdout.splitlines()

# 2) constrói árvore aninhada em dicts
tree = {}
for f in files:
    parts = f.split('/')
    node = tree
    for p in parts:
        node = node.setdefault(p, {})

# 3) função recursiva para imprimir com ├─, │  , └─
def print_tree(node, prefix=""):
    entries = sorted(node.items())
    for idx, (name, child) in enumerate(entries):
        is_last = idx == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(prefix + connector + name)
        if child:
            extension = "    " if is_last else "│   "
            print_tree(child, prefix + extension)

if __name__ == "__main__":
    print_tree(tree)
