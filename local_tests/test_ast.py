import ast
from ast import Module

# 1. Define your source code as a string
source_code = "x = 1 + 2"

# 2. Parse the string into an AST
tree: Module = ast.parse(source_code)

# 3. Inspect or debug the tree structure
print(ast.dump(tree, indent=4))
