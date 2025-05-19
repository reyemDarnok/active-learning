from pathlib import Path

import graphviz # type: ignore

doc_dir = Path(__file__).parent
source_dir = doc_dir / "sources"
build_dir = doc_dir / "created"

for dot_file in source_dir.rglob('*.dot'):
    build_file = build_dir / dot_file.relative_to(source_dir)
    graph = graphviz.Source(source=dot_file.read_text(), filename=build_file, format='svg')
    graph.render(directory=doc_dir) # type: ignore
