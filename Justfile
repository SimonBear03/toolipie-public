setup:
	python -m venv .venv && source .venv/bin/activate && pip install -e . && pre-commit install

docx:
	toolipie run md-to-docx --input examples/md --output output/docx --param template=src/toolipie/tools/md_to_docx/assets/presets/word_template.docx

pdf:
	toolipie run md-to-pdf --input examples/md --output output/pdf --param preset=a4_report
