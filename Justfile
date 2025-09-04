setup:
	python -m venv .venv && source .venv/bin/activate && pip install -e . && pre-commit install

docx:
	toolipie md-to-docx --input examples/md --output output/docx --template src/toolipie/tools/md_to_docx/assets/presets/word_template.docx

pdf:
	toolipie md-to-pdf --input examples/md --output output/pdf --preset a4_report
