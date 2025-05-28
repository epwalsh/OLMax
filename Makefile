.PHONY : check
check :
	black --check .
	isort --check .
	ruff check .
	mypy .

.PHONY : dev-install
dev-install :
	pip install -e .[dev] --config-settings editable_mode=compat
