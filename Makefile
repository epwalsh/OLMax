BEAKER_WORKSPACE = ai2/OLMo-pretraining-stability

.PHONY : check
check : style lint

.PHONY : style
style :
	black --check .
	isort --check .

.PHONY : lint
lint :
	ruff check .
	mypy .

.PHONY : test
test :
	pytest -v --color=yes --durations=3 src/test/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build

.PHONY : dev-install
dev-install :
	pip install -e .[dev] --config-settings editable_mode=compat

NVCR_TAG = 25.04

.PHONY : docker-image
docker-image :
	docker build -f src/Dockerfile --build-arg NVCR_TAG=$(NVCR_TAG) -t olmax-$(NVCR_TAG) .
	echo "Built image 'olmax-$(NVCR_TAG)', size: $$(docker inspect -f '{{ .Size }}' olmax-$(NVCR_TAG) | numfmt --to=si)"

.PHONY : beaker-image
beaker-image : docker-image
	./src/scripts/beaker/create_beaker_image.sh olmax-$(NVCR_TAG) olmax-$(NVCR_TAG) $(BEAKER_WORKSPACE)
