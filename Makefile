BEAKER_WORKSPACE = ai2/OLMo-pretraining-stability

.PHONY : check
check :
	black --check .
	isort --check .
	ruff check .
	mypy .

.PHONY : dev-install
dev-install :
	pip install -e .[dev] --config-settings editable_mode=compat

.PHONY : docker-image
docker-image :
	docker build -f src/Dockerfile -t olmax .
	echo "Built image 'olmax', size: $$(docker inspect -f '{{ .Size }}' olmax | numfmt --to=si)"

.PHONY : beaker-image
beaker-image : docker-image
	./src/scripts/beaker/create_beaker_image.sh olmax olmax $(BEAKER_WORKSPACE)
