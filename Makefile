.PHONY: check test test-all build clean docs dev

check:
	uv run ruff check murmur/
	uv run black --check murmur/
	uv run mypy murmur/ --ignore-missing-imports
	uv run pytest tests/unit/ -v

test:
	uv run pytest tests/unit/ -v --cov=murmur --cov-report=term-missing

test-all:
	MURMUR_INTEGRATION_TESTS=1 uv run pytest tests/ -v

build:
	uv build

clean:
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/

docs:
	cd docs && npm start

dev:
	uv run murmur run --config murmur.toml

fmt:
	uv run ruff check --fix murmur/
	uv run black murmur/

release:
	@echo "Tagging $(VERSION)..."
	git tag -s $(VERSION) -m "Release $(VERSION)"
	git push origin $(VERSION)
