build:
	python -m build

install:
	pip install -e .

test:
	python -m unittest tests/test_preprocessors.py
	python -m unittest tests/test_encoders.py
	python -m unittest tests/test_aggregators.py
	
upload:
	twine upload dist/*