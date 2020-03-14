# `make` is expected to be called from the directory that contains
# this Makefile

TAG := latest

rebuild: clean build-dist

build-dist:
	# Increase the build number
	python inc-build.py thomas/core/__build__

	# Build the PyPI package
	python setup.py sdist bdist_wheel

publish-test:
	# Uploading to test.pypi.org
	twine upload --repository testpypi dist/*

publish:
	# Uploading to pypi.org
	twine upload --repository pypi dist/*

docker-image:
	docker build \
	  -t thomas-core:${TAG} \
	  -t mellesies/thomas-core:${TAG} \
	  ./

docker-run:
	# Run the docker image
	docker run --rm -it -p 8888:8888 thomas-core:${TAG}

docker-push:
	mellesies/thomas-core:${TAG}

clean:
	# Cleaning ...
	-rm -r build
	-rm dist/*
