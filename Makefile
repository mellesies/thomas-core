# `make` is expected to be called from the directory that contains
# this Makefile

TAG ?= latest

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
	  .

docker-rebuild:
	docker build \
	--no-cache \
	  -t thomas-core:${TAG} \
	  -t mellesies/thomas-core:${TAG} \
	  .

docker-run:
	# Run the docker image and listen on port 9999
	docker run --rm -it -p 9999:8888 --name thomas-core thomas-core:${TAG}

docker-run-browser:
	# Run the docker image and listen on port 9999
	docker run -d --rm -it -p 9999:8888 --name thomas-core thomas-core:${TAG}

	# Open a browser
	sleep 2
	open http://localhost:9999/lab

docker-stop:
	# Stop the docker image
	docker stop thomas-core


docker-push:
	docker push mellesies/thomas-core:${TAG}

clean:
	# Cleaning ...
	-rm -r build
	-rm dist/*
