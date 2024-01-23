file_dir ?= $(realpath .)
build:
	docker build \
	-f Dockerfile \
	-t supersize-net \
	.

run:
	docker run \
	--runtime=nvidia \
	-v ./src:/src \
	-v ./data:/data \
	-ti supersize-net \
	/bin/bash