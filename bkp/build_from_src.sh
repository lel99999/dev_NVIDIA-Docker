/bin/bash

#
# This script requires buildkit: https://docs.docker.com/buildx/working-with-buildx/
#
IMAGE_NAME="nvcr.io/nvidia/cuda"
CUDA_VERSION="10.2"
OS="ubuntu18.04"
ARCHES="x86_64"
PLATFORM_ARG=`printf '%s ' '--platform'; for var in $(echo $ARCHES | sed "s/,/ /g"); do printf 'linux/%s,' "$var"; done | sed 's/,*$//g'`

cp NGC-DL-CONTAINER-LICENSE dist/${CUDA_VERSION}/${OS}/base/

docker buildx build --load ${PLATFORM_ARG} -t "${IMAGE_NAME}:${CUDA_VERSION}-base-${OS}" "dist/${CUDA_VERSION}/${OS}/base"
docker buildx build --load ${PLATFORM_ARG} -t "${IMAGE_NAME}:${CUDA_VERSION}-runtime-${OS}" --build-arg "IMAGE_NAME=${IMAGE_NAME}" "dist/${CUDA_VERSION}/${OS}/runtime"
docker buildx build --load ${PLATFORM_ARG} -t "${IMAGE_NAME}:${CUDA_VERSION}-devel-${OS}" --build-arg "IMAGE_NAME=${IMAGE_NAME}" "dist/${CUDA_VERSION}/${OS}/devel"

