#!/bin/bash

# Exit when an error happens instead of continue.
set -e

# Default values for flags.
DEBUG_TYPE="Release"
NUM_JOBS=${OPENSIM_BUILD_JOBS:-24}
MOCO="off"
ORG="nickbianco"
BRANCH="dc7e1f0a18905fcd17cbbee9f923a70b4cb9de99"
GENERATOR="Ninja"
PYTHON_ROOT_DIR=$1
WORKING_DIR="$(pwd)/opensim"
if [ -d "$WORKING_DIR" ]; then
    rm -rf "$WORKING_DIR"
fi
mkdir -p "$WORKING_DIR"

# Get opensim-core. Fetch the pinned commit directly instead of cloning the
# whole repo and checking it out afterwards: if that commit is later dropped
# from whatever branch it was on (rebase/branch deletion upstream), a normal
# clone stops including it and `git checkout $BRANCH` fails with "unable to
# read tree", even though the commit object itself still exists and remains
# directly fetchable by SHA.
mkdir -p "$WORKING_DIR/opensim-core"
cd "$WORKING_DIR/opensim-core"
git init -q
git remote add origin https://github.com/$ORG/opensim-core.git
git fetch --depth 1 origin $BRANCH
git checkout -q FETCH_HEAD

# The vendored numpy.i at this pinned commit still uses the Python 2 C API
# (PyString_Check/PyInt_Check/PyInt_AsLong) in its hand-written pytype_string()
# helper and array-dimension typemaps. Those symbols don't exist at all under
# Python 3 (CPython removed them, no compatibility shim), so building the
# SWIG-generated simbody/common/moco bindings against Python 3 fails with
# "was not declared in this scope". Patch it to the Python 3 equivalents.
NUMPY_I="$WORKING_DIR/opensim-core/Bindings/Python/swig/numpy.i"
sed -i \
    -e 's/PyString_Check/PyUnicode_Check/g' \
    -e 's/PyInt_AsLong/PyLong_AsLong/g' \
    -e 's/PyInt_Check/PyLong_Check/g' \
    "$NUMPY_I"

# Build opensim-core dependencies.
mkdir -p "$WORKING_DIR/opensim-core/dependencies/build"
cd "$WORKING_DIR/opensim-core/dependencies/build"
cmake "$WORKING_DIR/opensim-core/dependencies" -G"$GENERATOR" -DCMAKE_BUILD_TYPE=$DEBUG_TYPE -DCMAKE_INSTALL_PREFIX="$WORKING_DIR/opensim_dependencies_install/" -DSUPERBUILD_ezc3d=off -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR="$PYTHON_ROOT_DIR"
cmake . -LAH

# opensim-core's own dependencies/CMakeLists.txt pins Simbody to a raw commit
# on nickbianco's fork. That commit can become orphaned (unreachable from any
# branch) if the fork gets rebased/cleaned up upstream, which makes
# ExternalProject_Add's normal git clone fail with "reference is not a tree" /
# "unable to read tree", even though the commit object itself still exists
# and remains directly fetchable by SHA (same class of issue as the
# opensim-core checkout above). Work around it the same way: fetch Simbody by
# SHA ourselves into the exact source dir ExternalProject expects, then mark
# its git-clone step as already done via the stamp file it checks, so the
# build skips straight to configuring/building the pre-fetched source.
SIMBODY_GIT_URL="https://github.com/nickbianco/simbody.git"
SIMBODY_GIT_TAG="6cd18d8b8466b3e574377a6b5acd942af78c7a88"
SIMBODY_SOURCE_DIR="$WORKING_DIR/opensim-core/dependencies/simbody"
SIMBODY_STAMP_DIR="$WORKING_DIR/opensim-core/dependencies/build/simbody/stamp"

rm -rf "$SIMBODY_SOURCE_DIR"
mkdir -p "$SIMBODY_SOURCE_DIR"
(
    cd "$SIMBODY_SOURCE_DIR"
    git init -q
    git remote add origin "$SIMBODY_GIT_URL"
    git fetch --depth 1 origin "$SIMBODY_GIT_TAG"
    git checkout -q FETCH_HEAD
    if [ -f .gitmodules ]; then
        git submodule update --init --recursive
    fi
)
cp "$SIMBODY_STAMP_DIR/simbody-gitinfo.txt" "$SIMBODY_STAMP_DIR/simbody-gitclone-lastrun.txt"
touch "$SIMBODY_STAMP_DIR/simbody-gitclone-lastrun.txt"

cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS


# Build and install opensim-core.
mkdir -p "$WORKING_DIR/opensim-core/build"
cd "$WORKING_DIR/opensim-core/build"
cmake "$WORKING_DIR/opensim-core" -G"$GENERATOR" -DCMAKE_BUILD_TYPE=$DEBUG_TYPE -DOPENSIM_DEPENDENCIES_DIR="$WORKING_DIR/opensim_dependencies_install/" -DOPENSIM_C3D_PARSER=None -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX="$WORKING_DIR/opensim_core_install" -DOPENSIM_INSTALL_UNIX_FHS=off -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR="$PYTHON_ROOT_DIR"
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS
cmake --install .
