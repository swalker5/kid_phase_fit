# `common_utils` boilerplate

## Project structure

The `CMakeLists.txt` file contains documentation on how everything works
behind the scene and possible ways to structure the code.

## Build

```
    $ git clone /url/to/this/repo
    $ git submodule update --init --recursive --remote
    $ cd /path/to/cloned/repo
    $ mkdir build && cd build
    $ cmake ..
```

The built executables is in `<builddir>/bin/` by default.


## Notes

### On git submodule

If the common_utils package is managed as the git submodule.
After the project repo is cloned, the submodule does not
get populated until the `git submodule update ...` call is
made.

The `git submodule` by design is pinned to a particular commit
in the `detached head` state. In this state, one is not advised
to modify the content of the submodule.

However, the submodule is set up to track a remote branch, in this
case, `kids_dev`. One can checkout the head of this branch in
the submodule, so that any changes to the submodule can be committed
normally to the submodule repo.
