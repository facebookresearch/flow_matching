# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
from subprocess import check_call
from tempfile import TemporaryDirectory

this_dir = os.path.dirname(os.path.realpath(__file__))

remote = "git@github.com:fairinternal/flow_matching.git"
branch = "gh-pages"


with TemporaryDirectory() as tdir:
    local = os.path.join(tdir, "repo")
    shutil.copytree(os.path.join(this_dir, "build/html"), local)

    with open(os.path.join(local, ".nojekyll"), "w") as fout:
        print("", end="", file=fout)

    check_call(["git", "init", local])
    check_call(["git", "remote", "add", "origin", remote], cwd=local)
    check_call(["git", "checkout", "-b", branch], cwd=local)

    check_call(["git", "add", "--all"], cwd=local)
    check_call(["git", "commit", "-m", "Update github pages"], cwd=local)

    check_call(["git", "push", "--set-upstream", "origin", "gh-pages", "-f"], cwd=local)
