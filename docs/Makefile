# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

links:
	mkdir -p source/notebooks && ln -sfn $(ROOT_DIR)/../examples/standalone_flow_matching.ipynb source/notebooks/standalone_flow_matching.ipynb
	mkdir -p source/notebooks && ln -sfn $(ROOT_DIR)/../examples/2d_discrete_flow_matching.ipynb source/notebooks/2d_discrete_flow_matching.ipynb
	mkdir -p source/notebooks && ln -sfn $(ROOT_DIR)/../examples/2d_riemannian_flow_matching_flat_torus.ipynb source/notebooks/2d_riemannian_flow_matching_flat_torus.ipynb
	mkdir -p source/notebooks && ln -sfn $(ROOT_DIR)/../examples/2d_riemannian_flow_matching_sphere.ipynb source/notebooks/2d_riemannian_flow_matching_sphere.ipynb
	ln -sfn $(ROOT_DIR)/../assets/teaser.png  source/_images/teaser.png

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: export PYTHONPATH=../:./

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile	links
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

deploy:	html
	python deploy.py

serve:
	uvicorn server:app --reload --reload-include 'build/html/*.html'