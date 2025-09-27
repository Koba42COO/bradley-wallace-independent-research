# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for different output formats
html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

latexpdf:
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the PDF files are in $(BUILDDIR)/latex."

doctest:
	$(SPHINXBUILD) -b doctest $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/doctest
	@echo
	@echo "Testing of doctests in the sources finished."

coverage:
	$(SPHINXBUILD) -b coverage $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/coverage
	@echo
	@echo "Testing of coverage in the sources finished."

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf source/_autosummary/
	rm -rf source/generated/

# Full documentation build pipeline
all: clean html latexpdf doctest coverage
	@echo
	@echo "Full documentation build completed successfully!"
	@echo "HTML: $(BUILDDIR)/html/index.html"
	@echo "PDF: $(BUILDDIR)/latex/ConsciousnessCompressionEngine.pdf"

# Serve documentation locally for development
serve:
	cd $(BUILDDIR)/html && python -m http.server 8000

# Check for broken links
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/linkcheck

# Generate API documentation
api:
	sphinx-apidoc -f -o source/api ../consciousness_compression
	@echo "API documentation generated in source/api/"

# Build with live reload for development
livehtml:
	sphinx-autobuild -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
