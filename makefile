# --- IPython-plotly - makefile ---

# Constants:

# Path to notebooks
path_to_notebooks="./notebooks"

# Path to published notebooks (html -> plot.ly-ready)
path_to_published="./_published"

# Path to make scripts
path_to_makescripts="./_makescripts"

# Path to make data
path_to_makedata=$(path_to_makescripts)"/data"

# Path (relative or absolute) to streambed (only for `make publish`)
path_to_streambed="../streambed"

# Folders on streambed
includes=$(path_to_streambed)"/shelly/templates/api_docs/includes/ipython_notebooks"
image=$(path_to_streambed)"/shelly/api_docs/static/api_docs/image/ipython_notebooks"
urls=$(path_to_streambed)"/shelly/api_docs/urls/ipython_notebooks/urls.py"
sitemaps=$(path_to_streambed)"/shelly/api_docs/sitemaps/ipython_notebooks/sitemaps.py"

# Keyword arguments:
#
# 'nb' : name of notebook subfolder in question
#
# Examples:
#
# make <target> nb=<some-notebook-folder>
#
# N.B. Each target must be ran one notebook at a time
#
#-------------------------------------------------------------------------------

init:
	@mkdir $(path_to_notebooks)/$(nb)
	@cp $(path_to_makedata)/config-init.json $(path_to_notebooks)/$(nb)/config.json

run:
	@ipython $(path_to_makescripts)/trim.py $(nb)
	@cd $(path_to_notebooks)/$(nb) && ipython nbconvert --to html $(nb).tmp.ipynb
	@cd $(path_to_notebooks)/$(nb) && ipython nbconvert --to python $(nb).tmp.ipynb
	@cd $(path_to_notebooks)/$(nb) && mv $(nb).tmp.py $(nb).py
	
publish:
	@ipython $(path_to_makescripts)/publish.py $(nb)

run-publish: run publish

push:
	@rm -rf $(includes)/*
	@cp -R $(path_to_published)/includes/* $(includes)
	@rm -rf $(image)/*
	@cp -R $(path_to_published)/static/image/* $(image)
	@cp -f $(path_to_published)/urls.py $(urls)
	@cp -f $(path_to_published)/sitemaps.py $(sitemaps)
