# Adding a new notebook

## Step 1: Clone the repository

First, fork the repository on Github and clone your fork.
```
git clone https://github.com/<your-github-username>/IPython-plotly.git
```

Change into the into the repository directory.
```
cd IPython-plotly
```

Now checkout a new branch for your notebook:
```
git checkout -b my-super-cool-notebook
```

## Step 2: Install dependencies

Install pip if you don't have it already and then install the dependencies listed
in the `requirements.txt` file with:
```
pip install -r requirements.txt
```

## Step 3: Initialize a notebook directory

At this point you can manually create a notebook directory with a `config.json`
or run the following make target to do that for you:
```
make init nb=<notebook-id>
```

For example, `notebooks/aircraft_pitch` was initialized with this command:
```
make init nb='aircraft_pitch'
```

**As shown [below](#repo-structure), the `ipynb` file must have the same name
as its `notebooks/` sub-directory.**

## Step 4: Edit the configuration file

Now edit the generated configuration file,
`notebooks/<notebook-id>/config.json` and add in the unique metadata for your
notebook.

- **The the comments must be removed from the json file for the publishing
  scripts to run.**

- Sentence case are encouraged for the `title` and `title_short` fields

- If the `title` attributes is longer than 50 character, consider including
  `\n` to make the title appears on two lines.

- To include all cells in the published notebook set `cells: [0, "end"]`.
  Alternatively, set `cells: [0, -1]` to include all cells but the last.

## Step 5: Create the notebook

Navigate into your notebook's directory and start an IPython notebook server:
```
cd notebooks/<notebook-id>
ipython notebook
```

Create the notebook. Any resources such as images or data should be added to
this directory as shown in the [repository structure below](#repo-structure).

**Make sure that the `ipynb` file has been executed, i.e. all of the cells have
output.**

After the notebook is complete and IPython is shutdown, change back into the
root directory of the repository:
```
cd ../..
```

## Step 6: Generate the HTML and Python versions

First, edit `notebooks/references.json` and add your notebook id to the top of
the list. Now execute:

```
make run nb=<notebook-id>
```

This will generate some files in your notebook's directory:

- `<notebook-id>.tmp.ipynb`
- `<notebook-id>.tmp.html`
- `<notebook-id>.py`

For example,
```
make run nb='aircraft_pitch'
```
would have generated:

- `notebooks/aircraft_pitch/aircraft_pitch.ipynb.tmp.ipynb`
- `notebooks/aircraft_pitch/aircraft_pitch.ipynb.tmp.html`
- `notebooks/aircraft_pitch/aircraft_pitch.ipynb.py`

## Step 7: Generate the final versions

Finally, the final publishable versions are generated with:
```
make publish nb=<notebook-id>
```

This puts the html into publishable form, generates the `urls.py` and
`sitemaps.py` files and appends the config and references files with
auto-generatable fields.


## Step 8: Submit a pull request

At this point you should commit all of the  unignored files:

```
git add .
git commit -am "Added a new notebook."
```

Push your branch to your fork on Github:
```
git push origin my-super-cool-notebook
```

Now open a pull request from your fork on Github to the main repository.


## Step 6: Push to server

**This step is for plotly employees only.**

```
make push
```

Pushes the published content over to streambed.

# Extra information

## Config file

Each notebook directory must contain a configuration file:

`notebooks/<notebook-id>/config.json`

An example is available [here](_makescripts/data/config-init.json).

**Note that the comments must be removed for `make run` to work.**

## References file

`notebooks/references.json`

If the file looks like:

```json
{
    "notebooks": [
        "basemap",
        "collaborate"
    ]
}
```

Then add your notebook id to the top of the list:

```json
{
    "notebooks": [
        "<notebook-id>",
        "basemap",
        "collaborate"
    ]
}
```

- These *need* to be hard-coded in order to preserve the order in which they will
appear on the splash page.
- **As of Feb 19 2015**, please put the latest notebook should be the
  `notebooks[0]` item in order to appear at the top of the list on
  [/ipython-notebooks](https://plot.ly/ipython-notebooks/).

## Repo structure

```
IPython-plotly/
    README.md
    CONTRIBUTING.md
    requirements.txt
    notebooks/
        <notebook-id>/
            config.json
            <notebook-id>.ipynb
            <data>.* (optional)
            requirements.txt (auto-gen by `make run`)
            <notebook-id>.py (auto-gen by `make run`)
            <notebook-id>.zip (auto-gen if <data>.* by `make run`)
        <some-other-notebook-id>/
            ...
    makefile (setup, run, publish, push)
    _makescripts/
        *.py (scripts used `make`)
    _published/ (auto gen by `make publish`)
        sitemaps.py
        urls.py
        redirects.py
        includes/
            references.json
            <notebook-id>/
                references.json
                body.html
            <some-other-notebook-id>/
                ...
        static/
            image/
                <notebook-id>_image01.*
                ...
                <some-other-notebook-id>_image01.*
                ...
    .gitignore
```
