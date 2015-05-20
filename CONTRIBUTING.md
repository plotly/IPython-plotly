## General info


#### Repo structure

```
IPython-plotly/
    README.md
    CONTRIBUTING.md
    requirements.txt 
    notebooks/
        <some-notebook>/
            config.json
            <some-notebook>.ipynb
            <data>.* (optional)
            requirements.txt (auto-gen by `make run`)
            <some-notebook>.py (auto-gen by `make run`)
            <some-notebook>.zip (auto-gen if <data>.* by `make run`)
        <some-other-notebook>/ 
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
            <some-notebook>/
                references.json
                body.html
            <some-other-notebook>/
                ...
        static/
            image/
                <some-notebook>_image01.*
                ...
                <some-other-notebook>_image01.*
                ...
    .gitignore
```


#### Config file

`notebooks/<some-notebook>/config.json`

See [model](_makescripts/data/config-init.json).
 
#### References file

`notebooks/references.json` 

Simply fill in the notebook id:

```json
{
    "notebooks": [
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


## How to add a notebook?

#### Step 0: Make a directory and add ipynb file

Clone this repo:
```
git clone https://github.com/plotly/IPython-plotly.git
```

**As of Feb 19 2015**:

- As displayed [above](#repo-structure), the `ipynb` file must have the same
  name as its `notebooks/` sub-directory.

- Make sure that the `ipynb` file has been ran. The notebook is not ran in this
  process (possibly later), its in and out cell are only converted.

#### Step 1: Install requirements

```
pip install -r requirements.txt
pip install beautifulsoup4
```

#### Step 2: 

```
make init nb=<notebook-id>
```

For example,

```
Jacks-MacBook-Air:IPython-plotly jack$ sudo make init nb='aircraft_pitch'
```

Then, fill in the generated notebook `config.json` in the folder you created in Step 1.

- **Don't forget to remove the comments from the json file**

- Sentence case are encouraged for the `title` and `title_short` fields

- If the `title` attributes is longer than 50 character, consider including 
  `\n` to make the title appears on two lines.

- To include all cells in the published notebook set `cells: [0, "end"]`. Alternatively, set `cells: [0, -1]` to include all cells but the last.

#### Step 3:

Add your notebook name into `references.json` in the notebooks sub-directory then run:

```
make run nb=<notebook-id>
```

For example,
```
Jacks-MacBook-Air:IPython-plotly jack$ sudo make run nb='aircraft_pitch'
```

This creates an `html` and `py` version of the notebook


#### Step 4:

```
make publish nb=<notebook-id>
```

This puts the html into publishable form, generates the `urls.py` and
`sitemaps.py` files and appends the config and references files with
auto-generatable fields.


#### Step 5: 

Commit all unignored files and make a PR. 


#### Step 6 (for plotly employees only)

```
make push
```

Pushes the published content over to streambed
