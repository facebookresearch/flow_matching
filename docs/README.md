## How to build docs

Install `sphinx`

```
conda env update --file deps.yml
```

Build HTML

```
make html
```

Start server to view the html

```
cd build/html && python3 -m http.server <port>
```

To run auto-update the server when files change (`pip install fastapi[standard]`):

```
make serve
```

## Adding to Papers

The "/papers" page lists relevant papers.  To add, insert a bibtex citation to `source/refs.bib`.  The order in which citations are listed is the order that they will appear in the page.

## Deploy

To deploy the docs (in the current branch) to github pages, run `make deploy`
