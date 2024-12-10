# Release Instructions

Build a wheel:

```
pip wheel --no-deps . --wheel-dir dist
```

In your home directory, create `~/.pypirc` with the following:

```
[pypi]
username = __token__
password = <token>
```

Upload the wheel:

```
twine upload dist/*
```
