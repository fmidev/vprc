# Releasing

A release is a Git tag `vX.Y.Z` on `main`. The version comes from the tag via
hatch-vcs; there is no version number to edit in the source. Pushing the tag
starts two independent builds:

- [release.yml](../.github/workflows/release.yml) runs CI, builds the sdist
  and wheel, and publishes them to [PyPI](https://pypi.org/project/vprc/).
- Quay builds the container image `quay.io/fmi/vprc:vX.Y.Z` from the
  [Containerfile](../Containerfile).

## One-time setup

Publishing uses PyPI
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API
tokens are stored in GitHub.

1. On [PyPI](https://pypi.org/manage/account/publishing/), add a pending
   publisher for project `vprc`: owner `fmidev`, repository `vprc`, workflow
   `release.yml`, environment `pypi`.
2. On [TestPyPI](https://test.pypi.org/manage/account/publishing/), add the
   same with environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and
   `testpypi`. Adding required reviewers to `pypi` makes every PyPI upload wait
   for manual approval.

## Release checklist

1. Make sure CI passes on `main`.
2. Optionally refresh dependencies with `uv lock --upgrade`, run the tests and
   commit `uv.lock`.
3. Optionally build the container image locally against the newest base
   image, as Quay will:

   ```sh
   podman build --pull=newer -t vprc .
   ```
4. Optionally do a dry run: start the Release workflow manually from the
   Actions tab. It publishes the current commit to TestPyPI as a dev version
   such as `0.10.0.dev3`; check the
   [TestPyPI page](https://test.pypi.org/project/vprc/).
5. Tag and push:

   ```sh
   git tag -a v0.10.0 -m "vprc 0.10.0"
   git push origin v0.10.0
   ```

6. Approve the `pypi` deployment if reviewers are required, and create a
   GitHub release from the tag with generated release notes.
7. After Quay has built the image, update the image tag used by the Airflow
   deployment.

PyPI does not allow replacing a published version. If a release is broken,
fix it and release a new patch version.
