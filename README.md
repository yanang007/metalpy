<h1 align="center">
<img src="./branding/logo/metalpylogo.png" width="300" alt="METALpy">
</h1>

-------------

**METAL Essential Tools and Libraries (Python)** (**metalpy**),
is a collection of common python tools and libraries for SimPEG and other workflow in META Lab,
which currently includes:
1. **MEPA**: a general-purposed parallelization framework.
2. [**Mexin**](metalpy/mexin/README.md): a code injection framework for Python.
3. [**SCAB**](metalpy/scab/README.md): a collection of SimPEG related utilities and extensions.

Installation
------------
Metalpy can be installed using _pip_:

```console
pip install metalpy
```

<details><summary><b>Notes on installing with pip</b></summary>
<p>

metalpy includes SCAB, an extension to SimPEG,
whose dependencies will *not* be installed when running `pip` directly like this:

```console
pip install metalpy
```

Other supported commands are listed next:
```console
pip install "metalpy[scab]"      # Install requirements for SCAB module
pip install "metalpy[tests]"     # Install requirements for tests
pip install "metalpy[docs]"      # Install requirements for doc generation
pip install "metalpy[complete]"  # Install all requirements
pip install "metalpy[dev]"       # Install requirements for development
```

</p>
</details>
