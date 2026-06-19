def test_package_imports_and_has_version():
    import comparison

    assert isinstance(comparison.__version__, str)
    assert comparison.__version__
