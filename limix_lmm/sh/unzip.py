def unzip(filepath):
    import zipfile

    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(".")
