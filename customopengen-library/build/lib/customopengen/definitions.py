import pkg_resources


def templates_dir():
    """Directory where the templates are found (for internal use, mainly)"""
    return pkg_resources.resource_filename('customopengen', 'templates/')


def templates_subdir(subdir=None):
    """
    Directory where the templates are found and subfolder relative
    to that path(for internal use, mainly)
    """
    if subdir is None:
        return templates_dir()
    return pkg_resources.resource_filename('customopengen', 'templates/%s/' % subdir)


def original_icasadi_dir():
    """Directory where the original icasadi files are found (for internal use)"""
    return pkg_resources.resource_filename('customopengen', 'icasadi/')
