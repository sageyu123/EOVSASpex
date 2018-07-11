# M. Szydlarski [mikolaj.szydlarski@astro.uio.no] - SolarAlma' 2017

def import_modules(modules_folder, verbose=True):
    """
    Return dictionry with modules hadnlers

    Modules will be autmatically read from modules folder
    in which module file name match folder name

      modules_folder : string (folder name)
      verbose : bool (print debug information)
    """
    # from IPython import embed
    import os
    from importlib.machinery import SourceFileLoader
    modules = dict()

    for root, dirs, _ in os.walk(modules_folder):
        for module_name in dirs:
            if '__pycache__' in module_name:
                break  # ignore tmp folders
            file_name = root + module_name + '/' + module_name + '.py'
            if verbose:
                print('   [~] Importing name: ', file_name)
            loader = SourceFileLoader(module_name, file_name)
            # embed()
            try:
                handle = loader.load_module(module_name)
                modules[module_name] = handle
            except ImportError as err:
                print('')
                print('   [!] Failed to load Viewer module: %s' % file_name)
                print('   [!] Import error msg: ')
                print('   [!] ' + err.msg)
                print('')
                # raise [!] Do not raise, just print error!
            pass
        print('')
    return modules
