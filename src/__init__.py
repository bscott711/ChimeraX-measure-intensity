# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for registering commands,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1     # register_command called with BundleInfo and
    # CommandInfo instance instead of command name
    # (when api_version==0)

    # Override method
    @staticmethod
    def register_command(bi, ci, logger):
        from . import measure_commands
        measure_commands.register_distance_command(logger)
        measure_commands.register_intensity_command(logger)

        #from . import measure_distance, measure_intensity
        # measure_distance.register_command(logger)
        # measure_intensity.register_command(logger)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
