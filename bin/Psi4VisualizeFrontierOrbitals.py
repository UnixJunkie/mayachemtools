#!/usr/bin/env python
#
# File: Psi4VisualizeFrontierOrbitals.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2022 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using Psi4, an
# open source quantum chemistry software package, and RDKit, an open
# source toolkit for cheminformatics developed by Greg Landrum.
#
# This file is part of MayaChemTools.
#
# MayaChemTools is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# MayaChemTools is distributed in the hope that it will be useful, but without
# any warranty; without even the implied warranty of merchantability of fitness
# for a particular purpose.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MayaChemTools; if not, see <http://www.gnu.org/licenses/> or
# write to the Free Software Foundation Inc., 59 Temple Place, Suite 330,
# Boston, MA, 02111-1307, USA.
#

from __future__ import print_function

# Add local python path to the global path and import standard library modules...
import os
import sys;  sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), "..", "lib", "Python"))
import time
import re
import glob
import shutil
import multiprocessing as mp

# Psi4 imports...
if (hasattr(shutil, 'which') and shutil.which("psi4") is None):
    sys.stderr.write("\nWarning: Failed to find 'psi4' in your PATH indicating potential issues with your\n")
    sys.stderr.write("Psi4 environment. The 'import psi4' directive in the global scope of the script\n")
    sys.stderr.write("interferes with the multiprocessing functionality. It is imported later in the\n")
    sys.stderr.write("local scope during the execution of the script and may fail. Check/update your\n")
    sys.stderr.write("Psi4 environment and try again.\n\n")

# RDKit imports...
try:
    from rdkit import rdBase
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import RDKit module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your RDKit environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
try:
    from docopt import docopt
    import MiscUtil
    import Psi4Util
    import RDKitUtil
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import MayaChemTools module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your MayaChemTools environment and try again.\n\n")
    sys.exit(1)

ScriptName = os.path.basename(sys.argv[0])
Options = {}
OptionsInfo = {}

def main():
    """Start execution of the script."""
    
    MiscUtil.PrintInfo("\n%s (Psi4: Imported later; RDKit v%s; MayaChemTools v%s; %s): Starting...\n" % (ScriptName, rdBase.rdkitVersion, MiscUtil.GetMayaChemToolsVersion(), time.asctime()))
    
    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()
    
    # Retrieve command line arguments and options...
    RetrieveOptions()
    
    # Process and validate command line arguments and options...
    ProcessOptions()
    
    # Perform actions required by the script...
    GenerateAndVisualizeFrontierOrbitals()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def GenerateAndVisualizeFrontierOrbitals():
    """Generate and visualize frontier molecular orbitals."""
    
    if OptionsInfo["GenerateCubeFiles"]:
        GenerateFrontierOrbitals()
        
    if OptionsInfo["VisualizeCubeFiles"]:
        VisualizeFrontierOrbitals()

def GenerateFrontierOrbitals():
    """Generate cube files for frontier orbitals."""
    
    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["InfilePath"], **OptionsInfo["InfileParams"])
    
    MolCount, ValidMolCount, CubeFilesFailedCount = ProcessMolecules(Mols)
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during generation of cube files: %d" % CubeFilesFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + CubeFilesFailedCount))

def ProcessMolecules(Mols):
    """Process and generate frontier orbital cube files for molecules."""

    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols)

def ProcessMoleculesUsingSingleProcess(Mols):
    """Process and generate frontier orbitals cube files for molecules using a single process."""
    
    # Intialize Psi4...
    MiscUtil.PrintInfo("\nInitializing Psi4...")
    Psi4Handle = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = True, PrintHeader = True)
    OptionsInfo["psi4"] = Psi4Handle

    MiscUtil.PrintInfo("\nGenerating cube files for frontier orbitals...")
    
    (MolCount, ValidMolCount, CubeFilesFailedCount) = [0] * 3
    for Mol in Mols:
        MolCount += 1
        
        if not CheckAndValidateMolecule(Mol, MolCount):
            continue
        
        # Setup a Psi4 molecule...
        Psi4Mol = SetupPsi4Mol(Psi4Handle, Mol, MolCount)
        if Psi4Mol is None:
            continue
        
        ValidMolCount += 1
        
        CalcStatus = GenerateMolCubeFiles(Psi4Handle, Psi4Mol, Mol, MolCount)

        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Failed to generate cube files for molecule %s" % RDKitUtil.GetMolName(Mol, MolCount))
            
            CubeFilesFailedCount += 1
            continue
        
    return (MolCount, ValidMolCount, CubeFilesFailedCount)
    
def ProcessMoleculesUsingMultipleProcesses(Mols):
    """Process and generate frontier orbitals cube files for molecules using  multiple processes."""

    MiscUtil.PrintInfo("\nGenerating frontier orbitals using multiprocessing...")
    
    MPParams = OptionsInfo["MPParams"]
    
    # Setup data for initializing a worker process...
    InitializeWorkerProcessArgs = (MiscUtil.ObjectToBase64EncodedString(Options), MiscUtil.ObjectToBase64EncodedString(OptionsInfo))
    
    # Setup a encoded mols data iterable for a worker process...
    WorkerProcessDataIterable = RDKitUtil.GenerateBase64EncodedMolStrings(Mols)
    
    # Setup process pool along with data initialization for each process...
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("\nConfiguring multiprocessing using %s method..." % ("mp.Pool.imap()" if re.match("^Lazy$", MPParams["InputDataMode"], re.I) else "mp.Pool.map()"))
        MiscUtil.PrintInfo("NumProcesses: %s; InputDataMode: %s; ChunkSize: %s\n" % (MPParams["NumProcesses"], MPParams["InputDataMode"], ("automatic" if MPParams["ChunkSize"] is None else MPParams["ChunkSize"])))
    
    ProcessPool = mp.Pool(MPParams["NumProcesses"], InitializeWorkerProcess, InitializeWorkerProcessArgs)
    
    # Start processing...
    if re.match("^Lazy$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.imap(WorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    elif re.match("^InMemory$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.map(WorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    else:
        MiscUtil.PrintError("The value, %s, specified for \"--inputDataMode\" is not supported." % (MPParams["InputDataMode"]))

    # Print out Psi4 version in the main process...
    MiscUtil.PrintInfo("\nInitializing Psi4...\n")
    Psi4Handle  = Psi4Util.InitializePsi4(PrintVersion = True, PrintHeader = False)
    OptionsInfo["psi4"] = Psi4Handle
    
    (MolCount, ValidMolCount, CubeFilesFailedCount) = [0] * 3
    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, CalcStatus = Result
        
        if EncodedMol is None:
            continue
        
        ValidMolCount += 1

        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Failed to generate cube files for molecule %s" % RDKitUtil.GetMolName(Mol, MolCount))
            
            CubeFilesFailedCount += 1
            continue
        
    return (MolCount, ValidMolCount, CubeFilesFailedCount)

def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""
    
    global Options, OptionsInfo
    
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())
    
    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Psi4 is initialized in the worker process to avoid creation of redundant Psi4
    # output files for each process...
    OptionsInfo["Psi4Initialized"]  = False

def InitializePsi4ForWorkerProcess():
    """Initialize Psi4 for a worker process."""
    
    if OptionsInfo["Psi4Initialized"]:
        return

    OptionsInfo["Psi4Initialized"] = True
    
    # Update output file...
    OptionsInfo["Psi4RunParams"]["OutputFile"] = Psi4Util.UpdatePsi4OutputFileUsingPID(OptionsInfo["Psi4RunParams"]["OutputFile"], os.getpid())
    
    # Intialize Psi4...
    OptionsInfo["psi4"] = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = False, PrintHeader = True)
    
def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""
    
    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolIndex, EncodedMol = EncodedMolInfo
    
    CalcStatus = False
    
    if EncodedMol is None:
        return [MolIndex, None, CalcStatus]
    
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    MolCount = MolIndex + 1
    
    if not CheckAndValidateMolecule(Mol, MolCount):
        return [MolIndex, None, CalcStatus]
    
    # Setup a Psi4 molecule...
    Psi4Mol = SetupPsi4Mol(OptionsInfo["psi4"], Mol, MolCount)
    if Psi4Mol is None:
        return [MolIndex, None, CalcStatus]
    
    CalcStatus = GenerateMolCubeFiles(OptionsInfo["psi4"], Psi4Mol, Mol, MolCount)
    
    return [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CalcStatus]

def GenerateMolCubeFiles(Psi4Handle, Psi4Mol, Mol, MolNum = None):
    """Generate cube files for frontier orbitals."""
    
    #  Setup reference wave function...
    Reference = SetupReferenceWavefunction(Mol)
    Psi4Handle.set_options({'Reference': Reference})
    
    # Setup method name and basis set...
    MethodName, BasisSet = SetupMethodNameAndBasisSet(Mol)
    
    # Calculate single point energy to setup a wavefunction...
    Status, Energy, WaveFunction = Psi4Util.CalculateSinglePointEnergy(Psi4Handle, Psi4Mol, MethodName, BasisSet, ReturnWaveFunction = True, Quiet = OptionsInfo["QuietMode"])

    if not Status:
        PerformPsi4Cleanup(Psi4Handle)
        return (False)
    
    # Generate cube files...
    Status = GenerateCubeFilesForFrontierOrbitals(Psi4Handle, WaveFunction, Mol, MolNum)
    
    # Clean up
    PerformPsi4Cleanup(Psi4Handle)

    return (True)

def GenerateCubeFilesForFrontierOrbitals(Psi4Handle, WaveFunction, Mol, MolNum):
    """Generate cube files for frontier orbitals."""
    
    # Setup a temporary local directory to generate cube files...
    Status, CubeFilesDir, CubeFilesDirPath = SetupMolCubeFilesDir(Mol, MolNum)
    if not Status:
        return False
    
    # Generate cube files using psi4.cubeprop...
    Status, Psi4CubeFiles = GenerateCubeFilesUsingCubeprop(Psi4Handle, WaveFunction, Mol, MolNum, CubeFilesDir, CubeFilesDirPath)
    if not Status:
        return False

    # Copy and rename cube files...
    if not MoveAndRenameCubeFiles(Mol, MolNum, Psi4CubeFiles):
        return False

    # Delete temporary local directory...
    if os.path.isdir(CubeFilesDir):
        shutil.rmtree(CubeFilesDir)
    
    if not GenerateMolFile(Mol, MolNum):
        return False
    
    return True

def SetupMolCubeFilesDir(Mol, MolNum):
    """Setup a directory for generating cube files for a molecule."""
    
    MolPrefix = SetupMolPrefix(Mol, MolNum)

    CubeFilesDir = "%s_CubeFiles" % (MolPrefix)
    CubeFilesDirPath = os.path.join(os.getcwd(), CubeFilesDir)
    try:
        if os.path.isdir(CubeFilesDir):
            shutil.rmtree(CubeFilesDir)
        os.mkdir(CubeFilesDir)
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to create cube files: %s\n" % ErrMsg)
            MiscUtil.PrintWarning("Ignoring molecule: %s" % RDKitUtil.GetMolName(Mol, MolNum))
        return (False, None, None)
        
    return (True, CubeFilesDir, CubeFilesDirPath)

def GenerateCubeFilesUsingCubeprop(Psi4Handle, WaveFunction, Mol, MolNum, CubeFilesDir, CubeFilesDirPath):
    """Generate cube files using cubeprop."""
    
    Psi4CubeFilesParams = OptionsInfo["Psi4CubeFilesParams"]

    # Generate cube files...
    GridSpacing = Psi4CubeFilesParams["GridSpacing"]
    GridOverage = Psi4CubeFilesParams["GridOverage"]
    IsoContourThreshold = Psi4CubeFilesParams["IsoContourThreshold"]
    try:
        Psi4Handle.set_options({"CUBEPROP_TASKS": ['FRONTIER_ORBITALS'],
                                "CUBIC_GRID_SPACING": [GridSpacing, GridSpacing, GridSpacing],
                                "CUBIC_GRID_OVERAGE": [GridOverage, GridOverage, GridOverage],
                                "CUBEPROP_FILEPATH": CubeFilesDirPath,
                                "CUBEPROP_ISOCONTOUR_THRESHOLD": IsoContourThreshold})
        Psi4Handle.cubeprop(WaveFunction)
    except Exception as ErrMsg:
        shutil.rmtree(CubeFilesDir)
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to create cube files: %s\n" % ErrMsg)
            MiscUtil.PrintWarning("Ignoring molecule: %s" % RDKitUtil.GetMolName(Mol, MolNum))
        return (False, None)

    # Collect cube files...
    Psi4CubeFiles = glob.glob(os.path.join(CubeFilesDir, "Psi*.cube"))
    if len(Psi4CubeFiles) == 0:
        shutil.rmtree(CubeFilesDir)
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to create cube files for frontier orbitals...")
            MiscUtil.PrintWarning("Ignoring molecule: %s" % RDKitUtil.GetMolName(Mol, MolNum))
        return (False, None)
    
    return (True, Psi4CubeFiles)

def MoveAndRenameCubeFiles(Mol, MolNum, Psi4CubeFiles):
    """Move and rename cube files."""
    
    for Psi4CubeFileName in Psi4CubeFiles:
        try:
            NewCubeFileName = SetupMolCubeFileName(Mol, MolNum, Psi4CubeFileName)
            shutil.move(Psi4CubeFileName, NewCubeFileName)
        except Exception as ErrMsg:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Failed to move cube file: %s\n" % ErrMsg)
                MiscUtil.PrintWarning("Ignoring molecule: %s" % RDKitUtil.GetMolName(Mol, MolNum))
            return False
    
    return True

def GenerateMolFile(Mol, MolNum):
    """Generate a SD file for molecules corresponding to cube files."""

    Outfile = SetupMolFileName(Mol, MolNum)
    
    OutfileParams = {}
    Writer = RDKitUtil.MoleculesWriter(Outfile, **OutfileParams)
    if Writer is None:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintError("Failed to setup a writer for output fie %s " % Outfile)
        return False
    
    Writer.write(Mol)

    if Writer is not None:
        Writer.close()
    
    return True

def VisualizeFrontierOrbitals():
    """Visualize cube files for frontier orbitals."""
    
    MiscUtil.PrintInfo("\nVisualizing cube files for frontier orbitals...")

    if not OptionsInfo["GenerateCubeFiles"]:
        MiscUtil.PrintInfo("\nInitializing Psi4...\n")
        Psi4Handle  = Psi4Util.InitializePsi4(PrintVersion = True, PrintHeader = False)
        OptionsInfo["psi4"] = Psi4Handle
    
    # Setup a molecule reader...
    MiscUtil.PrintInfo("Processing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["InfilePath"], **OptionsInfo["InfileParams"])
    
    # Setup for writing a PyMOL PML file...
    Outfile = OptionsInfo["Outfile"]
    OutFH = open(Outfile, "w")
    if OutFH is None:
        MiscUtil.PrintError("Failed to open output fie %s " % Outfile)
    MiscUtil.PrintInfo("\nGenerating file %s..." % Outfile)

    # Setup header...
    WritePMLHeader(OutFH, ScriptName)
    WritePyMOLParameters(OutFH)

    # Process cube files for molecules...
    FirstMol = True
    (MolCount, ValidMolCount, CubeFilesMissingCount) = [0] * 3
    for Mol in Mols:
        MolCount += 1
        
        if Mol is None:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintInfo("\nProcessing molecule number %s..." % MolCount)
            continue
        ValidMolCount += 1
        
        MolName = RDKitUtil.GetMolName(Mol, MolCount)
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("\nProcessing molecule %s..." % MolName)
        
        # Retrieve cube files...
        CubeFilesStatus, CubeFilesInfo = RetrieveMolCubeFilesInfo(Mol, MolCount)
        if not CubeFilesStatus:
            CubeFilesMissingCount += 1
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Ignoring molecule with missing cube file...\n")
            continue
        
        # Setup PyMOL object names..
        PyMOLObjectNames = SetupPyMOLObjectNames(Mol, MolCount, CubeFilesInfo)

        # Setup molecule view...
        WriteMolView(OutFH, CubeFilesInfo, PyMOLObjectNames)
        
        # Setup orbital views...
        FirstOrbital = True
        for OrbitalID in CubeFilesInfo["OrbitalIDs"]:
            FirstSpin = True
            for SpinID in CubeFilesInfo["SpinIDs"][OrbitalID]:
                WriteOrbitalSpinView(OutFH, CubeFilesInfo, PyMOLObjectNames, OrbitalID, SpinID, FirstSpin)
                FirstSpin = False
        
            # Setup orbital level group...
            Enable, Action = [False, "open"]
            if FirstOrbital:
                FirstOrbital = False
                Enable, Action = [True, "open"]
            GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Orbitals"][OrbitalID]["OrbitalGroup"], PyMOLObjectNames["Orbitals"][OrbitalID]["OrbitalGroupMembers"], Enable, Action)

        # Setup mol level group...
        Enable, Action = [False, "close"]
        if FirstMol:
            FirstMol = False
            Enable, Action = [True, "open"]
        GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["MolGroup"], PyMOLObjectNames["MolGroupMembers"], Enable, Action)
    
    OutFH.close()
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules with missing cube files: %d" % CubeFilesMissingCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + CubeFilesMissingCount))

def WritePMLHeader(OutFH, ScriptName):
    """Write out PML header."""

    HeaderInfo = """\
#
# This file is automatically generated by the following Psi4 script available in
# MayaChemTools: %s
#
cmd.reinitialize() """ % (ScriptName)
    
    OutFH.write("%s\n" % HeaderInfo)

def WritePyMOLParameters(OutFH):
    """Write out PyMOL global parameters."""

    OutFH.write("""\n""\n"Setting up PyMOL gobal parameters..."\n""\n""")
    PMLCmds = []
    PMLCmds.append("""cmd.set("mesh_quality", %s)""" % (OptionsInfo["PyMOLViewParams"]["MeshQuality"]))
    PMLCmds.append("""cmd.set("mesh_width", %.2f)""" % (OptionsInfo["PyMOLViewParams"]["MeshWidth"]))
    
    PMLCmds.append("""cmd.set("surface_quality", %s)""" % (OptionsInfo["PyMOLViewParams"]["SurfaceQuality"]))
    PMLCmds.append("""cmd.set("transparency", %.2f)""" % (OptionsInfo["PyMOLViewParams"]["SurfaceTransparency"]))
    PML = "\n".join(PMLCmds)
    OutFH.write("%s\n" % PML)

    if OptionsInfo["PyMOLViewParams"]["SetupGlobalVolumeColorRamp"]:
        PyMOLViewParams = OptionsInfo["PyMOLViewParams"]
        GenerateAndWritePMLForVolumeColorRamp(OutFH, PyMOLViewParams["GlobalVolumeColorRampName"], PyMOLViewParams["ContourLevel1"], PyMOLViewParams["ContourColor1"], PyMOLViewParams["ContourLevel2"], PyMOLViewParams["ContourColor2"], PyMOLViewParams["VolumeContourWindowFactor"], PyMOLViewParams["VolumeColorRampOpacity"])

def WriteMolView(OutFH, CubeFilesInfo, PyMOLObjectNames):
    """Write out PML for viewing molecules."""

    MolName = CubeFilesInfo["MolName"]
    MolFile = CubeFilesInfo["MolFile"]
    
    OutFH.write("""\n""\n"Loading %s and setting up view for molecule..."\n""\n""" % MolFile)
    
    PyMOLViewParams = OptionsInfo["PyMOLViewParams"]
    PMLCmds = []
    
    # Molecule...
    Name = PyMOLObjectNames["Mol"]
    PMLCmds.append("""cmd.load("%s", "%s")""" % (MolFile, Name))
    PMLCmds.append("""cmd.hide("everything", "%s")""" % (Name))
    PMLCmds.append("""util.cbag("%s", _self = cmd)""" % (Name))
    PMLCmds.append("""cmd.show("sticks", "%s")""" % (Name))
    PMLCmds.append("""cmd.set("stick_radius", %s, "%s")""" % (PyMOLViewParams["DisplayStickRadius"], Name))
    if PyMOLViewParams["HideHydrogens"]:
        PMLCmds.append("""cmd.hide("(%s and hydro)")""" % (Name))
    if re.match("^Sticks$", PyMOLViewParams["DisplayMolecule"]):
        PMLCmds.append("""cmd.enable("%s")""" % (Name))
    else:
        PMLCmds.append("""cmd.disable("%s")""" % (Name))

    # Molecule ball and stick...
    Name = PyMOLObjectNames["MolBallAndStick"]
    PMLCmds.append("""cmd.load("%s", "%s")""" % (MolFile, Name))
    PMLCmds.append("""cmd.hide("everything", "%s")""" % (Name))
    PMLCmds.append("""util.cbag("%s", _self = cmd)""" % (Name))
    PMLCmds.append("""cmd.show("sphere", "%s")""" % (Name))
    PMLCmds.append("""cmd.show("sticks", "%s")""" % (Name))
    PMLCmds.append("""cmd.set("sphere_scale", %s, "%s")""" % (PyMOLViewParams["DisplaySphereScale"], Name))
    PMLCmds.append("""cmd.set("stick_radius", %s, "%s")""" % (PyMOLViewParams["DisplayStickRadius"], Name))
    if PyMOLViewParams["HideHydrogens"]:
        PMLCmds.append("""cmd.hide("(%s and hydro)")""" % (Name))
    if re.match("^BallAndStick$", PyMOLViewParams["DisplayMolecule"]):
        PMLCmds.append("""cmd.enable("%s")""" % (Name))
    else:
        PMLCmds.append("""cmd.disable("%s")""" % (Name))
    
    PML = "\n".join(PMLCmds)
    OutFH.write("%s\n" % PML)
    
    # MolAlone group...
    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["MolAloneGroup"], PyMOLObjectNames["MolAloneGroupMembers"], True, "open")

def WriteOrbitalSpinView(OutFH, CubeFilesInfo, PyMOLObjectNames, OrbitalID, SpinID, FirstSpin):
    """Write out PML for viewing spins in an orbital."""

    OutFH.write("""\n""\n"Setting up views for spin %s in orbital %s..."\n""\n""" % (SpinID, OrbitalID))
    
    # Cube...
    SpinCubeFile = CubeFilesInfo["FileName"][OrbitalID][SpinID]
    SpinCubeName = PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinCube"]
    PMLCmds = []
    PMLCmds.append("""cmd.load("%s", "%s")""" % (SpinCubeFile, SpinCubeName))
    PMLCmds.append("""cmd.disable("%s")""" % (SpinCubeName))
    PMLCmds.append("")
    PML = "\n".join(PMLCmds)
    OutFH.write("%s\n" % PML)

    ContourLevel1 = CubeFilesInfo["ContourLevel1"][OrbitalID][SpinID]
    ContourLevel2 = CubeFilesInfo["ContourLevel2"][OrbitalID][SpinID]
    
    ContourColor1 = CubeFilesInfo["ContourColor1"][OrbitalID][SpinID]
    ContourColor2 = CubeFilesInfo["ContourColor2"][OrbitalID][SpinID]
    
    ContourWindowFactor = CubeFilesInfo["ContourWindowFactor"][OrbitalID][SpinID]
    ContourColorOpacity = CubeFilesInfo["ContourColorOpacity"][OrbitalID][SpinID]

    SetupLocalVolumeColorRamp = CubeFilesInfo["SetupLocalVolumeColorRamp"][OrbitalID][SpinID]
    VolumeColorRampName = CubeFilesInfo["VolumeColorRampName"][OrbitalID][SpinID]
    
    # Volume ramp...
    if SetupLocalVolumeColorRamp:
        GenerateAndWritePMLForVolumeColorRamp(OutFH, VolumeColorRampName, ContourLevel1, ContourColor1, ContourLevel2, ContourColor2, ContourWindowFactor, ContourColorOpacity)
    
    # Volume...
    SpinVolumeName = PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinVolume"]
    PMLCmds = []
    PMLCmds.append("""cmd.volume("%s", "%s", "%s")""" % (SpinVolumeName, SpinCubeName, VolumeColorRampName))
    PMLCmds.append("""cmd.disable("%s")""" % (SpinVolumeName))
    PMLCmds.append("")
    PML = "\n".join(PMLCmds)
    OutFH.write("%s\n" % PML)
    
    # Mesh...
    SpinMeshName = PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshNegative"]
    PMLCmds = []
    PMLCmds.append("""cmd.isomesh("%s", "%s", "%s")""" % (SpinMeshName, SpinCubeName, ContourLevel1))
    PMLCmds.append("""util.color_deep("%s", "%s")""" % (ContourColor1, SpinMeshName))
    PMLCmds.append("""cmd.enable("%s")""" % (SpinMeshName))
    PMLCmds.append("")
    
    SpinMeshName = PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshPositive"]
    PMLCmds.append("""cmd.isomesh("%s", "%s", "%s")""" % (SpinMeshName, SpinCubeName, ContourLevel2))
    PMLCmds.append("""util.color_deep("%s", "%s")""" % (ContourColor2, SpinMeshName))
    PMLCmds.append("""cmd.enable("%s")""" % (SpinMeshName))
    PMLCmds.append("")
    PML = "\n".join(PMLCmds)
    OutFH.write("%s\n" % PML)

    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshGroup"], PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshGroupMembers"], False, "close")

    # Surface...
    SpinSurfaceName = PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceNegative"]
    PMLCmds = []
    PMLCmds.append("""cmd.isosurface("%s", "%s", "%s")""" % (SpinSurfaceName, SpinCubeName, ContourLevel1))
    PMLCmds.append("""util.color_deep("%s", "%s")""" % (ContourColor1, SpinSurfaceName))
    PMLCmds.append("""cmd.enable("%s")""" % (SpinSurfaceName))
    PMLCmds.append("")
    
    SpinSurfaceName = PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfacePositive"]
    PMLCmds.append("""cmd.isosurface("%s", "%s", "%s")""" % (SpinSurfaceName, SpinCubeName, ContourLevel2))
    PMLCmds.append("""util.color_deep("%s", "%s")""" % (ContourColor2, SpinSurfaceName))
    PMLCmds.append("""cmd.enable("%s")""" % (SpinSurfaceName))
    PMLCmds.append("")
    PML = "\n".join(PMLCmds)
    OutFH.write("%s\n" % PML)
    
    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceGroup"], PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceGroupMembers"], True, "close")

    Enable = True if FirstSpin else False
    Action = "open" if FirstSpin else "close"
    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroup"], PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroupMembers"], Enable, Action)
    
def GenerateAndWritePMLForVolumeColorRamp(OutFH, ColorRampName, ContourLevel1, ContourColor1, ContourLevel2, ContourColor2, ContourWindowFactor, ContourColorOpacity):
    """Write out PML for volume color ramp."""

    RampWindowOffset1 = abs(ContourLevel1 * ContourWindowFactor)
    RampWindowOffset2 = abs(ContourLevel2 * ContourWindowFactor)
    
    PML = """\
cmd.volume_ramp_new("%s", [%.4f, "%s", 0.00, %.4f, "%s", %s, %.4f, "%s", 0.00, %4f, "%s", 0.00, %.4f, "%s", %s, %4f, "%s", 0.00])""" % (ColorRampName, (ContourLevel1 - RampWindowOffset1), ContourColor1, ContourLevel1, ContourColor1, ContourColorOpacity, (ContourLevel1 + RampWindowOffset1), ContourColor1, (ContourLevel2 - RampWindowOffset2), ContourColor2, ContourLevel2, ContourColor2, ContourColorOpacity, (ContourLevel2 + RampWindowOffset2), ContourColor2)
    
    OutFH.write("%s\n" % PML)

def GenerateAndWritePMLForGroup(OutFH, GroupName, GroupMembersList, Enable, Action):
    """Generate and write PML for group."""
    
    OutFH.write("""\n""\n"Setting up group %s..."\n""\n""" % GroupName)
    
    PMLCmds = []
    
    GroupMembers = " ".join(GroupMembersList)
    PMLCmds.append("""cmd.group("%s", "%s")""" % (GroupName, GroupMembers))
    
    if Enable is not None:
        if Enable:
            PMLCmds.append("""cmd.enable("%s")""" % GroupName)
        else:
            PMLCmds.append("""cmd.disable("%s")""" % GroupName)
    
    if Action is not None:
        PMLCmds.append("""cmd.group("%s", action="%s")""" % (GroupName, Action))

    PML = "\n".join(PMLCmds)
    
    OutFH.write("%s\n" % PML)

def RetrieveMolCubeFilesInfo(Mol, MolNum):
    """Retrieve available cube files info for a molecule."""

    CubeFilesInfo = {}
    CubeFilesInfo["OrbitalIDs"] = []
    CubeFilesInfo["SpinIDs"] = {}
    CubeFilesInfo["FileName"] = {}

    CubeFilesInfo["IsocontourRangeMin"] = {}
    CubeFilesInfo["IsocontourRangeMax"] = {}
    
    CubeFilesInfo["ContourLevel1"] = {}
    CubeFilesInfo["ContourLevel2"] = {}
    CubeFilesInfo["ContourColor1"] = {}
    CubeFilesInfo["ContourColor2"] = {}
    
    CubeFilesInfo["ContourWindowFactor"] = {}
    CubeFilesInfo["ContourColorOpacity"] = {}
    
    CubeFilesInfo["VolumeColorRampName"] = {}
    CubeFilesInfo["SetupLocalVolumeColorRamp"] = {}
    
    # Setup cube mol info...
    CubeFilesInfo["MolPrefix"] = SetupMolPrefix(Mol, MolNum)
    CubeFilesInfo["MolName"] = SetupMolName(Mol, MolNum)
    CubeFilesInfo["MolFile"] = SetupMolFileName(Mol, MolNum)
    
    MolPrefix = CubeFilesInfo["MolPrefix"]
    CubeFilesCount = 0
    for OrbitalID in ["HOMO", "LUMO", "DOMO", "LVMO", "SOMO"]:
        CubeFiles = glob.glob("%s*%s.cube" % (MolPrefix, OrbitalID))
        if not len(CubeFiles):
            continue

        CubeFilesInfo["OrbitalIDs"].append(OrbitalID)
        CubeFilesInfo["SpinIDs"][OrbitalID] = []
        CubeFilesInfo["FileName"][OrbitalID] = {}
        
        CubeFilesInfo["IsocontourRangeMin"][OrbitalID] = {}
        CubeFilesInfo["IsocontourRangeMax"][OrbitalID] = {}
        
        CubeFilesInfo["ContourLevel1"][OrbitalID] = {}
        CubeFilesInfo["ContourLevel2"][OrbitalID] = {}
        CubeFilesInfo["ContourColor1"][OrbitalID] = {}
        CubeFilesInfo["ContourColor2"][OrbitalID] = {}
        
        CubeFilesInfo["ContourWindowFactor"][OrbitalID] = {}
        CubeFilesInfo["ContourColorOpacity"][OrbitalID] = {}
        
        CubeFilesInfo["VolumeColorRampName"][OrbitalID] = {}
        CubeFilesInfo["SetupLocalVolumeColorRamp"][OrbitalID] = {}
        
        AlphaSpinCount, BetaSpinCount, SpinCount, CurrentSpinCount = [0] * 4
        for CubeFile in CubeFiles:
            if re.match("%s_a_" % MolPrefix, CubeFile):
                SpinID = "Alpha"
                AlphaSpinCount += 1
                CurrentSpinCount = AlphaSpinCount
            elif re.match("%s_b_" % MolPrefix, CubeFile):
                SpinID = "Beta"
                BetaSpinCount += 1
                CurrentSpinCount = BetaSpinCount
            else:
                SpinID = "Spin"
                SpinCount += 1
                CurrentSpinCount = SpinCount

            # Set up a unique spin ID...
            if SpinID in CubeFilesInfo["FileName"][OrbitalID]:
                SpinID = "%s_%s" % (SpinID, CurrentSpinCount)
                
            CubeFilesCount += 1
            CubeFilesInfo["SpinIDs"][OrbitalID].append(SpinID)
            CubeFilesInfo["FileName"][OrbitalID][SpinID] = CubeFile
            
            IsocontourRangeMin, IsocontourRangeMax, ContourLevel1, ContourLevel2 = SetupIsocontourRangeAndContourLevels(CubeFile)
            
            CubeFilesInfo["IsocontourRangeMin"][OrbitalID][SpinID] = IsocontourRangeMin
            CubeFilesInfo["IsocontourRangeMax"][OrbitalID][SpinID] = IsocontourRangeMax
            
            CubeFilesInfo["ContourLevel1"][OrbitalID][SpinID] = ContourLevel1
            CubeFilesInfo["ContourLevel2"][OrbitalID][SpinID] = ContourLevel2
            CubeFilesInfo["ContourColor1"][OrbitalID][SpinID] = OptionsInfo["PyMOLViewParams"]["ContourColor1"]
            CubeFilesInfo["ContourColor2"][OrbitalID][SpinID] = OptionsInfo["PyMOLViewParams"]["ContourColor2"]
            
            CubeFilesInfo["ContourWindowFactor"][OrbitalID][SpinID] = OptionsInfo["PyMOLViewParams"]["VolumeContourWindowFactor"]
            CubeFilesInfo["ContourColorOpacity"][OrbitalID][SpinID] = OptionsInfo["PyMOLViewParams"]["VolumeColorRampOpacity"]
            
            VolumeColorRampName = SetupVolumeColorRampName(CubeFilesInfo)
            CubeFilesInfo["VolumeColorRampName"][OrbitalID][SpinID] = VolumeColorRampName
            CubeFilesInfo["SetupLocalVolumeColorRamp"][OrbitalID][SpinID] = False if OptionsInfo["PyMOLViewParams"]["SetupGlobalVolumeColorRamp"] else True
    
    Status = True if CubeFilesCount > 0 else False

    return (Status, CubeFilesInfo)

def SetupVolumeColorRampName(CubeFilesInfo):
    """Setup volume color ramp name and status."""
    
    PyMOLViewParams = OptionsInfo["PyMOLViewParams"]
    
    # Setup info for generating local volume color ramps for individual cube files...
    VolumeColorRampName = PyMOLViewParams["VolumeColorRamp"]
    if PyMOLViewParams["VolumeColorRampAuto"]:
        VolumeColorRampName = SetupDefaultVolumeRampName()
        if PyMOLViewParams["ContourLevel1Auto"] or PyMOLViewParams["ContourLevel2Auto"]:
            VolumeColorRampName = "%s_%s" % (CubeFilesInfo["MolPrefix"], VolumeColorRampName)
    
    return VolumeColorRampName

def SetupIsocontourRangeAndContourLevels(CubeFileName):
    """Setup isocontour range."""

    PyMOLViewParams = OptionsInfo["PyMOLViewParams"]
    
    # Setup isocontour range and contour levels...
    IsocontourRangeMin, IsocontourRangeMax = Psi4Util.RetrieveIsocontourRangeFromCubeFile(CubeFileName)

    DefaultIsocontourRange = 0.06
    if IsocontourRangeMin is None:
        IsocontourRangeMin = -DefaultIsocontourRange
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("Failed to retrieve isocontour range from the cube file. Setting min isocontour value to %s..." % (IsocontourRangeMin))
    
    if IsocontourRangeMax is None:
        IsocontourRangeMax = DefaultIsocontourRange
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("Failed to retrieve isocontour range from the cube file. Setting max isocontour value to %s..." % (IsocontourRangeMax))

    # Setup contour levels...
    ContourLevel = max(abs(IsocontourRangeMin), abs(IsocontourRangeMax)) * PyMOLViewParams["ContourLevelAutoAt"]
    if ContourLevel >= 0.01:
        ContourLevel = float("%.2f" % ContourLevel)
    elif ContourLevel >= 0.001:
        ContourLevel = float("%.3f" % ContourLevel)
    elif ContourLevel >= 0.0001:
        ContourLevel = float("%.4f" % ContourLevel)
    
    ContourLevel1 = -ContourLevel if PyMOLViewParams["ContourLevel1Auto"] else PyMOLViewParams["ContourLevel1"]
    ContourLevel2 = ContourLevel if PyMOLViewParams["ContourLevel2Auto"] else PyMOLViewParams["ContourLevel2"]
    
    if not OptionsInfo["QuietMode"]:
        if IsocontourRangeMin is not None and IsocontourRangeMax is not None:
            MiscUtil.PrintInfo("CubeFileName: %s; Isocontour range for %s percent of the density: %.4f to %.4f; ContourLevel1: %s; ContourLevel2: %s" % (CubeFileName, (100 * OptionsInfo["Psi4CubeFilesParams"]["IsoContourThreshold"]), IsocontourRangeMin, IsocontourRangeMax, ContourLevel1, ContourLevel2))

    return (IsocontourRangeMin, IsocontourRangeMax, ContourLevel1, ContourLevel2)

def SetupPyMOLObjectNames(Mol, MolNum, CubeFilesInfo):
    """Setup PyMOL object names."""
    
    PyMOLObjectNames = {}
    PyMOLObjectNames["Orbitals"] = {}
    PyMOLObjectNames["Spins"] = {}
    
    SetupPyMOLObjectNamesForMol(Mol, MolNum, CubeFilesInfo, PyMOLObjectNames)
    
    for OrbitalID in CubeFilesInfo["OrbitalIDs"]:
        SetupPyMOLObjectNamesForOrbital(Mol, MolNum, CubeFilesInfo, PyMOLObjectNames, OrbitalID)
        for SpinID in CubeFilesInfo["SpinIDs"][OrbitalID]:
            SetupPyMOLObjectNamesForSpin(Mol, MolNum, CubeFilesInfo, PyMOLObjectNames, OrbitalID, SpinID)
        
    return PyMOLObjectNames

def SetupPyMOLObjectNamesForMol(Mol, MolNum, CubeFilesInfo, PyMOLObjectNames):
    """Setup groups and objects for molecule."""

    MolFileRoot = CubeFilesInfo["MolPrefix"]
    
    MolGroupName = "%s" % MolFileRoot
    PyMOLObjectNames["MolGroup"] = MolGroupName
    PyMOLObjectNames["MolGroupMembers"] = []

    # Molecule alone group...
    MolAloneGroupName = "%s.Molecule" % (MolGroupName)
    PyMOLObjectNames["MolAloneGroup"] = MolAloneGroupName
    PyMOLObjectNames["MolGroupMembers"].append(MolAloneGroupName)
    
    PyMOLObjectNames["MolAloneGroupMembers"] = []

    # Molecule...
    MolName = "%s.Molecule" % (MolAloneGroupName)
    PyMOLObjectNames["Mol"] = MolName
    PyMOLObjectNames["MolAloneGroupMembers"].append(MolName)
    
    # BallAndStick...
    MolBallAndStickName = "%s.BallAndStick" % (MolAloneGroupName)
    PyMOLObjectNames["MolBallAndStick"] = MolBallAndStickName
    PyMOLObjectNames["MolAloneGroupMembers"].append(MolBallAndStickName)
    
def SetupPyMOLObjectNamesForOrbital(Mol, MolNum, CubeFilesInfo, PyMOLObjectNames, OrbitalID):
    """Setup groups and objects for orbital."""

    PyMOLObjectNames["Orbitals"][OrbitalID] = {}
    PyMOLObjectNames["Spins"][OrbitalID] = {}

    MolGroupName = PyMOLObjectNames["MolGroup"]
    
    # Setup orbital group...
    OrbitalGroupName = "%s.%s" % (MolGroupName, OrbitalID)
    PyMOLObjectNames["Orbitals"][OrbitalID]["OrbitalGroup"] = OrbitalGroupName
    PyMOLObjectNames["MolGroupMembers"].append(OrbitalGroupName)
    PyMOLObjectNames["Orbitals"][OrbitalID]["OrbitalGroupMembers"] = []
 
def SetupPyMOLObjectNamesForSpin(Mol, MolNum, CubeFilesInfo, PyMOLObjectNames, OrbitalID, SpinID):
    """Setup groups and objects for spin."""

    PyMOLObjectNames["Spins"][OrbitalID][SpinID] = {}
    
    OrbitalGroupName = PyMOLObjectNames["Orbitals"][OrbitalID]["OrbitalGroup"]
    
    # Setup spin group at orbital level...
    SpinGroupName = "%s.%s" % (OrbitalGroupName, SpinID)
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroup"] = SpinGroupName
    PyMOLObjectNames["Orbitals"][OrbitalID]["OrbitalGroupMembers"].append(SpinGroupName)
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroupMembers"] = []

    # Cube...
    SpinCubeName = "%s.Cube" % SpinGroupName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinCube"] = SpinCubeName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroupMembers"].append(SpinCubeName)
    
    # Volume...
    SpinVolumeName = "%s.Volume" % SpinGroupName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinVolume"] = SpinVolumeName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroupMembers"].append(SpinVolumeName)
    
    # Mesh...
    SpinMeshGroupName = "%s.Mesh" % SpinGroupName
    SpinMeshNegativeName = "%s.Negative_Phase" % SpinMeshGroupName
    SpinMeshPositiveName = "%s.Positive_Phase" % SpinMeshGroupName
    
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshGroup"] = SpinMeshGroupName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroupMembers"].append(SpinMeshGroupName)
    
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshNegative"] = SpinMeshNegativeName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshPositive"] = SpinMeshPositiveName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshGroupMembers"] = []
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshGroupMembers"].append(SpinMeshNegativeName)
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinMeshGroupMembers"].append(SpinMeshPositiveName)
    
    # Surface...
    SpinSurfaceGroupName = "%s.Surface" % SpinGroupName
    SpinSurfaceNegativeName = "%s.Negative_Phase" % SpinSurfaceGroupName
    SpinSurfacePositiveName = "%s.Positive_Phase" % SpinSurfaceGroupName
    
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceGroup"] = SpinSurfaceGroupName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinGroupMembers"].append(SpinSurfaceGroupName)
    
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceNegative"] = SpinSurfaceNegativeName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfacePositive"] = SpinSurfacePositiveName
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceGroupMembers"] = []
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceGroupMembers"].append(SpinSurfaceNegativeName)
    PyMOLObjectNames["Spins"][OrbitalID][SpinID]["SpinSurfaceGroupMembers"].append(SpinSurfacePositiveName)
    
def SetupMolFileName(Mol, MolNum):
    """Setup SD file name for a molecule."""

    MolPrefix = SetupMolPrefix(Mol, MolNum)
    MolFileName = "%s.sdf" % MolPrefix

    return MolFileName
    
def SetupMolCubeFileName(Mol, MolNum, Psi4CubeFileName):
    """Setup cube file name for a molecule."""

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Psi4CubeFileName)
    CubeFileName = "%s.%s" % (FileName, FileExt)

    # Replace Psi by MolPrefix...
    MolPrefix = SetupMolPrefix(Mol, MolNum)
    CubeFileName = re.sub("^Psi", MolPrefix, CubeFileName)
    
    # Replace prime (') and dprime ('') in the cube file names..
    CubeFileName = re.sub("\"", "dblprime", CubeFileName)
    CubeFileName = re.sub("\'", "prime", CubeFileName)

    return CubeFileName
    
def SetupMolPrefix(Mol, MolNum):
    """Get molecule prefix for generating files and directories."""

    MolNamePrefix = ''
    if Mol.HasProp("_Name"):
        MolNamePrefix = re.sub("[^a-zA-Z0-9]", "_", Mol.GetProp("_Name"))
    
    MolNumPrefix = "Mol%s" % MolNum

    if OptionsInfo["UseMolNumPrefix"] and OptionsInfo["UseMolNamePrefix"]:
        MolPrefix = MolNumPrefix
        if len(MolNamePrefix):
            MolPrefix = "%s_%s" % (MolNumPrefix, MolNamePrefix)
    elif OptionsInfo["UseMolNamePrefix"]:
        MolPrefix = MolNamePrefix if len(MolNamePrefix) else MolNumPrefix
    elif OptionsInfo["UseMolNumPrefix"]:
        MolPrefix = MolNumPrefix
    
    return MolPrefix

def SetupMolName(Mol, MolNum):
    """Get molecule name."""

    # Test for creating PyMOL object name for molecule...
    MolNamePrefix = ''
    if Mol.HasProp("_Name"):
        MolName = re.sub("[^a-zA-Z0-9]", "_", Mol.GetProp("_Name"))
    else:
        MolName = "Mol%s" % MolNum
    
    return MolName

def SetupDefaultVolumeRampName():
    """Setup a default name for a volume ramp."""

    return "psi4_cube_mo"

def SetupPsi4Mol(Psi4Handle, Mol, MolCount = None):
    """Setup a Psi4 molecule object."""
    
    MolGeometry = RDKitUtil.GetPsi4XYZFormatString(Mol, NoCom = True, NoReorient = True)
    
    try:
        Psi4Mol = Psi4Handle.geometry(MolGeometry)
    except Exception as ErrMsg:
        Psi4Mol = None
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to create Psi4 molecule from geometry string: %s\n" % ErrMsg)
            MolName = RDKitUtil.GetMolName(Mol, MolCount)
            MiscUtil.PrintWarning("Ignoring molecule: %s" % MolName)
    
    return Psi4Mol

def PerformPsi4Cleanup(Psi4Handle):
    """Perform clean up."""
    
    # Clean up after Psi4 run ...
    Psi4Handle.core.clean()
    
    # Clean up any leftover scratch files...
    if OptionsInfo["MPMode"]:
        Psi4Util.RemoveScratchFiles(Psi4Handle, OptionsInfo["Psi4RunParams"]["OutputFile"])

def CheckAndValidateMolecule(Mol, MolCount = None):
    """Validate molecule for Psi4 calculations."""
    
    if Mol is None:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("\nProcessing molecule number %s..." % MolCount)
        return False
    
    MolName = RDKitUtil.GetMolName(Mol, MolCount)
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("\nProcessing molecule %s..." % MolName)
    
    if RDKitUtil.IsMolEmpty(Mol):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Ignoring empty molecule: %s\n" % MolName)
        return False
    
    if not RDKitUtil.ValidateElementSymbols(RDKitUtil.GetAtomSymbols(Mol)):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Ignoring molecule containing invalid element symbols: %s\n" % MolName)
        return False
    
    # Check for 3D flag...
    if not Mol.GetConformer().Is3D():
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("3D tag is not set for molecule: %s\n" % MolName)
    
    # Check for missing hydrogens...
    if RDKitUtil.AreHydrogensMissingInMolecule(Mol):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Missing hydrogens in molecule: %s\n" % MolName)
    
    return True

def SetupMethodNameAndBasisSet(Mol):
    """Setup method name and basis set."""
    
    MethodName = OptionsInfo["MethodName"]
    if OptionsInfo["MethodNameAuto"]:
        MethodName = "B3LYP"
    
    BasisSet = OptionsInfo["BasisSet"]
    if OptionsInfo["BasisSetAuto"]:
        BasisSet = "6-31+G**" if RDKitUtil.IsAtomSymbolPresentInMol(Mol, "S") else "6-31G**"
    
    return (MethodName, BasisSet)

def SetupReferenceWavefunction(Mol):
    """Setup reference wavefunction."""
    
    Reference = OptionsInfo["Reference"]
    if OptionsInfo["ReferenceAuto"]:
        Reference = 'UHF' if (RDKitUtil.GetSpinMultiplicity(Mol) > 1) else 'RHF'
    
    return Reference

def CheckAndSetupOutfilesDir():
    """Check and setup outfiles directory."""
    
    if OptionsInfo["GenerateCubeFiles"]:
        if os.path.isdir(OptionsInfo["OutfilesDir"]):
            if not Options["--overwrite"]:
                MiscUtil.PrintError("The outfiles directory, %s, corresponding to output file specified, %s, for option \"-o, --outfile\" already exist. Use option \"--overwrite\" or \"--ov\"  and try again to generate and visualize cube files...\n" % (OptionsInfo["OutfilesDir"], OptionsInfo["Outfile"]))
    
    if OptionsInfo["VisualizeCubeFiles"]:
        if not Options["--overwrite"]:
            if  os.path.exists(os.path.join(OptionsInfo["OutfilesDir"], OptionsInfo["Outfile"])):
                MiscUtil.PrintError("The outfile file specified, %s, for option \"-o, --outfile\" already exist in outfiles dir, %s. Use option \"--overwrite\" or \"--ov\" to overwrite and try again to generate and visualize cube files.\n" % (OptionsInfo["Outfile"], OptionsInfo["OutfilesDir"]))
                    
        if not OptionsInfo["GenerateCubeFiles"]:
            if not os.path.isdir(OptionsInfo["OutfilesDir"]):
                MiscUtil.PrintError("The outfiles directory, %s, corresponding to output file specified, %s, for option \"-o, --outfile\" doesn't exist. Use value, GenerateCubeFiles or Both, for option \"--mode\" and try again to generate and visualize cube files...\n" % (OptionsInfo["OutfilesDir"], OptionsInfo["Outfile"]))
            
            CubeFiles = glob.glob(os.path.join(OptionsInfo["OutfilesDir"], "*.cube"))
            if not len(CubeFiles):
                MiscUtil.PrintError("The outfiles directory, %s, contains no cube files, \"*.cube\". Use value, GenerateCubeFiles or Both, for option \"--mode\" and try again to generate and visualize cube...\n" % (OptionsInfo["OutfilesDir"]))
    
    OutfilesDir = OptionsInfo["OutfilesDir"]
    if OptionsInfo["GenerateCubeFiles"]:
        if not os.path.isdir(OutfilesDir):
            MiscUtil.PrintInfo("\nCreating directory %s..." % OutfilesDir)
            os.mkdir(OutfilesDir)

    MiscUtil.PrintInfo("\nChanging directory to %s..." % OutfilesDir)
    os.chdir(OutfilesDir)

def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")
    
    # Validate options...
    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    OptionsInfo["InfilePath"] = os.path.abspath(Options["--infile"])
    
    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters("--infileParams", Options["--infileParams"], InfileName = Options["--infile"], ParamsDefaultInfo = ParamsDefaultInfoOverride)

    Outfile = Options["--outfile"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Outfile)
    
    OptionsInfo["Outfile"] = Outfile
    OptionsInfo["OutfileRoot"] = FileName
    OptionsInfo["OutfileExt"] = FileExt
    
    OutfilesDir = Options["--outfilesDir"]
    if re.match("^auto$", OutfilesDir, re.I):
        OutfilesDir = "%s_FrontierOrbitals" % OptionsInfo["OutfileRoot"]
    OptionsInfo["OutfilesDir"] = OutfilesDir
    
    OptionsInfo["Overwrite"] = Options["--overwrite"]
    
    # Method, basis set, and reference wavefunction...
    OptionsInfo["BasisSet"] = Options["--basisSet"]
    OptionsInfo["BasisSetAuto"] = True if re.match("^auto$", Options["--basisSet"], re.I) else False
    
    OptionsInfo["MethodName"] = Options["--methodName"]
    OptionsInfo["MethodNameAuto"] = True if re.match("^auto$", Options["--methodName"], re.I) else False
    
    OptionsInfo["Reference"] = Options["--reference"]
    OptionsInfo["ReferenceAuto"] = True if re.match("^auto$", Options["--reference"], re.I) else False
    
    # Run, options, and cube files parameters...
    OptionsInfo["Psi4OptionsParams"] = Psi4Util.ProcessPsi4OptionsParameters("--psi4OptionsParams", Options["--psi4OptionsParams"])
    OptionsInfo["Psi4RunParams"] = Psi4Util.ProcessPsi4RunParameters("--psi4RunParams", Options["--psi4RunParams"], InfileName = OptionsInfo["Infile"])

    ParamsDefaultInfoOverride = {}
    OptionsInfo["Psi4CubeFilesParams"] = Psi4Util.ProcessPsi4CubeFilesParameters("--psi4CubeFilesParams", Options["--psi4CubeFilesParams"], ParamsDefaultInfo = ParamsDefaultInfoOverride)
    
    ParamsDefaultInfoOverride = {"ContourColor1": "red", "ContourColor2": "blue", "ContourLevel1": "auto", "ContourLevel2": "auto", "ContourLevelAutoAt": 0.75, "DisplayMolecule": "BallAndStick", "DisplaySphereScale": 0.2, "DisplayStickRadius": 0.1, "HideHydrogens": True,  "MeshWidth": 0.5, "MeshQuality": 2, "SurfaceQuality": 2, "SurfaceTransparency": 0.25, "VolumeColorRamp": "auto", "VolumeColorRampOpacity": 0.2, "VolumeContourWindowFactor": 0.05}
    OptionsInfo["PyMOLViewParams"] = MiscUtil.ProcessOptionPyMOLCubeFileViewParameters("--pymolViewParams", Options["--pymolViewParams"], ParamsDefaultInfo = ParamsDefaultInfoOverride)
    
    SetupGlobalVolumeColorRamp = False
    GlobalVolumeColorRampName = OptionsInfo["PyMOLViewParams"]["VolumeColorRamp"]
    if OptionsInfo["PyMOLViewParams"]["VolumeColorRampAuto"]:
        GlobalVolumeColorRampName = SetupDefaultVolumeRampName()
        if not (OptionsInfo["PyMOLViewParams"]["ContourLevel1Auto"] or OptionsInfo["PyMOLViewParams"]["ContourLevel2Auto"]):
            SetupGlobalVolumeColorRamp = True
    OptionsInfo["PyMOLViewParams"]["GlobalVolumeColorRampName"] = GlobalVolumeColorRampName
    OptionsInfo["PyMOLViewParams"]["SetupGlobalVolumeColorRamp"] = SetupGlobalVolumeColorRamp

    Mode = Options["--mode"]
    if re.match("^GenerateCubeFiles$", Mode, re.I):
        GenerateCubeFiles = True
        VisualizeCubeFiles = False
    elif re.match("^VisualizeCubeFiles$", Mode, re.I):
        GenerateCubeFiles = False
        VisualizeCubeFiles = True
    else:
        GenerateCubeFiles = True
        VisualizeCubeFiles = True
    OptionsInfo["Mode"] = Mode
    OptionsInfo["GenerateCubeFiles"] = GenerateCubeFiles
    OptionsInfo["VisualizeCubeFiles"] = VisualizeCubeFiles
    
    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])
    
    OutfilesMolPrefix = Options["--outfilesMolPrefix"]
    if re.match("^MolName$", OutfilesMolPrefix, re.I):
        UseMolNamePrefix = True
        UseMolNumPrefix = False
    elif re.match("^MolNum$", OutfilesMolPrefix, re.I):
        UseMolNamePrefix = False
        UseMolNumPrefix = True
    else:
        UseMolNamePrefix = True
        UseMolNumPrefix = True
    OptionsInfo["OutfilesMolPrefix"] = OutfilesMolPrefix
    OptionsInfo["UseMolNamePrefix"] = UseMolNamePrefix
    OptionsInfo["UseMolNumPrefix"] = UseMolNumPrefix
    
    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False
    
    CheckAndSetupOutfilesDir()

def RetrieveOptions():
    """Retrieve command line arguments and options."""
    
    # Get options...
    global Options
    Options = docopt(_docoptUsage_)
    
    # Set current working directory to the specified directory...
    WorkingDir = Options["--workingdir"]
    if WorkingDir:
        os.chdir(WorkingDir)
    
    # Handle examples option...
    if "--examples" in Options and Options["--examples"]:
        MiscUtil.PrintInfo(MiscUtil.GetExamplesTextFromDocOptText(_docoptUsage_))
        sys.exit(0)

def ValidateOptions():
    """Validate option values."""
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol")

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "pml")
    
    MiscUtil.ValidateOptionTextValue("--mode", Options["--mode"], "GenerateCubeFiles VisualizeCubeFiles Both")
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--outfilesMolPrefix", Options["--outfilesMolPrefix"], "MolNum MolName Both")
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
 
# Setup a usage string for docopt...
_docoptUsage_ = """
Psi4VisualizeFrontierOrbitals.py - Visualize frontier molecular orbitals

Usage:
    Psi4VisualizeFrontierOrbitals.py [--basisSet <text>] [--infileParams <Name,Value,...>] [--methodName <text>]
                                     [--mode <GenerateCubeFiles, VisualizeCubeFiles, Both>] [--mp <yes or no>] [--mpParams <Name, Value,...>]
                                     [--outfilesDir <text>] [--outfilesMolPrefix <MolNum, MolName, Both> ] [--overwrite]
                                     [--psi4CubeFilesParams <Name,Value,...>] [--psi4OptionsParams <Name,Value,...>]
                                     [--psi4RunParams <Name,Value,...>] [--pymolViewParams <Name,Value,...>] [--quiet <yes or no>]
                                     [--reference <text>] [-w <dir>] -i <infile> -o <outfile> 
    Psi4VisualizeFrontierOrbitals.py -h | --help | -e | --examples

Description:
    Generate and visualize frontier molecular orbitals for molecules in input
    file. A set of cube files, corresponding to frontier orbitals, is generated for 
    molecules. The cube files are used to create a PyMOL visualization file for
    viewing volumes, meshes, and surfaces representing frontier molecular
    orbitals. An option is available to skip the generation of new cube files
    and use existing cube files to visualize frontier molecular orbitals.
    
    A Psi4 XYZ format geometry string is automatically generated for each molecule
    in input file. It contains atom symbols and 3D coordinates for each atom in a
    molecule. In addition, the formal charge and spin multiplicity are present in the
    the geometry string. These values are either retrieved from molecule properties
    named 'FormalCharge' and 'SpinMultiplicty' or dynamically calculated for a
    molecule.
    
    A set of cube and SD output files is generated for each molecule in input file
    as shown below:
        
        Ouput dir: <OutfileRoot>_FrontierOrbitals or <OutfilesDir>
        
        Closed-shell systems:
        
        <MolIDPrefix>.sdf
        <MolIDPrefix>*HOMO.cube
        <MolIDPrefix>*LUMO.cube
        
        Open-shell systems:
        
        <MolIDPrefix>.sdf
        <MolIDPrefix>*DOMO.cube
        <MolIDPrefix>*LVMO.cube
        <MolIDPrefix>*SOMO.cube
        
        <MolIDPrefix>: Mol<Num>_<MolName>, Mol<Num>, or <MolName>
        
        Abbreviations:
        
        HOMO: Highest Occupied Molecular Orbital
        LUMO: Lowest Unoccupied Molecular Orbital
    
        DOMO: Double Occupied Molecular Orbital
        SOMO: Singly Occupied Molecular Orbital
        LVMO: Lowest Virtual (Unoccupied) Molecular Orbital
        
    In addition, a <OutfileRoot>.pml is generated containing  frontier molecular
    orbitals for all molecules in input file.
    
    The supported input file formats are: Mol (.mol), SD (.sdf, .sd)
     
    The supported output file formats are: PyMOL script file (.pml)
    
    A variety of PyMOL groups and objects are  created for visualization of frontier
    molecular orbitals as shown below:
        
        Closed-shell systems:
        
        <MoleculeID>
            .Molecule
                .Molecule
                .BallAndStick
            .HOMO
                .Alpha
                    .Cube
                    .Volume
                    .Mesh
                        .Negative_Phase
                        .Positive_Phase
                    .Surface
                        .Negative_Phase
                        .Positive_Phase
            .LUMO
                .Alpha
                    .Cube
                    .Volume
                    .Mesh
                        .Negative_Phase
                        .Positive_Phase
                    .Surface
                        .Negative_Phase
                        .Positive_Phase
        <MoleculeID>
            .Molecule
                ... ... ...
            .HOMO
                ... ... ...
            .LUMO
                ... ... ...
        
        Open-shell systems:
        
        <MoleculeID>
            .Molecule
                .Molecule
                .BallAndStick
            .DOMO
                .Alpha
                    .Cube
                    .Volume
                    .Mesh
                        .Negative_Phase
                        .Positive_Phase
                    .Surface
                        .Negative_Phase
                        .Positive_Phase
                .Beta
                    .Cube
                    .Volume
                    .Mesh
                        .Negative_Phase
                        .Positive_Phase
                    .Surface
                        .Negative_Phase
                        .Positive_Phase
            .LVMO
                .Alpha
                    .Cube
                    .Volume
                    .Mesh
                        .Negative_Phase
                        .Positive_Phase
                    .Surface
                        .Negative_Phase
                        .Positive_Phase
                .Beta
                    .Cube
                    .Volume
                    .Mesh
                        .Negative_Phase
                        .Positive_Phase
                    .Surface
                        .Negative_Phase
                        .Positive_Phase
            .SOMO
                .Alpha
                    .Cube
                    .Volume
                    .Mesh
                        .Negative_Phase
                        .Positive_Phase
                    .Surface
                        .Negative_Phase
                        .Positive_Phase
                .Alpha_<Num>
                    ... ... ...
        <MoleculeID>
            .Molecule
                ... ... ...
            .DOMO
                ... ... ...
            .LVMO
                ... ... ...
            .SOMO
                ... ... ...

Options:
    -b, --basisSet <text>  [default: auto]
        Basis set to use for calculating single point energy before generating
        cube files for frontier molecular orbitals. Default: 6-31+G** for sulfur
        containing molecules; Otherwise, 6-31G** [ Ref 150 ]. The specified 
        value must be a valid Psi4 basis set. No validation is performed.
        
        The following list shows a representative sample of basis sets available
        in Psi4:
            
            STO-3G, 6-31G, 6-31+G, 6-31++G, 6-31G*, 6-31+G*,  6-31++G*, 
            6-31G**, 6-31+G**, 6-31++G**, 6-311G, 6-311+G, 6-311++G,
            6-311G*, 6-311+G*, 6-311++G*, 6-311G**, 6-311+G**, 6-311++G**,
            cc-pVDZ, cc-pCVDZ, aug-cc-pVDZ, cc-pVDZ-DK, cc-pCVDZ-DK, def2-SVP,
            def2-SVPD, def2-TZVP, def2-TZVPD, def2-TZVPP, def2-TZVPPD
            
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name.
    --infileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for reading
        molecules from files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD, MOL: removeHydrogens,no,sanitize,yes,strictParsing,yes
            
    -m, --methodName <text>  [default: auto]
        Method to use for calculating single point energy before generating
        cube files for frontier molecular orbitals. Default: B3LYP [ Ref 150 ].
        The specified value must be a valid Psi4 method name. No validation is
        performed.
        
        The following list shows a representative sample of methods available
        in Psi4:
            
            B1LYP, B2PLYP, B2PLYP-D3BJ, B2PLYP-D3MBJ, B3LYP, B3LYP-D3BJ,
            B3LYP-D3MBJ, CAM-B3LYP, CAM-B3LYP-D3BJ, HF, HF-D3BJ,  HF3c, M05,
            M06, M06-2x, M06-HF, M06-L, MN12-L, MN15, MN15-D3BJ,PBE, PBE0,
            PBEH3c, PW6B95, PW6B95-D3BJ, WB97, WB97X, WB97X-D, WB97X-D3BJ
            
    --mode <GenerateCubeFiles, VisualizeCubeFiles, or Both>  [default: Both]
        Generate and visualize cube files for frontier molecular orbitals. The 
        'VisualizeCubes' value skips the generation of new cube files and uses
         existing cube files for visualization of molecular orbitals. Multiprocessing
         is not supported during 'VisualizeCubeFiles' value of '--mode' option.
    --mp <yes or no>  [default: no]
        Use multiprocessing.
         
        By default, input data is retrieved in a lazy manner via mp.Pool.imap()
        function employing lazy RDKit data iterable. This allows processing of
        arbitrary large data sets without any additional requirements memory.
        
        All input data may be optionally loaded into memory by mp.Pool.map()
        before starting worker processes in a process pool by setting the value
        of 'inputDataMode' to 'InMemory' in '--mpParams' option.
        
        A word to the wise: The default 'chunkSize' value of 1 during 'Lazy' input
        data mode may adversely impact the performance. The '--mpParams' section
        provides additional information to tune the value of 'chunkSize'.
    --mpParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs to configure
        multiprocessing.
        
        The supported parameter names along with their default and possible
        values are shown below:
        
            chunkSize, auto
            inputDataMode, Lazy   [ Possible values: InMemory or Lazy ]
            numProcesses, auto   [ Default: mp.cpu_count() ]
        
        These parameters are used by the following functions to configure and
        control the behavior of multiprocessing: mp.Pool(), mp.Pool.map(), and
        mp.Pool.imap().
        
        The chunkSize determines chunks of input data passed to each worker
        process in a process pool by mp.Pool.map() and mp.Pool.imap() functions.
        The default value of chunkSize is dependent on the value of 'inputDataMode'.
        
        The mp.Pool.map() function, invoked during 'InMemory' input data mode,
        automatically converts RDKit data iterable into a list, loads all data into
        memory, and calculates the default chunkSize using the following method
        as shown in its code:
        
            chunkSize, extra = divmod(len(dataIterable), len(numProcesses) * 4)
            if extra: chunkSize += 1
        
        For example, the default chunkSize will be 7 for a pool of 4 worker processes
        and 100 data items.
        
        The mp.Pool.imap() function, invoked during 'Lazy' input data mode, employs
        'lazy' RDKit data iterable to retrieve data as needed, without loading all the
        data into memory. Consequently, the size of input data is not known a priori.
        It's not possible to estimate an optimal value for the chunkSize. The default 
        chunkSize is set to 1.
        
        The default value for the chunkSize during 'Lazy' data mode may adversely
        impact the performance due to the overhead associated with exchanging
        small chunks of data. It is generally a good idea to explicitly set chunkSize to
        a larger value during 'Lazy' input data mode, based on the size of your input
        data and number of processes in the process pool.
        
        The mp.Pool.map() function waits for all worker processes to process all
        the data and return the results. The mp.Pool.imap() function, however,
        returns the the results obtained from worker processes as soon as the
        results become available for specified chunks of data.
        
        The order of data in the results returned by both mp.Pool.map() and 
        mp.Pool.imap() functions always corresponds to the input data.
    -o, --outfile <outfile>
        Output file name for PyMOL PML file. The PML output file, along with cube
        files, is generated in a local directory corresponding to '--outfilesDir'
        option.
    --outfilesDir <text>  [default: auto]
        Directory name containing PML and cube files. Default:
       <OutfileRoot>_FrontierOrbitals. This directory must be present during
        'VisualizeCubeFiles' value of '--mode' option.
    --outfilesMolPrefix <MolNum, MolName, Both>  [default: Both]
        Molecule prefix to use for the names of cube files. Possible values:
        MolNum, MolName, or Both. By default, both molecule number and name
        are used. The format of molecule prefix is as follows: MolNum - Mol<Num>;
        MolName - <MolName>, Both: Mol<Num>_<MolName>. Empty molecule names
        are ignored. Molecule numbers are used for empty molecule names.
    --overwrite
        Overwrite existing files.
    --psi4CubeFilesParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        Psi4 cube files.
        
        The supported parameter names along with their default and possible
        values are shown below:
             
            gridSpacing, 0.2, gridOverage, 4.0, isoContourThreshold, 0.85
             
        gridSpacing: Grid spacing for generating cube files. Units: Bohr. A higher
        value reduces the size of the cube files on the disk. This option corresponds
        to Psi4 option CUBIC_GRID_SPACING.
        
        gridOverage: Grid overage for generating cube files. Units: Bohr.This option
        corresponds to Psi4 option CUBIC_GRID_OVERAGE.
        
        isoContourThreshold: IsoContour values for generating cube files that capture
        specified percent of the probability density using the least amount of grid
        points. Default: 0.85 (85%). This option corresponds to Psi4 option
        CUBEPROP_ISOCONTOUR_THRESHOLD.
    --psi4OptionsParams <Name,Value,...>  [default: none]
        A comma delimited list of Psi4 option name and value pairs for setting
        global and module options. The names are 'option_name' for global options
        and 'module_name__option_name' for options local to a module. The
        specified option names must be valid Psi4 names. No validation is
        performed.
        
        The specified option name and  value pairs are processed and passed to
        psi4.set_options() as a dictionary. The supported value types are float,
        integer, boolean, or string. The float value string is converted into a float.
        The valid values for a boolean string are yes, no, true, false, on, or off. 
    --psi4RunParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for configuring
        Psi4 jobs.
        
        The supported parameter names along with their default and possible
        values are shown below:
             
            MemoryInGB, 1
            NumThreads, 1
            OutputFile, auto   [ Possible  values: stdout, quiet, or FileName ]
            ScratchDir, auto   [ Possivle values: DirName]
            RemoveOutputFile, yes   [ Possible values: yes, no, true, or false]
            
        These parameters control the runtime behavior of Psi4.
        
        The default file name for 'OutputFile' is <InFileRoot>_Psi4.out. The PID
        is appended to output file name during multiprocessing as shown:
        <InFileRoot>_Psi4_<PIDNum>.out. The 'stdout' value for 'OutputType'
        sends Psi4 output to stdout. The 'quiet' or 'devnull' value suppresses
        all Psi4 output.
        
        The default 'Yes' value of 'RemoveOutputFile' option forces the removal
        of any existing Psi4 before creating new files to append output from
        multiple Psi4 runs.
        
        The option 'ScratchDir' is a directory path to the location of scratch
        files. The default value corresponds to Psi4 default. It may be used to
        override the deafult path.
    --pymolViewParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for visualizing
        cube files in PyMOL.
            
            contourColor1, red, contourColor2, blue,
            contourLevel1, auto, contourLevel2, auto,
            contourLevelAutoAt, 0.75,
            displayMolecule, BallAndStick, displaySphereScale, 0.2,
            displayStickRadius, 0.1, hideHydrogens, yes,
            meshWidth, 0.5, meshQuality, 2,
            surfaceQuality, 2, surfaceTransparency, 0.25,
            volumeColorRamp, auto, volumeColorRampOpacity,0.2
            volumeContourWindowFactor,0.05
            
        contourColor1 and contourColor2: Color to use for visualizing volumes,
        meshes, and surfaces corresponding to the negative and positive values
        in cube files. The specified values must be valid PyMOL color names. No
        validation is performed.
        
        contourLevel1 and contourLevel2: Contour levels to use for visualizing
        volumes, meshes, and surfaces corresponding to the negative and positive
        values in cube files. Default: auto. The specified values for contourLevel1
        and contourLevel2 must be negative and positive numbers.
        
        The contour levels are automatically calculated by default. The isocontour
        range for specified percent of the density is retrieved from the cube files.
        The contour levels are set at 'contourLevelAutoAt' of the absolute maximum
        value of the isocontour range. For example: contour levels are set to plus and
        minus 0.03 at 'contourLevelAutoAt' of 0.5 for isocontour range of -0.06 to
        0.06 covering specified percent of the density.
        
        contourLevelAutoAt: Set contour levels at specified fraction of the absolute
        maximum value of the isocontour range retrieved from  the cube files. This
        option is only used during the automatic calculations of the contour levels.
        
        hideHydrogens: Hide hydrogens in molecules. Default: yes. Possible
        values: yes or no.
        
        displayMolecule: Display mode for molecules. Possible values: Sticks or
        BallAndStick. Both displays objects are created for molecules.
        
        displaySphereScale: Sphere scale for displaying molecule during
        BallAndStick display.
        
        displayStickRadius: Stick radius  for displaying molecule during Sticks
        and BallAndStick display.
        
        hideHydrogens: Hide hydrogens in molecules. Default: yes. Possible
        values: yes or no.
        meshWidth: Line width for mesh lines to visualize cube files.
        
        meshQuality: Mesh quality for meshes to visualize cube files. The
        higher values represents better quality.
        
        meshWidth: Line width for mesh lines to visualize cube files.
        
        surfaceQuality: Surface quality for surfaces to visualize cube files.
        The higher values represents better quality.
        
        surfaceTransparency: Surface transparency for surfaces to visualize cube
        files.
        
        volumeColorRamp: Name of a PyMOL volume color ramp to use for visualizing
        cube files. Default name(s): <OutfielsMolPrefix>_psi4_cube_mo or psi4_cube_mo
        The default volume color ramps are automatically generated using contour
        levels and colors during 'auto' value of 'volumeColorRamp'. An explicitly
        specified value must be a valid PyMOL volume color ramp. No validation is
        preformed.
        
        VolumeColorRampOpacity: Opacity for generating volume color ramps
        for visualizing cube files. This value is equivalent to 1 minus Transparency.
        
        volumeContourWindowFactor: Fraction of contour level representing  window
        widths around contour levels during generation of volume color ramps for
        visualizing cube files. For example, the value of 0.05 implies a ramp window
        size of 0.0015 at contour level of 0.03.
    -q, --quiet <yes or no>  [default: no]
        Use quiet mode. The warning and information messages will not be printed.
    -r, --reference <text>  [default: auto]
        Reference wave function to use for calculating single point energy before
        generating cube files for frontier molecular orbitals. Default: RHF or UHF.
        The default values are Restricted Hartree-Fock (RHF) for closed-shell molecules
        with all electrons paired and Unrestricted artree-Fock (UHF) for open-shell
        molecules with unpaired electrons.
        
        The specified value must be a valid Psi4 reference wave function. No validation
        is performed. For example: ROHF, CUHF, RKS, etc.
        
        The spin multiplicity determines the default value of reference wave function
        for input molecules. It is calculated from number of free radical electrons using
        Hund's rule of maximum multiplicity defined as 2S + 1 where S is the total
        electron spin. The total spin is 1/2 the number of free radical electrons in a 
        molecule. The value of 'SpinMultiplicity' molecule property takes precedence
        over the calculated value of spin multiplicity.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To generate and visualize frontier molecular orbitals based on a single point 
    energy calculation using  B3LYP/6-31G** and B3LYP/6-31+G** for non-sulfur
    and sulfur containing molecules in a SD file with 3D structures, use RHF and
    UHF for closed-shell and open-shell molecules, and write a new PML file, type:

        % Psi4VisualizeFrontierOrbitals.py -i Psi4Sample3D.sdf
          -o Psi4Sample3DOut.pml

    To run the first example to only generate cube files and skip generation of 
    a PML file to visualize frontier molecular orbitals, type:

        % Psi4VisualizeFrontierOrbitals.py --mode GenerateCubeFiles
          -i Psi4Sample3D.sdf -o Psi4Sample3DOut.pml

    To run the first example to skip generation of cube files and use existing cube
    files to visualize frontier molecular orbitals and write out a PML file, type:

        % Psi4VisualizeFrontierOrbitals.py --mode VisualizeCubeFiles
          -i Psi4Sample3D.sdf -o Psi4Sample3DOut.pml

    To run the first example in multiprocessing mode on all available CPUs
    without loading all data into memory and write out a PML file, type:

        % Psi4VisualizeFrontierOrbitals.py --mp yes -i Psi4Sample3D.sdf
            -o Psi4Sample3DOut.pml

    To run the first example in multiprocessing mode on all available CPUs
    by loading all data into memory and write out a PML file, type:

        % Psi4VisualizeFrontierOrbitals.py  --mp yes --mpParams "inputDataMode,
            InMemory" -i Psi4Sample3D.sdf  -o Psi4Sample3DOut.pml

    To run the first example in multiprocessing mode on all available CPUs
    without loading all data into memory along with multiple threads for each
    Psi4 run and write out a SD file, type:

        % Psi4VisualizeFrontierOrbitals.py --mp yes --psi4RunParams
          "NumThreads,2" -i Psi4Sample3D.sdf -o Psi4Sample3DOut.pml

    To run the first example in using a specific set of parameters to generate and
    visualize frontier molecular orbitals and write out  a PML file, type:

        % Psi4VisualizeFrontierOrbitals.py  --mode both -m SCF -b aug-cc-pVDZ 
          --psi4CubeFilesParams "gridSpacing, 0.2, gridOverage, 4.0"
          --psi4RunParams "MemoryInGB, 2" --pymolViewParams "contourColor1,
          red, contourColor2, blue,contourLevel1, -0.04, contourLevel2, 0.04,
          contourLevelAutoAt, 0.75,volumeColorRamp, auto,
          volumeColorRampOpacity,0.25, volumeContourWindowFactor,0.05"
          -i Psi4Sample3D.sdf -o Psi4Sample3DOut.pml

Author:
    Manish Sud(msud@san.rr.com)

See also:
    Psi4PerformMinimization.py, Psi4GenerateConformers.py,
    Psi4VisualizeDualDescriptors.py, Psi4VisualizeElectrostaticPotential.py

Copyright:
    Copyright (C) 2022 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using Psi4, an
    open source quantum chemistry software package, and RDKit, an open
    source toolkit for cheminformatics developed by Greg Landrum.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
