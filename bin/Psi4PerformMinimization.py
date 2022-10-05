#!/usr/bin/env python
#
# File: Psi4PerformMinimization.py
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
    PerformMinimization()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def PerformMinimization():
    """Perform minimization."""
    
    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])
    
    # Set up a molecule writer...
    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["Outfile"])

    MolCount, ValidMolCount, MinimizationFailedCount, WriteFailedCount = ProcessMolecules(Mols, Writer)

    if Writer is not None:
        Writer.close()
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during conformation generation or minimization: %d" % MinimizationFailedCount)
    MiscUtil.PrintInfo("Number of molecules failed during writing: %d" % WriteFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + MinimizationFailedCount + WriteFailedCount))

def ProcessMolecules(Mols, Writer):
    """Process and minimize molecules."""
    
    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, Writer)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, Writer)

def ProcessMoleculesUsingSingleProcess(Mols, Writer):
    """Process and minimize molecules using a single process."""
    
    # Intialize Psi4...
    MiscUtil.PrintInfo("\nInitializing Psi4...")
    Psi4Handle = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = True, PrintHeader = True)
    OptionsInfo["psi4"] = Psi4Handle

    # Setup max iterations global variable...
    Psi4Util.UpdatePsi4OptionsParameters(Psi4Handle, {'GEOM_MAXITER': OptionsInfo["MaxIters"]})
    
    # Setup conversion factor for energy units...
    SetupEnergyConversionFactor(Psi4Handle)
    
    if OptionsInfo["SkipConformerGeneration"]:
        MiscUtil.PrintInfo("\nPerforming minimization without generation of conformers...")
    else:
        MiscUtil.PrintInfo("\nPerforming minimization with generation of conformers...")

    (MolCount, ValidMolCount, MinimizationFailedCount, WriteFailedCount) = [0] * 4
    for Mol in Mols:
        MolCount += 1

        if not CheckAndValidateMolecule(Mol, MolCount):
            continue

        # Setup 2D coordinates for SMILES input file...
        if OptionsInfo["SMILESInfileStatus"]:
            AllChem.Compute2DCoords(Mol)
        
        ValidMolCount += 1
        
        Mol, CalcStatus, ConfID, Energy = MinimizeMoleculeOrConformers(Psi4Handle, Mol, MolCount)

        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to calculate energy for molecule %s" % MolName)
            
            MinimizationFailedCount += 1
            continue
        
        Energy = "%.*f" % (OptionsInfo["Precision"], Energy)
        WriteStatus = WriteMolecule(Writer, Mol, MolCount, ConfID, Energy)
        
        if not WriteStatus:
            WriteFailedCount += 1
    
    return (MolCount, ValidMolCount, MinimizationFailedCount, WriteFailedCount)

def ProcessMoleculesUsingMultipleProcesses(Mols, Writer):
    """Process and minimize molecules using multiprocessing."""
    
    if OptionsInfo["MPLevelConformersMode"]:
        return ProcessMoleculesUsingMultipleProcessesAtConformersLevel(Mols, Writer)
    elif OptionsInfo["MPLevelMoleculesMode"]:
        return ProcessMoleculesUsingMultipleProcessesAtMoleculesLevel(Mols, Writer)
    else:
        MiscUtil.PrintError("The value, %s,  option \"--mpLevel\" is not supported." % (OptionsInfo["MPLevel"]))
        
def ProcessMoleculesUsingMultipleProcessesAtMoleculesLevel(Mols, Writer):
    """Process and minimize molecules using multiprocessing at molecules level."""

    if OptionsInfo["SkipConformerGeneration"]:
        MiscUtil.PrintInfo("\nPerforming minimization without generation of conformers using multiprocessing at molecules level...")
    else:
        MiscUtil.PrintInfo("\nPerforming minimization with generation of conformers using multiprocessing at molecules level...")

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
    
    (MolCount, ValidMolCount, MinimizationFailedCount, WriteFailedCount) = [0] * 4
    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, CalcStatus, ConfID, Energy = Result
        
        if EncodedMol is None:
            continue
        ValidMolCount += 1

        if not CalcStatus:
            MinimizationFailedCount += 1
            continue
            
        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
        Energy = "%.*f" % (OptionsInfo["Precision"], Energy)
        WriteStatus = WriteMolecule(Writer, Mol, MolCount, ConfID, Energy)
        
        if not WriteStatus:
            WriteFailedCount += 1
    
    return (MolCount, ValidMolCount, MinimizationFailedCount, WriteFailedCount)

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

def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""

    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolIndex, EncodedMol = EncodedMolInfo

    MolNum = MolIndex + 1
    CalcStatus = False
    ConfID = None
    Energy = None
    
    if EncodedMol is None:
        return [MolIndex, None, CalcStatus, ConfID, Energy]

    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    
    if not CheckAndValidateMolecule(Mol, MolNum):
        return [MolIndex, None, CalcStatus, ConfID, Energy]
    
    # Setup 2D coordinates for SMILES input file...
    if OptionsInfo["SMILESInfileStatus"]:
        AllChem.Compute2DCoords(Mol)
        
    Mol, CalcStatus, ConfID, Energy = MinimizeMoleculeOrConformers(OptionsInfo["psi4"], Mol, MolNum)

    return [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CalcStatus, ConfID, Energy]

def ProcessMoleculesUsingMultipleProcessesAtConformersLevel(Mols, Writer):
    """Process and minimize molecules using multiprocessing at conformers level."""

    MiscUtil.PrintInfo("\nPerforming minimization with generation of conformers using multiprocessing at conformers level...")

    (MolCount, ValidMolCount, MinimizationFailedCount, WriteFailedCount) = [0] * 4
    for Mol in Mols:
        MolCount += 1
        
        if not CheckAndValidateMolecule(Mol, MolCount):
            continue

        # Setup 2D coordinates for SMILES input file...
        if OptionsInfo["SMILESInfileStatus"]:
            AllChem.Compute2DCoords(Mol)

        ValidMolCount += 1
        
        Mol, CalcStatus, ConfID, Energy = ProcessConformersUsingMultipleProcesses(Mol, MolCount)

        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to calculate energy for molecule %s" % MolName)
            
            MinimizationFailedCount += 1
            continue
        
        Energy = "%.*f" % (OptionsInfo["Precision"], Energy)
        WriteStatus = WriteMolecule(Writer, Mol, MolCount, ConfID, Energy)
        
        if not WriteStatus:
            WriteFailedCount += 1
    
    return (MolCount, ValidMolCount, MinimizationFailedCount, WriteFailedCount)

def ProcessConformersUsingMultipleProcesses(Mol, MolNum):
    """Generate conformers and minimize them using multiple processes. """

    # Add hydrogens...
    Mol = Chem.AddHs(Mol)

    # Setup conformers...
    ConfIDs = EmbedMolecule(Mol, MolNum)
    if not len(ConfIDs):
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, MolNum)
            MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s: Embedding failed...\n" % MolName)
        return (Mol, False, None, None)

    MPParams = OptionsInfo["MPParams"]
    
    # Setup data for initializing a worker process...
    InitializeWorkerProcessArgs = (MiscUtil.ObjectToBase64EncodedString(Options), MiscUtil.ObjectToBase64EncodedString(OptionsInfo))

    # Setup a encoded mols data iterable for a worker process...
    MolIndex = MolNum - 1
    WorkerProcessDataIterable = RDKitUtil.GenerateBase64EncodedMolStringWithConfIDs(Mol, MolIndex, ConfIDs)

    # Setup process pool along with data initialization for each process...
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("\nConfiguring multiprocessing using %s method..." % ("mp.Pool.imap()" if re.match("^Lazy$", MPParams["InputDataMode"], re.I) else "mp.Pool.map()"))
        MiscUtil.PrintInfo("NumProcesses: %s; InputDataMode: %s; ChunkSize: %s\n" % (MPParams["NumProcesses"], MPParams["InputDataMode"], ("automatic" if MPParams["ChunkSize"] is None else MPParams["ChunkSize"])))
    
    ProcessPool = mp.Pool(MPParams["NumProcesses"], InitializeConformerWorkerProcess, InitializeWorkerProcessArgs)
    
    # Start processing...
    if re.match("^Lazy$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.imap(ConformerWorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    elif re.match("^InMemory$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.map(ConformerWorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    else:
        MiscUtil.PrintError("The value, %s, specified for \"--inputDataMode\" is not supported." % (MPParams["InputDataMode"]))
    
    CalcEnergyMap = {}
    CalcFailedCount = 0
    for Result in Results:
        MolIndex, EncodedMol, CalcStatus, ConfID, Energy = Result

        if EncodedMol is None:
            CalcFailedCount += 1
            continue
        
        if not CalcStatus:
            CalcFailedCount += 1
            continue
        
        # Retrieve minimized atom positions...
        MinimizedMol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        AtomPositions = RDKitUtil.GetAtomPositions(MinimizedMol, ConfID = ConfID)

        # Update atom positions...
        RDKitUtil.SetAtomPositions(Mol, AtomPositions, ConfID = ConfID)
        
        CalcEnergyMap[ConfID] = Energy
    
    SortedConfIDs = sorted(ConfIDs, key = lambda ConfID: CalcEnergyMap[ConfID])
    MinEnergyConfID = SortedConfIDs[0]
    MinEnergy = CalcEnergyMap[MinEnergyConfID]
    
    if CalcFailedCount:
        return (Mol, False, None, None)
    
    return (Mol, True, MinEnergyConfID, MinEnergy)
    
def InitializeConformerWorkerProcess(*EncodedArgs):
    """Initialize data for a conformer worker process."""
    
    global Options, OptionsInfo

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())
    
    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])
    
    # Psi4 is initialized in the worker process to avoid creation of redundant Psi4
    # output files for each process...
    OptionsInfo["Psi4Initialized"]  = False

def ConformerWorkerProcess(EncodedMolInfo):
    """Process data for a conformer worker process."""

    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolIndex, EncodedMol, ConfID = EncodedMolInfo

    MolNum = MolIndex + 1
    
    CalcStatus = False
    Energy = None
    
    if EncodedMol is None:
        return [MolIndex, None, CalcStatus, ConfID, Energy]
    
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Processing molecule %s conformer ID %s..." % (MolName, ConfID))
    
    Status, ConvergeStatus = MinimizeMoleculeUsingForceField(Mol, MolNum, ConfID)
    if not Status:
        return [MolIndex, EncodedMol, CalcStatus, ConfID, Energy]
    
    if ConvergeStatus != 0:
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, (MolIndex + 1))
            MiscUtil.PrintWarning("Minimization using forcefield failed to converge for molecule %s in %d steps. Try using higher value for \"maxIters\" in \"--confParams\" option...\n" % (MolName, OptionsInfo["ConfGenerationParams"]["MaxIters"]))
    
    # Perform Psi4 minimization...
    CalcStatus, Energy = MinimizeMoleculeUsingPsi4(OptionsInfo["psi4"], Mol, MolNum, ConfID)
    if not CalcStatus:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s\n" % (MolName))
            return [MolIndex, EncodedMol, False, ConfID, Energy]

    return [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CalcStatus, ConfID, Energy]

def InitializePsi4ForWorkerProcess():
    """Initialize Psi4 for a worker process."""
    
    if OptionsInfo["Psi4Initialized"]:
        return

    OptionsInfo["Psi4Initialized"] = True

    if OptionsInfo["MPLevelConformersMode"] and re.match("auto", OptionsInfo["Psi4RunParams"]["OutputFileSpecified"], re.I):
        # Run Psi4 in quiet mode during multiprocessing at Conformers level for 'auto' OutputFile...
        OptionsInfo["Psi4RunParams"]["OutputFile"] = "quiet"
    else:
        # Update output file...
        OptionsInfo["Psi4RunParams"]["OutputFile"] = Psi4Util.UpdatePsi4OutputFileUsingPID(OptionsInfo["Psi4RunParams"]["OutputFile"], os.getpid())
            
    # Intialize Psi4...
    OptionsInfo["psi4"] = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = False, PrintHeader = True)
    
    # Setup max iterations global variable...
    Psi4Util.UpdatePsi4OptionsParameters(OptionsInfo["psi4"], {'GEOM_MAXITER': OptionsInfo["MaxIters"]})
    
    # Setup conversion factor for energy units...
    SetupEnergyConversionFactor(OptionsInfo["psi4"])

def MinimizeMoleculeOrConformers(Psi4Handle, Mol, MolNum):
    """Minimize molecule or conformers."""

    ConfID = None
    if OptionsInfo["SkipConformerGeneration"]:
        Mol, CalcStatus, Energy = MinimizeMolecule(Psi4Handle, Mol, MolNum)
    else:
        Mol, CalcStatus, ConfID, Energy = GenerateAndMinimizeConformers(Psi4Handle, Mol, MolNum)

    return (Mol, CalcStatus, ConfID, Energy)

def GenerateAndMinimizeConformers(Psi4Handle, Mol, MolNum = None):
    """Generate and minimize conformers for a molecule to get the lowest energy conformer."""

    # Add hydrogens..
    Mol = Chem.AddHs(Mol)
    
    MolName = RDKitUtil.GetMolName(Mol, MolNum)

    # Setup conformers...
    ConfIDs = EmbedMolecule(Mol, MolNum)
    if not len(ConfIDs):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s: Embedding failed...\n" % MolName)
        return (Mol, False, None, None)
    
    # Minimize conformers...
    CalcEnergyMap = {}
    for ConfID in ConfIDs:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("Processing molecule %s conformer ID %s..." % (MolName, ConfID))
            
        # Perform forcefield minimization...
        Status, ConvergeStatus = MinimizeMoleculeUsingForceField(Mol, MolNum, ConfID)
        if not Status:
            return (Mol, False, None, None)
        
        if ConvergeStatus != 0:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Minimization using forcefield failed to converge for molecule %s in %d steps. Try using higher value for \"maxIters\" in \"--confParams\" option...\n" % (MolName, OptionsInfo["ConfGenerationParams"]["MaxIters"]))
        
        # Perform Psi4 minimization...
        CalcStatus, Energy = MinimizeMoleculeUsingPsi4(Psi4Handle, Mol, MolNum, ConfID)
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s\n" % (MolName))
            return (Mol, False, None, None)

        CalcEnergyMap[ConfID] = Energy

    # Select the lowest energy conformer..
    SortedConfIDs = sorted(ConfIDs, key = lambda ConfID: CalcEnergyMap[ConfID])
    MinEnergyConfID = SortedConfIDs[0]
    Energy = CalcEnergyMap[MinEnergyConfID]

    return (Mol, True, MinEnergyConfID, Energy)

def MinimizeMolecule(Psi4Handle, Mol, MolNum):
    """Minimize molecule."""

    # Skip conformation generation and forcefield minimization...
    CalcStatus, Energy = MinimizeMoleculeUsingPsi4(Psi4Handle, Mol, MolNum)
    
    return (Mol, CalcStatus, Energy)
    
def MinimizeMoleculeUsingPsi4(Psi4Handle, Mol, MolNum, ConfID = -1):
    """Minimize molecule using Psi4."""

    # Setup a Psi4Mol...
    Psi4Mol = SetupPsi4Mol(Psi4Handle, Mol, MolNum, ConfID)
    if Psi4Mol is None:
        return (False, None)
        
    #  Setup reference wave function...
    Reference = SetupReferenceWavefunction(Mol)
    Psi4Handle.set_options({'Reference': Reference})
    
    # Setup method name and basis set...
    MethodName, BasisSet = SetupMethodNameAndBasisSet(Mol)

    # Optimize geometry...
    Status, Energy, WaveFunction = Psi4Util.PerformGeometryOptimization(Psi4Handle, Psi4Mol, MethodName, BasisSet, ReturnWaveFunction = True, Quiet = OptionsInfo["QuietMode"])
    
    if not Status:
        PerformPsi4Cleanup(Psi4Handle)
        return (False, None)

    # Update atom positions...
    AtomPositions = Psi4Util.GetAtomPositions(Psi4Handle, WaveFunction, InAngstroms = True)
    RDKitUtil.SetAtomPositions(Mol, AtomPositions, ConfID = ConfID)

    # Convert energy units...
    if OptionsInfo["ApplyEnergyConversionFactor"]:
        Energy = Energy * OptionsInfo["EnergyConversionFactor"]
    
    # Clean up
    PerformPsi4Cleanup(Psi4Handle)
    
    return (True, Energy)

def MinimizeMoleculeUsingForceField(Mol, MolNum, ConfID = -1):
    """Minimize molecule using forcefield available in RDKit."""
    
    try:
        if OptionsInfo["ConfGenerationParams"]["UseUFF"]:
            ConvergeStatus = AllChem.UFFOptimizeMolecule(Mol, confId = ConfID, maxIters = OptionsInfo["ConfGenerationParams"]["MaxIters"])
        elif OptionsInfo["ConfGenerationParams"]["UseMMFF"]:
            ConvergeStatus = AllChem.MMFFOptimizeMolecule(Mol, confId = ConfID, maxIters = OptionsInfo["ConfGenerationParams"]["MaxIters"], mmffVariant = OptionsInfo["ConfGenerationParams"]["MMFFVariant"])
        else:
            MiscUtil.PrintError("Minimization couldn't be performed: Specified forcefield, %s, is not supported" % OptionsInfo["ConfGenerationParams"]["ForceField"])
    except (ValueError, RuntimeError, Chem.rdchem.KekulizeException) as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, MolNum)
            MiscUtil.PrintWarning("Minimization using forcefield couldn't be performed for molecule %s:\n%s\n" % (MolName, ErrMsg))
        return (False, None)
    
    return (True, ConvergeStatus)

def EmbedMolecule(Mol, MolNum = None):
    """Embed conformations."""

    ConfIDs = []
    
    MaxConfs = OptionsInfo["ConfGenerationParams"]["MaxConfs"]
    RandomSeed = OptionsInfo["ConfGenerationParams"]["RandomSeed"]
    EnforceChirality = OptionsInfo["ConfGenerationParams"]["EnforceChirality"]
    UseExpTorsionAnglePrefs = OptionsInfo["ConfGenerationParams"]["UseExpTorsionAnglePrefs"]
    UseBasicKnowledge = OptionsInfo["ConfGenerationParams"]["UseBasicKnowledge"]
    EmbedRMSDCutoff = OptionsInfo["ConfGenerationParams"]["EmbedRMSDCutoff"]

    try:
        ConfIDs = AllChem.EmbedMultipleConfs(Mol, numConfs = MaxConfs, randomSeed = RandomSeed, pruneRmsThresh = EmbedRMSDCutoff, enforceChirality = EnforceChirality, useExpTorsionAnglePrefs = UseExpTorsionAnglePrefs, useBasicKnowledge = UseBasicKnowledge)
    except ValueError as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, MolNum)
            MiscUtil.PrintWarning("Embedding failed  for molecule %s:\n%s\n" % (MolName, ErrMsg))
        ConfIDs = []

    if not OptionsInfo["QuietMode"]:
        if EmbedRMSDCutoff > 0:
            MiscUtil.PrintInfo("Generating initial conformation ensemble by distance geometry for %s - EmbedRMSDCutoff: %s; Size: %s; Size after RMSD filtering: %s" % (RDKitUtil.GetMolName(Mol, MolNum), EmbedRMSDCutoff, MaxConfs, len(ConfIDs)))
        else:
            MiscUtil.PrintInfo("Generating initial conformation ensemble by distance geometry for %s - EmbedRMSDCutoff: None; Size: %s" % (RDKitUtil.GetMolName(Mol, MolNum), len(ConfIDs)))
    
    return ConfIDs

def SetupPsi4Mol(Psi4Handle, Mol, MolNum, ConfID = -1):
    """Setup a Psi4 molecule object."""

    if OptionsInfo["RecenterAndReorient"]:
        MolGeometry = RDKitUtil.GetPsi4XYZFormatString(Mol, ConfID = ConfID, NoCom = False, NoReorient = False)
    else:
        MolGeometry = RDKitUtil.GetPsi4XYZFormatString(Mol, ConfID = ConfID, NoCom = True, NoReorient = True)
    
    try:
        Psi4Mol = Psi4Handle.geometry(MolGeometry)
    except Exception as ErrMsg:
        Psi4Mol = None
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to create Psi4 molecule from geometry string: %s\n" % ErrMsg)
            MolName = RDKitUtil.GetMolName(Mol, MolNum)
            MiscUtil.PrintWarning("Ignoring molecule: %s" % MolName)
    
    if OptionsInfo["Symmetrize"]:
        Psi4Mol.symmetrize(OptionsInfo["SymmetrizeTolerance"])
    
    return Psi4Mol

def PerformPsi4Cleanup(Psi4Handle):
    """Perform clean up."""

    # No need to perform any explicit clean by calling
    # Psi4Handle.core.opt_clean(). It's already done by Psi4 after
    # optimization...
    
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
    if OptionsInfo["SkipConformerGeneration"]:
        if not Mol.GetConformer().Is3D():
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("3D tag is not set for molecule: %s\n" % MolName)
    
    # Check for missing hydrogens...
    if OptionsInfo["SkipConformerGeneration"]:
        # Otherwise, Hydrogens are always added...
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

def SetupEnergyConversionFactor(Psi4Handle):
    """Setup converstion factor for energt units. The Psi4 energy units are Hartrees."""
    
    EnergyUnits = OptionsInfo["EnergyUnits"]
    
    ApplyConversionFactor = True
    if re.match("^kcal\/mol$", EnergyUnits, re.I):
        ConversionFactor = Psi4Handle.constants.hartree2kcalmol
    elif re.match("^kJ\/mol$", EnergyUnits, re.I):
        ConversionFactor = Psi4Handle.constants.hartree2kJmol
    elif re.match("^eV$", EnergyUnits, re.I):
        ConversionFactor = Psi4Handle.constants.hartree2ev
    else:
        ApplyConversionFactor = False
        ConversionFactor = 1.0
    
    OptionsInfo["ApplyEnergyConversionFactor"] = ApplyConversionFactor
    OptionsInfo["EnergyConversionFactor"] = ConversionFactor

def WriteMolecule(Writer, Mol, MolNum = None, ConfID = None, Energy = None):
    """Write molecule."""

    if OptionsInfo["EnergyOut"]  and Energy is not None:
        Mol.SetProp(OptionsInfo["EnergyDataFieldLabel"], Energy)

    try:
        if ConfID is None:
            Writer.write(Mol)
        else:
            Writer.write(Mol, confId = ConfID)
    except (ValueError, RuntimeError) as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, MolNum)
            MiscUtil.PrintWarning("Failed to write molecule %s:\n%s\n" % (MolName, ErrMsg))
        return False

    return True

def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")
    
    # Validate options...
    ValidateOptions()
    
    OptionsInfo["Infile"] = Options["--infile"]
    OptionsInfo["SMILESInfileStatus"] = True if  MiscUtil.CheckFileExt(Options["--infile"], "smi csv tsv txt") else False
    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters("--infileParams", Options["--infileParams"], InfileName = Options["--infile"], ParamsDefaultInfo = ParamsDefaultInfoOverride)
    
    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters("--outfileParams", Options["--outfileParams"])
    
    OptionsInfo["Overwrite"] = Options["--overwrite"]
    
    # Method, basis set, and reference wavefunction...
    OptionsInfo["BasisSet"] = Options["--basisSet"]
    OptionsInfo["BasisSetAuto"] = True if re.match("^auto$", Options["--basisSet"], re.I) else False
    
    OptionsInfo["MethodName"] = Options["--methodName"]
    OptionsInfo["MethodNameAuto"] = True if re.match("^auto$", Options["--methodName"], re.I) else False
    
    OptionsInfo["Reference"] = Options["--reference"]
    OptionsInfo["ReferenceAuto"] = True if re.match("^auto$", Options["--reference"], re.I) else False
    
    # Run and options parameters...
    OptionsInfo["Psi4OptionsParams"] = Psi4Util.ProcessPsi4OptionsParameters("--psi4OptionsParams", Options["--psi4OptionsParams"])
    OptionsInfo["Psi4RunParams"] = Psi4Util.ProcessPsi4RunParameters("--psi4RunParams", Options["--psi4RunParams"], InfileName = OptionsInfo["Infile"])

    # Conformer generation paramaters...
    OptionsInfo["ConfGeneration"] = Options["--confGeneration"]
    OptionsInfo["SkipConformerGeneration"] = False if re.match("^yes$", Options["--confGeneration"], re.I) else True

    if OptionsInfo["SkipConformerGeneration"]:
        if OptionsInfo["SMILESInfileStatus"]:
            MiscUtil.PrintError("The value, %s, specified for option \"-c --confGeneration\" is not allowed during, %s, value for option \"-i, --infile\" . The input file must be a 3D SD or MOL file. " % (Options["--confGeneration"], Options["--infile"]))
    
    ParamsDefaultInfoOverride = {"MaxConfs": 50, "MaxIters": 250}
    OptionsInfo["ConfGenerationParams"] = MiscUtil.ProcessOptionConformerParameters("--confParams", Options["--confParams"], ParamsDefaultInfoOverride)
    
    # Write energy parameters...
    OptionsInfo["EnergyOut"] = True if re.match("^yes$", Options["--energyOut"], re.I) else False
    OptionsInfo["EnergyUnits"] = Options["--energyUnits"]
    
    EnergyDataFieldLabel = Options["--energyDataFieldLabel"]
    if re.match("^auto$", EnergyDataFieldLabel, re.I):
        EnergyDataFieldLabel = "Psi4_Energy (%s)" % Options["--energyUnits"]
    OptionsInfo["EnergyDataFieldLabel"] = EnergyDataFieldLabel
    
    OptionsInfo["MaxIters"] = int(Options["--maxIters"])

    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])

    # Multiprocessing level...
    MPLevelMoleculesMode = False
    MPLevelConformersMode = False
    MPLevel = Options["--mpLevel"]
    if re.match("^Molecules$", MPLevel, re.I):
        MPLevelMoleculesMode = True
    elif re.match("^Conformers$", MPLevel, re.I):
        if  OptionsInfo["SkipConformerGeneration"]:
            MiscUtil.PrintError("The value, %s, specified for option \"--mpLevel\" is not allowed during, %s, value of option \"--confGeneration\"  . " % (MPLevel, Options["--confGeneration"]))
        MPLevelConformersMode = True
    else:
        MiscUtil.PrintError("The value, %s, specified for option \"--mpLevel\" is not valid. " % MPLevel)
    OptionsInfo["MPLevel"] = MPLevel
    OptionsInfo["MPLevelMoleculesMode"] = MPLevelMoleculesMode
    OptionsInfo["MPLevelConformersMode"] = MPLevelConformersMode
    
    OptionsInfo["Precision"] = int(Options["--precision"])
    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False

    RecenterAndReorient = Options["--recenterAndReorient"]
    if re.match("^auto$", RecenterAndReorient, re.I):
        RecenterAndReorient  = False if  OptionsInfo["SkipConformerGeneration"] else True
    else:
        RecenterAndReorient  = True if re.match("^yes$", RecenterAndReorient, re.I) else False
    OptionsInfo["RecenterAndReorient"] = RecenterAndReorient
    
    Symmetrize = Options["--symmetrize"]
    if re.match("^auto$", Symmetrize, re.I):
        Symmetrize  = True if  RecenterAndReorient else False
    else:
        Symmetrize  = True if re.match("^yes$", Symmetrize, re.I) else False
    OptionsInfo["Symmetrize"] = Symmetrize
    
    OptionsInfo["SymmetrizeTolerance"] = float(Options["--symmetrizeTolerance"])

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
    
    MiscUtil.ValidateOptionTextValue("-c, --confGeneration", Options["--confGeneration"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--energyOut", Options["--energyOut"], "yes no")
    MiscUtil.ValidateOptionTextValue("--energyUnits", Options["--energyUnits"], "Hartrees kcal/mol kJ/mol eV")
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol smi txt csv tsv")
    
    MiscUtil.ValidateOptionIntegerValue("--maxIters", Options["--maxIters"], {">": 0})
    
    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
    MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])
    
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionTextValue("--mpLevel", Options["--mpLevel"], "Molecules Conformers")
    
    MiscUtil.ValidateOptionIntegerValue("-p, --precision", Options["--precision"], {">": 0})
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--recenterAndReorient", Options["--recenterAndReorient"], "yes no auto")
    MiscUtil.ValidateOptionTextValue("--symmetrize", Options["--symmetrize"], "yes no auto")
    MiscUtil.ValidateOptionFloatValue("--symmetrizeTolerance", Options["--symmetrizeTolerance"], {">": 0})

# Setup a usage string for docopt...
_docoptUsage_ = """
Psi4PerformMinimization.py - Perform structure minimization

Usage:
    Psi4PerformMinimization.py [--basisSet <text>] [--confGeneration <yes or no> ] [--confParams <Name,Value,...>]
                               [--energyOut <yes or no>] [--energyDataFieldLabel <text>] [--energyUnits <text>]
                               [--infileParams <Name,Value,...>] [--maxIters <number>] [--methodName <text>]
                               [--mp <yes or no>] [--mpLevel <Molecules or Conformers>] [--mpParams <Name, Value,...>]
                               [ --outfileParams <Name,Value,...> ] [--overwrite] [--precision <number> ]
                               [--psi4OptionsParams <Name,Value,...>] [--psi4RunParams <Name,Value,...>]
                               [--quiet <yes or no>] [--reference <text>] [--recenterAndReorient <yes or no>]
                               [--symmetrize <yes or no>] [--symmetrizeTolerance <number>] [-w <dir>] -i <infile> -o <outfile> 
    Psi4PerformMinimization.py -h | --help | -e | --examples

Description:
    Generate 3D structures for molecules using a combination of distance geometry
    and forcefield minimization followed by geometry optimization using a quantum
    chemistry method. A set of initial 3D structures are generated for a molecule 
    employing distance geometry. The 3D structures in the conformation ensemble
    are sequentially minimized using forcefield and a quantum chemistry method.
    The conformation with lowest energy is selected to represent the final structure.
    An option is available to skip the generation of the conformation ensemble along
    with forcefield minimization and simply perform minimization on the initial 3D
    structure using a quantum chemistry method.

    A Psi4 XYZ format geometry string is automatically generated for each molecule
    in input file. It contains atom symbols and 3D coordinates for each atom in a
    molecule. In addition, the formal charge and spin multiplicity are present in the
    the geometry string. These values are either retrieved from molecule properties
    named 'FormalCharge' and 'SpinMultiplicty' or dynamically calculated for a
    molecule.

    The supported input file formats are: Mol (.mol), SD (.sdf, .sd), SMILES (.smi,
    .csv, .tsv, .txt)

    The supported output file formats are: SD (.sdf, .sd)

Options:
    -b, --basisSet <text>  [default: auto]
        Basis set to use for energy minimization. Default: 6-31+G** for sulfur
        containing molecules; Otherwise, 6-31G** [ Ref 150 ]. The specified 
        value must be a valid Psi4 basis set. No validation is performed.
        
        The following list shows a representative sample of basis sets available
        in Psi4:
            
            STO-3G, 6-31G, 6-31+G, 6-31++G, 6-31G*, 6-31+G*,  6-31++G*, 
            6-31G**, 6-31+G**, 6-31++G**, 6-311G, 6-311+G, 6-311++G,
            6-311G*, 6-311+G*, 6-311++G*, 6-311G**, 6-311+G**, 6-311++G**,
            cc-pVDZ, cc-pCVDZ, aug-cc-pVDZ, cc-pVDZ-DK, cc-pCVDZ-DK, def2-SVP,
            def2-SVPD, def2-TZVP, def2-TZVPD, def2-TZVPP, def2-TZVPPD
            
    -c, --confGeneration <yes or no>  [default: yes]
        Generate an initial 3D conformation ensemble using distance geometry and
        forcefield minimization before final geometry optimization by a specified
        method name and basis set. Possible values: yes or no.
        
        The 'No' value skips the generation of conformations employing distance
        geometry and forcefield minimization. The 3D structures in input file are
        minimized by a quantum method. It is not allowed for SMILES input file.
    --confParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        initial sets of 3D conformations for molecules. The 3D conformation ensemble
        is generated using distance geometry and forcefield functionality available
        in RDKit. The 3D structures in the conformation ensemble are subsequently
        minimized by a quantum chemistry method available in Psi4.
       
        The supported parameter names along with their default values are shown
        below:
            
            confMethod,ETKDG,
            forceField,MMFF, forceFieldMMFFVariant,MMFF94,
            enforceChirality,yes,embedRMSDCutoff,0.5,maxConfs,50,
            maxIters,250,randomSeed,auto
            
            confMethod,ETKDG   [ Possible values: SDG, ETDG, KDG, ETKDG ]
            forceField, MMFF   [ Possible values: UFF or MMFF ]
            forceFieldMMFFVariant,MMFF94   [ Possible values: MMFF94 or MMFF94s ]
            enforceChirality,yes   [ Possible values: yes or no ]
            embedRMSDCutoff,0.5   [ Possible values: number or None]
            
        confMethod: Conformation generation methodology for generating initial 3D
        coordinates. Possible values: Standard Distance Geometry (SDG), Experimental
        Torsion-angle preference with Distance Geometry (ETDG), basic Knowledge-terms
        with Distance Geometry (KDG) and Experimental Torsion-angle preference
        along with basic Knowledge-terms with Distance Geometry (ETKDG) [Ref 129] .
        
        forceField: Forcefield method to use for energy minimization. Possible
        values: Universal Force Field (UFF) [ Ref 81 ] or Merck Molecular Mechanics
        Force Field [ Ref 83-87 ] .
        
        enforceChirality: Enforce chirality for defined chiral centers during
        forcefield minimization.
        
        maxConfs: Maximum number of conformations to generate for each molecule
        during the generation of an initial 3D conformation ensemble using 
        conformation generation methodology. The conformations are minimized
        using the specified forcefield and a quantum chemistry method. The lowest
        energy conformation is written to the output file.
        
        embedRMSDCutoff: RMSD cutoff for retaining initial set conformers embedded
        using distance geometry and before forcefield minimization. All embedded
        conformers are kept for 'None' value. Otherwise, only those conformers which
        are different from each other by the specified RMSD cutoff, 0.5 by default,
        are kept. The first embedded conformer is always retained.
        
        maxIters: Maximum number of iterations to perform for each conformation
        during forcefield minimization.
        
        randomSeed: Seed for the random number generator for reproducing initial
        3D coordinates in a conformation ensemble. Default is to use a random seed.
    --energyOut <yes or no>  [default: yes]
        Write out energy values.
    --energyDataFieldLabel <text>  [default: auto]
        Energy data field label for writing energy values. Default: Psi4_Energy (<Units>). 
    --energyUnits <text>  [default: kcal/mol]
        Energy units. Possible values: Hartrees, kcal/mol, kJ/mol, or eV.
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
            SMILES: smilesColumn,1,smilesNameColumn,2,smilesDelimiter,space,
                smilesTitleLine,auto,sanitize,yes
            
        Possible values for smilesDelimiter: space, comma or tab.
    --maxIters <number>  [default: 50]
        Maximum number of iterations to perform for each molecule or conformer
        during energy minimization by a quantum chemistry method.
    -m, --methodName <text>  [default: auto]
        Method to use for energy minimization. Default: B3LYP [ Ref 150 ]. The
        specified value must be a valid Psi4 method name. No validation is
        performed.
        
        The following list shows a representative sample of methods available
        in Psi4:
            
            B1LYP, B2PLYP, B2PLYP-D3BJ, B2PLYP-D3MBJ, B3LYP, B3LYP-D3BJ,
            B3LYP-D3MBJ, CAM-B3LYP, CAM-B3LYP-D3BJ, HF, HF-D3BJ,  HF3c, M05,
            M06, M06-2x, M06-HF, M06-L, MN12-L, MN15, MN15-D3BJ,PBE, PBE0,
            PBEH3c, PW6B95, PW6B95-D3BJ, WB97, WB97X, WB97X-D, WB97X-D3BJ
            
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
    --mpLevel <Molecules or Conformers>  [default: Molecules]
        Perform multiprocessing at molecules or conformers level. Possible values:
        Molecules or Conformers. The 'Molecules' value starts a process pool at the
        molecules level. All conformers of a molecule are processed in a single
        process. The 'Conformers' value, however, starts a process pool at the 
        conformers level. Each conformer of a molecule is processed in an individual
        process in the process pool. The default Psi4 'OutputFile' is set to 'quiet'
        using '--psi4RunParams' for 'Conformers' level. Otherwise, it may generate
        a large number of Psi4 output files.
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
        Output file name.
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: kekulize,yes
            
    --overwrite
        Overwrite existing files.
    --precision <number>  [default: 6]
        Floating point precision for writing energy values.
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
        all Psi4 output. The 'OutputFile' is set to 'quiet' for 'auto' value during 
        'Conformers' of '--mpLevel' option.
        
        The default 'Yes' value of 'RemoveOutputFile' option forces the removal
        of any existing Psi4 before creating new files to append output from
        multiple Psi4 runs.
        
        The option 'ScratchDir' is a directory path to the location of scratch
        files. The default value corresponds to Psi4 default. It may be used to
        override the deafult path.
    -q, --quiet <yes or no>  [default: no]
        Use quiet mode. The warning and information messages will not be printed.
    -r, --reference <text>  [default: auto]
        Reference wave function to use for energy calculation. Default: RHF or UHF.
        The default values are Restricted Hartree-Fock (RHF) for closed-shell molecules
        with all electrons paired and Unrestricted Hartree-Fock (UHF) for open-shell
        molecules with unpaired electrons.
        
        The specified value must be a valid Psi4 reference wave function. No validation
        is performed. For example: ROHF, CUHF, RKS, etc.
        
        The spin multiplicity determines the default value of reference wave function
        for input molecules. It is calculated from number of free radical electrons using
        Hund's rule of maximum multiplicity defined as 2S + 1 where S is the total
        electron spin. The total spin is 1/2 the number of free radical electrons in a 
        molecule. The value of 'SpinMultiplicity' molecule property takes precedence
        over the calculated value of spin multiplicity.
    --recenterAndReorient <yes or no>  [default: auto]
        Recenter and reorient a molecule during creation of a Psi4 molecule from
        a geometry string. Default: 'No' during 'No' value of '--confGeneration';
        Otherwise, 'Yes'.
        
        The 'No' values allows the minimization of a molecule in its initial 3D
        coordinate space in input file or conformer ensemble.
    --symmetrize <yes or no>  [default: auto]
        Symmetrize molecules before energy minimization. Default: 'Yes' during
        'Yes' value of '--recenterAndReorient'; Otherwise, 'No'. The psi4 function,
        psi4mol.symmetrize( SymmetrizeTolerance), is called to symmetrize
        the molecule before calling psi4.optimize().
        
        The 'No' value of '--symmetrize' during 'Yes' value of '--recenterAndReorient'
        may cause psi4.optimize() to fail with a 'Point group changed...' error
        message.
    --symmetrizeTolerance <number>  [default: 0.01]
        Symmetry tolerance for '--symmetrize'.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To generate an initial conformer ensemble of up to 50 conformations using a
    combination of ETKDG distance geometry methodology, applying RMSD cutoff
    of 0.5 and MMFF forcefield minimization, followed by energy minimization
    using B3LYP/6-31G** and B3LYP/6-31+G** for non-sulfur and sulfur containing
    molecules in a SMILES file and  write out a SD file containing minimum energy
    structure corresponding to each molecule, type:

        % Psi4PerformMinimization.py -i Psi4Sample.smi -o Psi4SampleOut.sdf

    To run the first example in a quiet mode and write out a SD file, type:

        % Psi4PerformMinimization.py -q yes -i Psi4Sample.smi -o
          Psi4SampleOut.sdf

    To run the first example in multiprocessing mode at molecules level on all
    available CPUs without loading all data into memory and write out a SD file,
    type:

        % Psi4PerformMinimization.py --mp yes -i Psi4Sample.smi -o
          Psi4SampleOut.sdf

    To run the first example in multiprocessing mode at conformers level on all
    available CPUs without loading all data into memory and write out a SD file,
    type:

        % Psi4PerformMinimization.py --mp yes --mpLevel Conformers
          -i Psi4Sample.smi -o Psi4SampleOut.sdf

    To run the first example in multiprocessing mode at molecules level on all
    available CPUs by loading all data into memory and write out a SD file, type:

        % Psi4PerformMinimization.py  --mp yes --mpParams "inputDataMode,
          InMemory" -i Psi4Sample.smi -o Psi4SampleOut.sdf

    To run the first example in multiprocessing mode at molecules level on specific
    number of CPUs and chunk size without loading all data into memory and write
    out a SD file, type:

        % Psi4PerformMinimization.py  --mp yes --mpParams "inputDataMode,Lazy,
          numProcesses,4,chunkSize,8" -i Psi4Sample.smi -o Psi4SampleOut.sdf

    To generate an initial conformer ensemble of up to 20 conformations using a
    combination of ETKDG distance geometry methodology and MMFF94s forcefield
    minimization followed by energy minimization for a maxium of 20 iterations
    using B3LYP/6-31+G** molecules in a SMILES file and  write out a SD file
    containing minimum energy structure along with energy in specific units,
    type:

        % Psi4PerformMinimization.py --confGeneration yes --confParams
          "confMethod,ETKDG,forceField,MMFF, forceFieldMMFFVariant,MMFF94s,
          maxConfs,20,embedRMSDCutoff,0.5" --energyUnits "kJ/mol" -m B3LYP
          -b "6-31+G**" --maxIters 20 -i Psi4Sample.smi -o Psi4SampleOut.sdf

    To minimize molecules in a 3D files using B3LYP/6-31G** and B3LYP/6-31+G**
    for non-sulfur and sulfur containing molecules for a maximum of 25 iterations
    without generating any conformations and write out a SD file containing 
    minimum energy structures corresponding to each molecule, type:

        % Psi4PerformMinimization.py --confGeneration no --maxIters 25
          -i Psi4Sample3D.sdf -o Psi4Sample3DOut.sdf

    To run the first example for molecules in a CSV SMILES file, SMILES strings
    in column 1, name column 2, and write out a SD file, type:

        % Psi4PerformMinimization.py --infileParams "smilesDelimiter,comma,
          smilesTitleLine,yes,smilesColumn,1,smilesNameColumn,2"
          -i Psi4Sample.csv -o Psi4SampleOut.sdf

Author:

    Manish Sud(msud@san.rr.com)

See also:
    Psi4CalculateEnergy.py, Psi4CalculatePartialCharges.py, Psi4GenerateConformers.py

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
