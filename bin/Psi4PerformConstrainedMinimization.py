#!/usr/bin/env python
#
# File: Psi4PerformConstrainedMinimization.py
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
    from rdkit.Chem import AllChem, rdMolAlign
    from rdkit.Chem import rdFMCS
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
    PerformConstrainedMinimization()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def PerformConstrainedMinimization():
    """Perform constrained minimization."""
    
    # Read and validate reference molecule...
    RefMol = RetrieveReferenceMolecule()
    
    # Setup a molecule reader for input file...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])
    
    # Set up a molecule writer...
    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["Outfile"])

    MolCount, ValidMolCount, CoreScaffoldMissingCount, MinimizationFailedCount, WriteFailedCount = ProcessMolecules(RefMol, Mols, Writer)

    if Writer is not None:
        Writer.close()
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules with missing core scaffold: %d" % CoreScaffoldMissingCount)
    MiscUtil.PrintInfo("Number of molecules failed during conformation generation or minimization: %d" % MinimizationFailedCount)
    MiscUtil.PrintInfo("Number of molecules failed during writing: %d" % WriteFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + CoreScaffoldMissingCount + MinimizationFailedCount + WriteFailedCount))

def ProcessMolecules(RefMol, Mols, Writer):
    """Process and minimize molecules."""
    
    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(RefMol, Mols, Writer)
    else:
        return ProcessMoleculesUsingSingleProcess(RefMol, Mols, Writer)

def ProcessMoleculesUsingSingleProcess(RefMol, Mols, Writer):
    """Process and minimize molecules using a single process."""

    # Intialize Psi4...
    MiscUtil.PrintInfo("\nInitializing Psi4...")
    Psi4Handle = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = True, PrintHeader = True)
    OptionsInfo["psi4"] = Psi4Handle

    # Setup max iterations global variable...
    Psi4Util.UpdatePsi4OptionsParameters(Psi4Handle, {'GEOM_MAXITER': OptionsInfo["MaxIters"]})
    
    # Setup conversion factor for energy units...
    SetupEnergyConversionFactor(Psi4Handle)
    
    (MolCount, ValidMolCount, CoreScaffoldMissingCount, MinimizationFailedCount, WriteFailedCount) = [0] * 5
    
    for Mol in Mols:
        MolCount += 1

        if not CheckAndValidateMolecule(Mol, MolCount):
            continue

        # Setup 2D coordinates for SMILES input file...
        if OptionsInfo["SMILESInfileStatus"]:
            AllChem.Compute2DCoords(Mol)
        
        ValidMolCount += 1
        
        # Setup a reference molecule core containing common scaffold atoms...
        RefMolCore = SetupCoreScaffold(RefMol, Mol, MolCount)
        if RefMolCore is None:
            CoreScaffoldMissingCount += 1
            continue
        
        Mol, CalcStatus, Energy, ScaffoldEmbedRMSD = ConstrainAndMinimizeMolecule(Psi4Handle, Mol, RefMolCore, MolCount)
        
        if not CalcStatus:
            MinimizationFailedCount += 1
            continue
        
        WriteStatus = WriteMolecule(Writer, Mol, MolCount, Energy, ScaffoldEmbedRMSD)
        
        if not WriteStatus:
            WriteFailedCount += 1
        
    return (MolCount, ValidMolCount, CoreScaffoldMissingCount, MinimizationFailedCount, WriteFailedCount)

def ProcessMoleculesUsingMultipleProcesses(RefMol, Mols, Writer):
    """Process and minimize molecules using multiprocessing."""
    
    if OptionsInfo["MPLevelConformersMode"]:
        return ProcessMoleculesUsingMultipleProcessesAtConformersLevel(RefMol, Mols, Writer)
    elif OptionsInfo["MPLevelMoleculesMode"]:
        return ProcessMoleculesUsingMultipleProcessesAtMoleculesLevel(RefMol, Mols, Writer)
    else:
        MiscUtil.PrintError("The value, %s,  option \"--mpLevel\" is not supported." % (OptionsInfo["MPLevel"]))
        
def ProcessMoleculesUsingMultipleProcessesAtMoleculesLevel(RefMol, Mols, Writer):
    """Process and minimize molecules using multiprocessing at molecules level."""

    MiscUtil.PrintInfo("\nPerforming constrained minimization with generation of conformers using multiprocessing at molecules level...")
    
    MPParams = OptionsInfo["MPParams"]

    # Setup data for initializing a worker process...
    OptionsInfo["EncodedRefMol"] = RDKitUtil.MolToBase64EncodedMolString(RefMol)
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

    (MolCount, ValidMolCount, CoreScaffoldMissingCount, MinimizationFailedCount, WriteFailedCount) = [0] * 5

    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, CoreScaffoldMissingStatus, CalcStatus, Energy, ScaffoldEmbedRMSD  = Result
        
        if EncodedMol is None:
            continue
        ValidMolCount += 1

        if CoreScaffoldMissingStatus:
            CoreScaffoldMissingStatus += 1
            continue
        
        if not CalcStatus:
            MinimizationFailedCount += 1
            continue
            
        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        WriteStatus = WriteMolecule(Writer, Mol, MolCount, Energy, ScaffoldEmbedRMSD)
        
        if not WriteStatus:
            WriteFailedCount += 1
    
    return (MolCount, ValidMolCount, CoreScaffoldMissingCount, MinimizationFailedCount, WriteFailedCount)
    
def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""
    
    global Options, OptionsInfo

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())
    
    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])
    
    # Decode RefMol...
    OptionsInfo["RefMol"] = RDKitUtil.MolFromBase64EncodedMolString(OptionsInfo["EncodedRefMol"])
    
    # Psi4 is initialized in the worker process to avoid creation of redundant Psi4
    # output files for each process...
    OptionsInfo["Psi4Initialized"]  = False

def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""

    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolIndex, EncodedMol = EncodedMolInfo

    MolNum = MolIndex + 1
    CoreScaffoldMissingStatus = False
    CalcStatus = False
    Energy = None
    ScaffoldEmbedRMSD = None

    if EncodedMol is None:
        return [MolIndex, None, CoreScaffoldMissingStatus, CalcStatus, Energy, ScaffoldEmbedRMSD]

    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    if not CheckAndValidateMolecule(Mol, MolNum):
        return [MolIndex, None, CoreScaffoldMissingStatus, CalcStatus, Energy, ScaffoldEmbedRMSD]
    
    # Setup 2D coordinates for SMILES input file...
    if OptionsInfo["SMILESInfileStatus"]:
        AllChem.Compute2DCoords(Mol)
        
    # Setup a reference molecule core containing common scaffold atoms...
    RefMolCore = SetupCoreScaffold(OptionsInfo["RefMol"], Mol, MolNum)
    if RefMolCore is None:
        CoreScaffoldMissingStatus = True
        return [MolIndex, EncodedMol, CalcStatus, CoreScaffoldMissingStatus, Energy, ScaffoldEmbedRMSD]
    
    Mol, CalcStatus, Energy, ScaffoldEmbedRMSD = ConstrainAndMinimizeMolecule(OptionsInfo["psi4"], Mol, RefMolCore, MolNum)
    
    return [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CoreScaffoldMissingStatus, CalcStatus, Energy, ScaffoldEmbedRMSD]

def ProcessMoleculesUsingMultipleProcessesAtConformersLevel(RefMol, Mols, Writer):
    """Process and minimize molecules using multiprocessing at conformers level."""

    MiscUtil.PrintInfo("\nPerforming constrained minimization with generation of conformers using multiprocessing at conformers level...")
    
    (MolCount, ValidMolCount, CoreScaffoldMissingCount, MinimizationFailedCount, WriteFailedCount) = [0] * 5
    
    for Mol in Mols:
        MolCount += 1

        if not CheckAndValidateMolecule(Mol, MolCount):
            continue

        # Setup 2D coordinates for SMILES input file...
        if OptionsInfo["SMILESInfileStatus"]:
            AllChem.Compute2DCoords(Mol)
        
        ValidMolCount += 1
        
        # Setup a reference molecule core containing common scaffold atoms...
        RefMolCore = SetupCoreScaffold(RefMol, Mol, MolCount)
        if RefMolCore is None:
            CoreScaffoldMissingCount += 1
            continue

        Mol, CalcStatus, Energy, ScaffoldEmbedRMSD = ProcessConformersUsingMultipleProcesses(Mol, RefMolCore, MolCount)
        
        if not CalcStatus:
            MinimizationFailedCount += 1
            continue
        
        WriteStatus = WriteMolecule(Writer, Mol, MolCount, Energy, ScaffoldEmbedRMSD)
        
        if not WriteStatus:
            WriteFailedCount += 1
        
    return (MolCount, ValidMolCount, CoreScaffoldMissingCount, MinimizationFailedCount, WriteFailedCount)

def ProcessConformersUsingMultipleProcesses(Mol, RefMolCore, MolNum):
    """Generate conformers and minimize them using multiple processes. """
    
    # Add hydrogens...
    Mol = Chem.AddHs(Mol, addCoords = True)

    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    
    # Setup constrained conformers...
    MolConfs, MolConfsStatus = ConstrainEmbedAndMinimizeMoleculeUsingRDKit(Mol, RefMolCore, MolNum)
    if not MolConfsStatus:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Conformation generation couldn't be performed for molecule %s: Constrained embedding failed...\n" % MolName)
        return (Mol, False, None, None)
    
    MPParams = OptionsInfo["MPParams"]

    # Setup data for initializing a worker process...
    OptionsInfo["EncodedRefMolCore"] = RDKitUtil.MolToBase64EncodedMolString(RefMolCore)
    InitializeWorkerProcessArgs = (MiscUtil.ObjectToBase64EncodedString(Options), MiscUtil.ObjectToBase64EncodedString(OptionsInfo))

    # Setup a encoded mols data iterable for a worker process...
    WorkerProcessDataIterable = RDKitUtil.GenerateBase64EncodedMolStrings(MolConfs)
    
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

    ConfNums = []
    MolConfsMap = {}
    CalcEnergyMap = {}
    CalcFailedCount = 0
    
    for Result in Results:
        ConfNum, EncodedMolConf, CalcStatus, Energy = Result

        if EncodedMolConf is None:
            CalcFailedCount += 1
            continue
        
        if not CalcStatus:
            CalcFailedCount += 1
            continue
        
        MolConf = RDKitUtil.MolFromBase64EncodedMolString(EncodedMolConf)
        
        ConfNums.append(ConfNum)
        CalcEnergyMap[ConfNum] = Energy
        MolConfsMap[ConfNum] = MolConf
    
    if CalcFailedCount:
        return (Mol, False, None, None)
    
    SortedConfNums = sorted(ConfNums, key = lambda ConfNum: CalcEnergyMap[ConfNum])
    MinEnergyConfNum = SortedConfNums[0]
    
    MinEnergy = "%.*f" % (OptionsInfo["Precision"], CalcEnergyMap[MinEnergyConfNum])  if OptionsInfo["EnergyOut"] else None
    MinEnergyMolConf = MolConfsMap[MinEnergyConfNum]
    
    ScaffoldEmbedRMSD = "%.4f" % float(MinEnergyMolConf.GetProp('EmbedRMS')) if OptionsInfo["ScaffoldRMSDOut"] else None
    MinEnergyMolConf.ClearProp('EmbedRMS')

    return (MinEnergyMolConf, True, MinEnergy, ScaffoldEmbedRMSD)

def InitializeConformerWorkerProcess(*EncodedArgs):
    """Initialize data for a conformer worker process."""
    
    global Options, OptionsInfo

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())
    
    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])
    
    # Decode RefMol...
    OptionsInfo["RefMolCore"] = RDKitUtil.MolFromBase64EncodedMolString(OptionsInfo["EncodedRefMolCore"])
    
    # Psi4 is initialized in the worker process to avoid creation of redundant Psi4
    # output files for each process...
    OptionsInfo["Psi4Initialized"]  = False

def ConformerWorkerProcess(EncodedMolInfo):
    """Process data for a conformer worker process."""

    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolConfIndex, EncodedMolConf = EncodedMolInfo

    MolConfNum = MolConfIndex
    CalcStatus = False
    Energy = None
    
    if EncodedMolConf is None:
        return [MolConfIndex, None, CalcStatus, Energy]
    
    MolConf = RDKitUtil.MolFromBase64EncodedMolString(EncodedMolConf)
    MolConfName = RDKitUtil.GetMolName(MolConf, MolConfNum)

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Processing molecule %s conformer number %s..." % (MolConfName, MolConfNum))
    
    # Perform Psi4 constrained minimization...
    CalcStatus, Energy = ConstrainAndMinimizeMoleculeUsingPsi4(OptionsInfo["psi4"], MolConf, OptionsInfo["RefMolCore"], MolConfNum)
    if not CalcStatus:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s\n" % (MolName))
        return [MolConfIndex, EncodedMolConf, CalcStatus, Energy]

    return [MolConfIndex, RDKitUtil.MolToBase64EncodedMolString(MolConf, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CalcStatus, Energy]
    
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

def RetrieveReferenceMolecule():
    """Retrieve and validate reference molecule."""
    
    RefFile = OptionsInfo["RefFile"]
    
    MiscUtil.PrintInfo("\nProcessing file %s..." % (RefFile))
    OptionsInfo["InfileParams"]["AllowEmptyMols"] = False
    ValidRefMols, RefMolCount, ValidRefMolCount  = RDKitUtil.ReadAndValidateMolecules(RefFile, **OptionsInfo["InfileParams"])
    
    if ValidRefMolCount == 0:
        MiscUtil.PrintError("The reference file, %s, contains no valid molecules." % RefFile)
    elif ValidRefMolCount > 1:
        MiscUtil.PrintWarning("The reference file, %s, contains, %d, valid molecules. Using first molecule as the reference molecule..." % (RefFile, ValidRefMolCount))
    
    RefMol = ValidRefMols[0]

    if OptionsInfo["UseScaffoldSMARTS"]:
        ScaffoldPatternMol = Chem.MolFromSmarts(OptionsInfo["ScaffoldSMARTS"])
        if ScaffoldPatternMol is None:
            MiscUtil.PrintError("Failed to create scaffold pattern molecule. The scaffold SMARTS pattern, %s, specified using \"-s, --scaffold\" option is not valid." % (OptionsInfo["ScaffoldSMARTS"]))
        
        if not RefMol.HasSubstructMatch(ScaffoldPatternMol):
            MiscUtil.PrintError("The scaffold SMARTS pattern, %s, specified using \"-s, --scaffold\" option, is missing in the first valid reference molecule." % (OptionsInfo["ScaffoldSMARTS"]))
            
    return RefMol

def SetupCoreScaffold(RefMol, Mol, MolCount):
    """Setup a reference molecule core containing common scaffold atoms between
    a pair of molecules."""

    if OptionsInfo["UseScaffoldMCS"]:
        return SetupCoreScaffoldByMCS(RefMol, Mol, MolCount)
    elif OptionsInfo["UseScaffoldSMARTS"]:
        return SetupCoreScaffoldBySMARTS(RefMol, Mol, MolCount)
    else:
        MiscUtil.PrintError("The  value, %s, specified for  \"-s, --scaffold\" option is not supported." % (OptionsInfo["Scaffold"]))
        
def SetupCoreScaffoldByMCS(RefMol, Mol, MolCount):
    """Setup a reference molecule core containing common scaffold atoms between
    a pair of molecules using MCS."""
    
    MCSParams = OptionsInfo["MCSParams"]
    Mols = [RefMol, Mol]

    MCSResultObject = rdFMCS.FindMCS(Mols, maximizeBonds = MCSParams["MaximizeBonds"], threshold = MCSParams["Threshold"], timeout = MCSParams["TimeOut"], verbose = MCSParams["Verbose"], matchValences = MCSParams["MatchValences"], ringMatchesRingOnly = MCSParams["RingMatchesRingOnly"], completeRingsOnly = MCSParams["CompleteRingsOnly"], matchChiralTag = MCSParams["MatchChiralTag"], atomCompare = MCSParams["AtomCompare"], bondCompare = MCSParams["BondCompare"], seedSmarts = MCSParams["SeedSMARTS"]) 
    
    if MCSResultObject.canceled:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("MCS failed to identify a common core scaffold between reference moecule and input molecule %s. Specify a different set of parameters using \"-m, --mcsParams\" option and try again." % (RDKitUtil.GetMolName(Mol, MolCount)))
        return None
    
    CoreNumAtoms = MCSResultObject.numAtoms
    CoreNumBonds = MCSResultObject.numBonds
    
    SMARTSCore = MCSResultObject.smartsString
    
    if not len(SMARTSCore):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("MCS failed to identify a common core scaffold between reference moecule and input molecule %s. Specify a different set of parameters using \"-m, --mcsParams\" option and try again." % (RDKitUtil.GetMolName(Mol, MolCount)))
        return None
        
    if CoreNumAtoms < MCSParams["MinNumAtoms"]:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Number of atoms, %d, in core scaffold identified by MCS is less than, %d, as specified by \"minNumAtoms\" parameter in  \"-m, --mcsParams\" option." % (CoreNumAtoms, MCSParams["MinNumAtoms"]))
        return None
    
    if CoreNumBonds < MCSParams["MinNumBonds"]:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Number of bonds, %d, in core scaffold identified by MCS is less than, %d, as specified by \"minNumBonds\" parameter in  \"-m, --mcsParams\" option." % (CoreNumBonds, MCSParams["MinNumBonds"]))
        return None

    return GenerateCoreMol(RefMol, SMARTSCore)
    
def SetupCoreScaffoldBySMARTS(RefMol, Mol, MolCount):
    """Setup a reference molecule core containing common scaffold atoms between
    a pair of molecules using specified SMARTS."""
    
    if OptionsInfo["ScaffoldPatternMol"] is None:
        OptionsInfo["ScaffoldPatternMol"] = Chem.MolFromSmarts(OptionsInfo["ScaffoldSMARTS"])
        
    if not Mol.HasSubstructMatch(OptionsInfo["ScaffoldPatternMol"]):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("The scaffold SMARTS pattern, %s, specified using \"-s, --scaffold\" option is missing in input molecule,  %s." % (OptionsInfo["ScaffoldSMARTS"], RDKitUtil.GetMolName(Mol, MolCount)))
        return None

    return GenerateCoreMol(RefMol, OptionsInfo["ScaffoldSMARTS"])

def GenerateCoreMol(RefMol, SMARTSCore):
    """Generate core molecule for embedding. """

    # Create a molecule corresponding to core atoms...
    SMARTSCoreMol = Chem.MolFromSmarts(SMARTSCore)

    # Setup a ref molecule containing core atoms with dummy atoms as
    # attachment points for atoms around the core atoms...
    Core = AllChem.ReplaceSidechains(Chem.RemoveHs(RefMol), SMARTSCoreMol)

    # Delete any substructures containing dummy atoms..
    RefMolCore = AllChem.DeleteSubstructs(Core, Chem.MolFromSmiles('*'))
    RefMolCore.UpdatePropertyCache()
    
    return RefMolCore

def ConstrainAndMinimizeMolecule(Psi4Handle, Mol, RefMolCore, MolNum = None):
    """Constrain and minimize molecule."""

    # Add hydrogens...
    Mol = Chem.AddHs(Mol, addCoords = True)

    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    
    # Setup constrained conformers...
    MolConfs, MolConfsStatus = ConstrainEmbedAndMinimizeMoleculeUsingRDKit(Mol, RefMolCore, MolNum)
    if not MolConfsStatus:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Conformation generation couldn't be performed for molecule %s: Constrained embedding failed...\n" % MolName)
        return (Mol, False, None, None)

    # Minimize conformers...
    ConfNums = []
    CalcEnergyMap = {}
    MolConfsMap = {}
    
    for ConfNum, MolConf in enumerate(MolConfs):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("\nProcessing molecule %s conformer number %s..." % (MolName, ConfNum))
            
        CalcStatus, Energy = ConstrainAndMinimizeMoleculeUsingPsi4(Psi4Handle, MolConf, RefMolCore, MolNum)
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s\n" % (MolName))
            return (Mol, False, None, None)
        
        ConfNums.append(ConfNum)
        CalcEnergyMap[ConfNum] = Energy
        MolConfsMap[ConfNum] = MolConf
    
    SortedConfNums = sorted(ConfNums, key = lambda ConfNum: CalcEnergyMap[ConfNum])
    MinEnergyConfNum = SortedConfNums[0]
    
    MinEnergy = "%.*f" % (OptionsInfo["Precision"], CalcEnergyMap[MinEnergyConfNum])  if OptionsInfo["EnergyOut"] else None
    MinEnergyMolConf = MolConfsMap[MinEnergyConfNum]
    
    ScaffoldEmbedRMSD = "%.4f" % float(MinEnergyMolConf.GetProp('EmbedRMS')) if OptionsInfo["ScaffoldRMSDOut"] else None
    MinEnergyMolConf.ClearProp('EmbedRMS')
    
    return (MinEnergyMolConf, True, MinEnergy, ScaffoldEmbedRMSD)

def ConstrainAndMinimizeMoleculeUsingPsi4(Psi4Handle, Mol, RefMolCore, MolNum, ConfID = -1):
    """Minimize molecule using Psi4."""

    # Setup a list for constrained atoms...
    ConstrainedAtomIndices = SetupConstrainedAtomIndicesForPsi4(Mol, RefMolCore)
    if len(ConstrainedAtomIndices) == 0:
        return (False, None)

    # Setup a Psi4Mol...
    Psi4Mol = SetupPsi4Mol(Psi4Handle, Mol, MolNum, ConfID)
    if Psi4Mol is None:
        return (False, None)
        
    #  Setup reference wave function...
    Reference = SetupReferenceWavefunction(Mol)
    Psi4Handle.set_options({'Reference': Reference})
    
    # Setup method name and basis set...
    MethodName, BasisSet = SetupMethodNameAndBasisSet(Mol)

    # Setup freeze list for constrained atoms...
    FreezeXYZList = [("%s xyz" % AtomIdex) for AtomIdex in ConstrainedAtomIndices]
    FreezeXYZString = " %s " % " ".join(FreezeXYZList)
    Psi4Handle.set_options({"OPTKING__frozen_cartesian": FreezeXYZString})
    
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

def ConstrainEmbedAndMinimizeMoleculeUsingRDKit(Mol, RefMolCore, MolNum = None):
    """Constrain, embed, and minimize molecule."""

    # Setup forcefield function to use for constrained minimization...
    ForceFieldFunction = None
    ForceFieldName = None
    if OptionsInfo["ConfGenerationParams"]["UseUFF"]:
        ForceFieldFunction = lambda mol, confId = -1 : AllChem.UFFGetMoleculeForceField(mol, confId = confId)
        ForeceFieldName = "UFF"
    else:
        ForceFieldFunction = lambda mol, confId = -1 : AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant = OptionsInfo["ConfGenerationParams"]["MMFFVariant"]) , confId = confId)
        ForeceFieldName = "MMFF"

    if ForceFieldFunction is None:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to setup forcefield %s for molecule: %s\n" % (ForceFieldName, RDKitUtil.GetMolName(Mol, MolNum)))
        return (None, False)
    
    MaxConfs = OptionsInfo["ConfGenerationParams"]["MaxConfs"]
    EnforceChirality = OptionsInfo["ConfGenerationParams"]["EnforceChirality"]
    UseExpTorsionAnglePrefs = OptionsInfo["ConfGenerationParams"]["UseExpTorsionAnglePrefs"]
    UseBasicKnowledge = OptionsInfo["ConfGenerationParams"]["UseBasicKnowledge"]
    UseTethers = OptionsInfo["ConfGenerationParams"]["UseTethers"]

    MolConfs = []
    ConfIDs = [ConfID for ConfID in range(0, MaxConfs)]
    
    for ConfID in ConfIDs:
        try:
            MolConf = Chem.Mol(Mol)
            AllChem.ConstrainedEmbed(MolConf, RefMolCore, useTethers = UseTethers, coreConfId = -1, randomseed = ConfID, getForceField = ForceFieldFunction, enforceChirality = EnforceChirality, useExpTorsionAnglePrefs = UseExpTorsionAnglePrefs, useBasicKnowledge = UseBasicKnowledge)
        except (ValueError, RuntimeError, Chem.rdchem.KekulizeException)  as ErrMsg:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolNum)
                MiscUtil.PrintWarning("Constrained embedding couldn't  be performed for molecule %s:\n%s\n" % (RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
            return (None, False)
        
        MolConfs.append(MolConf)

    return FilterConstrainedConformationsByRMSD(Mol, MolConfs, MolNum)

def FilterConstrainedConformationsByRMSD(Mol, MolConfs, MolNum = None):
    """Filter contarained conformations by RMSD."""
    
    EmbedRMSDCutoff = OptionsInfo["ConfGenerationParams"]["EmbedRMSDCutoff"]
    if EmbedRMSDCutoff is None or EmbedRMSDCutoff <= 0:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("\nGenerating initial ensemble of  constrained conformations by distance geometry  and forcefield for %s - EmbedRMSDCutoff: None; Size: %s" % (RDKitUtil.GetMolName(Mol, MolNum), len(MolConfs)))
        return (MolConfs, True)

    FirstMolConf = True
    SelectedMolConfs = []
    for MolConfIndex, MolConf in enumerate(MolConfs):
        if FirstMolConf:
            FirstMolConf = False
            SelectedMolConfs.append(MolConf)
            continue
        
        # Compare RMSD against all selected conformers...
        ProbeMolConf = Chem.RemoveHs(Chem.Mol(MolConf))
        IgnoreConf = False
        for SelectedMolConfIndex, SelectedMolConf in enumerate(SelectedMolConfs):
            RefMolConf = Chem.RemoveHs(Chem.Mol(SelectedMolConf))
            CalcRMSD = rdMolAlign.AlignMol(ProbeMolConf, RefMolConf)
            
            if CalcRMSD < EmbedRMSDCutoff:
                IgnoreConf = True
                break
        
        if IgnoreConf:
            continue
        
        SelectedMolConfs.append(MolConf)
        
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("\nGenerating initial ensemble of constrained conformations by distance geometry and forcefield for %s - EmbedRMSDCutoff: %s; Size: %s; Size after RMSD filtering: %s" % (RDKitUtil.GetMolName(Mol, MolNum), EmbedRMSDCutoff, len(MolConfs), len(SelectedMolConfs)))
    
    return (SelectedMolConfs, True)

def SetupConstrainedAtomIndicesForPsi4(Mol, RefMolCore, ConstrainHydrogens = False):
    """Setup a list of atom indices to be constrained during Psi4 minimizaiton."""

    AtomIndices = []
    
    # Collect matched heavy atoms along with attached hydrogens...
    for AtomIndex in Mol.GetSubstructMatch(RefMolCore):
        Atom = Mol.GetAtomWithIdx(AtomIndex)
        if Atom.GetAtomicNum() == 1:
            continue
        
        AtomIndices.append(AtomIndex)
        for AtomNbr in Atom.GetNeighbors():
            if AtomNbr.GetAtomicNum() == 1:
                if ConstrainHydrogens:
                    AtomNbrIndex = AtomNbr.GetIdx()
                    AtomIndices.append(AtomNbrIndex)
    
    AtomIndices = sorted(AtomIndices)

    # Atom indices start from 1 for Psi4 instead 0 for RDKit...
    AtomIndices = [ AtomIndex + 1 for AtomIndex in AtomIndices]
    
    return AtomIndices
    
def SetupPsi4Mol(Psi4Handle, Mol, MolNum, ConfID = -1):
    """Setup a Psi4 molecule object."""

    # Turn off recentering and reorientation to perform optimization in the
    # constrained coordinate space...
    MolGeometry = RDKitUtil.GetPsi4XYZFormatString(Mol, ConfID = ConfID, NoCom = True, NoReorient = True)

    try:
        Psi4Mol = Psi4Handle.geometry(MolGeometry)
    except Exception as ErrMsg:
        Psi4Mol = None
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to create Psi4 molecule from geometry string: %s\n" % ErrMsg)
            MolName = RDKitUtil.GetMolName(Mol, MolNum)
            MiscUtil.PrintWarning("Ignoring molecule: %s" % MolName)

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

def WriteMolecule(Writer, Mol, MolNum = None, Energy = None, ScaffoldEmbedRMSD = None, ConfID = None,):
    """Write molecule."""

    if ScaffoldEmbedRMSD is not None:
        Mol.SetProp("CoreScaffoldEmbedRMSD", ScaffoldEmbedRMSD)
            
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

def ProcessMCSParameters():
    """Set up and process MCS parameters."""

    SetupMCSParameters()
    ProcessSpecifiedMCSParameters()

def SetupMCSParameters():
    """Set up default MCS parameters."""
    
    OptionsInfo["MCSParams"] = {"MaximizeBonds": True, "Threshold": 0.9, "TimeOut": 3600, "Verbose": False, "MatchValences": True, "MatchChiralTag": False, "RingMatchesRingOnly": True, "CompleteRingsOnly": True, "AtomCompare": rdFMCS.AtomCompare.CompareElements, "BondCompare": rdFMCS.BondCompare.CompareOrder, "SeedSMARTS": "", "MinNumAtoms": 1, "MinNumBonds": 0}
    
def ProcessSpecifiedMCSParameters():
    """Process specified MCS parameters."""

    if re.match("^auto$", OptionsInfo["SpecifiedMCSParams"], re.I):
        # Nothing to process...
        return
    
    # Parse specified parameters...
    MCSParams = re.sub(" ", "", OptionsInfo["SpecifiedMCSParams"])
    if not MCSParams:
        MiscUtil.PrintError("No valid parameter name and value pairs specified using \"-m, --mcsParams\" option.")

    MCSParamsWords = MCSParams.split(",")
    if len(MCSParamsWords) % 2:
        MiscUtil.PrintError("The number of comma delimited paramater names and values, %d, specified using \"-m, --mcsParams\" option must be an even number." % (len(MCSParamsWords)))
    
    # Setup  canonical parameter names...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(OptionsInfo["MCSParams"]):
        ValidParamNames.append(ParamName)
        CanonicalParamNamesMap[ParamName.lower()] = ParamName

    # Validate and set paramater names and value...
    for Index in range(0, len(MCSParamsWords), 2):
        Name = MCSParamsWords[Index]
        Value = MCSParamsWords[Index + 1]

        CanonicalName = Name.lower()
        if  not CanonicalName in CanonicalParamNamesMap:
            MiscUtil.PrintError("The parameter name, %s, specified using \"-m, --mcsParams\" option is not a valid name. Supported parameter names: %s" % (Name,  " ".join(ValidParamNames)))

        ParamName = CanonicalParamNamesMap[CanonicalName]
        if re.match("^Threshold$", ParamName, re.I):
            Value = float(Value)
            if Value <= 0.0 or Value > 1.0 :
                MiscUtil.PrintError("The parameter value, %s, specified using \"-m, --mcsParams\" option  for parameter, %s, is not a valid value. Supported values: > 0 and <= 1.0" % (Value, Name))
            ParamValue = Value
        elif re.match("^Timeout$", ParamName, re.I):
            Value = int(Value)
            if Value <= 0:
                MiscUtil.PrintError("The parameter value, %s, specified using \"-m, --mcsParams\" option  for parameter, %s, is not a valid value. Supported values: > 0" % (Value, Name))
            ParamValue = Value
        elif re.match("^MinNumAtoms$", ParamName, re.I):
            Value = int(Value)
            if Value < 1:
                MiscUtil.PrintError("The parameter value, %s, specified using \"-m, --mcsParams\" option  for parameter, %s, is not a valid value. Supported values: >= 1" % (Value, Name))
            ParamValue = Value
        elif re.match("^MinNumBonds$", ParamName, re.I):
            Value = int(Value)
            if Value < 0:
                MiscUtil.PrintError("The parameter value, %s, specified using \"-m, --mcsParams\" option  for parameter, %s, is not a valid value. Supported values: >=0 " % (Value, Name))
            ParamValue = Value
        elif re.match("^AtomCompare$", ParamName, re.I):
            if re.match("^CompareAny$", Value, re.I):
                ParamValue = rdFMCS.AtomCompare.CompareAny
            elif re.match("^CompareElements$", Value, re.I):
                ParamValue = Chem.rdFMCS.AtomCompare.CompareElements
            elif re.match("^CompareIsotopes$", Value, re.I):
                ParamValue = Chem.rdFMCS.AtomCompare.CompareIsotopes
            else:
                MiscUtil.PrintError("The parameter value, %s, specified using \"-m, --mcsParams\" option  for parameter, %s, is not a valid value. Supported values: CompareAny CompareElements CompareIsotopes" % (Value, Name))
        elif re.match("^BondCompare$", ParamName, re.I):
            if re.match("^CompareAny$", Value, re.I):
                ParamValue = Chem.rdFMCS.BondCompare.CompareAny
            elif re.match("^CompareOrder$", Value, re.I):
                ParamValue = rdFMCS.BondCompare.CompareOrder
            elif re.match("^CompareOrderExact$", Value, re.I):
                ParamValue = rdFMCS.BondCompare.CompareOrderExact
            else:
                MiscUtil.PrintError("The parameter value, %s, specified using \"-m, --mcsParams\" option  for parameter, %s, is not a valid value. Supported values: CompareAny CompareOrder CompareOrderExact" % (Value, Name))
        elif re.match("^SeedSMARTS$", ParamName, re.I):
            if not len(Value):
                MiscUtil.PrintError("The parameter value specified using \"-m, --mcsParams\" option  for parameter, %s, is empty. " % (Name))
            ParamValue = Value
        else:
            if not re.match("^(Yes|No|True|False)$", Value, re.I):
                MiscUtil.PrintError("The parameter value, %s, specified using \"-m, --mcsParams\" option  for parameter, %s, is not a valid value. Supported values: Yes No True False" % (Value, Name))
            ParamValue = False
            if re.match("^(Yes|True)$", Value, re.I):
                ParamValue = True
        
        # Set value...
        OptionsInfo["MCSParams"][ParamName] = ParamValue

def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")
    
    # Validate options...
    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    OptionsInfo["SMILESInfileStatus"] = True if  MiscUtil.CheckFileExt(Options["--infile"], "smi csv tsv txt") else False
    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters("--infileParams", Options["--infileParams"], InfileName = Options["--infile"], ParamsDefaultInfo = ParamsDefaultInfoOverride)

    OptionsInfo["RefFile"] = Options["--reffile"]

    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters("--outfileParams", Options["--outfileParams"])
    
    OptionsInfo["Overwrite"] = Options["--overwrite"]
    
    OptionsInfo["Scaffold"] = Options["--scaffold"]
    if re.match("^auto$", Options["--scaffold"], re.I):
        UseScaffoldMCS = True
        UseScaffoldSMARTS = False
        ScaffoldSMARTS = None
    else:
        UseScaffoldMCS = False
        UseScaffoldSMARTS = True
        ScaffoldSMARTS = OptionsInfo["Scaffold"]
    
    OptionsInfo["UseScaffoldMCS"] = UseScaffoldMCS
    OptionsInfo["UseScaffoldSMARTS"] = UseScaffoldSMARTS
    OptionsInfo["ScaffoldSMARTS"] = ScaffoldSMARTS
    OptionsInfo["ScaffoldPatternMol"] = None

    OptionsInfo["SpecifiedMCSParams"] = Options["--mcsParams"]
    ProcessMCSParameters()
    
    OptionsInfo["ScaffoldRMSDOut"] = True if re.match("^yes$", Options["--scaffoldRMSDOut"], re.I) else False
    
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
    ParamsDefaultInfoOverride = {"MaxConfs": 50}
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
        MPLevelConformersMode = True
    else:
        MiscUtil.PrintError("The value, %s, specified for option \"--mpLevel\" is not valid. " % MPLevel)
    OptionsInfo["MPLevel"] = MPLevel
    OptionsInfo["MPLevelMoleculesMode"] = MPLevelMoleculesMode
    OptionsInfo["MPLevelConformersMode"] = MPLevelConformersMode
    
    OptionsInfo["Precision"] = int(Options["--precision"])
    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False

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

    MiscUtil.ValidateOptionTextValue("--energyOut", Options["--energyOut"], "yes no")
    MiscUtil.ValidateOptionTextValue("--energyUnits", Options["--energyUnits"], "Hartrees kcal/mol kJ/mol eV")
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol smi txt csv tsv")
    
    MiscUtil.ValidateOptionFilePath("-r, --reffile", Options["--reffile"])
    MiscUtil.ValidateOptionFileExt("-r, --reffile", Options["--reffile"], "sdf sd mol")
    
    MiscUtil.ValidateOptionIntegerValue("--maxIters", Options["--maxIters"], {">": 0})
    
    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
    MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])
    
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionTextValue("--mpLevel", Options["--mpLevel"], "Molecules Conformers")
    
    MiscUtil.ValidateOptionIntegerValue("-p, --precision", Options["--precision"], {">": 0})
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--scaffoldRMSDOut", Options["--scaffoldRMSDOut"], "yes no")
    
# Setup a usage string for docopt...
_docoptUsage_ = """
Psi4PerformConstrainedMinimization.py - Perform constrained minimization

Usage:
    Psi4PerformConstrainedMinimization.py [--basisSet <text>] [--confParams <Name,Value,...>] [--energyOut <yes or no>]
                                          [--energyDataFieldLabel <text>] [--energyUnits <text>] [--infileParams <Name,Value,...>]
                                          [--maxIters <number>] [--methodName <text>] [--mcsParams <Name,Value,...>]
                                          [--mp <yes or no>] [--mpLevel <Molecules or Conformers>] [--mpParams <Name, Value,...>]
                                          [ --outfileParams <Name,Value,...> ] [--overwrite] [--precision <number> ]
                                          [--psi4OptionsParams <Name,Value,...>] [--psi4RunParams <Name,Value,...>]
                                          [--quiet <yes or no>]  [--reference <text>] [--scaffold <auto or SMARTS>]
                                          [--scaffoldRMSDOut <yes or no>] [-w <dir>] -i <infile> -r <reffile> -o <outfile> 
    Psi4PerformConstrainedMinimization.py -h | --help | -e | --examples

Description:
    Generate 3D structures for molecules by performing a constrained energy
    minimization against a reference molecule. The 3D structures for molecules
    are generated using a combination of distance geometry and forcefield
    minimization followed by geometry optimization using a quantum chemistry
    method.

    An initial set of 3D conformers are generated for input molecules using
    distance geometry. A common core scaffold, corresponding to a Maximum
    Common Substructure (MCS) or an explicit SMARTS pattern,  is identified
    between a pair of input and reference molecules. The core scaffold atoms in
    input molecules are aligned against the same atoms in the reference molecule.
    The energy of aligned structures are sequentially minimized using the forcefield
    and a quantum chemistry method. The conformation with the lowest energy is
    selected to represent the final structure.

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
        Basis set to use for constrained energy minimization. Default: 6-31+G**
        for sulfur containing molecules; Otherwise, 6-31G** [ Ref 150 ]. The specified 
        value must be a valid Psi4 basis set. No validation is performed.
        
        The following list shows a representative sample of basis sets available
        in Psi4:
            
            STO-3G, 6-31G, 6-31+G, 6-31++G, 6-31G*, 6-31+G*,  6-31++G*, 
            6-31G**, 6-31+G**, 6-31++G**, 6-311G, 6-311+G, 6-311++G,
            6-311G*, 6-311+G*, 6-311++G*, 6-311G**, 6-311+G**, 6-311++G**,
            cc-pVDZ, cc-pCVDZ, aug-cc-pVDZ, cc-pVDZ-DK, cc-pCVDZ-DK, def2-SVP,
            def2-SVPD, def2-TZVP, def2-TZVPD, def2-TZVPP, def2-TZVPPD
            
    --confParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        initial 3D coordinates for molecules in input file. A common core scaffold is
        identified between a pair of input and reference molecules. The atoms in
        common core scaffold of input molecules are aligned against the reference
        molecule followed by constrained energy minimization using forcefield
        available in RDKit. The 3D structures are subsequently constrained and 
        minimized by a quantum chemistry method available in Psi4.
        
        The supported parameter names along with their default values are shown
        below:
            
            confMethod,ETKDG,
            forceField,MMFF, forceFieldMMFFVariant,MMFF94,
            enforceChirality,yes,embedRMSDCutoff,0.5,maxConfs,50,
            useTethers,yes
            
            confMethod,ETKDG   [ Possible values: SDG, ETDG, KDG, ETKDG ]
            forceField, MMFF   [ Possible values: UFF or MMFF ]
            forceFieldMMFFVariant,MMFF94   [ Possible values: MMFF94 or MMFF94s ]
            enforceChirality,yes   [ Possible values: yes or no ]
            useTethers,yes   [ Possible values: yes or no ]
            
        confMethod: Conformation generation methodology for generating initial 3D
        coordinates. Possible values: Standard Distance Geometry (SDG), Experimental
        Torsion-angle preference with Distance Geometry (ETDG), basic Knowledge-terms
        with Distance Geometry (KDG) and Experimental Torsion-angle preference
        along with basic Knowledge-terms with Distance Geometry (ETKDG) [Ref 129] .
        
        forceField: Forcefield method to use for constrained energy minimization.
        Possible values: Universal Force Field (UFF) [ Ref 81 ] or Merck Molecular
        Mechanics Force Field [ Ref 83-87 ] .
        
        enforceChirality: Enforce chirality for defined chiral centers during
        forcefield minimization.
        
        maxConfs: Maximum number of conformations to generate for each molecule
        during the generation of an initial 3D conformation ensemble using conformation
        generation methodology. The conformations are constrained and minimized using
        the specified forcefield and a quantum chemistry method. The lowest energy
        conformation is written to the output file.
        
        embedRMSDCutoff: RMSD cutoff for retaining initial set of conformers embedded
        using distance geometry and forcefield minimization. All embedded conformers
        are kept for 'None' value. Otherwise, only those conformers which are different
        from each other by the specified RMSD cutoff, 0.5 by default, are kept. The first
        embedded conformer is always retained.
        
        useTethers: Use tethers to optimize the final embedded conformation by
        applying a series of extra forces to align matching atoms to the positions of
        the core atoms. Otherwise, use simple distance constraints during the
        optimization.
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
        Method to use for constrained energy minimization. Default: B3LYP [ Ref 150 ].
        The specified value must be a valid Psi4 method name. No validation is
        performed.
        
        The following list shows a representative sample of methods available
        in Psi4:
            
            B1LYP, B2PLYP, B2PLYP-D3BJ, B2PLYP-D3MBJ, B3LYP, B3LYP-D3BJ,
            B3LYP-D3MBJ, CAM-B3LYP, CAM-B3LYP-D3BJ, HF, HF-D3BJ,  HF3c, M05,
            M06, M06-2x, M06-HF, M06-L, MN12-L, MN15, MN15-D3BJ,PBE, PBE0,
            PBEH3c, PW6B95, PW6B95-D3BJ, WB97, WB97X, WB97X-D, WB97X-D3BJ
            
    --mcsParams <Name,Value,...>  [default: auto]
        Parameter values to use for identifying a maximum common substructure
        (MCS) in between a pair of reference and input molecules.In general, it is a
        comma delimited list of parameter name and value pairs. The supported
        parameter names along with their default values are shown below:
            
            atomCompare,CompareElements,bondCompare,CompareOrder,
            maximizeBonds,yes,matchValences,yes,matchChiralTag,no,
            minNumAtoms,1,minNumBonds,0,ringMatchesRingOnly,yes,
            completeRingsOnly,yes,threshold,1.0,timeOut,3600,seedSMARTS,none
            
        Possible values for atomCompare: CompareAny, CompareElements,
        CompareIsotopes. Possible values for bondCompare: CompareAny,
        CompareOrder, CompareOrderExact.
        
        A brief description of MCS parameters taken from RDKit documentation is
        as follows:
            
            atomCompare - Controls match between two atoms
            bondCompare - Controls match between two bonds
            maximizeBonds - Maximize number of bonds instead of atoms
            matchValences - Include atom valences in the MCS match
            matchChiralTag - Include atom chirality in the MCS match
            minNumAtoms - Minimum number of atoms in the MCS match
            minNumBonds - Minimum number of bonds in the MCS match
            ringMatchesRingOnly - Ring bonds only match other ring bonds
            completeRingsOnly - Partial rings not allowed during the match
            threshold - Fraction of the dataset that must contain the MCS
            seedSMARTS - SMARTS string as the seed of the MCS
            timeout - Timeout for the MCS calculation in seconds
            
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
    -r, --reffile <reffile>
        Reference input file name containing a 3D reference molecule. A common
        core scaffold must be present in a pair of an input and reference molecules.
        Otherwise, no constrained minimization is performed on the input molecule.
    --reference <text>  [default: auto]
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
    -s, --scaffold <auto or SMARTS>  [default: auto]
        Common core scaffold between a pair of input and reference molecules used for
        constrained minimization of molecules in input file. Possible values: Auto or a
        valid SMARTS pattern. The common core scaffold is automatically detected
        corresponding to the Maximum Common Substructure (MCS) between a pair of
        reference and input molecules. A valid SMARTS pattern may be optionally specified
        for the common core scaffold.
    --scaffoldRMSDOut <yes or no>  [default: No]
        Write out RMSD value for common core alignment between a pair of input and
        reference molecules.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To perform constrained energy minimization for molecules in a SMILES file against
    a reference 3D molecule in a SD file using a common core scaffold between pairs of
    input and reference molecules identified using MCS, generating up to 50 conformations
    using ETKDG methodology followed by initial MMFF forcefield minimization and final
    energy minimization using B3LYP/6-31G** and B3LYP/6-31+G** for non-sulfur and
    sulfur containing molecules, and write out a SD file containing minimum energy
    structure corresponding to each constrained molecule, type:

       %  Psi4PerformConstrainedMinimization.py  -i Psi4SampleAlkanes.smi
          -r Psi4SampleEthane3D.sdf  -o Psi4SampleAlkanesOut.sdf

    To run the first example in a quiet mode and write out a SD file, type:

       %  Psi4PerformConstrainedMinimization.py  -q yes
          -i Psi4SampleAlkanes.smi -r Psi4SampleEthane3D.sdf
          -o Psi4SampleAlkanesOut.sdf

    To run the first example in multiprocessing mode at molecules level on all
    available CPUs without loading all data into memory and write out a SD file,
    type:

       %  Psi4PerformConstrainedMinimization.py  --mp yes
          -i Psi4SampleAlkanes.smi -r Psi4SampleEthane3D.sdf
          -o Psi4SampleAlkanesOut.sdf

    To run the first example in multiprocessing mode at conformers level on all
    available CPUs without loading all data into memory and write out a SD file,
    type:

       %  Psi4PerformConstrainedMinimization.py  --mp yes --mpLevel Conformers
          -i Psi4SampleAlkanes.smi -r Psi4SampleEthane3D.sdf
          -o Psi4SampleAlkanesOut.sdf

    To run the first example in multiprocessing mode at molecules level on all
    available CPUs by loading all data into memory and write out a SD file, type:

       %  Psi4PerformConstrainedMinimization.py  --mp yes --mpParams
          "inputDataMode,Lazy,numProcesses,4,chunkSize,8"
          -i Psi4SampleAlkanes.smi -r Psi4SampleEthane3D.sdf
          -o Psi4SampleAlkanesOut.sdf

    To rerun the first example using an explicit SMARTS string for a common core
    scaffold and write out a SD file, type:

       %  Psi4PerformConstrainedMinimization.py  --scaffold "CC"
          -i Psi4SampleAlkanes.smi -r Psi4SampleEthane3D.sdf
          -o Psi4SampleAlkanesOut.sdf

    To run the first example using a specific set of parameters for generating
    an initial set of conformers followed by energy minimization using forcefield
    and a quantum chemistry method and write out a SD file type:

       %  Psi4PerformConstrainedMinimization.py  --confParams "
          confMethod,ETKDG,forceField,MMFF, forceFieldMMFFVariant,MMFF94s,
          maxConfs,20,embedRMSDCutoff,0.5" --energyUnits "kJ/mol" -m B3LYP
          -b "6-31+G**" --maxIters 20 -i Psi4SampleAlkanes.smi -r Psi4SampleEthane3D.sdf
          -o Psi4SampleAlkanesOut.sdf

    To run the first example for molecules in a CSV SMILES file, SMILES strings
    in column 1, name column 2, and write out a SD file, type:

       %  Psi4PerformConstrainedMinimization.py  --infileParams
          "smilesDelimiter,comma,smilesTitleLine,yes,smilesColumn,1,
           smilesNameColumn,2" -i Psi4SampleAlkanes.csv
           -r Psi4SampleEthane3D.sdf -o Psi4SampleAlkanesOut.sdf

Author:

    Manish Sud(msud@san.rr.com)

See also:
    Psi4CalculateEnergy.py, Psi4CalculatePartialCharges.py, Psi4GenerateConformers.py,
    Psi4GenerateConstrainedConformers.py, Psi4PerformMinimization.py

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
