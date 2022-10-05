#!/usr/bin/env python
#
# File: Psi4PerformTorsionScan.py
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

import matplotlib.pyplot as plt
import seaborn as sns

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
    from rdkit.Chem import rdMolAlign
    from rdkit.Chem import rdMolTransforms
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
    PerformTorsionScan()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def PerformTorsionScan():
    """Perform torsion scan."""
    
    # Setup a molecule reader for input file...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    OptionsInfo["InfileParams"]["AllowEmptyMols"] = True
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])
    
    PlotExt = OptionsInfo["OutPlotParams"]["OutExt"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    MiscUtil.PrintInfo("Generating output files %s_*.sdf, %s_*Torsion*Match*.sdf, %s_*Torsion*Match*Energies.csv, %s_*Torsion*Match*Plot.%s..." % (FileName, FileName, FileName, FileName, PlotExt))

    MolCount, ValidMolCount, MinimizationFailedCount, TorsionsMissingCount, TorsionsScanFailedCount = ProcessMolecules(Mols)
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during initial minimization: %d" % MinimizationFailedCount)
    MiscUtil.PrintInfo("Number of molecules without any matched torsions: %d" % TorsionsMissingCount)
    MiscUtil.PrintInfo("Number of molecules failed during torsion scan: %d" % TorsionsScanFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + TorsionsMissingCount + MinimizationFailedCount + TorsionsScanFailedCount))

def ProcessMolecules(Mols):
    """Process molecules to perform torsion scan."""

    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols)

def ProcessMoleculesUsingSingleProcess(Mols):
    """Process molecules to perform torsion scan using a single process."""

    # Intialize Psi4...
    MiscUtil.PrintInfo("\nInitializing Psi4...")
    Psi4Handle = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = True, PrintHeader = True)
    OptionsInfo["psi4"] = Psi4Handle

    # Setup max iterations global variable...
    Psi4Util.UpdatePsi4OptionsParameters(Psi4Handle, {'GEOM_MAXITER': OptionsInfo["MaxIters"]})
    
    # Setup conversion factor for energy units...
    SetupEnergyConversionFactor(Psi4Handle)
    
    MolInfoText = "first molecule"
    if not OptionsInfo["FirstMolMode"]:
        MolInfoText = "all molecules"

    if OptionsInfo["TorsionMinimize"]:
        MiscUtil.PrintInfo("\nPerforming torsion scan on %s by generating conformation ensembles for specific torsion angles and constrained energy minimization of the ensembles..." % (MolInfoText))
    else:
        MiscUtil.PrintInfo("\nPerforming torsion scan on %s by skipping generation of conformation ensembles for specific torsion angles and constrained energy minimization of the ensembles..." % (MolInfoText))
    
    SetupTorsionsPatternsInfo()
    
    (MolCount, ValidMolCount, TorsionsMissingCount, MinimizationFailedCount, TorsionsScanFailedCount) = [0] * 5

    for Mol in Mols:
        MolCount += 1

        if OptionsInfo["FirstMolMode"] and MolCount > 1:
            MolCount -= 1
            break
        
        if not CheckAndValidateMolecule(Mol, MolCount):
            continue

        # Setup 2D coordinates for SMILES input file...
        if OptionsInfo["SMILESInfileStatus"]:
            AllChem.Compute2DCoords(Mol)
        
        ValidMolCount += 1

        Mol, MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus = PerformMinimizationAndTorsionScan(Mol, MolCount)
        
        if not MinimizationCalcStatus:
            MinimizationFailedCount += 1
            continue
        
        if not TorsionsMatchStatus:
            TorsionsMissingCount += 1
            continue

        if not TorsionsScanCalcStatus:
            TorsionsScanFailedCount += 1
            continue

    return (MolCount, ValidMolCount, MinimizationFailedCount, TorsionsMissingCount, TorsionsScanFailedCount)

def ProcessMoleculesUsingMultipleProcesses(Mols):
    """Process and minimize molecules using multiprocessing."""
    
    if OptionsInfo["MPLevelTorsionAnglesMode"]:
        return ProcessMoleculesUsingMultipleProcessesAtTorsionAnglesLevel(Mols)
    elif OptionsInfo["MPLevelMoleculesMode"]:
        return ProcessMoleculesUsingMultipleProcessesAtMoleculesLevel(Mols)
    else:
        MiscUtil.PrintError("The value, %s,  option \"--mpLevel\" is not supported." % (OptionsInfo["MPLevel"]))
        
def ProcessMoleculesUsingMultipleProcessesAtMoleculesLevel(Mols):
    """Process molecules to perform torsion scan using multiprocessing at molecules level."""

    MolInfoText = "first molecule"
    if not OptionsInfo["FirstMolMode"]:
        MolInfoText = "all molecules"

    if OptionsInfo["TorsionMinimize"]:
        MiscUtil.PrintInfo("\nPerforming torsion scan on %s using multiprocessing at molecules level by generating conformation ensembles for specific torsion angles and constrained energy minimization of the ensembles..." % (MolInfoText))
    else:
        MiscUtil.PrintInfo("\nPerforming torsion scan %s using multiprocessing at molecules level by skipping generation of conformation ensembles for specific torsion angles and constrained energy minimization of the ensembles..." % (MolInfoText))
        
    MPParams = OptionsInfo["MPParams"]
    
    # Setup data for initializing a worker process...
    InitializeWorkerProcessArgs = (MiscUtil.ObjectToBase64EncodedString(Options), MiscUtil.ObjectToBase64EncodedString(OptionsInfo))

    if OptionsInfo["FirstMolMode"]:
        Mol = Mols[0]
        Mols = [Mol]

    # Setup a encoded mols data iterable for a worker process...
    WorkerProcessDataIterable = RDKitUtil.GenerateBase64EncodedMolStrings(Mols)

    # Setup process pool along with data initialization for each process...
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

    (MolCount, ValidMolCount, TorsionsMissingCount, MinimizationFailedCount, TorsionsScanFailedCount) = [0] * 5
    
    for Result in Results:
        MolCount += 1
        
        MolIndex, EncodedMol, MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus = Result
        
        if EncodedMol is None:
            continue
        ValidMolCount += 1
    
        if not MinimizationCalcStatus:
            MinimizationFailedCount += 1
            continue
        
        if not TorsionsMatchStatus:
            TorsionsMissingCount += 1
            continue

        if not TorsionsScanCalcStatus:
            TorsionsScanFailedCount += 1
            continue
        
        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
    return (MolCount, ValidMolCount, MinimizationFailedCount, TorsionsMissingCount, TorsionsScanFailedCount)
    
def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""
    
    global Options, OptionsInfo

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())

    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Initialize torsion patterns info...
    SetupTorsionsPatternsInfo()
    
    # Psi4 is initialized in the worker process to avoid creation of redundant Psi4
    # output files for each process...
    OptionsInfo["Psi4Initialized"]  = False

def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""

    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolIndex, EncodedMol = EncodedMolInfo

    MolNum = MolIndex + 1
    (MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus) = [False] * 3
    
    if EncodedMol is None:
        return [MolIndex, None, MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus]
    
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    if not CheckAndValidateMolecule(Mol, MolNum):
        return [MolIndex, None, MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus]
    
    # Setup 2D coordinates for SMILES input file...
    if OptionsInfo["SMILESInfileStatus"]:
        AllChem.Compute2DCoords(Mol)
    
    Mol, MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus = PerformMinimizationAndTorsionScan(Mol, MolNum)
    
    return [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus]

def ProcessMoleculesUsingMultipleProcessesAtTorsionAnglesLevel(Mols):
    """Process molecules to perform torsion scan using multiprocessing at torsion angles level."""

    MolInfoText = "first molecule"
    if not OptionsInfo["FirstMolMode"]:
        MolInfoText = "all molecules"

    if OptionsInfo["TorsionMinimize"]:
        MiscUtil.PrintInfo("\nPerforming torsion scan on %s using multiprocessing at torsion angles level by generating conformation ensembles for specific torsion angles and constrained energy minimization of the ensembles..." % (MolInfoText))
    else:
        MiscUtil.PrintInfo("\nPerforming torsion scan %s using multiprocessing at torsion angles level by skipping generation of conformation ensembles for specific torsion angles and constrained energy minimization of the ensembles..." % (MolInfoText))
    
    SetupTorsionsPatternsInfo()
    
    (MolCount, ValidMolCount, TorsionsMissingCount, MinimizationFailedCount, TorsionsScanFailedCount) = [0] * 5

    for Mol in Mols:
        MolCount += 1

        if OptionsInfo["FirstMolMode"] and MolCount > 1:
            MolCount -= 1
            break
        
        if not CheckAndValidateMolecule(Mol, MolCount):
            continue

        # Setup 2D coordinates for SMILES input file...
        if OptionsInfo["SMILESInfileStatus"]:
            AllChem.Compute2DCoords(Mol)
        
        ValidMolCount += 1

        Mol, MinimizationCalcStatus, TorsionsMatchStatus, TorsionsScanCalcStatus = PerformMinimizationAndTorsionScan(Mol, MolCount, UseMultiProcessingAtTorsionAnglesLevel = True)
        
        if not MinimizationCalcStatus:
            MinimizationFailedCount += 1
            continue
        
        if not TorsionsMatchStatus:
            TorsionsMissingCount += 1
            continue

        if not TorsionsScanCalcStatus:
            TorsionsScanFailedCount += 1
            continue

    return (MolCount, ValidMolCount, MinimizationFailedCount, TorsionsMissingCount, TorsionsScanFailedCount)

def ScanSingleTorsionInMolUsingMultipleProcessesAtTorsionAnglesLevel(Mol, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, MolNum):
    """Perform torsion scan for a molecule using multiple processses at torsion angles
    level along with constrained energy minimization.
    """
    
    if OptionsInfo["MPLevelMoleculesMode"]:
        MiscUtil.PrintError("Single torison scanning for a molecule is not allowed in multiprocessing mode at molecules level.\n")

    Mols, Angles = SetupMolsForSingleTorsionScanInMol(Mol, TorsionMatches, MolNum)

    MPParams = OptionsInfo["MPParams"]
    
    # Setup data for initializing a worker process...
    
    # Track and avoid encoding TorsionsPatternsInfo as it contains RDKit molecule object...
    TorsionsPatternsInfo = OptionsInfo["TorsionsPatternsInfo"]
    OptionsInfo["TorsionsPatternsInfo"] = None
    
    InitializeWorkerProcessArgs = (MiscUtil.ObjectToBase64EncodedString(Options), MiscUtil.ObjectToBase64EncodedString(OptionsInfo))
    
    # Restore TorsionsPatternsInfo...
    OptionsInfo["TorsionsPatternsInfo"] = TorsionsPatternsInfo

    # Setup a encoded mols data iterable for a worker process...
    WorkerProcessDataIterable = GenerateBase64EncodedMolStringsWithTorsionScanInfo(Mol, (MolNum -1), Mols, Angles, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches)
    
    # Setup process pool along with data initialization for each process...
    MiscUtil.PrintInfo("\nConfiguring multiprocessing using %s method..." % ("mp.Pool.imap()" if re.match("^Lazy$", MPParams["InputDataMode"], re.I) else "mp.Pool.map()"))
    MiscUtil.PrintInfo("NumProcesses: %s; InputDataMode: %s; ChunkSize: %s\n" % (MPParams["NumProcesses"], MPParams["InputDataMode"], ("automatic" if MPParams["ChunkSize"] is None else MPParams["ChunkSize"])))
    
    ProcessPool = mp.Pool(MPParams["NumProcesses"], InitializeTorsionAngleWorkerProcess, InitializeWorkerProcessArgs)
    
    # Start processing...
    if re.match("^Lazy$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.imap(TorsionAngleWorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    elif re.match("^InMemory$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.map(TorsionAngleWorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    else:
        MiscUtil.PrintError("The value, %s, specified for \"--inputDataMode\" is not supported." % (MPParams["InputDataMode"]))

    TorsionMols = []
    TorsionEnergies = []
    TorsionAngles = []
    
    for Result in Results:
        EncodedTorsionMol, CalcStatus, Angle, Energy = Result
        
        if not CalcStatus:
            return (Mol, False, None, None, None)

        if EncodedTorsionMol is None:
            return (Mol, False, None, None, None)
        TorsionMol = RDKitUtil.MolFromBase64EncodedMolString(EncodedTorsionMol)
        
        TorsionMols.append(TorsionMol)
        TorsionEnergies.append(Energy)
        TorsionAngles.append(Angle)
    
    return (Mol, True, TorsionMols, TorsionEnergies, TorsionAngles)

def InitializeTorsionAngleWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""
    
    global Options, OptionsInfo

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())

    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Initialize torsion patterns info...
    SetupTorsionsPatternsInfo()
    
    # Psi4 is initialized in the worker process to avoid creation of redundant Psi4
    # output files for each process...
    OptionsInfo["Psi4Initialized"]  = False

def TorsionAngleWorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""

    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolIndex, EncodedMol, EncodedTorsionMol, TorsionAngle, TorsionID, TorsionPattern, EncodedTorsionPatternMol, TorsionMatches = EncodedMolInfo

    if EncodedMol is None or EncodedTorsionMol is None or EncodedTorsionPatternMol is None:
        return (None, False, None, None)
    
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    TorsionMol = RDKitUtil.MolFromBase64EncodedMolString(EncodedTorsionMol)
    TorsionPatternMol = RDKitUtil.MolFromBase64EncodedMolString(EncodedTorsionPatternMol)
    
    TorsionMol, CalcStatus, Energy = MinimizeCalculateEnergyForTorsionMol(Mol, TorsionMol, TorsionAngle, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, (MolIndex + 1))

    return (RDKitUtil.MolToBase64EncodedMolString(TorsionMol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CalcStatus, TorsionAngle, Energy)

def GenerateBase64EncodedMolStringsWithTorsionScanInfo(Mol, MolIndex, TorsionMols, TorsionAngles, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, PropertyPickleFlags = Chem.PropertyPickleOptions.AllProps):
    """Set up an iterator for generating base64 encoded molecule string for
    a torsion in a molecule along with appropriate trosion scan information.
    """
    
    for Index, TorsionMol in enumerate(TorsionMols):
        yield [MolIndex, None, None, TorsionAngles[Index], TorsionID, TorsionPattern, None, TorsionMatches] if (Mol is None or TorsionMol is None) else [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags), RDKitUtil.MolToBase64EncodedMolString(TorsionMol, PropertyPickleFlags), TorsionAngles[Index], TorsionID, TorsionPattern, RDKitUtil.MolToBase64EncodedMolString(TorsionPatternMol, PropertyPickleFlags), TorsionMatches]

def InitializePsi4ForWorkerProcess():
    """Initialize Psi4 for a worker process."""
    
    if OptionsInfo["Psi4Initialized"]:
        return

    OptionsInfo["Psi4Initialized"] = True

    if OptionsInfo["MPLevelTorsionAnglesMode"] and re.match("auto", OptionsInfo["Psi4RunParams"]["OutputFileSpecified"], re.I):
        # Run Psi4 in quiet mode during multiprocessing at Torsions level for 'auto' OutputFile...
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

def PerformMinimizationAndTorsionScan(Mol, MolNum, UseMultiProcessingAtTorsionAnglesLevel = False):
    """Perform minimization and torsions scan."""

    if not OptionsInfo["Infile3D"]:
        # Add hydrogens...
        Mol = Chem.AddHs(Mol, addCoords = True)

        Mol, MinimizationCalcStatus = MinimizeMolecule(Mol, MolNum)
        if not MinimizationCalcStatus:
            return (Mol, False, False, False)
        
    TorsionsMolInfo = SetupTorsionsMolInfo(Mol, MolNum)
    if TorsionsMolInfo["NumOfMatches"] == 0:
        return (Mol, True, False, False)
    
    Mol, ScanCalcStatus = ScanAllTorsionsInMol(Mol, TorsionsMolInfo, MolNum, UseMultiProcessingAtTorsionAnglesLevel)
    if not ScanCalcStatus:
        return (Mol, True, True, False)
    
    return (Mol, True, True, True)

def ScanAllTorsionsInMol(Mol, TorsionsMolInfo,  MolNum, UseMultiProcessingAtTorsionAnglesLevel = False):
    """Perform scans on all torsions in a molecule."""

    if TorsionsMolInfo["NumOfMatches"] == 0:
        return Mol, True

    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    
    FirstTorsionMode = OptionsInfo["FirstTorsionMode"]
    TorsionsPatternsInfo = OptionsInfo["TorsionsPatternsInfo"]

    TorsionPatternCount, TorsionScanCount, TorsionMatchCount = [0] * 3
    TorsionMaxMatches = OptionsInfo["TorsionMaxMatches"]
    
    for TorsionID in TorsionsPatternsInfo["IDs"]:
        TorsionPatternCount +=  1
        TorsionPattern = TorsionsPatternsInfo["Pattern"][TorsionID]
        TorsionPatternMol = TorsionsPatternsInfo["Mol"][TorsionID]
        
        TorsionsMatches = TorsionsMolInfo["Matches"][TorsionID]

        if TorsionsMatches is None:
            continue
        
        if FirstTorsionMode and TorsionPatternCount > 1:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Already scaned first torsion pattern, \"%s\" for molecule %s during \"%s\" value of \"--modeTorsions\" option . Abandoning torsion scan...\n" % (TorsionPattern, MolName, OptionsInfo["ModeTorsions"]))
            break

        for Index, TorsionMatches in enumerate(TorsionsMatches):
            TorsionMatchNum = Index + 1
            TorsionMatchCount +=  1

            if TorsionMatchCount > TorsionMaxMatches:
                if not OptionsInfo["QuietMode"]:
                    MiscUtil.PrintWarning("Already scaned a maximum of %s torsion matches for molecule %s specified by \"--torsionMaxMatches\" option. Abandoning torsion scan...\n" % (TorsionMaxMatches, MolName))
                break

            TmpMol, TorsionScanStatus,  TorsionMols, TorsionEnergies, TorsionAngles = ScanSingleTorsionInMol(Mol, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches,  TorsionMatchNum, MolNum, UseMultiProcessingAtTorsionAnglesLevel)
            if not TorsionScanStatus:
                continue
            
            TorsionScanCount +=  1
            GenerateOutputFiles(Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles)
        
        if TorsionMatchCount > TorsionMaxMatches:
            break
    
    if TorsionScanCount:
        GenerateStartingTorsionScanStructureOutfile(Mol, MolNum)

    Status = True if TorsionScanCount else False
    
    return (Mol, Status)

def ScanSingleTorsionInMol(Mol, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, TorsionMatchNum, MolNum, UseMultiProcessingAtTorsionAnglesLevel):
    """Perform torsion scan for a molecule along with constrained energy minimization."""
    
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("\nProcessing torsion pattern, %s, match number, %s, in molecule %s..." % (TorsionPattern, TorsionMatchNum, RDKitUtil.GetMolName(Mol, MolNum)))
        
        if OptionsInfo["TorsionMinimize"]:
            MiscUtil.PrintInfo("Generating initial ensemble of constrained conformations by distance geometry and forcefield followed by Psi4 constraned minimization to select the lowest energy structure at specific torsion angles for molecule %s..." % (RDKitUtil.GetMolName(Mol, MolNum)))
        else:
            MiscUtil.PrintInfo("Calculating single point energy using Psi4 for molecule, %s, at specific torsion angles..." % (RDKitUtil.GetMolName(Mol, MolNum)))
    
    if UseMultiProcessingAtTorsionAnglesLevel:
        return ScanSingleTorsionInMolUsingMultipleProcessesAtTorsionAnglesLevel(Mol, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, MolNum)
    else:
        return ScanSingleTorsionInMolUsingSingleProcess(Mol, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, MolNum)

def ScanSingleTorsionInMolUsingSingleProcess(Mol, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, MolNum):
    """Perform torsion scan for a molecule using single processs along with constrained
    energy minimization."""

    TorsionMols = []
    TorsionEnergies = []
    TorsionAngles = []

    Mols, Angles = SetupMolsForSingleTorsionScanInMol(Mol, TorsionMatches, MolNum)
    
    for Index, Angle in enumerate(Angles):
        TorsionMol = Mols[Index]
        TorsionMol, CalcStatus, Energy = MinimizeCalculateEnergyForTorsionMol(Mol, TorsionMol, Angle, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, MolNum)
        
        if not CalcStatus:
            return (Mol, False, None, None, None)
        
        TorsionMols.append(TorsionMol)
        TorsionEnergies.append(Energy)
        TorsionAngles.append(Angle)
    
    return (Mol, True, TorsionMols, TorsionEnergies, TorsionAngles)

def SetupMolsForSingleTorsionScanInMol(Mol, TorsionMatches, MolNum = None):
    """Setup molecules corresponding to all torsion angles in a molecule."""

    StartAngle = OptionsInfo["TorsionStart"]
    StopAngle = OptionsInfo["TorsionStop"]
    StepSize = OptionsInfo["TorsionStep"]
    
    AtomIndex1, AtomIndex2, AtomIndex3, AtomIndex4 = TorsionMatches

    TorsionMols = []
    
    TorsionAngles = [Angle for Angle in range(StartAngle, StopAngle, StepSize)]
    TorsionAngles.append(StopAngle)

    for Angle in TorsionAngles:
        TorsionMol = Chem.Mol(Mol)
        TorsionMolConf = TorsionMol.GetConformer(0)
        
        rdMolTransforms.SetDihedralDeg(TorsionMolConf, AtomIndex1, AtomIndex2, AtomIndex3, AtomIndex4, Angle)
        TorsionMols.append(TorsionMol)

    return (TorsionMols, TorsionAngles)

def MinimizeCalculateEnergyForTorsionMol(Mol, TorsionMol, TorsionAngle, TorsionID, TorsionPattern, TorsionPatternMol, TorsionMatches, MolNum):
    """"Calculate energy of a torsion molecule by performing an optional constrained
    energy minimzation.
    """

    if OptionsInfo["TorsionMinimize"]:
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, MolNum)
            MiscUtil.PrintInfo("\nProcessing torsion angle %s for molecule %s..." % (TorsionAngle, MolName))
        
        # Perform constrained minimization...
        TorsionMatchesMol = RDKitUtil.MolFromSubstructureMatch(TorsionMol, TorsionPatternMol, TorsionMatches)
        TorsionMol, CalcStatus, Energy = ConstrainAndMinimizeMolecule(TorsionMol, TorsionAngle, TorsionMatchesMol, TorsionMatches, MolNum)
        
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolNum)
                MiscUtil.PrintWarning("Failed to perform constrained minimization for molecule %s with torsion angle set to %s during torsion scan for torsion pattern %s. Abandoning torsion scan..." % (MolName, TorsionAngle, TorsionPattern))
            return (TorsionMol, False, None)
    else:
        # Setup a Psi4 molecule...
        Psi4Mol = SetupPsi4Mol(OptionsInfo["psi4"], TorsionMol, MolNum)
        if Psi4Mol is None:
            return (TorsionMol, False, None)
            
        # Calculate single point Psi4 energy...
        CalcStatus, Energy = CalculateEnergyUsingPsi4(OptionsInfo["psi4"], Psi4Mol, TorsionMol, MolNum)
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolNum)
                MiscUtil.PrintWarning("Failed to calculate Psi4 energy for molecule %s with torsion angle set to %s during torsion scan for torsion pattern %s. Abandoning torsion scan..." % (MolName, TorsionAngle, TorsionPattern))
            return (TorsionMol, False, None)
        
    return (TorsionMol, CalcStatus, Energy)
    
def SetupTorsionsMolInfo(Mol, MolNum = None):
    """Setup torsions info for a molecule."""

    TorsionsPatternsInfo = OptionsInfo["TorsionsPatternsInfo"]
    
    # Initialize...
    TorsionsMolInfo = {}
    TorsionsMolInfo["IDs"] = []
    TorsionsMolInfo["NumOfMatches"] = 0
    TorsionsMolInfo["Matches"] = {}
    for TorsionID in TorsionsPatternsInfo["IDs"]:
        TorsionsMolInfo["IDs"].append(TorsionID)
        TorsionsMolInfo["Matches"][TorsionID] = None
    
    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    UseChirality = OptionsInfo["UseChirality"]
    
    for TorsionID in TorsionsPatternsInfo["IDs"]:
        # Match torsions..
        TorsionPattern = TorsionsPatternsInfo["Pattern"][TorsionID]
        TorsionPatternMol = TorsionsPatternsInfo["Mol"][TorsionID]
        TorsionsMatches = RDKitUtil.FilterSubstructureMatchesByAtomMapNumbers(Mol, TorsionPatternMol, Mol.GetSubstructMatches(TorsionPatternMol, useChirality = UseChirality))

        # Validate tosion matches...
        ValidTorsionsMatches = []
        for Index, TorsionMatch in enumerate(TorsionsMatches):
            if len(TorsionMatch) != 4:
                if not OptionsInfo["QuietMode"]:
                    MiscUtil.PrintWarning("Ignoring invalid torsion match to atom indices, %s, for torsion pattern, %s, in molecule %s: It must match exactly 4 atoms." % (TorsionMatch, TorsionPattern, MolName))
                continue
            
            if not RDKitUtil.AreAtomIndicesSequentiallyConnected(Mol, TorsionMatch):
                if not OptionsInfo["QuietMode"]:
                    MiscUtil.PrintInfo("")
                    MiscUtil.PrintWarning("Invalid torsion match to atom indices, %s, for torsion pattern, %s, in molecule %s: Matched atom indices must be sequentially connected." % (TorsionMatch, TorsionPattern, MolName))
                    MiscUtil.PrintWarning("Reordering matched atom indices in a sequentially connected manner...")
                
                Status, ReorderdTorsionMatch = RDKitUtil.ReorderAtomIndicesInSequentiallyConnectedManner(Mol, TorsionMatch)
                if Status:
                    TorsionMatch = ReorderdTorsionMatch
                    if not OptionsInfo["QuietMode"]:
                        MiscUtil.PrintWarning("Successfully reordered torsion match to atom indices, %s, for torsion pattern, %s, in molecule %s: Matched atom indices are now sequentially connected." % (TorsionMatch, TorsionPattern, MolName))
                else:
                    if not OptionsInfo["QuietMode"]:
                        MiscUtil.PrintWarning("Ignoring torsion match. Failed to reorder torsion match to atom indices, %s, for torsion pattern, %s, in molecule %s: Matched atom indices are not sequentially connected." % (TorsionMatch, TorsionPattern, MolName))
                    continue
            
            Bond = Mol.GetBondBetweenAtoms(TorsionMatch[1], TorsionMatch[2])
            if Bond.IsInRing():
                if not OptionsInfo["QuietMode"]:
                    MiscUtil.PrintWarning("Ignoring invalid torsion match to atom indices, %s, for torsion pattern, %s, in molecule %s: Matched atom indices, %s and %s, are not allowed to be in a ring." % (TorsionMatch, TorsionPattern, MolName, TorsionMatch[1], TorsionMatch[2]))
                continue
            
            # Filter matched torsions...
            if OptionsInfo["FilterTorsionsByAtomIndicesMode"]:
                InvalidAtomIndices = []
                for AtomIndex in TorsionMatch:
                    if AtomIndex not in OptionsInfo["TorsionsFilterByAtomIndicesList"]:
                        InvalidAtomIndices.append(AtomIndex)
                if len(InvalidAtomIndices):
                    if not OptionsInfo["QuietMode"]:
                        MiscUtil.PrintWarning("Ignoring invalid torsion match to atom indices, %s, for torsion pattern, %s, in molecule %s: Matched atom indices, %s,  must be present in the list, %s, specified using option \"--torsionsFilterbyAtomIndices\"." % (TorsionMatch, TorsionPattern, MolName, InvalidAtomIndices, OptionsInfo["TorsionsFilterByAtomIndicesList"]))
                    continue
            
            ValidTorsionsMatches.append(TorsionMatch)
        
        # Track valid matches...
        if len(ValidTorsionsMatches):
            TorsionsMolInfo["NumOfMatches"] += len(ValidTorsionsMatches)
            TorsionsMolInfo["Matches"][TorsionID] = ValidTorsionsMatches
        
    if TorsionsMolInfo["NumOfMatches"] == 0:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to match any torsions  in molecule %s" % (MolName))

    return TorsionsMolInfo

def SetupTorsionsPatternsInfo():
    """Setup torsions patterns info."""

    TorsionsPatternsInfo = {}
    TorsionsPatternsInfo["IDs"] = []
    TorsionsPatternsInfo["Pattern"] = {}
    TorsionsPatternsInfo["Mol"] = {}

    TorsionID = 0
    for TorsionPattern in OptionsInfo["TorsionPatternsList"]:
        TorsionID += 1
        
        TorsionMol = Chem.MolFromSmarts(TorsionPattern)
        if TorsionMol is None:
            MiscUtil.PrintError("Failed to create torsion pattern molecule. The torsion SMILES/SMARTS pattern, \"%s\", specified using \"-t, --torsions\" option is not valid." % (TorsionPattern))
        
        TorsionsPatternsInfo["IDs"].append(TorsionID)
        TorsionsPatternsInfo["Pattern"][TorsionID] = TorsionPattern
        TorsionsPatternsInfo["Mol"][TorsionID] = TorsionMol

    OptionsInfo["TorsionsPatternsInfo"] = TorsionsPatternsInfo
    
def MinimizeMolecule(Mol, MolNum = None):
    """Minimize molecule."""

    return GenerateAndMinimizeConformersUsingForceField(Mol, MolNum)

def GenerateAndMinimizeConformersUsingForceField(Mol, MolNum = None):
    """Generate and minimize conformers for a molecule to get the lowest energy conformer
    as the minimized structure."""

    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    
    # Setup conformers...
    ConfIDs = EmbedMolecule(Mol, MolNum)
    if not len(ConfIDs):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s: Embedding failed...\n" % MolName)
        return (Mol, False)
    
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Performing initial minimization of molecule, %s, using forcefield by generating a conformation ensemble and selecting the lowest energy conformer - EmbedRMSDCutoff: %s; Size: %s; Size after RMSD filtering: %s" % (MolName, OptionsInfo["ConfGenerationParams"]["EmbedRMSDCutoff"], OptionsInfo["ConfGenerationParams"]["MaxConfs"], len(ConfIDs)))
    
    # Minimize conformers...
    CalcEnergyMap = {}
    for ConfID in ConfIDs:
        # Perform forcefield minimization...
        Status, ConvergeStatus = MinimizeMoleculeUsingForceField(Mol, MolNum, ConfID)
        if not Status:
            return (Mol, False)
        
        EnergyStatus, Energy = CalculateEnergyUsingForceField(Mol, ConfID)
        if not EnergyStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolNum)
                MiscUtil.PrintWarning("Failed to retrieve calculated energy for conformation number %d of molecule %s. Try again after removing any salts or cleaing up the molecule...\n" % (ConfID, MolName))
            return (Mol, False)
        
        if ConvergeStatus != 0:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Minimization using forcefield failed to converge for molecule %s in %d steps. Try using higher value for \"maxIters\" in \"--confParams\" option...\n" % (MolName, OptionsInfo["ConfGenerationParams"]["MaxIters"]))
        
        CalcEnergyMap[ConfID] = Energy
    
    SortedConfIDs = sorted(ConfIDs, key = lambda ConfID: CalcEnergyMap[ConfID])
    MinEnergyConfID = SortedConfIDs[0]
        
    for ConfID in [Conf.GetId() for Conf in Mol.GetConformers()]:
        if ConfID == MinEnergyConfID:
            continue
        Mol.RemoveConformer(ConfID)
    
    # Set ConfID to 0 for MinEnergyConf...
    Mol.GetConformer(MinEnergyConfID).SetId(0)

    return (Mol, True)

def ConstrainAndMinimizeMolecule(Mol, TorsionAngle, RefMolCore, RefMolMatches, MolNum = None):
    """Constrain and minimize molecule."""

    # TorsionMol, CalcStatus, Energy
    MolName = RDKitUtil.GetMolName(Mol, MolNum)

    # Setup constrained conformers...
    MolConfs, MolConfsStatus = ConstrainEmbedAndMinimizeMoleculeUsingRDKit(Mol, RefMolCore, RefMolMatches, MolNum)
    if not MolConfsStatus:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Conformation generation couldn't be performed for molecule %s: Constrained embedding failed...\n" % MolName)
        return (Mol, False, None)

    # Minimize conformers...
    ConfNums = []
    CalcEnergyMap = {}
    MolConfsMap = {}
    
    for ConfNum, MolConf in enumerate(MolConfs):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintInfo("\nPerforming constrained minimization using Psi4 for molecule, %s, conformer number, %s, at torsion angle %s..." % (MolName, ConfNum, TorsionAngle))

        CalcStatus, Energy = ConstrainAndMinimizeMoleculeUsingPsi4(OptionsInfo["psi4"], MolConf, RefMolCore, RefMolMatches, MolNum)
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Minimization couldn't be performed for molecule %s\n" % (MolName))
            return (Mol, False, None)
        
        ConfNums.append(ConfNum)
        CalcEnergyMap[ConfNum] = Energy
        MolConfsMap[ConfNum] = MolConf
    
    SortedConfNums = sorted(ConfNums, key = lambda ConfNum: CalcEnergyMap[ConfNum])
    MinEnergyConfNum = SortedConfNums[0]
    
    MinEnergy = CalcEnergyMap[MinEnergyConfNum]
    MinEnergyMolConf = MolConfsMap[MinEnergyConfNum]
    
    MinEnergyMolConf.ClearProp('EmbedRMS')
    
    return (MinEnergyMolConf, True, MinEnergy)

def ConstrainAndMinimizeMoleculeUsingPsi4(Psi4Handle, Mol, RefMolCore, RefMolMatches, MolNum, ConfID = -1):
    """Minimize molecule using Psi4."""

    # Setup a list for constrained atoms...
    ConstrainedAtomIndices = SetupConstrainedAtomIndicesForPsi4(Mol, RefMolCore, RefMolMatches)
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
    
def ConstrainEmbedAndMinimizeMoleculeUsingRDKit(Mol, RefMolCore, RefMolMatches, MolNum = None):
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
    
    MaxConfs = OptionsInfo["ConfGenerationParams"]["MaxConfsTorsions"]
    EnforceChirality = OptionsInfo["ConfGenerationParams"]["EnforceChirality"]
    UseExpTorsionAnglePrefs = OptionsInfo["ConfGenerationParams"]["UseExpTorsionAnglePrefs"]
    UseBasicKnowledge = OptionsInfo["ConfGenerationParams"]["UseBasicKnowledge"]
    UseTethers = OptionsInfo["ConfGenerationParams"]["UseTethers"]

    MolConfs = []
    ConfIDs = [ConfID for ConfID in range(0, MaxConfs)]
    
    for ConfID in ConfIDs:
        try:
            MolConf = Chem.Mol(Mol)
            RDKitUtil.ConstrainAndEmbed(MolConf, RefMolCore, coreMatchesMol = RefMolMatches, useTethers = UseTethers, coreConfId = -1, randomseed = ConfID, getForceField = ForceFieldFunction, enforceChirality = EnforceChirality, useExpTorsionAnglePrefs = UseExpTorsionAnglePrefs, useBasicKnowledge = UseBasicKnowledge)
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

    return ConfIDs

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

def CalculateEnergyUsingForceField(Mol, ConfID = None):
    """Calculate energy."""

    Status = True
    Energy = None

    if ConfID is None:
        ConfID = -1
    
    if OptionsInfo["ConfGenerationParams"]["UseUFF"]:
        UFFMoleculeForcefield = AllChem.UFFGetMoleculeForceField(Mol, confId = ConfID)
        if UFFMoleculeForcefield is None:
            Status = False
        else:
            Energy = UFFMoleculeForcefield.CalcEnergy()
    elif OptionsInfo["ConfGenerationParams"]["UseMMFF"]:
        MMFFMoleculeProperties = AllChem.MMFFGetMoleculeProperties(Mol, mmffVariant = OptionsInfo["ConfGenerationParams"]["MMFFVariant"])
        MMFFMoleculeForcefield = AllChem.MMFFGetMoleculeForceField(Mol, MMFFMoleculeProperties, confId = ConfID)
        if MMFFMoleculeForcefield is None:
            Status = False
        else:
            Energy = MMFFMoleculeForcefield.CalcEnergy()
    else:
        MiscUtil.PrintError("Couldn't retrieve conformer energy: Specified forcefield, %s, is not supported" % OptionsInfo["ConfGenerationParams"]["ForceField"])
    
    return (Status, Energy)

def CalculateEnergyUsingPsi4(Psi4Handle, Psi4Mol, Mol, MolNum = None):
    """Calculate single point energy using Psi4."""
    
    Status = False
    Energy = None
    
    #  Setup reference wave function...
    Reference = SetupReferenceWavefunction(Mol)
    Psi4Handle.set_options({'Reference': Reference})
    
    # Setup method name and basis set...
    MethodName, BasisSet = SetupMethodNameAndBasisSet(Mol)
    
    Status, Energy = Psi4Util.CalculateSinglePointEnergy(Psi4Handle, Psi4Mol, MethodName, BasisSet, Quiet = OptionsInfo["QuietMode"])
    
    # Convert energy units...
    if Status:
        if OptionsInfo["ApplyEnergyConversionFactor"]:
            Energy = Energy * OptionsInfo["EnergyConversionFactor"]

    # Clean up
    PerformPsi4Cleanup(Psi4Handle)
    
    return (Status, Energy)

def SetupConstrainedAtomIndicesForPsi4(Mol, RefMolCore, RefMolMatches, ConstrainHydrogens = False):
    """Setup a list of atom indices to be constrained during Psi4 minimizaiton."""

    AtomIndices = []

    if RefMolMatches is None:
        ConstrainAtomIndices = Mol.GetSubstructMatch(RefMolCore)
    else:
        ConstrainAtomIndices = RefMolMatches
        
    # Collect matched heavy atoms along with attached hydrogens...
    for AtomIndex in ConstrainAtomIndices:
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

    if OptionsInfo["Infile3D"]:
        if not Mol.GetConformer().Is3D():
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("3D tag is not set for molecule: %s\n" % MolName)
    
    if OptionsInfo["Infile3D"]:
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

def GenerateStartingTorsionScanStructureOutfile(Mol, MolNum):
    """Write out the structure of molecule used for starting tosion scan."""
    
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    MolName = GetOutputFileMolName(Mol, MolNum)
    
    Outfile  = "%s_%s.%s" % (FileName, MolName, FileExt)
    
    # Set up a molecule writer...
    Writer = RDKitUtil.MoleculesWriter(Outfile, **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintWarning("Failed to setup a writer for output fie %s " % Outfile)
        return
    
    Writer.write(Mol)
    
    if Writer is not None:
        Writer.close()

def GenerateOutputFiles(Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles):
    """Generate output files."""
    
    StructureOutfile, EnergyTextOutfile, PlotOutfile = SetupOutputFileNames(Mol, MolNum, TorsionID, TorsionMatchNum)
    
    GenerateScannedTorsionsStructureOutfile(StructureOutfile, Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles)
    GenerateEnergyTextOutfile(EnergyTextOutfile, Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles)
    GeneratePlotOutfile(PlotOutfile, Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles)

def GenerateScannedTorsionsStructureOutfile(Outfile, Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles):
    """Write out structures generated after torsion scan along with associated data."""

    # Set up a molecule writer...
    Writer = RDKitUtil.MoleculesWriter(Outfile, **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintWarning("Failed to setup a writer for output fie %s " % Outfile)
        return
    
    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    
    RelativeTorsionEnergies = SetupRelativeEnergies(TorsionEnergies)
    for Index, TorsionMol in enumerate(TorsionMols):
        TorsionAngle = "%s" % TorsionAngles[Index]
        TorsionMol.SetProp("Torsion_Angle", TorsionAngle)
        
        TorsionEnergy = "%.*f" % (OptionsInfo["Precision"], TorsionEnergies[Index])
        TorsionMol.SetProp(OptionsInfo["EnergyDataFieldLabel"], TorsionEnergy)

        RelativeTorsionEnergy = "%.*f" % (OptionsInfo["Precision"], RelativeTorsionEnergies[Index])
        TorsionMol.SetProp(OptionsInfo["EnergyRelativeDataFieldLabel"], RelativeTorsionEnergy)

        TorsionMolName = "%s_Deg%s" % (MolName, TorsionAngle)
        TorsionMol.SetProp("_Name", TorsionMolName)
        
        Writer.write(TorsionMol)
        
    if Writer is not None:
        Writer.close()
    
def GenerateEnergyTextOutfile(Outfile, Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles):
    """Write out torsion angles and energies."""

    # Setup a writer...
    Writer = open(Outfile, "w")
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % Outfile)

    # Write headers...
    Writer.write("TorsionAngle,%s,%s\n" % (OptionsInfo["EnergyDataFieldLabel"], OptionsInfo["EnergyRelativeDataFieldLabel"]))

    RelativeTorsionEnergies = SetupRelativeEnergies(TorsionEnergies)
    for Index, TorsionAngle in enumerate(TorsionAngles):
        TorsionEnergy = "%.*f" % (OptionsInfo["Precision"], TorsionEnergies[Index])
        RelativeTorsionEnergy = "%.*f" % (OptionsInfo["Precision"], RelativeTorsionEnergies[Index])
        Writer.write("%d,%s,%s\n" % (TorsionAngle, TorsionEnergy, RelativeTorsionEnergy))

    if Writer is not None:
        Writer.close()
    
def GeneratePlotOutfile(Outfile, Mol, MolNum, TorsionID, TorsionMatchNum, TorsionMols, TorsionEnergies, TorsionAngles):
    """Generate a plot corresponding to torsion angles and energies."""

    OutPlotParams = OptionsInfo["OutPlotParams"]

    # Initialize seaborn and matplotlib paramaters...
    if not OptionsInfo["OutPlotInitialized"]:
        OptionsInfo["OutPlotInitialized"] = True
        RCParams = {"figure.figsize":(OutPlotParams["Width"], OutPlotParams["Height"]),
                    "axes.titleweight": OutPlotParams["TitleWeight"],
                    "axes.labelweight": OutPlotParams["LabelWeight"]}
        sns.set(context = OutPlotParams["Context"], style = OutPlotParams["Style"], palette = OutPlotParams["Palette"], font = OutPlotParams["Font"], font_scale = OutPlotParams["FontScale"], rc = RCParams)

    # Create a new figure...
    plt.figure()

    if OptionsInfo["OutPlotRelativeEnergy"]: 
        TorsionEnergies = SetupRelativeEnergies(TorsionEnergies)
    
    # Draw plot...
    PlotType = OutPlotParams["Type"]
    if re.match("linepoint", PlotType, re.I):
        Axis = sns.lineplot(x = TorsionAngles, y = TorsionEnergies, marker = "o",  legend = False)
    elif re.match("scatter", PlotType, re.I):
        Axis = sns.scatterplot(x = TorsionAngles, y = TorsionEnergies, legend = False)
    elif re.match("line", PlotType, re.I):
        Axis = sns.lineplot(x = TorsionAngles, y = TorsionEnergies, legend = False)
    else:
        MiscUtil.PrintError("The value, %s, specified for \"type\" using option \"--outPlotParams\" is not supported. Valid plot types: linepoint, scatter or line" % (PlotType))

    # Setup title and labels...
    Title = OutPlotParams["Title"]
    if OptionsInfo["OutPlotTitleTorsionSpec"]:
        TorsionPattern = OptionsInfo["TorsionsPatternsInfo"]["Pattern"][TorsionID]
        Title = "%s: %s" % (OutPlotParams["Title"], TorsionPattern)

    # Set labels and title...
    Axis.set(xlabel = OutPlotParams["XLabel"], ylabel = OutPlotParams["YLabel"], title = Title)
    
    # Save figure...
    plt.savefig(Outfile)

    # Close the plot...
    plt.close()

def SetupRelativeEnergies(Energies):
    """Set up a list of relative energies."""
    
    SortedEnergies = sorted(Energies)
    MinEnergy = SortedEnergies[0]
    RelativeEnergies = [(Energy - MinEnergy) for Energy in Energies]

    return RelativeEnergies
    
def SetupOutputFileNames(Mol, MolNum, TorsionID, TorsionMatchNum):
    """Setup names of output files."""
    
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    MolName = GetOutputFileMolName(Mol, MolNum)
    
    OutfileRoot  = "%s_%s_Torsion%s_Match%s" % (FileName, MolName, TorsionID, TorsionMatchNum)
    
    StructureOutfile = "%s.%s" % (OutfileRoot, FileExt)
    EnergyTextOutfile = "%s_Energies.csv" % (OutfileRoot)
    PlotExt = OptionsInfo["OutPlotParams"]["OutExt"]
    PlotOutfile = "%s_Plot.%s" % (OutfileRoot, PlotExt)

    return (StructureOutfile, EnergyTextOutfile, PlotOutfile)

def GetOutputFileMolName(Mol, MolNum):
    """Get output file prefix."""
    
    MolName = "Mol%s" % MolNum
    if OptionsInfo["OutfileMolName"]:
        MolName = re.sub("[^a-zA-Z0-9]", "_", RDKitUtil.GetMolName(Mol, MolNum), re.I)

    return MolName

def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["ModeMols"] = Options["--modeMols"]
    OptionsInfo["FirstMolMode"] = True if re.match("^First$", Options["--modeMols"], re.I) else False
    
    OptionsInfo["ModeTorsions"] = Options["--modeTorsions"]
    OptionsInfo["FirstTorsionMode"] = True if re.match("^First$", Options["--modeTorsions"], re.I) else False
    
    OptionsInfo["Infile"] = Options["--infile"]
    OptionsInfo["SMILESInfileStatus"] = True if  MiscUtil.CheckFileExt(Options["--infile"], "smi csv tsv txt") else False
    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters("--infileParams", Options["--infileParams"], InfileName = Options["--infile"], ParamsDefaultInfo = ParamsDefaultInfoOverride)
    OptionsInfo["Infile3D"] = True if re.match("^yes$", Options["--infile3D"], re.I) else False
    
    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters("--outfileParams", Options["--outfileParams"])
    
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
    ParamsDefaultInfoOverride = {"MaxConfs": 250, "MaxConfsTorsions": 50}
    OptionsInfo["ConfGenerationParams"] = MiscUtil.ProcessOptionConformerParameters("--confParams", Options["--confParams"], ParamsDefaultInfoOverride)
    
    # Energy units and label...
    OptionsInfo["EnergyUnits"] = Options["--energyUnits"]
    
    EnergyDataFieldLabel = Options["--energyDataFieldLabel"]
    if re.match("^auto$", EnergyDataFieldLabel, re.I):
        EnergyDataFieldLabel = "Psi4_Energy (%s)" % Options["--energyUnits"]
    OptionsInfo["EnergyDataFieldLabel"] = EnergyDataFieldLabel
    
    EnergyRelativeDataFieldLabel = Options["--energyRelativeDataFieldLabel"]
    if re.match("^auto$", EnergyRelativeDataFieldLabel, re.I):
        EnergyRelativeDataFieldLabel = "Psi4_Relative_Energy (%s)" % Options["--energyUnits"]
    OptionsInfo["EnergyRelativeDataFieldLabel"] = EnergyRelativeDataFieldLabel

    # Plot parameters...
    OptionsInfo["OutfileMolName"] = True if re.match("^yes$", Options["--outfileMolName"], re.I) else False
    OptionsInfo["OutPlotRelativeEnergy"] = True if re.match("^yes$", Options["--outPlotRelativeEnergy"], re.I) else False
    OptionsInfo["OutPlotTitleTorsionSpec"] = True if re.match("^yes$", Options["--outPlotTitleTorsionSpec"], re.I) else False
    
    # The default width and height, 10.0 and 7.5, map to aspect raito of 16/9 (1.778)...
    if OptionsInfo["OutPlotRelativeEnergy"]:
        EnergyLabel = "Relative Energy (%s)" % OptionsInfo["EnergyUnits"]
    else:
        EnergyLabel = "Energy (%s)" % OptionsInfo["EnergyUnits"]
        
    DefaultValues = {'Type': 'linepoint', 'Width': 10.0, 'Height': 5.6, 'Title': 'Psi4 Torsion Scan', 'XLabel': 'Torsion Angle (degrees)', 'YLabel': EnergyLabel}
    OptionsInfo["OutPlotParams"] = MiscUtil.ProcessOptionSeabornPlotParameters("--outPlotParams", Options["--outPlotParams"], DefaultValues)
    if not re.match("^(linepoint|scatter|Line)$", OptionsInfo["OutPlotParams"]["Type"], re.I):
        MiscUtil.PrintError("The value, %s, specified for \"type\" using option \"--outPlotParams\" is not supported. Valid plot types: linepoint, scatter or line" % (OptionsInfo["OutPlotParams"]["Type"]))
    
    OptionsInfo["OutPlotInitialized"] = False
    
    OptionsInfo["Overwrite"] = Options["--overwrite"]

    OptionsInfo["MaxIters"] = int(Options["--maxIters"])
    
    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])
    
    # Multiprocessing level...
    MPLevelMoleculesMode = False
    MPLevelTorsionAnglesMode = False
    MPLevel = Options["--mpLevel"]
    if re.match("^Molecules$", MPLevel, re.I):
        MPLevelMoleculesMode = True
    elif re.match("^TorsionAngles$", MPLevel, re.I):
        MPLevelTorsionAnglesMode = True
    else:
        MiscUtil.PrintError("The value, %s, specified for option \"--mpLevel\" is not valid. " % MPLevel)
    OptionsInfo["MPLevel"] = MPLevel
    OptionsInfo["MPLevelMoleculesMode"] = MPLevelMoleculesMode
    OptionsInfo["MPLevelTorsionAnglesMode"] = MPLevelTorsionAnglesMode
    
    OptionsInfo["Precision"] = int(Options["--precision"])
    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False
    
    # Procsss and validate specified SMILES/SMARTS torsion patterns...
    TorsionPatterns = Options["--torsions"]
    TorsionPatternsList = []
    for TorsionPattern in TorsionPatterns.split(","):
        TorsionPattern = TorsionPattern.strip()
        if not len(TorsionPattern):
            MiscUtil.PrintError("Empty value specified for SMILES/SMARTS pattern in  \"-t, --torsions\" option: %s" % TorsionPatterns)
        
        TorsionMol = Chem.MolFromSmarts(TorsionPattern)
        if TorsionMol is None:
            MiscUtil.PrintError("Failed to create torsion pattern molecule. The torsion SMILES/SMARTS pattern, \"%s\", specified using \"-t, --torsions\" option, \"%s\",  is not valid." % (TorsionPattern, TorsionPatterns))
        TorsionPatternsList.append(TorsionPattern)
    
    OptionsInfo["TorsionPatterns"] = TorsionPatterns
    OptionsInfo["TorsionPatternsList"] = TorsionPatternsList
    
    # Process and validate any specified torsion atom indices for filtering torsion matches...
    TorsionsFilterByAtomIndices =  Options["--torsionsFilterbyAtomIndices"]
    TorsionsFilterByAtomIndicesList = []
    if not re.match("^None$", TorsionsFilterByAtomIndices, re.I):
        for AtomIndex in TorsionsFilterByAtomIndices.split(","):
            AtomIndex = AtomIndex.strip()
            if not MiscUtil.IsInteger(AtomIndex):
                MiscUtil.PrintError("The value specified, %s, for option \"--torsionsFilterbyAtomIndices\" must be an integer." % AtomIndex)
            AtomIndex = int(AtomIndex)
            if AtomIndex < 0:
                MiscUtil.PrintError("The value specified, %s, for option \"--torsionsFilterbyAtomIndices\" must be >= 0." % AtomIdex)
            TorsionsFilterByAtomIndicesList.append(AtomIndex)

        if len(TorsionsFilterByAtomIndicesList) < 4:
            MiscUtil.PrintError("The number of values, %s,  specified, %s, for option \"--torsionsFilterbyAtomIndices\" must be >=4." % (len(TorsionsFilterByAtomIndicesList), TorsionsFilterByAtomIndices))
            
    OptionsInfo["TorsionsFilterByAtomIndices"] = TorsionsFilterByAtomIndices
    OptionsInfo["TorsionsFilterByAtomIndicesList"] = TorsionsFilterByAtomIndicesList
    OptionsInfo["FilterTorsionsByAtomIndicesMode"] = True if len(TorsionsFilterByAtomIndicesList) > 0 else False
    
    OptionsInfo["TorsionMaxMatches"] = int(Options["--torsionMaxMatches"])
    OptionsInfo["TorsionMinimize"] = True if re.match("^yes$", Options["--torsionMinimize"], re.I) else False

    TorsionRange = Options["--torsionRange"]
    TorsionRangeWords = TorsionRange.split(",")
    
    TorsionStart = int(TorsionRangeWords[0])
    TorsionStop = int(TorsionRangeWords[1])
    TorsionStep = int(TorsionRangeWords[2])
    
    if TorsionStart >= TorsionStop:
        MiscUtil.PrintError("The start value, %d, specified for option \"--torsionRange\" in string \"%s\" must be less than stop value, %s." % (TorsionStart, Options["--torsionRange"], TorsionStop))
    if TorsionStep == 0:
        MiscUtil.PrintError("The step value, %d, specified for option \"--torsonRange\" in string \"%s\" must be > 0." % (TorsionStep, Options["--torsionRange"]))
    if TorsionStep >= (TorsionStop - TorsionStart):
        MiscUtil.PrintError("The step value, %d, specified for option \"--torsonRange\" in string \"%s\" must be less than, %s." % (TorsionStep, Options["--torsionRange"], (TorsionStop - TorsionStart)))
    
    if TorsionStart < 0:
        if TorsionStart < -180:
            MiscUtil.PrintError("The start value, %d, specified for option \"--torsionRange\" in string \"%s\" must be  >= -180 to use scan range from -180 to 180." % (TorsionStart, Options["--torsionRange"]))
        if TorsionStop > 180:
            MiscUtil.PrintError("The stop value, %d, specified for option \"--torsionRange\" in string \"%s\" must be <= 180 to use scan range from -180 to 180." % (TorsionStop, Options["--torsionRange"]))
    else:
        if TorsionStop > 360:
            MiscUtil.PrintError("The stop value, %d, specified for option \"--torsionRange\" in string \"%s\" must be  <= 360 to use scan range from 0 to 360." % (TorsionStop, Options["--torsionRange"]))
    
    OptionsInfo["TorsionRange"] = TorsionRange
    OptionsInfo["TorsionStart"] = TorsionStart
    OptionsInfo["TorsionStop"] = TorsionStop
    OptionsInfo["TorsionStep"] = TorsionStep
    
    OptionsInfo["UseChirality"] = True if re.match("^yes$", Options["--useChirality"], re.I) else False
    
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

    MiscUtil.ValidateOptionTextValue("--energyUnits", Options["--energyUnits"], "Hartrees kcal/mol kJ/mol eV")
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol smi txt csv tsv")
    MiscUtil.ValidateOptionTextValue("--infile3D", Options["--infile3D"], "yes no")

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
    MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])
    
    if not Options["--overwrite"]:
        FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
        FileNames = glob.glob("%s_*" % FileName)
        if len(FileNames):
            MiscUtil.PrintError("The outfile names, %s_*, generated from file specified, %s, for option \"-o, --outfile\" already exist. Use option \"--overwrite\" or \"--ov\"  and try again.\n" % (FileName, Options["--outfile"]))
    
    MiscUtil.ValidateOptionTextValue("--outPlotRelativeEnergy", Options["--outPlotRelativeEnergy"], "yes no")
    MiscUtil.ValidateOptionTextValue("--outPlotTitleTorsionSpec", Options["--outPlotTitleTorsionSpec"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--outfileMolName ", Options["--outfileMolName"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--modeMols", Options["--modeMols"], "First All")
    MiscUtil.ValidateOptionTextValue("--modeTorsions", Options["--modeTorsions"], "First All")
    
    MiscUtil.ValidateOptionIntegerValue("--maxIters", Options["--maxIters"], {">": 0})
    
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionTextValue("--mpLevel", Options["--mpLevel"], "Molecules TorsionAngles")
    
    MiscUtil.ValidateOptionIntegerValue("-p, --precision", Options["--precision"], {">": 0})
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
    
    MiscUtil.ValidateOptionIntegerValue("--torsionMaxMatches", Options["--torsionMaxMatches"], {">": 0})
    MiscUtil.ValidateOptionTextValue("--torsionMinimize", Options["--torsionMinimize"], "yes no")
    MiscUtil.ValidateOptionNumberValues("--torsionRange", Options["--torsionRange"], 3, ",", "integer", {})
    
    MiscUtil.ValidateOptionTextValue("--useChirality", Options["--useChirality"], "yes no")
    
# Setup a usage string for docopt...
_docoptUsage_ = """
Psi4PerformTorsionScan.py - Perform torsion scan

Usage:
    Psi4PerformTorsionScan.py [--basisSet <text>] [--confParams <Name,Value,...>] [--energyDataFieldLabel <text>]
                              [--energyRelativeDataFieldLabel <text>] [--energyUnits <text>] [--infile3D <yes or no>]
                              [--infileParams <Name,Value,...>] [--maxIters <number>] [--methodName <text>]
                              [--modeMols <First or All>] [--modeTorsions <First or All>] [--mp <yes or no>]
                              [--mpLevel <Molecules or TorsionAngles>] [--mpParams <Name,Value,...>]
                              [--outfileMolName <yes or no>] [--outfileParams <Name,Value,...>] [--outPlotParams <Name,Value,...>]
                              [--outPlotRelativeEnergy <yes or no>] [--outPlotTitleTorsionSpec <yes or no>] [--overwrite]
                              [--precision <number>] [--psi4OptionsParams <Name,Value,...>] [--psi4RunParams <Name,Value,...>]
                              [--quiet <yes or no>] [--reference <text>]  [--torsionsFilterbyAtomIndices <Index1, Index2, ...>]
                              [--torsionMaxMatches <number>] [--torsionMinimize <yes or no>] [--torsionRange <Start,Stop,Step>]
                              [--useChirality <yes or no>] [-w <dir>] -t <torsions> -i <infile>  -o <outfile> 
    Psi4PerformTorsionScan.py -h | --help | -e | --examples

Description:
    Perform torsion scan for molecules around torsion angles specified using
    SMILES/SMARTS patterns. A molecule is optionally minimized before performing
    a torsion scan using a forcefield. A set of initial 3D structures are generated for
    a molecule by scanning the torsion angle across the specified range and updating
    the 3D coordinates of the molecule. A conformation ensemble is optionally generated
    for each 3D structure representing a specific torsion angle using a combination of
    distance geometry and forcefield followed by constrained geometry optimization
    using a quantum chemistry method. The conformation with the lowest energy is
    selected to represent the torsion angle. An option is available to skip the generation
    of the conformation ensemble and simply calculate the energy for the initial 3D
    structure for a specific torsion torsion angle using a quantum chemistry method.
    
    The torsions are specified using SMILES or SMARTS patterns. A substructure match
    is performed to select torsion atoms in a molecule. The SMILES pattern match must
    correspond to four torsion atoms. The SMARTS patterns containing atom map numbers
    may match  more than four atoms. The atom map numbers, however, must match
    exactly four torsion atoms. For example: [s:1][c:2]([aX2,cH1])!@[CX3:3](O)=[O:4] for
    thiophene esters and carboxylates as specified in Torsion Library (TorLib) [Ref 146].

    A Psi4 XYZ format geometry string is automatically generated for each molecule
    in input file. It contains atom symbols and 3D coordinates for each atom in a
    molecule. In addition, the formal charge and spin multiplicity are present in the
    the geometry string. These values are either retrieved from molecule properties
    named 'FormalCharge' and 'SpinMultiplicty' or dynamically calculated for a
    molecule.
    
    A set of four output files is generated for each torsion match in each
    molecule. The names of the output files are generated using the root of
    the specified output file. They may either contain sequential molecule
    numbers or molecule names as shown below:
        
        <OutfileRoot>_Mol<Num>.sdf
        <OutfileRoot>_Mol<Num>_Torsion<Num>_Match<Num>.sdf
        <OutfileRoot>_Mol<Num>_Torsion<Num>_Match<Num>_Energies.csv
        <OutfileRoot>_Mol<Num>_Torsion<Num>_Match<Num>_Plot.<ImgExt>
        
        or
        
        <OutfileRoot>_<MolName>.sdf
        <OutfileRoot>_<MolName>_Torsion<Num>_Match<Num>.sdf
        <OutfileRoot>_<MolName>_Torsion<Num>_Match<Num>_Energies.csv
        <OutfileRoot>_<MolName>_Torsion<Num>_Match<Num>_Plot.<ImgExt>
        
    The supported input file formats are: Mol (.mol), SD (.sdf, .sd), SMILES (.smi,
    .csv, .tsv, .txt)

    The supported output file formats are: SD (.sdf, .sd)

Options:
    -b, --basisSet <text>  [default: auto]
        Basis set to use for energy calculation or constrained energy minimization.
        Default: 6-31+G** for sulfur containing molecules; Otherwise, 6-31G** [ Ref 150 ].
        The specified value must be a valid Psi4 basis set. No validation is performed.
        
        The following list shows a representative sample of basis sets available
        in Psi4:
            
            STO-3G, 6-31G, 6-31+G, 6-31++G, 6-31G*, 6-31+G*,  6-31++G*, 
            6-31G**, 6-31+G**, 6-31++G**, 6-311G, 6-311+G, 6-311++G,
            6-311G*, 6-311+G*, 6-311++G*, 6-311G**, 6-311+G**, 6-311++G**,
            cc-pVDZ, cc-pCVDZ, aug-cc-pVDZ, cc-pVDZ-DK, cc-pCVDZ-DK, def2-SVP,
            def2-SVPD, def2-TZVP, def2-TZVPD, def2-TZVPP, def2-TZVPPD
            
    --confParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        initial 3D coordinates for molecules in input file at specific torsion angles. A
        conformation ensemble is optionally generated for each 3D structure
        representing a specific torsion angle using a combination of distance geometry
        and forcefield followed by constrained geometry optimization using a quantum
        chemistry method. The conformation with the lowest energy is selected to
        represent the torsion angle.
        
        The supported parameter names along with their default values are shown
        below:
            
            confMethod,ETKDG,
            forceField,MMFF, forceFieldMMFFVariant,MMFF94,
            enforceChirality,yes,embedRMSDCutoff,0.5,maxConfs,250,
            maxConfsTorsions,50,useTethers,yes
            
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
        
        forceField: Forcefield method to use for energy minimization. Possible values:
        Universal Force Field (UFF) [ Ref 81 ] or Merck Molecular Mechanics Force
        Field [ Ref 83-87 ] .
        
        enforceChirality: Enforce chirality for defined chiral centers during
        forcefield minimization.
        
        maxConfs: Maximum number of conformations to generate for each molecule
        during the generation of an initial 3D conformation ensemble using a conformation
        generation methodology. The conformations are minimized using the specified
        forcefield. The lowest energy structure is selected for performing the torsion scan.
        
        maxConfsTorsion: Maximum number of 3D conformations to generate for
        conformation ensemble representing a specific torsion. The conformations are
        constrained at specific torsions angles and minimized using the specified forcefield
        and a quantum chemistry method. The lowest energy conformation is selected to
        calculate final torsion energy and written to the output file.
        
        embedRMSDCutoff: RMSD cutoff for retaining initial set of conformers embedded
        using distance geometry and forcefield minimization. All embedded conformers
        are kept for 'None' value. Otherwise, only those conformers which are different
        from each other by the specified RMSD cutoff, 0.5 by default, are kept. The first
        embedded conformer is always retained.
        
        useTethers: Use tethers to optimize the final embedded conformation by
        applying a series of extra forces to align matching atoms to the positions of
        the core atoms. Otherwise, use simple distance constraints during the
        optimization.
    --energyDataFieldLabel <text>  [default: auto]
        Energy data field label for writing energy values. Default: Psi4_Energy (<Units>). 
    --energyRelativeDataFieldLabel <text>  [default: auto]
        Relative energy data field label for writing energy values. Default:
        Psi4_Relative_Energy (<Units>). 
    --energyUnits <text>  [default: kcal/mol]
        Energy units. Possible values: Hartrees, kcal/mol, kJ/mol, or eV.
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name.
    --infile3D <yes or no>  [default: no]
        Skip generation and minimization of initial 3D structures for molecules in
        input file containing 3D coordinates.
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
        during constrained energy minimization by a quantum chemistry method.
    -m, --methodName <text>  [default: auto]
        Method to use for energy calculation or constrained energy minimization.
        Default: B3LYP [ Ref 150 ]. The specified value must be a valid Psi4 method
        name. No validation is performed.
        
        The following list shows a representative sample of methods available
        in Psi4:
            
            B1LYP, B2PLYP, B2PLYP-D3BJ, B2PLYP-D3MBJ, B3LYP, B3LYP-D3BJ,
            B3LYP-D3MBJ, CAM-B3LYP, CAM-B3LYP-D3BJ, HF, HF-D3BJ,  HF3c, M05,
            M06, M06-2x, M06-HF, M06-L, MN12-L, MN15, MN15-D3BJ,PBE, PBE0,
            PBEH3c, PW6B95, PW6B95-D3BJ, WB97, WB97X, WB97X-D, WB97X-D3BJ
            
    --modeMols <First or All>  [default: First]
        Perform torsion scan for the first molecule or all molecules in input
        file.
    --modeTorsions <First or All>  [default: First]
        Perform torsion scan for the first or all specified torsion pattern in
        molecules up to a maximum number of matches for each torsion
        specification as indicated by '--torsionMaxMatches' option. 
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
    --mpLevel <Molecules or TorsionAngles>  [default: Molecules]
        Perform multiprocessing at molecules or torsion angles level. Possible values:
        Molecules or TorsionAngles. The 'Molecules' value starts a process pool at the
        molecules level. All torsion angles of a molecule are processed in a single
        process. The 'TorsionAngles' value, however, starts a process pool at the 
        torsion angles level. Each torsion angle in a torsion match for a molecule is
        processed in an individual process in the process pool.
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
        Output file name. The output file root is used for generating the names
        of the output files corresponding to structures, energies, and plots during
        the torsion scan.
    --outfileMolName <yes or no>  [default: no]
        Append molecule name to output file root during the generation of the names
        for output files. The default is to use <MolNum>. The non alphabetical
        characters in molecule names are replaced by underscores.
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: kekulize,yes
            
    --outPlotParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        plots using Seaborn module. The supported parameter names along with their
        default values are shown below:
            
            type,linepoint,outExt,svg,width,10,height,5.6,
            title,auto,xlabel,auto,ylabel,auto,titleWeight,bold,labelWeight,bold
            style,darkgrid,palette,deep,font,sans-serif,fontScale,1,
            context,notebook
            
        Possible values:
            
            type: linepoint, scatter, or line. Both points and lines are drawn
                for linepoint plot type.
            outExt: Any valid format supported by Python module Matplotlib.
                For example: PDF (.pdf), PNG (.png), PS (.ps), SVG (.svg)
            titleWeight, labelWeight: Font weight for title and axes labels.
                Any valid value.
            style: darkgrid, whitegrid, dark, white, ticks
            palette: deep, muted, pastel, dark, bright, colorblind
            font: Any valid font name
            
    --outPlotRelativeEnergy <yes or no>  [default: yes]
        Plot relative energies in the torsion plot. The minimum energy value is
        subtracted from energy values to calculate relative energies.
    --outPlotTitleTorsionSpec <yes or no>  [default: yes]
        Append torsion specification to the title of the torsion plot.
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
    --reference <text>  [default: auto]
        Reference wave function to use for energy calculation or constrained energy
        minimization. Default: RHF or UHF. The default values are Restricted Hartree-Fock
        (RHF) for closed-shell molecules with all electrons paired and Unrestricted
        Hartree-Fock (UHF) for open-shell molecules with unpaired electrons.
        
        The specified value must be a valid Psi4 reference wave function. No validation
        is performed. For example: ROHF, CUHF, RKS, etc.
        
        The spin multiplicity determines the default value of reference wave function
        for input molecules. It is calculated from number of free radical electrons using
        Hund's rule of maximum multiplicity defined as 2S + 1 where S is the total
        electron spin. The total spin is 1/2 the number of free radical electrons in a 
        molecule. The value of 'SpinMultiplicity' molecule property takes precedence
        over the calculated value of spin multiplicity.
    -t, --torsions <SMILES/SMARTS,...,...>
        SMILES/SMARTS patterns corresponding to torsion specifications. It's a 
        comma delimited list of valid SMILES/SMART patterns.
        
        A substructure match is performed to select torsion atoms in a molecule.
        The SMILES pattern match must correspond to four torsion atoms. The
        SMARTS patterns containing atom map numbers  may match  more than four
        atoms. The atom map numbers, however, must match exactly four torsion
        atoms. For example: [s:1][c:2]([aX2,cH1])!@[CX3:3](O)=[O:4] for thiophene
        esters and carboxylates as specified in Torsion Library (TorLib) [Ref 146].
    --torsionsFilterbyAtomIndices <Index1, Index2, ...>  [default: none]
        Comma delimited list of atom indices for filtering torsion matches
        corresponding to torsion specifications  "-t, --torsions". The atom indices
        must be valid. No explicit validation is performed. The list must contain at
        least 4 atom indices.
        
        The torsion atom indices, matched by "-t, --torsions" specifications, must be
        present in the list. Otherwise, the torsion matches are ignored.
    --torsionMaxMatches <number>  [default: 5]
        Maximum number of torsions to match for each torsion specification in a
        molecule.
    --torsionMinimize <yes or no>  [default: no]
        Perform constrained energy minimization on a conformation ensemble
        for  a specific torsion angle and select the lowest energy conformation
        representing the torsion angle. A conformation ensemble is generated for
        each 3D structure representing a specific torsion angle using a combination
        of distance geometry and forcefield followed by constrained geometry
        optimization using a quantum chemistry method.
    --torsionRange <Start,Stop,Step>  [default: 0,360,5]
        Start, stop, and step size angles in degrees for a torsion scan. In addition,
        you may specify values using start and stop angles from -180 to 180.
    --useChirality <yes or no>  [default: no]
        Use chirrality during substructure matches for identification of torsions.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To perform a torsion scan on the first molecule in a SMILES file using a minimum
    energy structure of the molecule selected from an initial ensemble of conformations
    generated using distance geometry and forcefield, skip generation of conformation
    ensembles for specific torsion angles and constrained energy minimization of the
    ensemble, calculating single point at a specific torsion angle energy using B3LYP/6-31G**
    and B3LYP/6-31+G** for non-sulfur and sulfur containing molecules, generate output files
    corresponding to structure, energy and torsion plot, type:
    
        % Psi4PerformTorsionScan.py  -t "CCCC" -i Psi4SampleTorsionScan.smi 
          -o SampleOut.sdf

    To run the previous example on the first molecule in a SD file containing 3D
    coordinates and skip the generations of initial 3D structure, type: 
    
        % Psi4PerformTorsionScan.py  -t "CCCC"  --infile3D yes
          -i Psi4SampleTorsionScan3D.sdf  -o SampleOut.sdf

    To run the first example on all molecules in a SD file, type:
    
        % Psi4PerformTorsionScan.py  -t "CCCC" --modeMols All
          -i Psi4SampleTorsionScan.sdf -o SampleOut.sdf

    To run the first example on all molecules in a SD file containing 3D
    coordinates and skip the generation of initial 3D structures, type: 
    
        % Psi4PerformTorsionScan.py  -t "CCCC"  --infile3D yes
          --modeMols All -i Psi4SampleTorsionScan3D.sdf  -o SampleOut.sdf

    To perform a torsion scan on the first molecule in a SMILES file using a minimum
    energy structure of the molecule selected from an initial ensemble of conformations
    generated using distance geometry and forcefield,  generate up to 50 conformations
    for specific torsion angles using ETKDG methodology followed by initial MMFF
    forcefield minimization and final energy minimization using B3LYP/6-31G** and
    B3LYP/6-31+G** for non-sulfur and sulfur containing molecules, generate output files
    corresponding to minimum energy structure, energy and torsion plot, type:

        % Psi4PerformTorsionScan.py  -t "CCCC" --torsionMinimize Yes
           -i Psi4SampleTorsionScan.smi -o SampleOut.sdf

    To run the previous example on all molecules in a SD file, type:
    
        % Psi4PerformTorsionScan.py  -t "CCCC" --modeMols All
           --torsionMinimize Yes -i Psi4SampleTorsionScan.sdf -o SampleOut.sdf

    To run the previous example on all molecules in a SD file containing 3D
    coordinates and skip the generation of initial 3D structures, type:
    
        % Psi4PerformTorsionScan.py  -t "CCCC" --modeMols All
           --infile3D yes --modeMols All  --torsionMinimize Yes
           -i Psi4SampleTorsionScan.sdf -o SampleOut.sdf

    To run the previous example in multiprocessing mode at molecules level
    on all available CPUs without loading all data into memory and write out
    a SD file, type:

        % Psi4PerformTorsionScan.py  -t "CCCC" -i Psi4SampleTorsionScan.smi 
          -o SampleOut.sdf --modeMols All --torsionMinimize Yes --mp yes
    
    To run the previous example in multiprocessing mode at torsion angles level
    on all available CPUs without loading all data into memory and write out
    a SD file, type:

        % Psi4PerformTorsionScan.py  -t "CCCC" -i Psi4SampleTorsionScan.smi 
          -o SampleOut.sdf --modeMols All --torsionMinimize Yes --mp yes
          --mpLevel TorsionAngles
    
    To run the previous example in multiprocessing mode on all available CPUs
    by loading all data into memory and write out a SD file, type:

        % Psi4PerformTorsionScan.py  -t "CCCC" -i Psi4SampleTorsionScan.smi 
          -o SampleOut.sdf --modeMols All --torsionMinimize Yes --mp yes
          --mpParams "inputDataMode,InMemory"
    
    To run the previous example in multiprocessing mode on specific number of
    CPUs and chunk size without loading all data into memory and write out a SD file,
    type:

        % Psi4PerformTorsionScan.py  -t "CCCC" -i Psi4SampleTorsionScan.smi 
          -o SampleOut.sdf --modeMols All --torsionMinimize Yes --mp yes
          --mpParams "inputDataMode,Lazy,numProcesses,4,chunkSize,8"

Author:
    Manish Sud(msud@san.rr.com)

See also:
    Psi4CalculateEnergy.py, Psi4GenerateConformers.py,
    Psi4GenerateConstrainedConformers.py, Psi4PerformConstrainedMinimization.py

Copyright:
    Copyright (C) 2022 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using RDKit, an
    open source toolkit for cheminformatics developed by Greg Landrum.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
