#!/usr/bin/env python
#
# File: Psi4CalculatePartialCharges.py
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
    CalculatePartialCharges()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def CalculatePartialCharges():
    """Calculate partial atomic charges."""
    
    CheckSupportForRESPCalculations()
    
    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])
    
    # Set up a molecule writer...
    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["Outfile"])

    MolCount, ValidMolCount, CalcFailedCount = ProcessMolecules(Mols, Writer)

    if Writer is not None:
        Writer.close()
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during calculation of partial charges: %d" % CalcFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + CalcFailedCount))

def ProcessMolecules(Mols, Writer):
    """Process molecules and calculate partial charges."""
    
    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, Writer)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, Writer)

def ProcessMoleculesUsingSingleProcess(Mols, Writer):
    """Process molecules and calculate partial charges using a single process."""
    
    # Intialize Psi4...
    MiscUtil.PrintInfo("\nInitializing Psi4...")
    Psi4Handle = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = True, PrintHeader = True)
    OptionsInfo["psi4"] = Psi4Handle

    # Initialize RESP...
    OptionsInfo["resp"] = SetupRESP()
    
    MiscUtil.PrintInfo("\nCalculating partial atomic charges...")
    
    (MolCount, ValidMolCount, CalcFailedCount) = [0] * 3
    for Mol in Mols:
        MolCount += 1
        
        if not CheckAndValidateMolecule(Mol, MolCount):
            continue
        
        # Setup a Psi4 molecule...
        Psi4Mol = SetupPsi4Mol(Psi4Handle, Mol, MolCount)
        if Psi4Mol is None:
            continue
        
        ValidMolCount += 1
        
        # Retrieve charges...
        CalcStatus, PartialCharges = CalculateMolPartialCharges(Psi4Handle, Psi4Mol, Mol, MolCount)

        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to calculate partial atomic charges for molecule %s" % MolName)
            
            CalcFailedCount += 1
            continue
        
        WriteMolPartialCharges(Writer, Mol, PartialCharges)
    
    return (MolCount, ValidMolCount, CalcFailedCount)

def ProcessMoleculesUsingMultipleProcesses(Mols, Writer):
    """Process molecules and calculate partial charges using a multiprocessing."""

    MiscUtil.PrintInfo("\nCalculating partial atomic charges using multiprocessing...")
    
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
    
    (MolCount, ValidMolCount, CalcFailedCount) = [0] * 3
    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, CalcStatus, PartialCharges = Result
        
        if EncodedMol is None:
            continue
        
        ValidMolCount += 1

        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to calculate partial atomic charges for molecule %s" % MolName)
            
            CalcFailedCount += 1
            continue
        
        WriteMolPartialCharges(Writer, Mol, PartialCharges)
    
    return (MolCount, ValidMolCount, CalcFailedCount)

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
    PartialCharges = None
    
    if EncodedMol is None:
        return [MolIndex, None, CalcStatus, PartialCharges]
    
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    MolCount = MolIndex + 1
    
    if not CheckAndValidateMolecule(Mol, MolCount):
        return [MolIndex, None, CalcStatus, PartialCharges]
    
    # Setup a Psi4 molecule...
    Psi4Mol = SetupPsi4Mol(OptionsInfo["psi4"], Mol, MolCount)
    if Psi4Mol is None:
        return [MolIndex, None, CalcStatus, PartialCharges]
    
    CalcStatus, PartialCharges = CalculateMolPartialCharges(OptionsInfo["psi4"], Psi4Mol, Mol, MolCount)
    
    return [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CalcStatus, PartialCharges]

def CalculateMolPartialCharges(Psi4Handle, Psi4Mol, Mol, MolNum):
    """Calculate partial atomic charges for a molecule."""

    if OptionsInfo["RESPChargesTypeMode"]:
        return CalculateRespMolPartialCharges(Psi4Handle, Psi4Mol, Mol, MolNum)
    else:
        return CalculateNonRespMolPartialCharges(Psi4Handle, Psi4Mol, Mol, MolNum)
    
def CalculateNonRespMolPartialCharges(Psi4Handle, Psi4Mol, Mol, MolNum):
    """Calculate non-RESP partial atomic charges for a molecule."""
    
    Status = False
    PartialCharges = []
    
    #  Setup reference wave function...
    Reference = SetupReferenceWavefunction(Mol)
    Psi4Handle.set_options({'Reference': Reference})
    
    # Setup method name and basis set...
    MethodName, BasisSet = SetupMethodNameAndBasisSet(Mol)

    # Calculate single point energy to setup a wavefunction...
    Status, Energy, WaveFunction = Psi4Util.CalculateSinglePointEnergy(Psi4Handle, Psi4Mol, MethodName, BasisSet, ReturnWaveFunction = True, Quiet = OptionsInfo["QuietMode"])

    if not Status:
        PerformPsi4Cleanup(Psi4Handle)
        return (False, PartialCharges)
    
    # Calculate atomic point charges...
    try:
        Psi4Handle.oeprop(WaveFunction, OptionsInfo["ChargesPropType"])
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to calculate partial atomic charges:\n%s" % ErrMsg)
        PerformPsi4Cleanup(Psi4Handle)
        return (False, PartialCharges)
    
    AtomicPointCharges = WaveFunction.atomic_point_charges().np.tolist()
    
    # Clean up...
    PerformPsi4Cleanup(Psi4Handle)
    
    # Format charges...
    PartialCharges = ["%.*f" % (OptionsInfo["Precision"], float(Value)) for Value in AtomicPointCharges]

    return (Status, PartialCharges)

def CalculateRespMolPartialCharges(Psi4Handle, Psi4Mol, Mol, MolNum):
    """Calculate RESP partial atomic charges for a molecule."""

    PartialCharges = []

    RESPHandle = OptionsInfo["resp"]
    RESPOptions = OptionsInfo["ChargesRespOtions"]
    
    # Setup method name and basis set...
    MethodName, BasisSet = SetupMethodNameAndBasisSet(Mol)
    RESPOptions["METHOD_ESP"] = MethodName
    RESPOptions["BASIS_ESP"] = BasisSet

    # Calculate RESP charges...
    try:
        RESPCalcResults =  RESPHandle.resp([Psi4Mol], RESPOptions)
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to calculate RESP partial atomic charges:\n%s" % ErrMsg)
        RemoveRESPGridFiles(MolNum)
        return (False, PartialCharges)
    
    ESPCharges = RESPCalcResults[0]
    RESPCharges = RESPCalcResults[1]
    
    PerformRESPCleanup(MolNum)
    
    # Format charges...
    PartialCharges = ["%.*f" % (OptionsInfo["Precision"], float(Value)) for Value in RESPCharges]
    
    return (True, PartialCharges)

def PerformRESPCleanup(MolNum):
    """Peform RESP cleanup."""

    if OptionsInfo["ChargesRespParams"]["RemoveGridFiles"]:
        RemoveRESPGridFiles(MolNum)
    else:
        RenameRESPGridFiles(MolNum)

def RemoveRESPGridFiles(MolNum):
    """Remove RESP grid files."""

    try:
        for GridFile in ["1_default_grid.dat", "1_default_grid_esp.dat", "results.out"]:
            if os.path.isfile(GridFile):
                os.remove(GridFile)
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to remove RESP results/grid files: %s\n" % ErrMsg)

def RenameRESPGridFiles(MolNum):
    """Rename RESP grid files."""

    try:
        MolPrefix = "Mol%s" % MolNum
        GridFiles = ["1_default_grid.dat", "1_default_grid_esp.dat", "results.out"]
        NewGridFiles = ["%s_grid.dat" % MolPrefix, "%s_grid_esp.dat" % MolPrefix, "%s_resp_results.out" % MolPrefix]
        for GridFile, NewGridFile in zip(GridFiles, NewGridFiles):
            if os.path.isfile(GridFile):
                shutil.move(GridFile, NewGridFile)
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to move RESP results/grid files: %s\n" % ErrMsg)

def WriteMolPartialCharges(Writer, Mol, PartialCharges):
    """Write out partial atomic charges for a molecule."""

    if PartialCharges is None:
        return
    
    if OptionsInfo["AtomAliasesFormatMode"]:
        for Atom, PartialCharge in zip(Mol.GetAtoms(), PartialCharges):
            Atom.SetProp('molFileAlias', PartialCharge)
    else:
        ChargesValues = "\n".join(PartialCharges)
        Mol.SetProp(OptionsInfo["DataFieldLabel"], ChargesValues)
    
    Writer.write(Mol)
    
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

def SetupRESP():
    """Load resp and return its handle."""
    
    if not OptionsInfo["RESPChargesTypeMode"]:
        return None
        
    try:
        import resp
    except ImportError as ErrMsg:
        sys.stderr.write("\nFailed to import Psi4 module/package resp: %s\n" % ErrMsg)
        sys.stderr.write("Check/update your Psi4 environment and try again.\n\n")
        sys.exit(1)

    return resp
    
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

def ProcessOptionChargesRespParameters():
    """Process charges RSEP parameters option."""

    ParamsOptionName = "--chargesRespParams"
    ParamsOptionValue = Options["--chargesRespParams"]

    VDWScaleFactors = [1.4, 1.6, 1.8, 2.0]
    VDWRadii = {'H': 1.20, 'HE': 1.20, 'LI': 1.37, 'BE': 1.45, 'B': 1.45, 'C': 1.50, 'N': 1.50, 'O': 1.40, 'F': 1.35, 'NE': 1.30, 'NA': 1.57, 'MG': 1.36, 'AL': 1.24, 'SI': 1.17, 'P': 1.80, 'S': 1.75, 'CL': 1.70}
    
    ParamsInfo = {"MaxIter": 25, "RestrainHydrogens": False, "RemoveGridFiles": True, "RespA": 0.0005, "RespB": 0.1, "Tolerance": 1e-5, "VDWRadii": VDWRadii, "VDWScaleFactors": VDWScaleFactors, "VDWPointDensity": 1.0}

    if re.match("^auto$", ParamsOptionValue, re.I):
        SetupChargesRespOptions(ParamsInfo)
        return
    
    # Setup a canonical paramater names...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(ParamsInfo):
        ValidParamNames.append(ParamName)
        CanonicalParamNamesMap[ParamName.lower()] = ParamName
    
    ParamsOptionValue = ParamsOptionValue.strip()
    if not ParamsOptionValue:
        MiscUtil.PrintError("No valid parameter name and value pairs specified using \"%s\" option" % ParamsOptionName)
    
    ParamsOptionValueWords = ParamsOptionValue.split(",")
    if len(ParamsOptionValueWords) % 2:
        MiscUtil.PrintError("The number of comma delimited paramater names and values, %d, specified using \"%s\" option must be an even number." % (len(ParamsOptionValueWords), ParamsOptionName))
    
    # Validate paramater name and value pairs...
    for Index in range(0, len(ParamsOptionValueWords), 2):
        Name = ParamsOptionValueWords[Index].strip()
        Value = ParamsOptionValueWords[Index + 1].strip()

        CanonicalName = Name.lower()
        if  not CanonicalName in CanonicalParamNamesMap:
            MiscUtil.PrintError("The parameter name, %s, specified using \"%s\" is not a valid name. Supported parameter names: %s" % (Name, ParamsOptionName, " ".join(ValidParamNames)))

        ParamName = CanonicalParamNamesMap[CanonicalName]
        ParamValue = Value
        
        if re.match("^MaxIter$", ParamName, re.I):
            if not MiscUtil.IsInteger(Value):
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option must be an integer." % (Value, Name, ParamsOptionName))
            Value = int(Value)
            if Value <= 0:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: > 0" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^(RestrainHydrogens|RemoveGridFiles)$", ParamName, re.I):
            if not re.match("^(yes|no)$", Value, re.I):
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: yes or no" % (Value, Name, ParamsOptionName))
            ParamValue = True if re.match("^yes$", Value, re.I) else False
        elif re.match("^(RespA|RespB|VDWPointDensity)$", ParamName, re.I):
            if not MiscUtil.IsNumber(Value):
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option must be a float." % (Value, Name, ParamsOptionName))
            Value = float(Value)
            if Value <= 0:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: > 0" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^Tolerance$", ParamName, re.I):
            if not MiscUtil.IsNumber(Value):
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option must be a float." % (Value, Name, ParamsOptionName))
            Value = float(Value)
            if Value < 0:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: >= 0" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^VDWScaleFactors$", ParamName, re.I):
            ScaleFactorsValue = Value.strip()
            if not ScaleFactorsValue:
                MiscUtil.PrintError("No parameter value specified for parameter name, %s, using \"%s\" option." % (Name, ParamsOptionName))
            ScaleFactorsWords = ScaleFactorsValue.split()
            
            ScaleFactors = []
            LastScaleFactor = 0.0
            for ScaleFactor in ScaleFactorsWords:
                if not MiscUtil.IsNumber(ScaleFactor):
                    MiscUtil.PrintError("The value, %s, in parameter value, %s, specified for parameter name, %s, using \"%s\" option must be a float." % (ScaleFactor, Value, Name, ParamsOptionName))
                ScaleFactor = float(ScaleFactor)
                if ScaleFactor <= 0:
                    MiscUtil.PrintError("The value, %s, in parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: > 0" % (ScaleFactor, Value, Name, ParamsOptionName))
                if len(ScaleFactors):
                    if ScaleFactor <= LastScaleFactor:
                        MiscUtil.PrintError("The value, %s, in parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. It must be greater than the previous value, %s, specified for the scale factor." % (ScaleFactor, Value, Name, ParamsOptionName, LastScaleFactor))
                        
                LastScaleFactor = ScaleFactor
                ScaleFactors.append(ScaleFactor)
            
            ParamValue = ScaleFactors
        elif re.match("^VDWRadii$", ParamName, re.I):
            RadiiValue = Value.strip()
            if not RadiiValue:
                MiscUtil.PrintError("No parameter value specified for parameter name, %s, using \"%s\" option." % (Name, ParamsOptionName))
            RadiiWords = RadiiValue.split()
            if len(RadiiWords) % 2:
                MiscUtil.PrintError("The number of space delimited values, %s, in parameter value, %s, specified for parameter name, %s, using \"%s\" option is not valid. It must be an even number." % (len(RadiiWords), Value, Name, ParamsOptionName))
            
            for RadiiWordsIndex in range(0, len(RadiiWords), 2):
                ElementSymbol = RadiiWords[RadiiWordsIndex].upper()
                VDWRadius = RadiiWords[RadiiWordsIndex + 1]

                if not MiscUtil.IsNumber(VDWRadius):
                    MiscUtil.PrintError("The vdw radius value, %s, in parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid." % (VDWRadius, Value, Name, ParamsOptionName))
                    
                if not RDKitUtil.IsValidElementSymbol(ElementSymbol.capitalize()):
                    MiscUtil.PrintWarning("The element symbol, %s, in parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid." % (ElementSymbol, Value, Name, ParamsOptionName))
                VDWRadii[ElementSymbol] = float(VDWRadius)
            
            ParamValue = VDWRadii
        else:
            ParamValue = Value
            
        # Set value...
        ParamsInfo[ParamName] = ParamValue
    
    SetupChargesRespOptions(ParamsInfo)

def SetupChargesRespOptions(ParamsInfo):
    """Setup options for calculating RESP charges."""

    # Initialize ESP options...
    ChargesRespOtions = {}
    
    ChargesRespOtions["METHOD_ESP"] = None
    ChargesRespOtions["BASIS_ESP"] = None

    # Setup ESP options...
    ParamNameToESPOptionID = {"MaxIter": "MAX_IT", "RespA": "RESP_A", "RespB": "RESP_B", "Tolerance": "TOLER", "VDWRadii": "VDW_RADII", "VDWScaleFactors": "VDW_SCALE_FACTORS", "VDWPointDensity": "VDW_POINT_DENSITY"}
    
    for ParamName in ParamNameToESPOptionID:
        ESPOptionID = ParamNameToESPOptionID[ParamName]
        ChargesRespOtions[ESPOptionID] = ParamsInfo[ParamName]

    # Setup IHFREE option...
    ChargesRespOtions["IHFREE"] = False if ParamsInfo["RestrainHydrogens"] else True

    OptionsInfo["ChargesRespParams"] = ParamsInfo
    OptionsInfo["ChargesRespOtions"] = ChargesRespOtions

def CheckSupportForRESPCalculations():
    """Check support for RESP calculations."""

    if not OptionsInfo["MPMode"]:
        return
    
    if not OptionsInfo["RESPChargesTypeMode"]:
        return
    
    MiscUtil.PrintInfo("")
    MiscUtil.PrintError("Multiprocessing is not supported during the calculation of RSEP partial atomic\ncharges. The RESP module is not conducive for multiprocessing. The names of the results\nand grid output files are not unique for molecules during the RESP calculations spread\nacross multiple processes.")

def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")
    
    # Validate options...
    ValidateOptions()
    
    OptionsInfo["Infile"] = Options["--infile"]
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

    ChargesType = Options["--chargesType"]
    ChargesPropType = None
    RESPChargesTypeMode = False
    if re.match("^Mulliken$", ChargesType, re.I):
        ChargesType = 'Mulliken'
        ChargesPropType = 'MULLIKEN_CHARGES'
    elif re.match("^Lowdin$", ChargesType, re.I):
        ChargesType = 'Lowdin'
        ChargesPropType = 'LOWDIN_CHARGES'
    elif re.match("^RESP$", ChargesType, re.I):
        ChargesType = 'RESP'
        ChargesPropType = None
        RESPChargesTypeMode = True
    else:
        MiscUtil.PrintError("The value, %s, specified for charge mode is not supported. " % Options["--chargesType"])
        
    OptionsInfo["ChargesType"] = ChargesType
    OptionsInfo["ChargesPropType"] = ChargesPropType
    OptionsInfo["RESPChargesTypeMode"] = RESPChargesTypeMode

    ProcessOptionChargesRespParameters()
    
    AtomAliasesFormatMode = True
    if re.match("^DataField", Options["--chargesSDFormat"], re.I):
        AtomAliasesFormatMode = False
    OptionsInfo["AtomAliasesFormatMode"] = AtomAliasesFormatMode

    DataFieldLabel = Options["--dataFieldLabel"]
    if re.match("^auto$", DataFieldLabel, re.I):
        DataFieldLabel = "Psi4_%s_Charges (a.u.)" % Options["--chargesType"]
    OptionsInfo["DataFieldLabel"] = DataFieldLabel
    
    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])
    
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

    MiscUtil.ValidateOptionTextValue("-c, --chargesType", Options["--chargesType"], "Mulliken Lowdin RESP")
    MiscUtil.ValidateOptionTextValue("--chargesSDFormat", Options["--chargesSDFormat"], "AtomAliases DataField")
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol")
    
    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
    MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])
    
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
    
    MiscUtil.ValidateOptionIntegerValue("-p, --precision", Options["--precision"], {">": 0})
    
# Setup a usage string for docopt...
_docoptUsage_ = """
Psi4CalculatePartialCharges.py - Calculate partial atomic charges

Usage:
    Psi4CalculatePartialCharges.py [--basisSet <text>] [--chargesType <Mulliken or Lowdin>] [--chargesRespParams <Name,Value,...>]
                                   [--chargesSDFormat <AtomAliases or DataField>] [--dataFieldLabel <text>] [--infileParams <Name,Value,...>]
                                   [--methodName <text>] [--mp <yes or no>] [--mpParams <Name, Value,...>] [ --outfileParams <Name,Value,...> ]
                                   [--overwrite] [--precision <number>] [--psi4OptionsParams <Name,Value,...>] [--psi4RunParams <Name,Value,...>] 
                                   [--quiet <yes or no>] [--reference <text>] [-w <dir>] -i <infile> -o <outfile> 
    Psi4CalculatePartialCharges.py -h | --help | -e | --examples

Description:
    Calculate partial atomic charges for molecules using a specified method name
    and basis set. The molecules must have 3D coordinates in input file. The molecular
    geometry is not optimized before the calculation. In addition, hydrogens must
    be present for all molecules in input file. A single point energy calculation is 
    performed before calculating the partial atomic charges. The 3D coordinates
    are not modified during the calculation.
    
    A Psi4 XYZ format geometry string is automatically generated for each molecule
    in input file. It contains atom symbols and 3D coordinates for each atom in a
    molecule. In addition, the formal charge and spin multiplicity are present in the
    the geometry string. These values are either retrieved from molecule properties
    named 'FormalCharge' and 'SpinMultiplicty' or dynamically calculated for a
    molecule.

    The supported input file formats are: Mol (.mol), SD (.sdf, .sd)

    The supported output file formats are: SD (.sdf, .sd)

Options:
    -b, --basisSet <text>  [default: auto]
        Basis set to use for calculating single point energy before partial atomic
        charges. Default: 6-31+G** for sulfur containing molecules; Otherwise,
        6-31G** [ Ref 150 ]. The specified value must be a valid Psi4 basis set.
        No validation is performed.
        
        The following list shows a representative sample of basis sets available
        in Psi4:
            
            STO-3G, 6-31G, 6-31+G, 6-31++G, 6-31G*, 6-31+G*,  6-31++G*, 
            6-31G**, 6-31+G**, 6-31++G**, 6-311G, 6-311+G, 6-311++G,
            6-311G*, 6-311+G*, 6-311++G*, 6-311G**, 6-311+G**, 6-311++G**,
            cc-pVDZ, cc-pCVDZ, aug-cc-pVDZ, cc-pVDZ-DK, cc-pCVDZ-DK, def2-SVP,
            def2-SVPD, def2-TZVP, def2-TZVPD, def2-TZVPP, def2-TZVPPD
            
    -c, --chargesType <Mulliken, Lowdin, or RESP>  [default: Mulliken]
        Type of partial atomic charges to calculate. Possible values: Mulliken, Lowdin,
        or RESP [ Ref 158 ]. Multiprocessing is not supported during the calculation
        of RSEP charges. In addition, the RSEP calculation relies on the presence of
        the RESP Psi4 Plugin in your environment.
    --chargesRespParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value  pairs for calculating
        RESP [ Ref 158 ] charges. A space is used as a delimiter for multiple values in
        a name and value pair. The supported parameter names, along with
        their default values, are shown below:
            
            maxIter, 25
            restrainHydrogens, no
            removeGridFiles, yes
            respA, 0.0005
            respB, 0.1
            tolerance, 1e-5
            vdwRadii, auto
            vdwScaleFactors, 1.4 1.6 1.8 2.0
            vdwPointDensity, 1.0
            
        maxIter: Maximum number of iterations to perform during charge fitting.
         
        restrainHydrogens: Restrain hydrogens during charge fitting.
        
        removeGridFiles: Keep or remove the following ESP grid and output files:
        1_default_grid.dat, 1_default_grid_esp.dat, results.out. The output
        files are removed by default. You may optionally keep the output files. The
        output files are automatically renamed to the following file for 'No' value of
        'removeGridFiles': Mol<MolNum>_grid.dat, Mol<MolNum>_grid_esp.dat,
        Mol<MolNum>_resp_results.out.
        
        respA: Scale factor defining the  asymptotic limits of the strength of the
        restraint.
        
        respB: The 'tightness' of the hyperbola around its minimum for the
        restraint.
        
        tolerance: Tolerance for charges during charge fitting to the ESP.
        
        vdwRadii: vdw radii for elements in angstroms. It's a space delimited list of
        element symbol and radius value pairs. The default list is shown below:
            
            H 1.20 He 1.20 Li 1.37 Be 1.45 B 1.45 C 1.50 N 1.50 O 1.40 F 1.35
            Ne 1.30 Na 1.57 Mg 1.36 Al 1.24 Si 1.17P 1.80 S 1.75 Cl 1.7
            
        You may specify all or a subset of element symbol and vdw radius pairs to
        update the default values.
        
        vdwScaleFactors: The vdw radii are scaled by the scale factors to set the
        grid points at the shells for calculating the ESP using quantum methodology.
        The default number of shells is 4 and corresponds to the number of vdw
        scale factors.The 'shell' points are written to a grid file for calculating the ESP.
        
        vdwPointDensity: Approximate number of points to generate per square
        angstrom surface area.
    --chargesSDFormat <AtomAliases or DataField>  [default: AtomAliases]
        Format for writing out partial atomic charges to SD file. Possible values:
        AtomAliases or DataField.
        
        The charges are stored as atom property named 'molFileAlias' for
        'AtomAliases' format and may be retrieved using the RDKit function
        'GetProp' for atoms: Aotm.GetProp('molFileAliases').
        
        The charges are stored under a data field label specified using
        '-d, --dataFieldLabel' for 'DataField' format and may be retrieved using the
        RDKit function 'GetProp' for molecules.
    -d, --dataFieldLabel <text>  [default: auto]
        Data field label to use for storing charged in SD file during 'DataField' value
        of '-c, --chargesSDFormat'. Default: Psi4_<ChargesType>_Charges (a.u.)
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
        Method to use for calculating single point energy before partial atomic
        charges. Default: B3LYP [ Ref 150 ]. The specified value must be a valid
        Psi4 method name. No validation is performed.
        
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
    --precision <number>  [default: 4]
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
        all Psi4 output.
        
        The default 'Yes' value of 'RemoveOutputFile' option forces the removal
        of any existing Psi4 before creating new files to append output from
        multiple Psi4 runs.
        
        The option 'ScratchDir' is a directory path to the location of scratch
        files. The default value corresponds to Psi4 default. It may be used to
        override the deafult path.
    -q, --quiet <yes or no>  [default: no]
        Use quiet mode. The warning and information messages will not be printed.
    -r, --reference <text>  [default: auto]
        Reference wave function to use for calculating single point energy before
        partial atomic charges. Default: RHF or UHF. The default values are Restricted
        Hartree-Fock (RHF) for closed-shell molecules with all electrons paired and
        Unrestricted Hartree-Fock (UHF) for open-shell molecules with unpaired electrons.
        
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
    To calculate Mulliken partial atomic charges using  B3LYP/6-31G** and
    B3LYP/6-31+G** for non-sulfur and sulfur containing molecules in a SD
    file with 3D structures, use RHF and UHF for closed-shell and open-shell
    molecules, and write a new SD file, type:

        % Psi4CalculatePartialCharges.py  -i Psi4Sample3D.sdf 
          -o Psi4Sample3DOut.sdf

    To run the first example for calculating RESP charges using a default set of
    parameters for the RESP calculation and write out a SD file, type:

        % Psi4CalculatePartialCharges.py  --chargesType RESP
           -i Psi4Sample3D.sdf -o Psi4Sample3DOut.sdf

    To run the first example for calculating RESP charges using an explicit set
    of specific parameters for the RESP calculation and write out a SD file, type:

        % Psi4CalculatePartialCharges.py  --chargesType RESP
           --chargesRespParams "respA, 0.0005, respB, 0.1, vdwScaleFactors,
           1.4 1.6 1.8 2.0" -i Psi4Sample3D.sdf -o Psi4Sample3DOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    without loading all data into memory and write out a SD file, type:

        % Psi4CalculatePartialCharges.py --mp yes -i Psi4Sample3D.sdf
          -o Psi4Sample3DOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    by loading all data into memory and write out a SD file, type:

        % Psi4CalculatePartialCharges.py  --mp yes --mpParams "inputDataMode,
          InMemory" -i Psi4Sample3D.sdf  -o Psi4Sample3DOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    without loading all data into memory along with multiple threads for each
    Psi4 run and write out a SD file, type:

        % Psi4CalculatePartialCharges.py --mp yes --psi4RunParams "NumThreads,2"
           -i Psi4Sample3D.sdf -o Psi4Sample3DOut.sdf

    To run the first example for writing out charges to a new SD file under a
    datafield instead of storing them as atom property, type:

        % Psi4CalculatePartialCharges.py  --chargesSDFormat DataField
          -i Psi4Sample3D.sdf  -o Psi4Sample3DOut.sdf

    To calculate specific partial atomic charges using a specific method  and basis
    set for molecules in a SD ontaining 3D structures and write them out to a specific
    datafield in a new SD file, type:

        % Psi4CalculatePartialCharges.py  -c Lowdin -m SCF -b aug-cc-pVDZ
          --chargesSDFormat DataField --dataFieldLabel "Lowdin_Charges"
          -i Psi4Sample3D.sdf  -o Psi4Sample3DOut.sdf

Author:
    Manish Sud(msud@san.rr.com)

See also:
    Psi4CalculateEnergy.py, Psi4PerformMinimization.py, Psi4GenerateConformers.py

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
