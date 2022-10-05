#!/usr/bin/env python
#
# File: Psi4CalculateProperties.py
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
import math
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

PropertyNamesMap = {}

def main():
    """Start execution of the script."""
    
    MiscUtil.PrintInfo("\n%s (Psi4: Imported later; RDKit v%s; MayaChemTools v%s; %s): Starting...\n" % (ScriptName, rdBase.rdkitVersion, MiscUtil.GetMayaChemToolsVersion(), time.asctime()))
    
    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()
    
    # Retrieve command line arguments and options...
    RetrieveOptions()
    
    # Process and validate command line arguments and options...
    ProcessOptions()
    
    # Perform actions required by the script...
    CalculateProperties()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def CalculateProperties():
    """Calculate properties."""
    
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
    MiscUtil.PrintInfo("Number of molecules failed during property calculation: %d" % CalcFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + CalcFailedCount))

def ProcessMolecules(Mols, Writer):
    """Process and calculate properties of molecules."""
    
    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, Writer)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, Writer)

def ProcessMoleculesUsingSingleProcess(Mols, Writer):
    """Process and calculate properties of molecules using a single process."""

    # Intialize Psi4...
    MiscUtil.PrintInfo("\nInitializing Psi4...")
    Psi4Handle = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = True, PrintHeader = True)
    OptionsInfo["psi4"] = Psi4Handle
    
    # Setup conversion factor for energy units...
    SetupEnergyConversionFactor(Psi4Handle)
    
    MiscUtil.PrintInfo("\nCalculating properties...")
    
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

        CalcStatus, CalculatedValues = CalculateMolProperties(Psi4Handle, Psi4Mol, Mol, MolCount)

        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Failed to calculate properties for molecule %s" % RDKitUtil.GetMolName(Mol, MolCount))
            
            CalcFailedCount += 1
            continue

        WritePropertyValues(Writer, Mol, CalculatedValues)
    
    return (MolCount, ValidMolCount, CalcFailedCount)

def ProcessMoleculesUsingMultipleProcesses(Mols, Writer):
    """Process and calculate properties of molecules using  process."""
    
    MiscUtil.PrintInfo("\nCalculating properties using multiprocessing...")
    
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
        MolIndex, EncodedMol, CalcStatus, CalculatedValues = Result
        
        if EncodedMol is None:
            continue
        
        ValidMolCount += 1

        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to calculate properties for molecule %s" % MolName)
            
            CalcFailedCount += 1
            continue
        
        WritePropertyValues(Writer, Mol, CalculatedValues)
    
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
    
    SetupPropertyNamesInfo(PrintInfo = False)

def InitializePsi4ForWorkerProcess():
    """Initialize Psi4 for a worker process."""
    
    if OptionsInfo["Psi4Initialized"]:
        return

    OptionsInfo["Psi4Initialized"] = True
    
    # Update output file...
    OptionsInfo["Psi4RunParams"]["OutputFile"] = Psi4Util.UpdatePsi4OutputFileUsingPID(OptionsInfo["Psi4RunParams"]["OutputFile"], os.getpid())
    
    # Intialize Psi4...
    OptionsInfo["psi4"] = Psi4Util.InitializePsi4(Psi4RunParams = OptionsInfo["Psi4RunParams"], Psi4OptionsParams = OptionsInfo["Psi4OptionsParams"], PrintVersion = False, PrintHeader = True)
    
    # Setup conversion factor for energy units...
    SetupEnergyConversionFactor(OptionsInfo["psi4"])
    
def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""
    
    if not OptionsInfo["Psi4Initialized"]:
        InitializePsi4ForWorkerProcess()
    
    MolIndex, EncodedMol = EncodedMolInfo
    
    CalcStatus = False
    CalculatedValues = None
    
    if EncodedMol is None:
        return [MolIndex, None, CalcStatus, CalculatedValues]
    
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    MolCount = MolIndex + 1
    
    if not CheckAndValidateMolecule(Mol, MolCount):
        return [MolIndex, None, CalcStatus, CalculatedValues]
    
    # Setup a Psi4 molecule...
    Psi4Mol = SetupPsi4Mol(OptionsInfo["psi4"], Mol, MolCount)
    if Psi4Mol is None:
        return [MolIndex, None, CalcStatus, CalculatedValues]
    
    CalcStatus, CalculatedValues = CalculateMolProperties(OptionsInfo["psi4"], Psi4Mol, Mol, MolCount)
    
    return [MolIndex, RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps), CalcStatus, CalculatedValues]

def CalculateMolProperties(Psi4Handle, Psi4Mol, Mol, MolNum = None):
    """Calculate properties."""
    
    Status = False
    CalculatedValues = []
    
    #  Setup reference wave function...
    Reference = SetupReferenceWavefunction(Mol)
    Psi4Handle.set_options({'Reference': Reference})
    
    # Setup method name and basis set...
    MethodName, BasisSet = SetupMethodNameAndBasisSet(Mol)
    
    # Calculate single point energy to setup a wavefunction...
    Status, Energy, WaveFunction = Psi4Util.CalculateSinglePointEnergy(Psi4Handle, Psi4Mol, MethodName, BasisSet, ReturnWaveFunction = True, Quiet = OptionsInfo["QuietMode"])

    if not Status:
        PerformPsi4Cleanup(Psi4Handle)
        return (False, CalculatedValues)
    
    # Calculate properties...
    CalculatedValues = CalculatePropertyValues(Psi4Handle, WaveFunction, Mol, MolNum)
    
    # Clean up
    PerformPsi4Cleanup(Psi4Handle)

    return (Status, CalculatedValues)

def CalculatePropertyValues(Psi4Handle, WaveFunction, Mol, MolNum):
    """Calculate property values."""
    
    return [PropertyNamesMap["CalcFunction"][Name](Name, Psi4Handle, WaveFunction, Mol, MolNum) for Name in OptionsInfo["SpecifiedPropertyNames"]]

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

def WritePropertyValues(Writer, Mol, CalculatedValues):
    """Write out property values for a molecule."""
    
    if CalculatedValues is None:
        return
    
    for NameIndex, Name in enumerate(OptionsInfo["SpecifiedPropertyNames"]):
        PropertyValues = CalculatedValues[NameIndex]
        if re.match("^(MayerIndices|WibergLowdinIndices)$", Name, re.I):
            # Handle multiple lines values...
            Label = PropertyNamesMap["CalcValueLabels"][Name][0]
            FormattedValues = [" ".join(Values) for Values in PropertyValues]
            Mol.SetProp(Label, "\n".join(FormattedValues))
        else:
            for LabelIndex, Label in enumerate(PropertyNamesMap["CalcValueLabels"][Name]):
                Mol.SetProp(Label, PropertyValues[LabelIndex])
        
    Writer.write(Mol)
    
def CalculateDipole(PropertyName, Psi4Handle, WaveFunction, Mol, MolNum):
    """Calculate dipole values."""

    Values = ["NA"] * 4

    try:
        Title = "OEPropDipole"
        Psi4Handle.oeprop(WaveFunction, "DIPOLE", title = Title)
        
        DipoleX = WaveFunction.variable("%s DIPOLE X" % Title)
        DipoleY = WaveFunction.variable("%s DIPOLE Y" % Title)
        DipoleZ = WaveFunction.variable("%s DIPOLE Z" % Title)
        Dipole = math.sqrt(DipoleX*DipoleX + DipoleY*DipoleY + DipoleZ*DipoleZ)

        Values = [DipoleX, DipoleY, DipoleZ, Dipole]
        Values = ["%.*f" % (OptionsInfo["Precision"], Value) for Value in Values]
    except Exception as ErrMsg:
        Values = ["NA"] * 4
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Psi4 failed to calculate %s values for molecule %s:\n%s\n" % (PropertyName, RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
    
    return (Values)
    
def CalculateQuadrupole(PropertyName, Psi4Handle, WaveFunction, Mol, MolNum):
    """Calculate quadrupole values."""

    Values = ["NA"] * 6
    
    try:
        Title = "OEPropQuadrupole"
        Psi4Handle.oeprop(WaveFunction, "QUADRUPOLE", title = Title)
        
        QuadrupoleXX = WaveFunction.variable("%s QUADRUPOLE XX" % Title)
        QuadrupoleYY = WaveFunction.variable("%s QUADRUPOLE YY" % Title)
        QuadrupoleZZ = WaveFunction.variable("%s QUADRUPOLE ZZ" % Title)
        QuadrupoleXY = WaveFunction.variable("%s QUADRUPOLE XY" % Title)
        QuadrupoleXZ = WaveFunction.variable("%s QUADRUPOLE XZ" % Title)
        QuadrupoleYZ = WaveFunction.variable("%s QUADRUPOLE YZ" % Title)

        Values = [QuadrupoleXX, QuadrupoleYY, QuadrupoleZZ, QuadrupoleXY, QuadrupoleXZ, QuadrupoleYZ]
        Values = ["%.*f" % (OptionsInfo["Precision"], Value) for Value in Values]
    except Exception as ErrMsg:
        Values = ["NA"] * 6
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Psi4 failed to calculate %s values for molecule %s:\n%s\n" % (PropertyName, RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
    
    return (Values)

def CalculateLumoHomoEnergyGap(PropertyName, Psi4Handle, WaveFunction, Mol, MolNum):
    """Calculate HOMO and LUMO energy values along with the energy gap."""

    Values = ["NA"] * 3
    
    try:
        HOMOEnergy = WaveFunction.epsilon_a_subset("AO", "ALL").np[WaveFunction.nalpha()]
        LUMOEnergy = WaveFunction.epsilon_a_subset("AO", "ALL").np[WaveFunction.nalpha() + 1]
        EnergyGap = LUMOEnergy - HOMOEnergy
        
        Values = [HOMOEnergy, LUMOEnergy, EnergyGap]
        
        if OptionsInfo["ApplyEnergyConversionFactor"]:
            Values = [ Value * OptionsInfo["EnergyConversionFactor"] for Value in Values]
        Values = ["%.*f" % (OptionsInfo["Precision"], Value) for Value in Values]
    except Exception as ErrMsg:
        Values = ["NA"] * 3
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Psi4 failed to calculate %s values for molecule %s:\n%s\n" % (PropertyName, RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
    
    return (Values)

def CalculateMayerIndices(PropertyName, Psi4Handle, WaveFunction, Mol, MolNum):
    """Calculate mayer indices."""

    Values = ["NA"]
    
    try:
        Psi4Handle.oeprop(WaveFunction, "MAYER_INDICES")
        MayerIndicesList = WaveFunction.array_variable("MAYER_INDICES").np.tolist()

        # MayerIndicesList a list of lists corresponding to n x n matrix corresponding
        #  to number of atoms in a molecule...
        Values = []
        for IndicesRow in MayerIndicesList:
            IndicesRowFormatted = ["%.*f" % (OptionsInfo["Precision"], Value) for Value in IndicesRow]
            Values.append(IndicesRowFormatted)
    except Exception as ErrMsg:
        Values = ["NA"]
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Psi4 failed to calculate %s values for molecule %s:\n%s\n" % (PropertyName, RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
    
    return (Values)
    
def CalculateWibergLowdinIndices(PropertyName, Psi4Handle, WaveFunction, Mol, MolNum):
    """Calculate mayer indices."""

    Values = ["NA"]
    
    try:
        Psi4Handle.oeprop(WaveFunction, "WIBERG_LOWDIN_INDICES")
        WibergLowdinIndices = WaveFunction.array_variable("WIBERG_LOWDIN_INDICES").np.tolist()
        
        # WibergLowdinIndices a list of lists corresponding to n x n matrix corresponding
        #  to number of atoms in a molecule...
        Values = []
        for IndicesRow in WibergLowdinIndices:
            IndicesRowFormatted = ["%.*f" % (OptionsInfo["Precision"], Value) for Value in IndicesRow]
            Values.append(IndicesRowFormatted)
    except Exception as ErrMsg:
        Values = ["NA"]
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Psi4 failed to calculate %s values for molecule %s:\n%s\n" % (PropertyName, RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
    
    return (Values)

def ProcessSpecifiedPropertyNames():
    """Process specified property names."""
    
    SetupPropertyNamesInfo()
    PropertyNames = PropertyNamesMap["Names"]
    
    OptionsInfo["SpecifiedPropertyNames"] = []
    
    SpecifiedNames = re.sub(" ", "", OptionsInfo["Property"])
    if not SpecifiedNames:
        MiscUtil.PrintError("No valid property names specifed  using \ -p, --property\" option")
    
    if re.match("^All$", SpecifiedNames, re.I):
        OptionsInfo["SpecifiedPropertyNames"] = PropertyNames
        return

    # Validate propery names...
    CanonicalPropertyNamesMap = {}
    for Name in PropertyNames:
        CanonicalPropertyNamesMap[Name.lower()] = Name

    SpecifiedNamesWords = SpecifiedNames.split(",")
    for Name in SpecifiedNamesWords:
        CanonicalName = Name.lower()
        if CanonicalName not in CanonicalPropertyNamesMap:
            MiscUtil.PrintError("The property name specified, %s, using \"-p, --property\" option is not a valid name." % Name)
        
        PropertyName = CanonicalPropertyNamesMap[CanonicalName]
        OptionsInfo["SpecifiedPropertyNames"].append(PropertyName)

def ProcessListPropertyNames():
    """List available property names."""
    
    SetupPropertyNamesInfo()
    
    MiscUtil.PrintInfo("\nListing available property names...")
    Delimiter = ", "
    MiscUtil.PrintInfo("\n%s" % (Delimiter.join(PropertyNamesMap["Names"])))

    MiscUtil.PrintInfo("\nMultiple property values may be calculated for each propery name as shown below:\n")
    for PropertyName in PropertyNamesMap["Names"]:
        MiscUtil.PrintInfo("%s: %s" % (PropertyName, Delimiter.join(PropertyNamesMap["CalcValueLabels"][PropertyName])))
        
    MiscUtil.PrintInfo("")

def SetupPropertyNamesInfo(PrintInfo = True):
    """Setup information for available property names."""

    if PrintInfo:
        MiscUtil.PrintInfo("\nRetrieving information for avalible property names...")
    
    PropertyNamesMap["Names"] = ["Dipole", "Quadrupole", "LumoHomoEnergyGap", "MayerIndices", "WibergLowdinIndices"]
    
    PropertyNamesMap["CalcValueLabels"] = {}
    PropertyNamesMap["CalcFunction"] = {}

    Name = "Dipole"
    PropertyNamesMap["CalcValueLabels"][Name] = ["DiopleX", "DiopleY", "DiopleZ", "Diople (Debye)"]
    PropertyNamesMap["CalcFunction"][Name] = CalculateDipole
    
    Name = "Quadrupole"
    PropertyNamesMap["CalcValueLabels"][Name] = ["QuadrupoleXX", "QuadrupoleYY", "QuadrupoleZZ", "QuadrupoleXY", "QuadrupoleXZ", "QuadrupoleYZ"]
    PropertyNamesMap["CalcFunction"][Name] = CalculateQuadrupole
    
    Name = "LumoHomoEnergyGap"
    PropertyNamesMap["CalcValueLabels"][Name] = ["HOMOEnergy", "LUMOEnergy", "LUMO-HOMOGap (%s)" % OptionsInfo["EnergyUnits"]]
    PropertyNamesMap["CalcFunction"][Name] = CalculateLumoHomoEnergyGap
    
    Name = "MayerIndices"
    PropertyNamesMap["CalcValueLabels"]["MayerIndices"] = ["MayerIndices"]
    PropertyNamesMap["CalcFunction"][Name] = CalculateMayerIndices
    
    Name = "WibergLowdinIndices"
    PropertyNamesMap["CalcValueLabels"][Name] = ["WibergLowdinIndices"]
    PropertyNamesMap["CalcFunction"][Name] = CalculateWibergLowdinIndices
    
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
    
    # Energy units...
    OptionsInfo["EnergyUnits"] = Options["--energyUnits"]
    
    # Run and options parameters...
    OptionsInfo["Psi4OptionsParams"] = Psi4Util.ProcessPsi4OptionsParameters("--psi4OptionsParams", Options["--psi4OptionsParams"])
    OptionsInfo["Psi4RunParams"] = Psi4Util.ProcessPsi4RunParameters("--psi4RunParams", Options["--psi4RunParams"], InfileName = OptionsInfo["Infile"])
    
    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])
    
    OptionsInfo["Precision"] = int(Options["--precision"])
    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False

    OptionsInfo["Property"] = Options["--property"]
    ProcessSpecifiedPropertyNames()
    
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
    
    # Handle listing of propery names...
    if  Options["--list"]:
        ProcessListPropertyNames()
        sys.exit(0)

def ValidateOptions():
    """Validate option values."""
    
    MiscUtil.ValidateOptionTextValue("--energyUnits", Options["--energyUnits"], "Hartrees kcal/mol kJ/mol eV")
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol")
    
    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
    MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])
    
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionIntegerValue("-p, --precision", Options["--precision"], {">": 0})
    
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
    
# Setup a usage string for docopt...
_docoptUsage_ = """
Psi4CalculateProperties.py - Calculate properties

Usage:
    Psi4CalculateProperties.py [--basisSet <text>] [--energyUnits <text>] [--infileParams <Name,Value,...>]
                             [--methodName <text>] [--mp <yes or no>] [--mpParams <Name, Value,...>] 
                             [--outfileParams <Name,Value,...> ] [--overwrite] [--property <All or Name1,Name2,Name3,...>]
                             [--precision <number> ] [--psi4OptionsParams <Name,Value,...>] [--psi4RunParams <Name,Value,...>]
                             [--quiet <yes or no>] [--reference <text>] [-w <dir>] -i <infile> -o <outfile> 
    Psi4CalculateProperties.py -l | --list
    Psi4CalculateProperties.py -h | --help | -e | --examples

Description:
    Calculate properties for molecules using a specified method name and basis
    set. The molecules must have 3D coordinates in input file. The molecular
    geometry is not optimized before the calculation. In addition, hydrogens must
    be present for all molecules in input file. A single point energy calculation is 
    performed before calculating properties. The 3D coordinates are not modified
    during the calculation.

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
        Basis set to use for calculating single point energy before calculating
        properties. Default: 6-31+G** for sulfur containing molecules; Otherwise,
        6-31G** [ Ref 150 ]. The specified value must be a valid Psi4 basis set.
        No validation is performed.
        
        The following list shows a representative sample of basis sets available
        in Psi4:
            
            STO-3G, 6-31G, 6-31+G, 6-31++G, 6-31G*, 6-31+G*,  6-31++G*, 
            6-31G**, 6-31+G**, 6-31++G**, 6-311G, 6-311+G, 6-311++G,
            6-311G*, 6-311+G*, 6-311++G*, 6-311G**, 6-311+G**, 6-311++G**,
            cc-pVDZ, cc-pCVDZ, aug-cc-pVDZ, cc-pVDZ-DK, cc-pCVDZ-DK, def2-SVP,
            def2-SVPD, def2-TZVP, def2-TZVPD, def2-TZVPP, def2-TZVPPD
            
    --energyUnits <text>  [default: kcal/mol]
        Energy units for writing out LUMO and HOMO energies and their energy gap.
        Possible values: Hartrees, kcal/mol, kJ/mol, or eV.
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
            
    -l, --list
        List property names without performing any calculations.
    -m, --methodName <text>  [default: auto]
        Method to use for calculating single point energy before calculating
        properties. Default: B3LYP [ Ref 150 ]. The specified value must be a valid
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
    -p, --property <All or Name1,Name2,Name3,...>  [default: Dipole]
        Comma delimited lists of properties properties to calculate. Default:
        'Dipole'. The following properties may be calculated for molecules:
            
            Dipole,Quadrupole,LumoHomoEnergyGap,MayerIndices,
            WibergLowdinIndices
            
        Multiple property values are calculated for a single property name as
        shown below:
            
            Dipole: DipoleX, DipoleY, DipoleZ,Dipole
            Quadrupole: QuadrupoleXX, QuadrupoleYY, QuadrupoleZZ,
                QuadrupoleXY, QuadrupoleXZ, QuadrupoleYZ
            LumoHomoEnergyGap: HOMOEnergy, LUMOEnergy, LUMO-HOMOGap
            
    --precision <number>  [default: 4]
        Floating point precision for writing values of calculated properties.
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
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To calculate dipole based on a single point energy calculation using  B3LYP/6-31G**
    and B3LYP/6-31+G** for non-sulfur and sulfur containing molecules in a SD file
    with 3D structures, use RHF and UHF for closed-shell and open-shell molecules,
    and write a new SD file, type:

        % Psi4CalculateProperties.py -i Psi4Sample3D.sdf -o Psi4Sample3DOut.sdf

    To run the first example for calculating all available properties and write out a 
    SD file, type:

        % Psi4CalculateProperties.py -p All -i Psi4Sample3D.sdf
          -o Psi4Sample3DOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    without loading all data into memory and write out a SD file, type:

        % Psi4CalculateProperties.py --mp yes -i Psi4Sample3D.sdf
          -o Psi4Sample3DOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    by loading all data into memory and write out a SD file, type:

        % Psi4CalculateProperties.py  --mp yes --mpParams "inputDataMode,
          InMemory" -i Psi4Sample3D.sdf  -o Psi4Sample3DOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    without loading all data into memory along with multiple threads for each
    Psi4 run and write out a SD file, type:

        % Psi4CalculateProperties.py --mp yes --psi4RunParams "NumThreads,2"
           -i Psi4Sample3D.sdf -o Psi4Sample3DOut.sdf

    To calculate a specific set of properties based on a single point energy using
    a specific method and basis set for molecules in a SD containing 3D structures
    and write a new SD file, type:

        % Psi4CalculateProperties.py  -p "Dipole,LumoHomoEnergyGap" -m SCF
          -b aug-cc-pVDZ -i Psi4Sample3D.sdf -o Psi4Sample3DOut.sdf

Author:
    Manish Sud(msud@san.rr.com)

See also:
    Psi4CalculatePartialCharges.py, Psi4PerformMinimization.py, Psi4GenerateConformers.py

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
