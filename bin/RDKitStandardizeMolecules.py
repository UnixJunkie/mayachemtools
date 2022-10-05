#!/usr/bin/env python
#
# File: RDKitStandardizeMolecules.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2022 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using RDKit, an
# open source toolkit for cheminformatics developed by Greg Landrum.
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
import multiprocessing as mp

# RDKit imports...
try:
    from rdkit import rdBase
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem import AllChem
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import RDKit module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your RDKit environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
try:
    from docopt import docopt
    import MiscUtil
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
    
    MiscUtil.PrintInfo("\n%s (RDKit v%s; MayaChemTools v%s; %s): Starting...\n" % (ScriptName, rdBase.rdkitVersion, MiscUtil.GetMayaChemToolsVersion(), time.asctime()))
    
    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()
    
    # Retrieve command line arguments and options...
    RetrieveOptions()
    
    # Process and validate command line arguments and options...
    ProcessOptions()
    
    # Perform actions required by the script...
    StandardizeMolecules()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def StandardizeMolecules():
    """Stanardize molecules."""
    
    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])
    
    # Set up a molecule writer...
    Writer = SetupMoleculeWriter()

    MolCount, ValidMolCount, StandardizationFailedCount = ProcessMolecules(Mols, Writer)

    if Writer is not None:
        Writer.close()
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during standardization: %d" % StandardizationFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + StandardizationFailedCount))
    
    MiscUtil.PrintInfo("\nNumber of standardized molecules: %d" % (ValidMolCount - StandardizationFailedCount))

def ProcessMolecules(Mols, Writer):
    """Process and standardize molecules."""

    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, Writer)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, Writer)

def ProcessMoleculesUsingSingleProcess(Mols,  Writer):
    """Process and standardize molecules using a single process."""

    MiscUtil.PrintInfo("\nStandardizing molecules...")
    
    Compute2DCoords = OptionsInfo["OutfileParams"]["Compute2DCoords"]
    SetSMILESMolProps = OptionsInfo["OutfileParams"]["SetSMILESMolProps"]

    # Set up standardize...
    SetupStandardize()

    (MolCount, ValidMolCount, StandardizationFailedCount) = [0] * 3
    FirstMol = True
    for Mol in Mols:
        MolCount += 1
        
        if Mol is None:
            continue
        
        if RDKitUtil.IsMolEmpty(Mol):
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Ignoring empty molecule: %s" % MolName)
            continue
        
        ValidMolCount += 1
        if FirstMol:
            FirstMol = False
            if SetSMILESMolProps:
                RDKitUtil.SetWriterMolProps(Writer, Mol)
        
        StandardizedMol,  StandardizationStatus = PerformStandardization(Mol, MolCount)
        if not StandardizationStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to standardize molecule %s" % MolName)
            
            StandardizationFailedCount += 1
            continue
        
        WriteMolecule(Writer, StandardizedMol, Compute2DCoords)
    
    return (MolCount, ValidMolCount, StandardizationFailedCount)
    
def ProcessMoleculesUsingMultipleProcesses(Mols, Writer):
    """Process and standardize molecules using  multiprocessing."""
    
    MiscUtil.PrintInfo("\nStandardize molecules using multiprocessing...")
    
    MPParams = OptionsInfo["MPParams"]
    Compute2DCoords = OptionsInfo["OutfileParams"]["Compute2DCoords"]
    
    # Setup data for initializing a worker process...
    InitializeWorkerProcessArgs = (MiscUtil.ObjectToBase64EncodedString(Options), MiscUtil.ObjectToBase64EncodedString(OptionsInfo))

    # Setup a encoded mols data iterable for a worker process by pickling only public
    # and private molecule properties...
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
    
    SetSMILESMolProps = OptionsInfo["OutfileParams"]["SetSMILESMolProps"]
    
    (MolCount, ValidMolCount, StandardizationFailedCount) = [0] * 3
    FirstMol = True
    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, StandardizationStatus = Result
        
        if EncodedMol is None:
            continue
        ValidMolCount += 1
        
        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
        if FirstMol:
            FirstMol = False
            if SetSMILESMolProps:
                RDKitUtil.SetWriterMolProps(Writer, Mol)
        
        if not StandardizationStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to standardize molecule %s" % MolName)
            
            StandardizationFailedCount += 1
            continue
        
        WriteMolecule(Writer, Mol, Compute2DCoords)
    
    return (MolCount, ValidMolCount, StandardizationFailedCount)

def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""

    global Options, OptionsInfo
    
    MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())

    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Set up standardize...
    SetupStandardize()

def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""
    
    MolIndex, EncodedMol = EncodedMolInfo
    
    if EncodedMol is None:
        return [MolIndex, None, False]
        
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    if RDKitUtil.IsMolEmpty(Mol):
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, (MolIndex + 1))
            MiscUtil.PrintWarning("Ignoring empty molecule: %s" % MolName)
        return [MolIndex, None, False]
    
    Mol, StandardizationStatus = PerformStandardization(Mol, (MolIndex + 1))
    EncodedMol = RDKitUtil.MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps)
    
    return [MolIndex, EncodedMol, StandardizationStatus]
    
def PerformStandardization(Mol, MolNum):
    """Perform standardization and return a standardized mol along with the status of
    the standardization."""

    try:
        # Step 1: Cleanup...
        if OptionsInfo["MethodologyParams"]["Cleanup"]:
            Mol = CleanupMolecule(Mol)
        
        # Step2: Get largest fragment...
        if OptionsInfo["MethodologyParams"]["RemoveFragments"]:
            Mol = ChooseLargestMoleculeFragment(Mol)
        
        # Step3: Neutralize...
        if OptionsInfo["MethodologyParams"]["Neutralize"]:
            Mol = NeutralizeMolecule(Mol)
        
        # Step4: Canonicalize tautomer...
        if OptionsInfo["MethodologyParams"]["CanonicalizeTautomer"]:
            Mol = CanonicalizeMoleculeTautomer(Mol)
        
        Status = True
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to standardize molecule %s: %s" % (RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
        Status = False

    return (Mol, Status)

def CleanupMolecule(Mol):
    """Clean up molecule."""
    
    if OptionsInfo["StandardizeParams"]["CleanupRemoveHydrogens"]:
        Mol = Chem.RemoveHs(Mol)
        
    if OptionsInfo["StandardizeParams"]["CleanupDisconnectMetals"]:
        # Disconnect metal atoms that are defined as covalently bonded to non-metals...
        Mol = OptionsInfo["StandardizeObjects"]["MetalDisconnector"].Disconnect(Mol)
    
    if OptionsInfo["StandardizeParams"]["CleanupNormalize"]:
        # Apply normalization transforms to correct functional groups and recombine charges...
        Mol = rdMolStandardize.Normalize(Mol, OptionsInfo["CleanupParams"])
    
    if OptionsInfo["StandardizeParams"]["CleanupReionize"]:
        # Ensure the strongest acid groups ionize first in partially ionized molecules...
        Mol = rdMolStandardize.Reionize(Mol, OptionsInfo["CleanupParams"])
    
    if OptionsInfo["StandardizeParams"]["CleanupAssignStereo"]:
        # Assign stereochemistry
        Chem.AssignStereochemistry(Mol, force=OptionsInfo["StandardizeParams"]["CleanupAssignStereoForce"], cleanIt=OptionsInfo["StandardizeParams"]["CleanupAssignStereoCleanIt"])
    
    Mol.UpdatePropertyCache()

    return Mol

def ChooseLargestMoleculeFragment(Mol):
    """Choose largest molecule fragment. """

    return OptionsInfo["StandardizeObjects"]["LargestFragmentChooser"].choose(Mol)

def NeutralizeMolecule(Mol):
    """Neutralize molecule."""
    
    return OptionsInfo["StandardizeObjects"]["Uncharger"].uncharge(Mol)

def CanonicalizeMoleculeTautomer(Mol):
    """Canonicalize molecule tautomer."""
    
    return OptionsInfo["StandardizeObjects"]["TautomerEnumerator"].Canonicalize(Mol)

def SetupStandardize():
    """Setup RDKit standardize objects to perform standardization."""

    OptionsInfo["StandardizeObjects"] = {}
    
    OptionsInfo["CleanupParams"] = SetupStandardizeCleanupParameters()
    
    if OptionsInfo["MethodologyParams"]["Cleanup"]:
        if OptionsInfo["StandardizeParams"]["CleanupDisconnectMetals"]:
            OptionsInfo["StandardizeObjects"]["MetalDisconnector"] = rdMolStandardize.MetalDisconnector()
    
    if OptionsInfo["MethodologyParams"]["RemoveFragments"]:
        OptionsInfo["StandardizeObjects"]["LargestFragmentChooser"] = rdMolStandardize.LargestFragmentChooser(OptionsInfo["CleanupParams"])
    
    if OptionsInfo["MethodologyParams"]["Neutralize"]:
        OptionsInfo["StandardizeObjects"]["Uncharger"] = rdMolStandardize.Uncharger(OptionsInfo["CleanupParams"].doCanonical)

    if OptionsInfo["MethodologyParams"]["CanonicalizeTautomer"]:
        OptionsInfo["StandardizeObjects"]["TautomerEnumerator"] = rdMolStandardize.TautomerEnumerator(OptionsInfo["CleanupParams"])

def SetupStandardizeCleanupParameters():
    """Setup standardize clean up parameters for RDKit. """

    CleanupParams = rdMolStandardize.CleanupParameters()
    StandardizeParams = OptionsInfo["StandardizeParams"]
    
    if StandardizeParams["AcidBaseFile"] is not None:
        CleanupParams.acidbaseFile = StandardizeParams["AcidBaseFile"]
    if StandardizeParams["FragmentFile"] is not None:
        CleanupParams.acidbaseFile = StandardizeParams["FragmentFile"]
    if StandardizeParams["NormalizationsFile"] is not None:
        CleanupParams.normalizationsFile = StandardizeParams["NormalizationsFile"]
    if StandardizeParams["TautomerTransformsFile"] is not None:
        CleanupParams.tautomerTransformsFile = StandardizeParams["TautomerTransformsFile"]
    
    CleanupParams.maxRestarts = StandardizeParams["CleanupNormalizeMaxRestarts"]
    
    CleanupParams.doCanonical = StandardizeParams["DoCanonical"]
    
    CleanupParams.largestFragmentChooserUseAtomCount = StandardizeParams["LargestFragmentChooserUseAtomCount"]
    CleanupParams.largestFragmentChooserCountHeavyAtomsOnly = StandardizeParams["LargestFragmentChooserCountHeavyAtomsOnly"]
    
    CleanupParams.preferOrganic = StandardizeParams["PreferOrganic"]
    
    CleanupParams.maxTautomers = StandardizeParams["MaxTautomers"]
    CleanupParams.maxTransforms = StandardizeParams["MaxTransforms"]
    CleanupParams.tautomerRemoveBondStereo = StandardizeParams["TautomerRemoveBondStereo"]
    CleanupParams.tautomerRemoveIsotopicHs = StandardizeParams["TautomerRemoveIsotopicHs"]
    CleanupParams.tautomerRemoveSp3Stereo = StandardizeParams["TautomerRemoveSp3Stereo"]
    CleanupParams.tautomerReassignStereo = StandardizeParams["TautomerReassignStereo"]
    
    return CleanupParams

def WriteMolecule(Writer, Mol, Compute2DCoords):
    """Write out molecule."""
    
    if OptionsInfo["CountMode"]:
        return
    
    if Compute2DCoords:
        AllChem.Compute2DCoords(Mol)
    
    Writer.write(Mol)

def SetupMoleculeWriter():
    """Setup a molecule writer."""
    
    Writer = None
    if OptionsInfo["CountMode"]:
        return Writer

    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["Outfile"])
    
    return Writer

def ProcessMethodologyParameters():
    """Process methodology parameters. """

    ParamsDefaultInfo = {"Cleanup": ["bool", True], "RemoveFragments": ["bool", True], "Neutralize": ["bool", True], "CanonicalizeTautomer": ["bool", True]}
    OptionsInfo["MethodologyParams"] = MiscUtil.ProcessOptionNameValuePairParameters("--methodologyParams", Options["--methodologyParams"], ParamsDefaultInfo)
    
def ProcessStandardizationParameters():
    """Process standardization parameters. """

    ParamsDefaultInfo = {"AcidBaseFile": ["file", None], "FragmentFile": ["file", None], "NormalizationsFile": ["file", None], "TautomerTransformsFile": ["file", None], "CleanupRemoveHydrogens": ["bool", True], "CleanupDisconnectMetals": ["bool", True], "CleanupNormalize": ["bool", True], "CleanupNormalizeMaxRestarts": ["int", 200], "CleanupReionize": ["bool", True], "CleanupAssignStereo": ["bool", True], "CleanupAssignStereoCleanIt": ["bool", True], "CleanupAssignStereoForce": ["bool", True], "DoCanonical": ["bool", True], "LargestFragmentChooserUseAtomCount": ["bool", True], "LargestFragmentChooserCountHeavyAtomsOnly": ["bool", False], "PreferOrganic": ["bool", False], "MaxTautomers": ["int", 1000], "MaxTransforms": ["int", 1000], "TautomerRemoveBondStereo": ["bool", True], "TautomerRemoveIsotopicHs": ["bool", True], "TautomerRemoveSp3Stereo": ["bool", True], "TautomerReassignStereo": ["bool", True]}

    OptionsInfo["StandardizeParams"] = MiscUtil.ProcessOptionNameValuePairParameters("--standardizeParams", Options["--standardizeParams"], ParamsDefaultInfo)
    
    #  Validate numerical values...
    for ParamName in ["CleanupNormalizeMaxRestarts", "MaxTautomers", "MaxTransforms"]:
        ParamValue = OptionsInfo["StandardizeParams"][ParamName]
        if  ParamValue <= 0:
            MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"-s, --standardizeParams\" option is not a valid value. Supported values: > 0" % (ParamValue, ParamName))

def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()
    
    OptionsInfo["Infile"] = Options["--infile"]
    ParamsDefaultInfoOverride = {'RemoveHydrogens': False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters("--infileParams", Options["--infileParams"], Options["--infile"], ParamsDefaultInfo = ParamsDefaultInfoOverride)
    
    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters("--outfileParams", Options["--outfileParams"], Options["--infile"], Options["--outfile"])

    OptionsInfo["Overwrite"] = Options["--overwrite"]

    OptionsInfo["CountMode"] = False
    if re.match("^count$", Options["--mode"], re.I):
        OptionsInfo["CountMode"] = True
    
    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])

    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False

    ProcessMethodologyParameters()
    ProcessStandardizationParameters()

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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol smi txt csv tsv")
    
    if Options["--outfile"]:
        MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd smi")
        MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
        MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])

    MiscUtil.ValidateOptionTextValue("--mode", Options["--mode"], "standardize count")
    if re.match("^standardize$", Options["--mode"], re.I):
        if not Options["--outfile"]:
            MiscUtil.PrintError("The outfile must be specified using \"-o, --outfile\" during \"standardize\" value of \"--mode\" option")
    
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
    
# Setup a usage string for docopt...
_docoptUsage_ = """
RDKitStandardizeMolecules.py - Standardize molecules

Usage:
    RDKitStandardizeMolecules.py [--infileParams <Name,Value,...>] [--methodologyParams <Name,Value,...>]
                                 [--mode <standardize or count>] [--mp <yes or no>] [--mpParams <Name,Value,...>]
                                 [--outfileParams <Name,Value,...> ] [--overwrite] [--standardizeParams <Name,Value,...>]
                                 [--quiet <yes or no>] [-w <dir>] [-o <outfile>] -i <infile>
    RDKitStandardizeMolecules.py -h | --help | -e | --examples

Description:
    Standardize molecules and write them out to an output file or simply count
    the number of molecules to be standardized. The standardization methodology
    consists of the following 4 steps executed in a sequential manner:
        
        1. Cleanup molecules
        2. Keep largest fragment
        3. Neutralize molecules
        4. Select canonical tautomer
        
    The molecules are cleaned up by performing the following actions:
        
        1. Remove hydrogens
        2. Disconnect metal atoms - Disconnect metal atoms covalently bonded
            to non-metals
        3. Normalize - Normalize functional groups and recombine charges
        4. Reionize - Ionize strongest acid groups first in partially
            ionized molecules
        5. Assign stereochemistry
        
    You may optionally skip any cleanup action during standardization.

    The supported input file formats are: SD (.sdf, .sd), SMILES (.smi., csv, .tsv, .txt)

    The supported output file formats are: SD (.sdf, .sd), SMILES (.smi)

Options:
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
    -m, --mode <standardize or count>  [default: standardize]
        Specify whether to standardize molecules and write them out or simply
        count the number of molecules being standardized.
    --methodologyParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs to control
        the execution of different steps in the standardization methodology. The
        supported parameter names along with their default values are shown
        below:
            
            cleanup,yes,removeFragments,yes,neutralize,yes,
            canonicalizeTautomer,yes
            
        The standardization methodology consists of the following 4 steps executed
        in a sequential manner starting from step 1:
            
            1. cleanup
            2. removeFragments
            3. neutralize
            4. canonicalizeTautomer
            
        You may optionally skip the execution of any standardization step.
        
        The step1, cleanup, performs the following actions:
            
            1. Remove hydrogens
            2. Disconnect metal atoms - Disconnect metal atoms covalently bonded
                to non-metals
            3. Normalize - Normalize functional groups and recombine charges
            4. Reionize - Ionize strongest acid groups first in partially
                ionized molecules
            5. Assign stereochemistry
            
        You may optionally skip any cleanup action using '-s, --standardize' option.
        
        The step2, removeFragments, employs rdMolStandardize.FragmentParent()
        function to keep the largest fragment.
        
        The step3, neutralize, uses rdMolStandardize.Uncharger().uncharge()
        function to neutralize molecules by adding/removing hydrogens.
        
        The step4, canonicalizeTautomer, relies on Canonicalize() function availabe via
        rdMolStandardize.TautomerEnumerator() to select a canonical tautomer.
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
            
            SD: compute2DCoords,auto,kekulize,yes
            SMILES: smilesKekulize,no,smilesDelimiter,space, smilesIsomeric,yes,
                smilesTitleLine,yes,smilesMolName,yes,smilesMolProps,no
            
        Default value for compute2DCoords: yes for SMILES input file; no for all other
        file types.
    --overwrite
        Overwrite existing files.
    -q, --quiet <yes or no>  [default: no]
        Use quiet mode. The warning and information messages will not be printed.
    -s, --standardizeParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for standardizing
        molecules. The supported parameter names along with their default values
        are shown below:
            
            acidbaseFile,none,fragmentFile,none,normalizationsFile,none,
            tautomerTransformsFile,none,
            cleanupRemoveHydrogens,yes,cleanupDisconnectMetals,yes,
            cleanupNormalize,yes,cleanupNormalizeMaxRestarts,200,
            cleanupReionize,yes,cleanupAssignStereo,yes,
            cleanupAssignStereoCleanIt,yes,cleanupAssignStereoForce,yes
            largestFragmentChooserUseAtomCount,yes,
            largestFragmentChooserCountHeavyAtomsOnly,no,preferOrganic,no,
            doCanonical,yes,
            maxTautomers,1000,maxTransforms,1000,
            tautomerRemoveBondStereo,yes,tautomerRemoveIsotopicHs,yes
            tautomerRemoveSp3Stereo,yes,tautomerReassignStereo,yes
            
        A brief description of the standardization parameters, taken from RDKit
        documentation, is as follows:
            
            acidbaseFile - File containing acid and base definitions
            fragmentFile - File containing fragment definitions
            normalizationsFile - File conataining normalization transformations
            tautomerTransformsFile - File containing tautomer transformations
            
            cleanupRemoveHydrogens - Remove hydrogens druring cleanup
            cleanupDisconnectMetals - Disconnect metal atoms covalently bonded
                to non-metals during cleanup
            cleanupNormalize - Normalize functional groups and recombine
                charges during cleanup
            cleanupNormalizeMaxRestarts - Maximum number of restarts during
                normalization step of cleanup
            cleanupReionize -Ionize strongest acid groups first in partially
                ionized molecules during cleanup
            cleanupAssignStereo - Assign stererochemistry during cleanup
            cleanupAssignStereoCleanIt - Clean property _CIPCode during
                assign stereochemistry 
            cleanupAssignStereoForce - Always perform stereochemistry
                calculation during assign stereochemistry
            
            largestFragmentChooserUseAtomCount - Use atom count as main
                criterion before molecular weight to determine largest fragment
                in LargestFragmentChooser
            largestFragmentChooserCountHeavyAtomsOnly - Count only heavy
                atoms to determine largest fragment in LargestFragmentChooser
            preferOrganic - Prefer organic fragments over  inorganic ones when
                choosing fragments
            
            doCanonical - Apply atom-order dependent normalizations in a
                canonical order during uncharging
            
            maxTautomers - Maximum number of tautomers to generate
            maxTransforms - Maximum number of transforms to apply during
                tautomer enumeration
            tautomerRemoveBondStereo - Remove stereochemistry from double bonds
                involved in tautomerism
            tautomerRemoveIsotopicHs: Remove isotopic Hs from centers involved in tautomerism
            tautomerRemoveSp3Stereo - Remove stereochemistry from sp3 centers
                involved in tautomerism
            tautomerReassignStereo - AssignStereochemistry on all generated tautomers
            
        The default value is set to none for the following  file name parameters:
        acidbaseFile, fragmentFile, normalizationsFile, and tautomerTransformsFile.
        The script relies on RDKit to automatically load appropriate acid base and
        fragment definitions along with normalization and tautomer transformations
        from a set of internal catalogs.
        
        Note: The fragmentFile doesn't appear to be used by the RDKit method
        rdMolStandardize.FragmentParent() to find largest fragment.
            
        The contents  of various standardization definitions and transformations files
        are described below:
            
            acidbaseFile - File containing acid and base definitions
            
                // Name     Acid                 Base
                -OSO3H      OS(=O)(=O)[OH]       OS(=O)(=O)[O-]
                -SO3H       [!O]S(=O)(=O)[OH]    [!O]S(=O)(=O)[O-]
                -OSO2H      O[SD3](=O)[OH]       O[SD3](=O)[O-]
                ... ... ...
        
            fragmentFile - File containing fragment definitions
            
                // Name     SMARTS
                hydrogen     [H]
                fluorine     [F]
                chlorine     [Cl]
                ... ... ...
        
            normalizationsFile - File conataining normalization transformations
            
                // Name     SMIRKS
                Sulfone to S(=O)(=O)        [S+2:1]([O-:2])([O-:3])>>
                    [S+0:1](=[O-0:2])(=[O-0:3])
                Pyridine oxide to n+O-     [n:1]=[O:2]>>[n+:1][O-:2]
                ... ... ...
        
            tautomerTransformsFile - File containing tautomer transformations
            
                // Name                SMARTS   Bonds  Charges
                1,3 (thio)keto/enol f  [CX4!H0]-[C]=[O,S,Se,Te;X1]
                1,3 (thio)keto/enol r  [O,S,Se,Te;X2!H0]-[C]=[C]
                1,5 (thio)keto/enol f  [CX4,NX3;!H0]-[C]=[C][CH0]=[O,S,Se,Te;X1]
                ... ... ...
            
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To standardize molecules in a SMILES file by executing all standardization
    steps and write out a SMILES file, type:

        % RDKitStandardizeMolecules.py -i Sample.smi -o SampleOut.smi

    To standardize molecules in a SD file by executing all standardization
    steps, performing standardization in multiprocessing mode on all available
    CPUs without loading all data into memory, and write out and write out a
    SD file, type:

        % RDKitStandardizeMolecules.py --mp yes -i Sample.sdf -o SampleOut.sdf

    To standardize molecules in a SMILES file by executing  all standardization
    steps, performing standardization in multiprocessing mode on all available
    CPUs by loading all data into memory, and write out and write out a
    SMILES file, type:

        % RDKitStandardizeMolecules.py --mp yes --mpParams "inputDataMode,
          InMemory" -i Sample.smi -o SampleOut.smi
    
    To standardize molecules in a SMILES file by executing  all standardization
    steps, performing standardization in multiprocessing mode on specific number
    of CPUs and chunk size without loading all data into memory, and write out a
    a SMILES file, type:

        % RDKitStandardizeMolecules.py --mp yes --mpParams "inputDataMode,Lazy,
          numProcesses,4,chunkSize,8" -i Sample.smi -o SampleOut.smi

    To count number of molecules to be standardized without generating any
    output file, type:

        % RDKitStandardizeMolecules.py -m count -i Sample.sdf

    To standardize molecules in a SD file by executing specific standardization
    steps along with explicit values for various parameters to control the
    standardization behavior, and write out a SD file, type:

        % RDKitStandardizeMolecules.py --methodologyParams "cleanup,yes,
          removeFragments,yes,neutralize,yes,canonicalizeTautomer,yes"
          --standardizeParams "cleanupRemoveHydrogens,yes,
          cleanupDisconnectMetals,yes,cleanupNormalize,yes,
          cleanupNormalizeMaxRestarts,200,cleanupReionize,yes,
          cleanupAssignStereo,yes,largestFragmentChooserUseAtomCount,yes,
          doCanonical,yes,maxTautomers,1000"
          -i Sample.sdf -o SampleOut.sdf

    To standardize molecules in a CSV SMILES file, SMILES strings in column 1,
    name in column 2, and generate output SD file, type:

        % RDKitStandardizeMolecules.py --infileParams 
          "smilesDelimiter,comma,smilesTitleLine,yes,smilesColumn,1,
          smilesNameColumn,2" --outfileParams "compute2DCoords,yes"
          -i SampleSMILES.csv -o SampleOut.sdf

Author:
    Manish Sud(msud@san.rr.com)

See also:
    RDKitConvertFileFormat.py, RDKitEnumerateTautomers.py,
    RDKitRemoveDuplicateMolecules.py, RDKitRemoveInvalidMolecules.py,
    RDKitRemoveSalts.py, RDKitSearchFunctionalGroups.py, RDKitSearchSMARTS.py

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
