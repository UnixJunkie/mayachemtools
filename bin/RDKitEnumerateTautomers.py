#!/usr/bin/env python
#
# File: RDKitEnumerateTautomers.py
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
    EnumerateTautomers()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def EnumerateTautomers():
    """Enunmerate tautomers."""
    
    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])
    
    # Set up a molecule writer...
    Writer = SetupMoleculeWriter()

    MolCount, ValidMolCount, TautomerizationFailedCount, TautomersCount, MinTautomersCount, MaxTautomersCount = ProcessMolecules(Mols, Writer)

    if Writer is not None:
        Writer.close()
    
    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during tautomerization: %d" % TautomerizationFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + TautomerizationFailedCount))
    
    MiscUtil.PrintInfo("\nNumber of tautomerized molecules: %d" % (ValidMolCount - TautomerizationFailedCount))
    
    MiscUtil.PrintInfo("\nTotal number of tautomers for molecules: %d" % TautomersCount)
    MiscUtil.PrintInfo("Minumum number of tautomers for a molecule: %d" % MinTautomersCount)
    MiscUtil.PrintInfo("Maxiumum number of tautomers for a molecule: %d" % MaxTautomersCount)
    MiscUtil.PrintInfo("Average number of tautomers for a molecule: %.1f" % (TautomersCount/(ValidMolCount - TautomerizationFailedCount)))

def ProcessMolecules(Mols, Writer):
    """Process molecules."""

    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, Writer)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, Writer)

def ProcessMoleculesUsingSingleProcess(Mols,  Writer):
    """Process and generate tautomers for molecules using a single process."""

    MiscUtil.PrintInfo("\nEnumerating tatutomers...")
    
    Compute2DCoords = OptionsInfo["OutfileParams"]["Compute2DCoords"]
    SetSMILESMolProps = OptionsInfo["OutfileParams"]["SetSMILESMolProps"]

    # Set up tautomer enumerator...
    TautomerEnumerator = SetupTautomerEnumerator()

    (MolCount, ValidMolCount, TautomerizationFailedCount, TautomersCount) = [0] * 4
    (MinTautomersCount, MaxTautomersCount) = [sys.maxsize, 0]
    FirstTautomerMol = True
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
        
        TautomerMols,  TautomerizationStatus = EnumerateMolTautomers(Mol, TautomerEnumerator, MolCount)
        if not TautomerizationStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to tautomerize molecule %s" % MolName)
            
            TautomerizationFailedCount += 1
            continue
        
        if FirstTautomerMol:
            FirstTautomerMol = False
            if SetSMILESMolProps:
                RDKitUtil.SetWriterMolProps(Writer, TautomerMols[0])

        # Track tautomer count...
        TautomerMolsCount = len(TautomerMols)
        TautomersCount += TautomerMolsCount
        if TautomerMolsCount < MinTautomersCount:
            MinTautomersCount = TautomerMolsCount
        if TautomerMolsCount > MaxTautomersCount:
            MaxTautomersCount = TautomerMolsCount
        
        WriteMolTautomers(Writer, Mol, MolCount, Compute2DCoords, TautomerMols)
    
    return (MolCount, ValidMolCount, TautomerizationFailedCount, TautomersCount, MinTautomersCount, MaxTautomersCount)
    
def ProcessMoleculesUsingMultipleProcesses(Mols, Writer):
    """Process and enumerate tautomer of molecules using  multiprocessing."""
    
    MiscUtil.PrintInfo("\nEnumerating tatutomers using multiprocessing...")
    
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
    
    (MolCount, ValidMolCount, TautomerizationFailedCount, TautomersCount) = [0] * 4
    (MinTautomersCount, MaxTautomersCount) = [sys.maxsize, 0]
    FirstTautomerMol = True
    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, TautomerizationStatus, EncodedTautomerMols = Result
        
        if EncodedMol is None:
            continue
        ValidMolCount += 1
        
        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
        TautomerMols = []
        if EncodedTautomerMols is not None:
            TautomerMols = [RDKitUtil.MolFromBase64EncodedMolString(EncodedTautomerMol) for EncodedTautomerMol in EncodedTautomerMols]
            
        if not TautomerizationStatus:
            if not OptionsInfo["QuietMode"]:
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to tautomerize molecule %s" % MolName)
            
            TautomerizationFailedCount += 1
            continue

        if FirstTautomerMol:
            FirstTautomerMol = False
            if SetSMILESMolProps:
                RDKitUtil.SetWriterMolProps(Writer, TautomerMols[0])

        # Track tautomer count...
        TautomerMolsCount = len(TautomerMols)
        TautomersCount += TautomerMolsCount
        if TautomerMolsCount < MinTautomersCount:
            MinTautomersCount = TautomerMolsCount
        if TautomerMolsCount > MaxTautomersCount:
            MaxTautomersCount = TautomerMolsCount
        
        WriteMolTautomers(Writer, Mol, MolCount, Compute2DCoords, TautomerMols)
    
    return (MolCount, ValidMolCount, TautomerizationFailedCount, TautomersCount, MinTautomersCount, MaxTautomersCount)

def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""

    global Options, OptionsInfo
    
    MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())

    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Set up tautomer enumerator...
    OptionsInfo["TautomerEnumerator"] = SetupTautomerEnumerator()

def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""
    
    MolIndex, EncodedMol = EncodedMolInfo
    
    if EncodedMol is None:
        return [MolIndex, None, False, None]
        
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    if RDKitUtil.IsMolEmpty(Mol):
        if not OptionsInfo["QuietMode"]:
            MolName = RDKitUtil.GetMolName(Mol, (MolIndex + 1))
            MiscUtil.PrintWarning("Ignoring empty molecule: %s" % MolName)
        return [MolIndex, None, False, None]
    
    TautomerMols,  TautomerizationStatus = EnumerateMolTautomers(Mol, OptionsInfo["TautomerEnumerator"], (MolIndex + 1))
    
    EncodedTautomerMols = None
    if TautomerMols is not None:
        EncodedTautomerMols = [RDKitUtil.MolToBase64EncodedMolString(TautomerMol, PropertyPickleFlags = Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps) for TautomerMol in TautomerMols]
    
    return [MolIndex, EncodedMol, TautomerizationStatus, EncodedTautomerMols]
    
def EnumerateMolTautomers(Mol, TautomerEnumerator, MolNum):
    """Enumerate tautomers of a molecule and return a list of tatutomers
    along with the status of tautomerization."""

    TautomerMols, Status, TautomerScores  = [None, False, None]
    try:
        TautomerMols = [TautomerMol for TautomerMol in TautomerEnumerator.Enumerate(Mol)]

        if OptionsInfo["ScoreTautomers"]:
            TautomerScores = [TautomerEnumerator.ScoreTautomer(TautomerMol) for TautomerMol in TautomerMols]

        if OptionsInfo["SortTautomers"]:
            TautomerMols, TautomerScores = SortMolTautomers(Mol, TautomerEnumerator, TautomerMols, TautomerScores)

        # Set tautomer score...
        if TautomerScores is not None:
            for Index, TautomerMol in enumerate(TautomerMols):
                TautomerMol.SetProp("Tautomer_Score", "%.1f" % TautomerScores[Index])
                
        Status = True
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to tautomerize molecule %s: %s" % (RDKitUtil.GetMolName(Mol, MolNum), ErrMsg))
        TautomerMols, Status = [None, False]

    return (TautomerMols,  Status)

def SortMolTautomers(Mol, TautomerEnumerator, TautomerMols, TautomerScores = None):
    """Sort tatutomers by SMILES string and place canonical tautomer at the top
    of the list."""

    CanonicalTautomer = TautomerEnumerator.Canonicalize(Mol)
    CanonicalTautomerSmiles = Chem.MolToSmiles(CanonicalTautomer)
    if TautomerScores is None:
        CanonicalTautomerScore = None
    else:
        CanonicalTautomerScore = TautomerEnumerator.ScoreTautomer(CanonicalTautomer)

    TautomerSmiles = [Chem.MolToSmiles(TautomerMol) for TautomerMol in TautomerMols]
    if TautomerScores is None:
        SortedResults = sorted((Smiles,  TautomerMol) for Smiles,  TautomerMol in zip(TautomerSmiles, TautomerMols) if Smiles != CanonicalTautomerSmiles)
    else:
        SortedResults = sorted((Smiles,  TautomerMol, TautomerScore) for Smiles,  TautomerMol, TautomerScore in zip(TautomerSmiles, TautomerMols, TautomerScores) if Smiles != CanonicalTautomerSmiles)
    
    SortedTautomerMols = [CanonicalTautomer]
    if TautomerScores is None:
        SortedTautomerMols += [TautomerMol for Smiles,  TautomerMol in SortedResults]
    else:
        SortedTautomerMols += [TautomerMol for Smiles,  TautomerMol, TautomerScore in SortedResults]
    
    if TautomerScores is None:
        SortedTautomerScores = None
    else:
        SortedTautomerScores = [CanonicalTautomerScore]
        SortedTautomerScores += [TautomerScore for Smiles,  TautomerMol, TautomerScore in SortedResults]

    return (SortedTautomerMols, SortedTautomerScores)

def WriteMolTautomers(Writer, Mol, MolNum, Compute2DCoords, TautomerMols):
    """Write out tautomers of a  molecule."""

    if TautomerMols is None:
        return
    
    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    
    for Index, TautomerMol in enumerate(TautomerMols):
        SetupTautomerMolName(TautomerMol, MolName, (Index + 1))

        if Compute2DCoords:
            AllChem.Compute2DCoords(Mol)
    
        Writer.write(TautomerMol)

def SetupTautomerMolName(Mol, MolName, TautomerCount):
    """Set tautomer mol name."""

    TautomerName = "%s_Taut%d" % (MolName, TautomerCount)
    Mol.SetProp("_Name", TautomerName)

def SetupTautomerEnumerator():
    """Setup tautomer enumerator. """
    
    TautomerParams  = SetupTautomerizationParameters()
    
    return rdMolStandardize.TautomerEnumerator(TautomerParams)
    
def SetupTautomerizationParameters():
    """Setup tautomerization parameters for RDKit using cleanup parameters."""

    Params = rdMolStandardize.CleanupParameters()
    TautomerizationParams = OptionsInfo["TautomerizationParams"]
    
    if TautomerizationParams["TautomerTransformsFile"] is not None:
        Params.tautomerTransformsFile = TautomerizationParams["TautomerTransformsFile"]
    
    Params.maxTautomers = TautomerizationParams["MaxTautomers"]
    Params.maxTransforms = TautomerizationParams["MaxTransforms"]
    Params.tautomerRemoveBondStereo = TautomerizationParams["TautomerRemoveBondStereo"]
    Params.tautomerRemoveIsotopicHs = TautomerizationParams["TautomerRemoveIsotopicHs"]
    Params.tautomerRemoveSp3Stereo = TautomerizationParams["TautomerRemoveSp3Stereo"]
    Params.tautomerReassignStereo = TautomerizationParams["TautomerReassignStereo"]
    
    return Params

def SetupMoleculeWriter():
    """Setup a molecule writer."""
    
    Writer = None

    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["Outfile"])
    
    return Writer

def ProcessTautomerizationParameters():
    """Process tautomerizationparameters. """

    ParamsDefaultInfo = {"TautomerTransformsFile": ["file", None], "MaxTautomers": ["int", 1000], "MaxTransforms": ["int", 1000], "TautomerRemoveBondStereo": ["bool", True], "TautomerRemoveIsotopicHs": ["bool", True], "TautomerRemoveSp3Stereo": ["bool", True], "TautomerReassignStereo": ["bool", True]}

    OptionsInfo["TautomerizationParams"] = MiscUtil.ProcessOptionNameValuePairParameters("--tautomerizationParams", Options["--tautomerizationParams"], ParamsDefaultInfo)
    
    #  Validate numerical values...
    for ParamName in ["MaxTautomers", "MaxTransforms"]:
        ParamValue = OptionsInfo["TautomerizationParams"][ParamName]
        if  ParamValue <= 0:
            MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"-t, --tautomerizationParams\" option is not a valid value. Supported values: > 0" % (ParamValue, ParamName))

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

    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])

    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False
    
    OptionsInfo["ScoreTautomers"] = True if re.match("^yes$", Options["--scoreTautomers"], re.I) else False
    OptionsInfo["SortTautomers"] = True if re.match("^yes$", Options["--sortTautomers"], re.I) else False

    ProcessTautomerizationParameters()
    
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

    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--scoreTautomers", Options["--scoreTautomers"], "yes no")
    MiscUtil.ValidateOptionTextValue("--sortTautomers", Options["--sortTautomers"], "yes no")
    
# Setup a usage string for docopt...
_docoptUsage_ = """
RDKitEnumerateTautomers.py - Enumerate tautomers of molecules

Usage:
    RDKitEnumerateTautomers.py [--infileParams <Name,Value,...>] [--mp <yes or no>] [--mpParams <Name,Value,...>]
                               [--outfileParams <Name,Value,...> ] [--overwrite] [--quiet <yes or no>] [--scoreTautomers <yes or no>]
                               [--sortTautomers <yes or no>] [--tautomerizationParams <Name,Value,...>] [-w <dir>] -i <infile> -o <outfile>
    RDKitEnumerateTautomers.py -h | --help | -e | --examples

Description:
    Enumerate tautomers [ Ref 159 ] for molecules and write them out to an output file.
    The tautomer enumerator generates both protomers and valence tautomers. You
    may optionally calculate tautomer scores and sort tautomers by SMILES string. The
    canonical tautomer is placed at the top during sorting.

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
    --scoreTautomers <yes or no>  [default: no]
        Calculate and write out tautomer scores [ Ref 159 ].
    --sortTautomers <yes or no>  [default: no]
        Sort tatutomers of a molecule by SMILES string and place canonical tautomer
        at the top of the list.
    -t, --tautomerizationParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for enumerating
        tautomers of molecules. The supported parameter names along with their
        default values are shown below:
            
            tautomerTransformsFile,none,
            maxTautomers,1000,maxTransforms,1000,
            tautomerRemoveBondStereo,yes,tautomerRemoveIsotopicHs,yes
            tautomerRemoveSp3Stereo,yes,tautomerReassignStereo,yes
            
        A brief description of the tatutomerization parameters, taken from RDKit
        documentation, is as follows:
            
            tautomerTransformsFile - File containing tautomer transformations
            
            maxTautomers - Maximum number of tautomers to generate
            maxTransforms - Maximum number of transforms to apply during
                tautomer enumeration
            tautomerRemoveBondStereo - Remove stereochemistry from double bonds
                involved in tautomerism
            tautomerRemoveIsotopicHs: Remove isotopic Hs from centers involved in tautomerism
            tautomerRemoveSp3Stereo - Remove stereochemistry from sp3 centers
                involved in tautomerism
            tautomerReassignStereo - AssignStereochemistry on all generated tautomers
            
        The default value is set to none for the 'tautomerTransformsFile' parameter. The
        script relies on RDKit to automatically load appropriate tautomer transformations
        from a set of internal catalog.
        
        The contents  of transformation file are described below:
            
            tautomerTransformsFile - File containing tautomer transformations
            
                // Name                SMARTS   Bonds  Charges
                1,3 (thio)keto/enol f  [CX4!H0]-[C]=[O,S,Se,Te;X1]
                1,3 (thio)keto/enol r  [O,S,Se,Te;X2!H0]-[C]=[C]
                1,5 (thio)keto/enol f  [CX4,NX3;!H0]-[C]=[C][CH0]=[O,S,Se,Te;X1]
                ... ... ...
            
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To enumerate tautomers of molecules in a SMILES file and write out a SMILES
    file, type: 

        % RDKitEnumerateTautomers.py -i Sample.smi -o SampleOut.smi

    To enumerate tautomers of molecules in a SD file, calculate tautomer scores,
    sort tautomers, and write out a SD file, type:

        % RDKitEnumerateTautomers.py --scoreTautomers yes --sortTautomers yes
          -i Sample.sdf -o SampleOut.sdf

    To enumerate tautomers of molecules in a SD fie , calculate tautomer
    scores, sort tautomers, and write out a SMILES file, type:

        % RDKitEnumerateTautomers.py --scoreTautomers yes  --sortTautomers yes
          --outfileParams "smilesMolProps,yes" -i Sample.smi -o SampleOut.smi

    To enumerate tautomers of  molecules in a SD file, performing enumeration in
    multiprocessing mode on all available CPUs without loading all data into
    memory, and write out a SD file, type:

        % RDKitEnumerateTautomers.py --mp yes -i Sample.sdf -o SampleOut.sdf

    To enumerate tautomers of  molecules in a SD file, performing enumeration in
    multiprocessing mode on specific number of CPUs and chunk size without loading
    all data into memory, and write out a SD file, type:

        % RDKitEnumerateTautomers.py --mp yes --mpParams "inputDataMode,Lazy,
          numProcesses,4,chunkSize,8" -i Sample.sdf -o SampleOut.sdf

    To enumerate tautomers of  molecules in a SD file using specific values of
    parameters to contol the enumeration behavior, and write out a SD file, type:

        % RDKitEnumerateTautomers.py  -t "maxTautomers,1000,maxTransforms,1000,
          tautomerRemoveBondStereo,yes,tautomerRemoveIsotopicHs,yes,
          tautomerRemoveSp3Stereo,yes,tautomerReassignStereo,yes"
          --scoreTautomers yes --sortTautomers yes -i Sample.sdf -o SampleOut.sdf

    To enumerate tautomers for molecules in a CSV SMILES file, SMILES strings in column 1,
    name in column 2, and generate output SD file, type:

        % RDKitEnumerateTautomers.py --infileParams 
          "smilesDelimiter,comma,smilesTitleLine,yes,smilesColumn,1,
          smilesNameColumn,2" --outfileParams "compute2DCoords,yes"
          -i SampleSMILES.csv -o SampleOut.sdf

Author:
    Manish Sud(msud@san.rr.com)

See also:
    RDKitConvertFileFormat.py, RDKitRemoveDuplicateMolecules.py,
    RDKitRemoveInvalidMolecules.py, RDKitRemoveSalts.py,
    RDKitSearchFunctionalGroups.py, RDKitSearchSMARTS.py,
    RDKitStandardizeMolecules.py

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
