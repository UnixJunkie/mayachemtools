#
# File: Psi4Util.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2022 Manish Sud. All rights reserved.
#
# The functionality available in this file is implemented using Psi4, an open
# source quantum chemistry software package.
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

import os
import sys
import re
import glob

import MiscUtil

__all__ = ["CalculateSinglePointEnergy", "InitializePsi4", "JoinMethodNameAndBasisSet", "ListPsi4RunParamaters", "RetrieveIsocontourRangeFromCubeFile", "RetrieveMinAndMaxValueFromCubeFile", "PerformGeometryOptimization", "ProcessPsi4CubeFilesParameters", "ProcessPsi4OptionsParameters", "ProcessPsi4RunParameters", "RemoveScratchFiles", "UpdatePsi4OptionsParameters", "UpdatePsi4RunParameters", "UpdatePsi4OutputFileUsingPID"]

def InitializePsi4(Psi4RunParams = None,  Psi4OptionsParams = None, PrintVersion = False, PrintHeader = False):
    """Import Psi4 module and configure it for running Psi4 jobs.
    
    Arguments:
        Psi4RunParams (dict): Runtime parameter name and value pairs.
        Psi4OptionsParams (dict): Option name and value pairs. This is simply
            passed to ps4.set_options().      
        PrintVersion (boolean): Print version number.
        PrintHeader (boolean): Print header information.

    Returns:
        Object: Psi4 module reference.

    """
    
    # Import Psi4...
    try:
        import psi4
    except ImportError as ErrMsg:
        sys.stderr.write("\nFailed to import Psi4 module/package: %s\n" % ErrMsg)
        sys.stderr.write("Check/update your Psi4 environment and try again.\n\n")
        sys.exit(1)

    Psi4Handle = psi4
    
    if PrintVersion:
        MiscUtil.PrintInfo("Importing Psi4 module (Psi4 v%s)...\n" % (Psi4Handle.__version__))

    # Update Psi4 run paramaters...
    if Psi4RunParams is not None:
        UpdatePsi4RunParameters(Psi4Handle, Psi4RunParams)

    # Update Psi4 options...
    if Psi4OptionsParams is not None:
        UpdatePsi4OptionsParameters(Psi4Handle, Psi4OptionsParams)
        
    # Print header after updating Psi4 run parameters...
    if PrintHeader:
        Psi4Handle.print_header()
    
    return Psi4Handle
    
def CalculateSinglePointEnergy(psi4, Molecule, Method, BasisSet, ReturnWaveFunction = False, Quiet = False):
    """Calculate single point electronic energy in Hartrees using a specified
    method and basis set.

    Arguments:
        psi4 (Object): Psi4 module reference.
        Molecule (Object): Psi4 molecule object.
        Method (str): A valid method name.
        BasisSet (str): A valid basis set.
        ReturnWaveFunction (boolean): Return wave function.
        Quiet (boolean): Flag to print error message.

    Returns:
        float: Total electronic energy in Hartrees.
        (float, psi4 object): Energy and wavefuction.

    """
    
    Status = False
    Energy, WaveFunction = [None] * 2

    try:
        MethodAndBasisSet = JoinMethodNameAndBasisSet(Method, BasisSet)
        if ReturnWaveFunction:
            Energy, WaveFunction = psi4.energy(MethodAndBasisSet, molecule = Molecule, return_wfn = True)
        else:
            Energy = psi4.energy(MethodAndBasisSet, molecule = Molecule, return_wfn = False)
        Status = True
    except Exception as ErrMsg:
        if not Quiet:
            MiscUtil.PrintWarning("Psi4Util.CalculateSinglePointEnergy: Failed to calculate energy:\n%s\n" % ErrMsg)
    
    return (Status, Energy, WaveFunction) if ReturnWaveFunction else (Status, Energy)
    
def PerformGeometryOptimization(psi4, Molecule, Method, BasisSet, ReturnWaveFunction = True, Quiet = False):
    """Perform geometry optimization using a specified method and basis set.
    
    Arguments:
        psi4 (Object): Psi4 module reference.
        Molecule (Object): Psi4 molecule object.
        Method (str): A valid method name.
        BasisSet (str): A valid basis set.
        ReturnWaveFunction (boolean): Return wave function.
        Quiet (boolean): Flag to print error message.

    Returns:
        float: Total electronic energy in Hartrees.
        (float, psi4 object): Energy and wavefuction.

    """
    
    Status = False
    Energy, WaveFunction = [None] * 2

    try:
        MethodAndBasisSet = JoinMethodNameAndBasisSet(Method, BasisSet)
        if ReturnWaveFunction:
            Energy, WaveFunction = psi4.optimize(MethodAndBasisSet, molecule = Molecule, return_wfn = True)
        else:
            Energy = psi4.optimize(MethodAndBasisSet, molecule = Molecule, return_wfn = False)
        Status = True
    except Exception as ErrMsg:
        if not Quiet:
            MiscUtil.PrintWarning("Psi4Util.PerformGeometryOptimization: Failed to perform geometry optimization:\n%s\n" % ErrMsg)
    
    return (Status, Energy, WaveFunction) if ReturnWaveFunction else (Status, Energy)
    
def JoinMethodNameAndBasisSet(MethodName, BasisSet):
    """Join method name and basis set using a backslash delimiter.
    An empty basis set specification is ignored.

    Arguments:
        MethodName (str): A valid method name.
        BasisSet (str): A valid basis set or an empty string.

    Returns:
        str: MethodName/BasisSet or MethodName

    """
    
    return MethodName if MiscUtil.IsEmpty(BasisSet) else "%s/%s" % (MethodName, BasisSet)
    
def GetAtomPositions(psi4, WaveFunction, InAngstroms = True):
    """Retrieve a list of lists containing coordinates of all atoms in the
    molecule available in Psi4 wave function. By default, the atom positions
    are returned in Angstroms. The Psi4 default is Bohr.

    Arguments:
        psi4 (Object): Psi4 module reference.
        WaveFunction (Object): Psi4 wave function reference.
        InAngstroms (bool): True - Positions in Angstroms; Otherwise, in Bohr.

    Returns:
        None or list : List of lists containing atom positions.

    Examples:

        for AtomPosition in Psi4Util.GetAtomPositions(Psi4Handle, WaveFunction):
            print("X: %s; Y: %s; Z: %s" % (AtomPosition[0], AtomPosition[1],
                AtomPosition[2]))

    """

    if WaveFunction is None:
        return None
    
    AtomPositions = WaveFunction.molecule().geometry().to_array()
    if InAngstroms:
        AtomPositions = AtomPositions * psi4.constants.bohr2angstroms
    
    return AtomPositions.tolist()

def ListPsi4RunParamaters(psi4):
    """List values for a key set of the following Psi4 runtime parameters:
    Memory, NumThreads, OutputFile, ScratchDir, DataDir.
    
    Arguments:
        psi4 (object): Psi4 module reference.

    Returns:
        None

    """
    
    MiscUtil.PrintInfo("\nListing Psi4 run options:")
    
    # Memory in bytes...
    Memory = psi4.get_memory()
    MiscUtil.PrintInfo("Memory: %s (B); %s (MB)" % (Memory, Memory/(1024*1024)))
    
    # Number of threads...
    NumThreads = psi4.get_num_threads()
    MiscUtil.PrintInfo("NumThreads: %s " % (NumThreads))
    
    # Output file...
    OutputFile = psi4.core.get_output_file()
    MiscUtil.PrintInfo("OutputFile: %s " % (OutputFile))
    
    # Scratch dir...
    psi4_io = psi4.core.IOManager.shared_object()
    ScratchDir = psi4_io.get_default_path()
    MiscUtil.PrintInfo("ScratchDir: %s " % (ScratchDir))
    
    # Data dir...
    DataDir = psi4.core.get_datadir()
    MiscUtil.PrintInfo("DataDir: %s " % (DataDir))

def UpdatePsi4OptionsParameters(psi4, OptionsInfo):
    """Update Psi4 options using psi4.set_options().
    
    Arguments:
        psi4 (object): Psi4 module reference.
        OptionsInfo (dictionary) : Option name and value pairs for setting
            global and module options.

    Returns:
        None

    """
    if OptionsInfo is None:
        return

    if len(OptionsInfo) == 0:
        return

    try:
        psi4.set_options(OptionsInfo)
    except Exception as ErrMsg:
        MiscUtil.PrintWarning("Psi4Util.UpdatePsi4OptionsParameters: Failed to set Psi4 options\n%s\n" % ErrMsg)

def UpdatePsi4RunParameters(psi4, RunParamsInfo):
    """Update Psi4 runtime parameters. The supported parameter names along with
    their default values are as follows: MemoryInGB: 1; NumThreads: 1, OutputFile:
    stdout; ScratchDir: auto; RemoveOutputFile: True.

    Arguments:
        psi4 (object): Psi4 module reference.
        RunParamsInfo (dictionary) : Parameter name and value pairs for
            configuring Psi4 jobs.

    Returns:
        None

    """

    # Set default values for possible arguments...
    Psi4RunParams = {"MemoryInGB": 1, "NumThreads": 1, "OutputFile": "stdout",  "ScratchDir" : "auto", "RemoveOutputFile": True}
    
    # Set specified values for possible arguments...
    for Param in Psi4RunParams:
        if Param in RunParamsInfo:
            Psi4RunParams[Param] = RunParamsInfo[Param]
    
    # Memory...
    Memory = int(Psi4RunParams["MemoryInGB"]*1024*1024*1024)
    psi4.core.set_memory_bytes(Memory, True)
    
    # Number of threads...
    psi4.core.set_num_threads(Psi4RunParams["NumThreads"], quiet = True)

    # Output file...
    OutputFile = Psi4RunParams["OutputFile"]
    if not re.match("^stdout$", OutputFile, re.I):
        # Possible values: stdout, quiet, devnull, or filename
        if re.match("^(quiet|devnull)$", OutputFile, re.I):
            # Psi4 output is redirected to /dev/null after call to be_quiet function...
            psi4.core.be_quiet()
        else:
            # Delete existing output file at the start of the first Psi4 run...
            if Psi4RunParams["RemoveOutputFile"]:
                if os.path.isfile(OutputFile):
                    os.remove(OutputFile)
                    
            # Append to handle output from multiple Psi4 runs for molecules in
            # input file...
            Append = True
            psi4.core.set_output_file(OutputFile, Append)

    # Scratch directory...
    ScratchDir = Psi4RunParams["ScratchDir"]
    if not re.match("^auto$", ScratchDir, re.I):
        if not os.path.isdir(ScratchDir):
            MiscUtil.PrintError("ScratchDir is not a directory: %s" % ScratchDir)
        psi4.core.IOManager.shared_object().set_default_path(os.path.abspath(os.path.expanduser(ScratchDir)))

def ProcessPsi4OptionsParameters(ParamsOptionName, ParamsOptionValue):
    """Process parameters for setting up Psi4 options and return a map
    containing processed parameter names and values.
    
    ParamsOptionValue is a comma delimited list of Psi4 option name and value
    pairs for setting global and module options. The names are 'option_name'
    for global options and 'module_name__option_name' for options local to a
    module. The specified option names must be valid Psi4 names. No validation
    is performed.
    
    The specified option name and  value pairs are processed and passed to
    psi4.set_options() as a dictionary. The supported value types are float,
    integer, boolean, or string. The float value string is converted into a float.
    The valid values for a boolean string are yes, no, true, false, on, or off. 

    Arguments:
        ParamsOptionName (str): Command line input parameters option name.
        ParamsOptionValue (str): Comma delimited list of parameter name and value pairs.

    Returns:
        dictionary: Processed parameter name and value pairs.

    """

    OptionsInfo = {}
    
    if re.match("^(auto|none)$", ParamsOptionValue, re.I):
        return None

    ParamsOptionValue = ParamsOptionValue.strip()
    if not ParamsOptionValue:
        PrintError("No valid parameter name and value pairs specified using \"%s\" option" % ParamsOptionName)
    
    ParamsOptionValueWords = ParamsOptionValue.split(",")
    if len(ParamsOptionValueWords) % 2:
        MiscUtil.PrintError("The number of comma delimited paramater names and values, %d, specified using \"%s\" option must be an even number." % (len(ParamsOptionValueWords), ParamsOptionName))
    
    # Validate paramater name and value pairs...
    for Index in range(0, len(ParamsOptionValueWords), 2):
        Name = ParamsOptionValueWords[Index].strip()
        Value = ParamsOptionValueWords[Index + 1].strip()
        
        if  MiscUtil.IsInteger(Value):
            Value = int(Value)
        elif MiscUtil.IsFloat(Value):
            Value = float(Value)
        
        OptionsInfo[Name] = Value

    return OptionsInfo

def ProcessPsi4RunParameters(ParamsOptionName, ParamsOptionValue, InfileName = None, ParamsDefaultInfo = None):
    """Process parameters for Psi4 runs and return a map containing processed
    parameter names and values.
    
    ParamsOptionValue a comma delimited list of parameter name and value pairs
    for configuring Psi4 jobs.
    
    The supported parameter names along with their default and possible
    values are shown below:
    
    MemoryInGB,1,NumThreads,1,OutputFile,auto,ScratchDir,auto,
    RemoveOutputFile,yes
            
    Possible  values: OutputFile - stdout, quiet, or FileName; OutputFile -
    DirName; RemoveOutputFile - yes, no, true, or false
    
    These parameters control the runtime behavior of Psi4.
    
    The default for 'OutputFile' is a file name <InFileRoot>_Psi4.out. The PID
    is appened the output file name during multiprocessing. The 'stdout' value
    for 'OutputType' sends Psi4 output to stdout. The 'quiet' or 'devnull' value
    suppresses all Psi4 output.
    
    The default 'Yes' value of 'RemoveOutputFile' option forces the removal
    of any existing Psi4 before creating new files to append output from
    multiple Psi4 runs.
    
    The option 'ScratchDir' is a directory path to the location of scratch
    files. The default value corresponds to Psi4 default. It may be used to
    override the deafult path.

    Arguments:
        ParamsOptionName (str): Command line Psi4 run parameters option name.
        ParamsOptionValues (str): Comma delimited list of parameter name and value pairs.
        InfileName (str): Name of input file.
        ParamsDefaultInfo (dict): Default values to override for selected parameters.

    Returns:
        dictionary: Processed parameter name and value pairs.

    Notes:
        The parameter name and values specified in ParamsOptionValues are validated before
        returning them in a dictionary.

    """

    ParamsInfo = {"MemoryInGB": 1, "NumThreads": 1, "OutputFile": "auto",  "ScratchDir" : "auto", "RemoveOutputFile": True}
    
    # Setup a canonical paramater names...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(ParamsInfo):
        ValidParamNames.append(ParamName)
        CanonicalParamNamesMap[ParamName.lower()] = ParamName
    
    # Update default values...
    if ParamsDefaultInfo is not None:
        for ParamName in ParamsDefaultInfo:
            if ParamName not in ParamsInfo:
                MiscUtil.PrintError("The default parameter name, %s, specified using \"%s\" to function ProcessPsi4RunParameters is not a valid name. Supported parameter names: %s" % (ParamName, ParamsDefaultInfo, " ".join(ValidParamNames)))
            ParamsInfo[ParamName] = ParamsDefaultInfo[ParamName]
    
    if re.match("^auto$", ParamsOptionValue, re.I):
        # No specific parameters to process except for parameters with possible auto value...
        _ProcessPsi4RunAutoParameters(ParamsInfo, ParamsOptionName, ParamsOptionValue, InfileName)
        return ParamsInfo
    
    ParamsOptionValue = ParamsOptionValue.strip()
    if not ParamsOptionValue:
        PrintError("No valid parameter name and value pairs specified using \"%s\" option" % ParamsOptionName)
    
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
        
        if re.match("^MemoryInGB$", ParamName, re.I):
            Value = float(Value)
            if Value <= 0:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: > 0" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^NumThreads$", ParamName, re.I):
            Value = int(Value)
            if Value <= 0:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: > 0" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^ScratchDir$", ParamName, re.I):
            if not re.match("^auto$", Value, re.I):
                if not os.path.isdir(Value):
                    MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. It must be a directory name." % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^RemoveOutputFile$", ParamName, re.I):
            if re.match("^(yes|true)$", Value, re.I):
                Value = True
            elif re.match("^(no|false)$", Value, re.I):
                Value = False
            else:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: yes, no, true, or false" % (Value, Name, ParamsOptionName))
            ParamValue = Value
            
        # Set value...
        ParamsInfo[ParamName] = ParamValue
    
    # Handle paramaters with possible auto values...
    _ProcessPsi4RunAutoParameters(ParamsInfo, ParamsOptionName, ParamsOptionValue, InfileName)

    return ParamsInfo

def _ProcessPsi4RunAutoParameters(ParamsInfo, ParamsOptionName, ParamsOptionValue, InfileName):
    """Process parameters with possible auto values.
    """
    
    Value = ParamsInfo["OutputFile"]
    ParamValue = Value
    if re.match("^auto$", Value, re.I):
        if InfileName is not None:
            # Use InfileName to setup output file. The OutputFile name is automatically updated using
            # PID during multiprocessing...
            InfileDir, InfileRoot, InfileExt = MiscUtil.ParseFileName(InfileName)
            OutputFile = "%s_Psi4.out" % (InfileRoot)
        else:
            OutputFile = "Psi4.out"
    elif re.match("^(devnull|quiet)$", Value, re.I):
        OutputFile = "quiet"
    else:
        # It'll be treated as a filename and processed later...
        OutputFile = Value
    
    ParamsInfo["OutputFile"] = OutputFile

    # OutputFileSpecified is used to track the specified value of the paramater.
    # It may be used by the calling function to dynamically override the value of
    # OutputFile to suprress the Psi4 output based on the initial value.
    ParamsInfo["OutputFileSpecified"] = ParamValue
    
def ProcessPsi4CubeFilesParameters(ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo = None):
    """Process parameters for Psi4 runs and return a map containing processed
    parameter names and values.
    
    ParamsOptionValue is a comma delimited list of parameter name and value pairs
    for generating cube files.
    
    The supported parameter names along with their default and possible
    values are shown below:
    
    GridSpacing, 0.2, GridOverage, 4.0, IsoContourThreshold, 0.85
    
    GridSpacing: Units: Bohr. A higher value reduces the size of the cube files
    on the disk. This option corresponds to Psi4 option CUBIC_GRID_SPACING.
    
    GridOverage: Units: Bohr.This option corresponds to Psi4 option
    CUBIC_GRID_OVERAGE.
        
     IsoContourThreshold captures specified percent of the probability density
     using the least amount of grid points. This option corresponds to Psi4 option
     CUBEPROP_ISOCONTOUR_THRESHOLD.

    Arguments:
        ParamsOptionName (str): Command line Psi4 cube files option name.
        ParamsOptionValues (str): Comma delimited list of parameter name and value pairs.
        ParamsDefaultInfo (dict): Default values to override for selected parameters.

    Returns:
        dictionary: Processed parameter name and value pairs.

    """

    ParamsInfo = {"GridSpacing": 0.2, "GridOverage":  4.0, "IsoContourThreshold": 0.85}
    
    # Setup a canonical paramater names...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(ParamsInfo):
        ValidParamNames.append(ParamName)
        CanonicalParamNamesMap[ParamName.lower()] = ParamName
    
    # Update default values...
    if ParamsDefaultInfo is not None:
        for ParamName in ParamsDefaultInfo:
            if ParamName not in ParamsInfo:
                MiscUtil.PrintError("The default parameter name, %s, specified using \"%s\" to function ProcessPsi4CubeFilesParameters not a valid name. Supported parameter names: %s" % (ParamName, ParamsDefaultInfo, " ".join(ValidParamNames)))
            ParamsInfo[ParamName] = ParamsDefaultInfo[ParamName]
    
    if re.match("^auto$", ParamsOptionValue, re.I):
        # No specific parameters to process except for parameters with possible auto value...
        _ProcessPsi4CubeFilesAutoParameters(ParamsInfo, ParamsOptionName, ParamsOptionValue)
        return ParamsInfo
    
    ParamsOptionValue = ParamsOptionValue.strip()
    if not ParamsOptionValue:
        PrintError("No valid parameter name and value pairs specified using \"%s\" option" % ParamsOptionName)
    
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
        
        if re.match("^(GridSpacing|GridOverage)$", ParamName, re.I):
            if not MiscUtil.IsFloat(Value):
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option must be a float." % (Value, Name, ParamsOptionName))
            Value = float(Value)
            if Value <= 0:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: > 0" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^IsoContourThreshold$", ParamName, re.I):
            if not MiscUtil.IsFloat(Value):
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option must be a float." % (Value, Name, ParamsOptionName))
            Value = float(Value)
            if Value <= 0 or Value > 1:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: >= 0 and <= 1" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        
        # Set value...
        ParamsInfo[ParamName] = ParamValue
    
    # Handle paramaters with possible auto values...
    _ProcessPsi4CubeFilesAutoParameters(ParamsInfo, ParamsOptionName, ParamsOptionValue)

    return ParamsInfo

def _ProcessPsi4CubeFilesAutoParameters(ParamsInfo, ParamsOptionName, ParamsOptionValue):
    """Process parameters with possible auto values.
    """
    
    # No auto parameters to process
    return

def RetrieveIsocontourRangeFromCubeFile(CubeFileName):
    """Retrieve isocontour range values from the cube file. The range
    values are retrieved from the second line in the cube file after
    the string 'Isocontour range'.
    
    Arguments:
        CubeFileName (str): Cube file name.

    Returns:
        float: Minimum range value.
        float: Maximum range value.

    """

    IsocontourRangeMin, IsocontourRangeMax = [None] * 2
    
    CubeFH = open(CubeFileName, "r")
    if CubeFH is None:
        MiscUtil.PrintError("Couldn't open cube file: %s.\n" % (CubeFileName))

    # Look for isocontour range in the first 2 comments line...
    RangeLine = None
    LineCount = 0
    for Line in CubeFH:
        LineCount += 1
        Line = Line.rstrip()
        if re.search("Isocontour range", Line, re.I):
            RangeLine = Line
            break
        
        if LineCount >= 2:
            break
    CubeFH.close()

    if RangeLine is None:
        return (IsocontourRangeMin, IsocontourRangeMax)
    
    LineWords = RangeLine.split(":")
    
    ContourRangeWord = LineWords[-1]
    ContourRangeWord = re.sub("(\(|\)| )", "", ContourRangeWord)

    ContourLevel1, ContourLevel2 = ContourRangeWord.split(",")
    ContourLevel1 = float(ContourLevel1)
    ContourLevel2 = float(ContourLevel2)
    
    if ContourLevel1 < ContourLevel2:
        IsocontourRangeMin = ContourLevel1
        IsocontourRangeMax = ContourLevel2
    else:
        IsocontourRangeMin = ContourLevel2
        IsocontourRangeMax = ContourLevel1
        
    return (IsocontourRangeMin, IsocontourRangeMax)
    
def RetrieveMinAndMaxValueFromCubeFile(CubeFileName):
    """Retrieve minimum and maxmimum grid values from the cube file.
    
    Arguments:
        CubeFileName (str): Cube file name.

    Returns:
        float: Minimum value.
        float: Maximum value.

    """

    MinValue, MaxValue = [sys.float_info.max, sys.float_info.min]
    
    CubeFH = open(CubeFileName, "r")
    if CubeFH is None:
        MiscUtil.PrintError("Couldn't open cube file: %s.\n" % (CubeFileName))

    # Ignore first two comments lines:
    #
    # The first two lines of the header are comments, they are generally ignored by parsing packages or used as two default labels.
    #
    # Ignore lines upto the last section of the header lines:
    #
    # The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    # The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector.
    # The last section in the header is one line for each atom consisting of 5 numbers, the first is the atom number, the second
    # is the charge, and the last three are the x,y,z coordinates of the atom center.
    #
    Line = CubeFH.readline()
    Line = CubeFH.readline()
    Line = CubeFH.readline()
    CubeFH.close()
    
    Line = Line.strip()
    LineWords = Line.split()
    NumOfAtoms = int(LineWords[0])

    HeaderLinesCount = 6 + NumOfAtoms

    # Ignore header lines...
    CubeFH = open(CubeFileName, "r")
    LineCount = 0
    for Line in CubeFH:
        LineCount += 1
        if LineCount >= HeaderLinesCount:
            break
    
    # Process values....
    for Line in CubeFH:
        Line = Line.strip()
        for Value in Line.split():
            Value = float(Value)
            
            if Value < MinValue:
                MinValue = Value
            if Value > MaxValue:
                MaxValue = Value
    
    return (MinValue, MaxValue)

def UpdatePsi4OutputFileUsingPID(OutputFile, PID = None):
    """Append PID to output file name. The PID is automatically retrieved
    during None value of PID.
    
    Arguments:
        OutputFile (str): Output file name.
        PID (int): Process ID or None.

    Returns:
        str: Update output file name. Format: <OutFieRoot>_<PID>.<OutFileExt>

    """
    
    if re.match("stdout|devnull|quiet", OutputFile, re.I):
        return OutputFile

    if PID is None:
        PID = os.getpid()
    
    FileDir, FileRoot, FileExt = MiscUtil.ParseFileName(OutputFile)
    OutputFile = "%s_PID%s.%s" % (FileRoot, PID, FileExt)
    
    return OutputFile
    
def RemoveScratchFiles(psi4, OutputFile, PID = None):
    """Remove any leftover scratch files associated with the specified output
    file. The file specification, <OutfileRoot>.*<PID>.* is used to collect and
    remove files from the scratch directory. In addition, the file
    psi.<PID>.clean, in current directory is removed.
    
    Arguments:
        psi4 (object): psi4 module reference.
        OutputFile (str): Output file name.
        PID (int): Process ID or None.

    Returns:
        None

    """
    
    if re.match("stdout|devnull|quiet", OutputFile, re.I):
        # Scratch files are associated to stdout prefix...
        OutputFile = "stdout"
    
    if PID is None:
        PID = os.getpid()
    
    OutfileDir, OutfileRoot, OutfileExt = MiscUtil.ParseFileName(OutputFile)
    
    ScratchOutfilesSpec = os.path.join(psi4.core.IOManager.shared_object().get_default_path(), "%s.*%s.*" % (OutfileRoot, PID))
    for ScratchFile in glob.glob(ScratchOutfilesSpec):
        os.remove(ScratchFile)

    # Remove any psi.<PID>.clean in the current directory...
    ScratchFile = os.path.join(os.getcwd(), "psi.%s.clean" % (PID))
    if os.path.isfile(ScratchFile):
        os.remove(ScratchFile)
