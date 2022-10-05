#!/usr/bin/env python
#
# File: RDKitFilterTorsionStrainEnergyAlerts.py
# Author: Manish Sud <msud@san.rr.com>
#
# Collaborator: Pat Walters
#
# Copyright (C) 2022 Manish Sud. All rights reserved.
#
# This script uses the torsion strain energy library developed by Gu, S.;
# Smith, M. S.; Yang, Y.; Irwin, J. J.; Shoichet, B. K. [ Ref 153 ].
#
# The torsion strain enegy library is based on the Torsion Library jointly
# developed by the University of Hamburg, Center for Bioinformatics,
# Hamburg, Germany and F. Hoffmann-La-Roche Ltd., Basel, Switzerland.
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
import glob
import multiprocessing as mp
import math

# RDKit imports...
try:
    from rdkit import rdBase
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import RDKit module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your RDKit environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
try:
    from docopt import docopt
    import MiscUtil
    import RDKitUtil
    import TorsionLibraryUtil
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import MayaChemTools module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your MayaChemTools environment and try again.\n\n")
    sys.exit(1)

ScriptName = os.path.basename(sys.argv[0])
Options = {}
OptionsInfo = {}

TorsionLibraryInfo = {}

def main():
    """Start execution of the script."""
    
    MiscUtil.PrintInfo("\n%s (RDKit v%s; MayaChemTools v%s; %s): Starting...\n" % (ScriptName, rdBase.rdkitVersion, MiscUtil.GetMayaChemToolsVersion(), time.asctime()))
    
    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()
    
    # Retrieve command line arguments and options...
    RetrieveOptions()
    
    if  Options["--list"]:
        # Handle listing of torsion library information...
        ProcessListTorsionLibraryOption()
    else:
        # Process and validate command line arguments and options...
        ProcessOptions()
        
        # Perform actions required by the script...
        PerformFiltering()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def PerformFiltering():
    """Filter molecules using SMARTS torsion rules in the torsion strain energy
    library file."""

    # Process torsion library info...
    ProcessTorsionLibraryInfo()
    
    # Set up a pattern molecule for rotatable bonds...
    RotBondsPatternMol = Chem.MolFromSmarts(OptionsInfo["RotBondsSMARTSPattern"])
    
    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols  = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])
    
    MolCount, ValidMolCount, RemainingMolCount, WriteFailedCount = ProcessMolecules(Mols, RotBondsPatternMol)

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during writing: %d" % WriteFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + WriteFailedCount))

    MiscUtil.PrintInfo("\nNumber of remaining molecules: %d" % RemainingMolCount)
    MiscUtil.PrintInfo("Number of filtered molecules: %d" % (ValidMolCount - RemainingMolCount))

def ProcessMolecules(Mols, RotBondsPatternMol):
    """Process and filter molecules."""
    
    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, RotBondsPatternMol)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, RotBondsPatternMol)

def ProcessMoleculesUsingSingleProcess(Mols, RotBondsPatternMol):
    """Process and filter molecules using a single process."""
    
    SetupTorsionLibraryInfo()
    
    MiscUtil.PrintInfo("\nFiltering molecules...")
    
    OutfileFilteredMode = OptionsInfo["OutfileFilteredMode"]

    # Set up writers...
    OutfilesWriters = SetupOutfilesWriters()
    
    WriterRemaining = OutfilesWriters["WriterRemaining"]
    WriterFiltered = OutfilesWriters["WriterFiltered"]
    WriterAlertSummary = OutfilesWriters["WriterAlertSummary"]
    
    # Initialize alerts summary info...
    TorsionAlertsSummaryInfo = InitializeTorsionAlertsSummaryInfo()

    (MolCount, ValidMolCount, RemainingMolCount, WriteFailedCount, FilteredMolWriteCount) = [0] * 5
    for Mol in Mols:
        MolCount += 1
        
        if Mol is None:
            continue
        
        if RDKitUtil.IsMolEmpty(Mol):
            MiscUtil.PrintWarning("Ignoring empty molecule: %s" % RDKitUtil.GetMolName(Mol, MolCount))
            continue

        # Check for 3D flag...
        if not Mol.GetConformer().Is3D():
            MiscUtil.PrintWarning("3D tag is not set. Ignoring molecule: %s\n" % RDKitUtil.GetMolName(Mol, MolCount))
            continue
        
        ValidMolCount += 1
        
        # Identify torsion library alerts for rotatable bonds..
        RotBondsAlertsStatus, RotBondsAlertsInfo = IdentifyTorsionLibraryAlertsForRotatableBonds(Mol, RotBondsPatternMol)

        TrackTorsionAlertsSummaryInfo(TorsionAlertsSummaryInfo, RotBondsAlertsInfo)
        
        # Write out filtered and remaining molecules...
        WriteStatus = True
        if RotBondsAlertsStatus:
            if OutfileFilteredMode:
                WriteStatus = WriteMolecule(WriterFiltered, Mol, RotBondsAlertsInfo)
                if WriteStatus:
                    FilteredMolWriteCount += 1
        else:
            RemainingMolCount += 1
            WriteStatus = WriteMolecule(WriterRemaining, Mol, RotBondsAlertsInfo)
        
        if not WriteStatus:
            WriteFailedCount += 1

    WriteTorsionAlertsSummaryInfo(WriterAlertSummary, TorsionAlertsSummaryInfo)
    CloseOutfilesWriters(OutfilesWriters)

    if FilteredMolWriteCount:
        WriteTorsionAlertsFilteredByRulesInfo(TorsionAlertsSummaryInfo)
    
    return (MolCount, ValidMolCount, RemainingMolCount, WriteFailedCount)

def ProcessMoleculesUsingMultipleProcesses(Mols, RotBondsPatternMol):
    """Process and filter molecules using multiprocessing."""

    MiscUtil.PrintInfo("\nFiltering molecules using multiprocessing...")
    
    MPParams = OptionsInfo["MPParams"]
    OutfileFilteredMode = OptionsInfo["OutfileFilteredMode"]

    # Set up writers...
    OutfilesWriters = SetupOutfilesWriters()
    
    WriterRemaining = OutfilesWriters["WriterRemaining"]
    WriterFiltered = OutfilesWriters["WriterFiltered"]
    WriterAlertSummary = OutfilesWriters["WriterAlertSummary"]
    
    # Initialize alerts summary info...
    TorsionAlertsSummaryInfo = InitializeTorsionAlertsSummaryInfo()

    # Setup data for initializing a worker process...
    MiscUtil.PrintInfo("\nEncoding options info and rotatable bond pattern molecule...")
    OptionsInfo["EncodedRotBondsPatternMol"] = RDKitUtil.MolToBase64EncodedMolString(RotBondsPatternMol)
    InitializeWorkerProcessArgs = (MiscUtil.ObjectToBase64EncodedString(Options), MiscUtil.ObjectToBase64EncodedString(OptionsInfo))

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
    
    (MolCount, ValidMolCount, RemainingMolCount, WriteFailedCount, FilteredMolWriteCount) = [0] * 5
    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, RotBondsAlertsStatus, RotBondsAlertsInfo = Result
        
        if EncodedMol is None:
            continue
        ValidMolCount += 1
        
        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        
        TrackTorsionAlertsSummaryInfo(TorsionAlertsSummaryInfo, RotBondsAlertsInfo)
        
        # Write out filtered and remaining molecules...
        WriteStatus = True
        if RotBondsAlertsStatus:
            if OutfileFilteredMode:
                WriteStatus = WriteMolecule(WriterFiltered, Mol, RotBondsAlertsInfo)
                if WriteStatus:
                    FilteredMolWriteCount += 1
        else:
            RemainingMolCount += 1
            WriteStatus = WriteMolecule(WriterRemaining, Mol, RotBondsAlertsInfo)

        if not WriteStatus:
            WriteFailedCount += 1
    
    WriteTorsionAlertsSummaryInfo(WriterAlertSummary, TorsionAlertsSummaryInfo)
    CloseOutfilesWriters(OutfilesWriters)

    if FilteredMolWriteCount:
        WriteTorsionAlertsFilteredByRulesInfo(TorsionAlertsSummaryInfo)
    
    return (MolCount, ValidMolCount, RemainingMolCount, WriteFailedCount)

def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""
    
    global Options, OptionsInfo
    
    MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())
    
    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Decode RotBondsPatternMol...
    OptionsInfo["RotBondsPatternMol"] = RDKitUtil.MolFromBase64EncodedMolString(OptionsInfo["EncodedRotBondsPatternMol"])
    
    # Setup torsion library...
    RetrieveTorsionLibraryInfo(Quiet = True)
    SetupTorsionLibraryInfo(Quiet = True)

def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""

    MolIndex, EncodedMol = EncodedMolInfo
    
    if EncodedMol is None:
        return [MolIndex, None, False, None]
    
    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    if RDKitUtil.IsMolEmpty(Mol):
        MolName = RDKitUtil.GetMolName(Mol, (MolIndex + 1))
        MiscUtil.PrintWarning("Ignoring empty molecule: %s" % MolName)
        return [MolIndex, None, False, None]
        
    # Check for 3D flag...
    if not Mol.GetConformer().Is3D():
        MolName = RDKitUtil.GetMolName(Mol, (MolIndex + 1))
        MiscUtil.PrintWarning("3D tag is not set. Ignoring molecule: %s\n" % MolName)
        return [MolIndex, None, False, None]
    
    # Identify torsion library alerts for rotatable bonds..
    RotBondsAlertsStatus, RotBondsAlertsInfo = IdentifyTorsionLibraryAlertsForRotatableBonds(Mol, OptionsInfo["RotBondsPatternMol"])

    return [MolIndex, EncodedMol, RotBondsAlertsStatus, RotBondsAlertsInfo]
    
def  IdentifyTorsionLibraryAlertsForRotatableBonds(Mol, RotBondsPatternMol):
    """Identify rotatable bonds for torsion library alerts."""
    
    # Identify rotatable bonds...
    RotBondsStatus, RotBondsInfo = TorsionLibraryUtil.IdentifyRotatableBondsForTorsionLibraryMatch(TorsionLibraryInfo, Mol, RotBondsPatternMol)
    
    if not RotBondsStatus:
        return (False, None)

    # Identify alerts for rotatable bonds...
    RotBondsAlertsStatus, RotBondsAlertsInfo = MatchRotatableBondsToTorsionLibrary(Mol, RotBondsInfo)
    
    return (RotBondsAlertsStatus, RotBondsAlertsInfo)

def MatchRotatableBondsToTorsionLibrary(Mol, RotBondsInfo):
    """Match rotatable bond to torsion library."""

    # Initialize...
    RotBondsAlertsInfo = InitializeRotatableBondsAlertsInfo()
    
    # Match rotatable bonds to torsion library...
    for ID in RotBondsInfo["IDs"]:
        AtomIndices = RotBondsInfo["AtomIndices"][ID]
        HierarchyClass = RotBondsInfo["HierarchyClass"][ID]

        MatchStatus, MatchInfo = MatchRotatableBondToTorsionLibrary(Mol, AtomIndices, HierarchyClass)
        TrackRotatableBondsAlertsInfo(RotBondsAlertsInfo, ID, AtomIndices, MatchStatus, MatchInfo)
        
    RotBondsAlertsStatus = SetupRotatableBondsAlertStatusTotalStrainEnergiesInfo(RotBondsAlertsInfo)

    return (RotBondsAlertsStatus, RotBondsAlertsInfo)

def InitializeRotatableBondsAlertsInfo():
    """Initialize alerts information for rotatable bonds."""
    
    RotBondsAlertsInfo = {}
    RotBondsAlertsInfo["IDs"] = []

    for DataLabel in ["RotBondsAlertsStatus", "TotalEnergy", "TotalEnergyLowerBound", "TotalEnergyUpperBound", "AnglesNotObservedCount", "MaxSingleEnergy", "MaxSingleEnergyAlertsCount"]:
        RotBondsAlertsInfo[DataLabel] = None
    
    for DataLabel in ["MatchStatus", "MaxSingleEnergyAlertStatus", "AtomIndices", "TorsionAtomIndices", "TorsionAngle", "HierarchyClassName", "HierarchySubClassName", "TorsionRuleNodeID", "TorsionRuleSMARTS", "EnergyMethod", "AngleNotObserved", "Energy", "EnergyLowerBound", "EnergyUpperBound"]:
        RotBondsAlertsInfo[DataLabel] = {}
        
    return RotBondsAlertsInfo

def TrackRotatableBondsAlertsInfo(RotBondsAlertsInfo, ID, AtomIndices, MatchStatus, MatchInfo):
    """Track alerts information for rotatable bonds."""
    
    if MatchInfo is None or len(MatchInfo) == 0:
        TorsionAtomIndices, TorsionAngle, HierarchyClassName, HierarchySubClassName, TorsionRuleNodeID, TorsionRuleSMARTS, EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = [None] * 11
    else:
        TorsionAtomIndices, TorsionAngle, HierarchyClassName, HierarchySubClassName, TorsionRuleNodeID, TorsionRuleSMARTS, EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = MatchInfo
    
    # Track torsion match information...
    RotBondsAlertsInfo["IDs"].append(ID)
    RotBondsAlertsInfo["MatchStatus"][ID] = MatchStatus
    RotBondsAlertsInfo["AtomIndices"][ID] = AtomIndices
    RotBondsAlertsInfo["TorsionAtomIndices"][ID] = TorsionAtomIndices
    RotBondsAlertsInfo["TorsionAngle"][ID] = TorsionAngle
    RotBondsAlertsInfo["HierarchyClassName"][ID] = HierarchyClassName
    RotBondsAlertsInfo["HierarchySubClassName"][ID] = HierarchySubClassName
    RotBondsAlertsInfo["TorsionRuleNodeID"][ID] = TorsionRuleNodeID
    RotBondsAlertsInfo["TorsionRuleSMARTS"][ID] = TorsionRuleSMARTS
    RotBondsAlertsInfo["EnergyMethod"][ID] = EnergyMethod
    RotBondsAlertsInfo["AngleNotObserved"][ID] = AngleNotObserved
    RotBondsAlertsInfo["Energy"][ID] = Energy
    RotBondsAlertsInfo["EnergyLowerBound"][ID] = EnergyLowerBound
    RotBondsAlertsInfo["EnergyUpperBound"][ID] = EnergyUpperBound

def SetupRotatableBondsAlertStatusTotalStrainEnergiesInfo(RotBondsAlertsInfo):
    """Setup rotatable bonds alert status along with total strain energies."""

    # Initialize...
    RotBondsAlertsStatus = False
    TotalEnergy, TotalEnergyLowerBound, TotalEnergyUpperBound, AnglesNotObservedCount  = [None, None, None, None]
    MaxSingleEnergy, MaxSingleEnergyAlertsCount  = [None, None]

    # Initialize max single energy alert status...
    for ID in RotBondsAlertsInfo["IDs"]:
        RotBondsAlertsInfo["MaxSingleEnergyAlertStatus"][ID] = None
    
    # Check for torsion angles not obervered in the strain library...
    AnglesNotObservedCount = 0
    for ID in RotBondsAlertsInfo["IDs"]:
        AngleNotObserved = RotBondsAlertsInfo["AngleNotObserved"][ID]
        if AngleNotObserved is not None and AngleNotObserved:
            AnglesNotObservedCount += 1

    # Setup alert status for rotable bonds...
    if AnglesNotObservedCount > 0:
        if OptionsInfo["FilterTorsionsNotObserved"]:
            RotBondsAlertsStatus = True
    else:
        TotalEnergy = 0.0
        for ID in RotBondsAlertsInfo["IDs"]:
            Energy = RotBondsAlertsInfo["Energy"][ID]
            TotalEnergy += Energy
            if OptionsInfo["TotalEnergyMode"]:
                if TotalEnergy > OptionsInfo["TotalEnergyCutoff"]:
                    RotBondsAlertsStatus = True
                    break
            elif OptionsInfo["MaxSingleEnergyMode"]:
                if Energy > OptionsInfo["MaxSingleEnergyCutoff"]:
                    RotBondsAlertsStatus = True
                    break
            elif OptionsInfo["TotalOrMaxSingleEnergyMode"]:
                if TotalEnergy > OptionsInfo["TotalEnergyCutoff"] or Energy > OptionsInfo["MaxSingleEnergyCutoff"]:
                    RotBondsAlertsStatus = True
                    break
    
        # Setup energy infomation...
        TotalEnergy, TotalEnergyLowerBound, TotalEnergyUpperBound = [0.0, 0.0, 0.0]
        if OptionsInfo["MaxSingleEnergyMode"] or OptionsInfo["TotalOrMaxSingleEnergyMode"]:
            MaxSingleEnergy, MaxSingleEnergyAlertsCount = [0.0, 0]
        
        for ID in RotBondsAlertsInfo["IDs"]:
            Energy = RotBondsAlertsInfo["Energy"][ID]
            
            # Setup total energy along with the lower and upper bounds...
            TotalEnergy += Energy
            TotalEnergyLowerBound += RotBondsAlertsInfo["EnergyLowerBound"][ID]
            TotalEnergyUpperBound += RotBondsAlertsInfo["EnergyUpperBound"][ID]
        
            # Setup max single energy and max single energy alerts count...
            if OptionsInfo["MaxSingleEnergyMode"] or OptionsInfo["TotalOrMaxSingleEnergyMode"]:
                MaxSingleEnergyAlertStatus = False
                
                if Energy > MaxSingleEnergy:
                    MaxSingleEnergy = Energy
                    if Energy > OptionsInfo["MaxSingleEnergyCutoff"]:
                        MaxSingleEnergyAlertStatus = True
                        MaxSingleEnergyAlertsCount += 1
                
                RotBondsAlertsInfo["MaxSingleEnergyAlertStatus"][ID] = MaxSingleEnergyAlertStatus
    
    RotBondsAlertsInfo["RotBondsAlertsStatus"] = RotBondsAlertsStatus
    
    RotBondsAlertsInfo["TotalEnergy"] = TotalEnergy
    RotBondsAlertsInfo["TotalEnergyLowerBound"] = TotalEnergyLowerBound
    RotBondsAlertsInfo["TotalEnergyUpperBound"] = TotalEnergyUpperBound
    
    RotBondsAlertsInfo["AnglesNotObservedCount"] = AnglesNotObservedCount
    
    RotBondsAlertsInfo["MaxSingleEnergy"] = MaxSingleEnergy
    RotBondsAlertsInfo["MaxSingleEnergyAlertsCount"] = MaxSingleEnergyAlertsCount

    return RotBondsAlertsStatus
    
def MatchRotatableBondToTorsionLibrary(Mol, RotBondAtomIndices, RotBondHierarchyClass):
    """Match rotatable bond to torsion library."""

    if TorsionLibraryUtil.IsSpecificHierarchyClass(TorsionLibraryInfo, RotBondHierarchyClass):
        MatchStatus, MatchInfo = MatchRotatableBondAgainstSpecificHierarchyClass(Mol, RotBondAtomIndices, RotBondHierarchyClass)
        if not MatchStatus:
            MatchStatus, MatchInfo = MatchRotatableBondAgainstGenericHierarchyClass(Mol, RotBondAtomIndices, RotBondHierarchyClass)
    else:
        MatchStatus, MatchInfo = MatchRotatableBondAgainstGenericHierarchyClass(Mol, RotBondAtomIndices, RotBondHierarchyClass)

    return (MatchStatus, MatchInfo)

def MatchRotatableBondAgainstSpecificHierarchyClass(Mol, RotBondAtomIndices, RotBondHierarchyClass):
    """Match rotatable bond against a specific hierarchy class."""

    HierarchyClassElementNode = None
    if RotBondHierarchyClass in TorsionLibraryInfo["SpecificClasses"]["ElementNode"]:
        HierarchyClassElementNode = TorsionLibraryInfo["SpecificClasses"]["ElementNode"][RotBondHierarchyClass]
    
    if HierarchyClassElementNode is None:
        return (False, None, None, None)

    TorsionLibraryUtil.TrackHierarchyClassElementNode(TorsionLibraryInfo, HierarchyClassElementNode)
    MatchStatus, MatchInfo = ProcessElementForRotatableBondMatch(Mol, RotBondAtomIndices, HierarchyClassElementNode)
    TorsionLibraryUtil.RemoveLastHierarchyClassElementNodeFromTracking(TorsionLibraryInfo)
    
    return (MatchStatus, MatchInfo)

def MatchRotatableBondAgainstGenericHierarchyClass(Mol, RotBondAtomIndices, RotBondHierarchyClass):
    """Match rotatable bond against a generic hierarchy class."""
    
    HierarchyClassElementNode = TorsionLibraryUtil.GetGenericHierarchyClassElementNode(TorsionLibraryInfo)
    if HierarchyClassElementNode is None:
        return (False, None)

    TorsionLibraryUtil.TrackHierarchyClassElementNode(TorsionLibraryInfo, HierarchyClassElementNode)
    
    #  Match hierarchy subclasses before matching torsion rules...
    MatchStatus, MatchInfo = MatchRotatableBondAgainstGenericHierarchySubClasses(Mol, RotBondAtomIndices, HierarchyClassElementNode)
    
    if not MatchStatus:
        MatchStatus, MatchInfo = MatchRotatableBondAgainstGenericHierarchyTorsionRules(Mol, RotBondAtomIndices, HierarchyClassElementNode)
    
    TorsionLibraryUtil.RemoveLastHierarchyClassElementNodeFromTracking(TorsionLibraryInfo)
    
    return (MatchStatus, MatchInfo)

def MatchRotatableBondAgainstGenericHierarchySubClasses(Mol, RotBondAtomIndices, HierarchyClassElementNode):
    """Match rotatable bond againat generic hierarchy subclasses."""

    for ElementChildNode in HierarchyClassElementNode:
        if ElementChildNode.tag != "hierarchySubClass":
            continue
        
        SubClassMatchStatus = ProcessHierarchySubClassElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementChildNode)
        
        if SubClassMatchStatus:
            MatchStatus, MatchInfo = ProcessElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementChildNode)
            
            if MatchStatus:
                return (MatchStatus, MatchInfo)
        
    return(False, None)

def MatchRotatableBondAgainstGenericHierarchyTorsionRules(Mol, RotBondAtomIndices, HierarchyClassElementNode):
    """Match rotatable bond againat torsion rules generic hierarchy class."""
    
    for ElementChildNode in HierarchyClassElementNode:
        if ElementChildNode.tag != "torsionRule":
            continue
        
        MatchStatus, MatchInfo = ProcessTorsionRuleElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementChildNode)
        
        if MatchStatus:
            return (MatchStatus, MatchInfo)

    return(False, None)
    
def ProcessElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementNode):
    """Process element node to recursively match rotatable bond against hierarchy
    subclasses and torsion rules."""
    
    for ElementChildNode in ElementNode:
        if ElementChildNode.tag == "hierarchySubClass":
            SubClassMatchStatus = ProcessHierarchySubClassElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementChildNode)
            
            if SubClassMatchStatus:
                TorsionLibraryUtil.TrackHierarchySubClassElementNode(TorsionLibraryInfo, ElementChildNode)
                
                MatchStatus, MatchInfo = ProcessElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementChildNode)
                if MatchStatus:
                    TorsionLibraryUtil.RemoveLastHierarchySubClassElementNodeFromTracking(TorsionLibraryInfo)
                    return (MatchStatus, MatchInfo)
            
                TorsionLibraryUtil.RemoveLastHierarchySubClassElementNodeFromTracking(TorsionLibraryInfo)
            
        elif ElementChildNode.tag == "torsionRule":
            MatchStatus, MatchInfo = ProcessTorsionRuleElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementChildNode)
            
            if MatchStatus:
                return (MatchStatus, MatchInfo)
    
    return (False, None)

def ProcessHierarchySubClassElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementNode):
    """Process hierarchy subclass element to match rotatable bond."""
    
    # Setup subclass SMARTS pattern mol...
    SubClassPatternMol = TorsionLibraryUtil.SetupHierarchySubClassElementPatternMol(TorsionLibraryInfo, ElementNode)
    if SubClassPatternMol is None:
        return False

    # Match SMARTS pattern...
    SubClassPatternMatches = RDKitUtil.FilterSubstructureMatchesByAtomMapNumbers(Mol, SubClassPatternMol, Mol.GetSubstructMatches(SubClassPatternMol, useChirality = False))
    if len(SubClassPatternMatches) == 0:
        return False

    # Match rotatable bond indices...
    RotBondAtomIndex1, RotBondAtomIndex2 = RotBondAtomIndices
    MatchStatus = False
    for SubClassPatternMatch in SubClassPatternMatches:
        if len(SubClassPatternMatch) == 2:
            # Matched to pattern containing map atom numbers ":2" and ":3"...
            CentralAtomsIndex1, CentralAtomsIndex2 = SubClassPatternMatch
        elif len(SubClassPatternMatch) == 4:
            # Matched to pattern containing map atom numbers ":1", ":2", ":3" and ":4"...
            CentralAtomsIndex1 = SubClassPatternMatch[1]
            CentralAtomsIndex2 = SubClassPatternMatch[2]
        elif len(SubClassPatternMatch) == 3:
            SubClassSMARTSPattern = ElementNode.get("smarts")
            if TorsionLibraryUtil.DoesSMARTSContainsMappedAtoms(SubClassSMARTSPattern, [":2", ":3", ":4"]):
                # Matched to pattern containing map atom numbers ":2", ":3" and ":4"...
                CentralAtomsIndex1 = SubClassPatternMatch[0]
                CentralAtomsIndex2 = SubClassPatternMatch[1]
            else:
                # Matched to pattern containing map atom numbers ":1", ":2" and ":3"...
                CentralAtomsIndex1 = SubClassPatternMatch[1]
                CentralAtomsIndex2 = SubClassPatternMatch[2]
        else:
            continue

        if CentralAtomsIndex1 != CentralAtomsIndex2:
            if ((CentralAtomsIndex1 == RotBondAtomIndex1 and CentralAtomsIndex2 == RotBondAtomIndex2) or (CentralAtomsIndex1 == RotBondAtomIndex2 and CentralAtomsIndex2 == RotBondAtomIndex1)):
                MatchStatus = True
                break
    
    return (MatchStatus)

def ProcessTorsionRuleElementForRotatableBondMatch(Mol, RotBondAtomIndices, ElementNode):
    """Process torsion rule element to match rotatable bond."""

    #  Retrieve torsions matched to rotatable bond...
    TorsionAtomIndicesList, TorsionAnglesList = MatchTorsionRuleToRotatableBond(Mol, RotBondAtomIndices, ElementNode)
    if TorsionAtomIndicesList is None:
        return (False, None)
    
    # Setup torsion angles and enery bin information for matched torsion rule...
    TorsionRuleAnglesInfo = TorsionLibraryUtil.SetupTorsionRuleAnglesInfo(TorsionLibraryInfo, ElementNode)
    if TorsionRuleAnglesInfo is None:
        return (False, None)

    # Setup highest strain energy for matched torsions...
    TorsionAtomIndices, TorsionAngle, EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = SelectHighestStrainEnergyTorsionForRotatableBond(TorsionRuleAnglesInfo, TorsionAtomIndicesList, TorsionAnglesList)
    
    # Setup hierarchy class and subclass names...
    HierarchyClassName, HierarchySubClassName = TorsionLibraryUtil.SetupHierarchyClassAndSubClassNamesForRotatableBond(TorsionLibraryInfo)
    
    # Setup rule node ID...
    TorsionRuleNodeID = ElementNode.get("NodeID")
    
    # Setup SMARTS...
    TorsionRuleSMARTS = ElementNode.get("smarts")
    if " " in TorsionRuleSMARTS:
        TorsionRuleSMARTS = TorsionRuleSMARTS.replace(" ", "")

    # Setup match info...
    MatchInfo = [TorsionAtomIndices, TorsionAngle, HierarchyClassName, HierarchySubClassName, TorsionRuleNodeID, TorsionRuleSMARTS, EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound]
    
    # Setup match status...
    MatchStatus = True
    
    return (MatchStatus, MatchInfo)

def SelectHighestStrainEnergyTorsionForRotatableBond(TorsionRuleAnglesInfo, TorsionAtomIndicesList, TorsionAnglesList):
    """Select highest strain energy torsion matched to a rotatable bond."""
    
    TorsionAtomIndices, TorsionAngle, EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = [None] * 7
    ValidEnergyValue, ValidCurrentEnergyValue = [False] * 2
    
    FirstTorsion = True
    for Index in range(0, len(TorsionAtomIndicesList)):
        CurrentTorsionAtomIndices = TorsionAtomIndicesList[Index]
        CurrentTorsionAngle = TorsionAnglesList[Index]

        if FirstTorsion:
            FirstTorsion = False
            EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = SetupStrainEnergyForRotatableBond(TorsionRuleAnglesInfo, CurrentTorsionAngle)
            TorsionAtomIndices = CurrentTorsionAtomIndices
            TorsionAngle = CurrentTorsionAngle
            ValidEnergyValue = IsEnergyValueValid(Energy)
            continue

        # Select highest strain energy...
        CurrentEnergyMethod, CurrentAngleNotObserved, CurrentEnergy, CurrentEnergyLowerBound, CurrentEnergyUpperBound = SetupStrainEnergyForRotatableBond(TorsionRuleAnglesInfo, CurrentTorsionAngle)
        ValidCurrentEnergyValue = IsEnergyValueValid(CurrentEnergy)
        
        UpdateValues = False
        if ValidEnergyValue and ValidCurrentEnergyValue:
            if CurrentEnergy > Energy:
                UpdateValues = True
        elif ValidCurrentEnergyValue:
            if not ValidEnergyValue:
                UpdateValues = True
        
        if UpdateValues:
            EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = [CurrentEnergyMethod, CurrentAngleNotObserved, CurrentEnergy, CurrentEnergyLowerBound, CurrentEnergyUpperBound]
            TorsionAtomIndices = CurrentTorsionAtomIndices
            TorsionAngle = CurrentTorsionAngle
    
    return (TorsionAtomIndices, TorsionAngle, EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound)

def IsEnergyValueValid(Value):
    """Check for valid energy value."""

    return False if (Value is None or math.isnan(Value) or math.isinf(Value)) else True

def SetupStrainEnergyForRotatableBond(TorsionRuleAnglesInfo, TorsionAngle):
    """Setup strain energy for rotatable bond."""

    if TorsionRuleAnglesInfo["EnergyMethodExact"]:
        return (SetupStrainEnergyForRotatableBondByExactMethod(TorsionRuleAnglesInfo, TorsionAngle))
    elif TorsionRuleAnglesInfo["EnergyMethodApproximate"]:
        return (SetupStrainEnergyForRotatableBondByApproximateMethod(TorsionRuleAnglesInfo, TorsionAngle))
    else:
        EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = [None, None, None, None, None]
        return (EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound)

def SetupStrainEnergyForRotatableBondByExactMethod(TorsionRuleAnglesInfo, TorsionAngle):
    """Setup strain energy for rotatable bond by exact method."""
    
    EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = ["Exact", None, None, None, None]

    # Map angle to energy bin numbers...
    BinNum = math.ceil(TorsionAngle / 10) + 17
    PreviousBinNum = (BinNum + 35) % 36
    
    # Bin angle from -170 to 180 by the right end points...
    BinAngleRightSide = (BinNum - 17) * 10
    
    # Angle offset towards the left of the bin from the right end point...
    AngleOffset = TorsionAngle - BinAngleRightSide
    
    BinEnergy = TorsionRuleAnglesInfo["HistogramEnergy"][BinNum]
    PreviousBinEnergy = TorsionRuleAnglesInfo["HistogramEnergy"][PreviousBinNum]
    Energy =  BinEnergy + (BinEnergy - PreviousBinEnergy)/10.0 * AngleOffset
    
    BinEnergyLowerBound = TorsionRuleAnglesInfo["HistogramEnergyLowerBound"][BinNum]
    PreviousBinEnergyLowerBound = TorsionRuleAnglesInfo["HistogramEnergyLowerBound"][PreviousBinNum]
    EnergyLowerBound =  BinEnergyLowerBound + (BinEnergyLowerBound - PreviousBinEnergyLowerBound)/10.0 * AngleOffset
    
    BinEnergyUpperBound = TorsionRuleAnglesInfo["HistogramEnergyUpperBound"][BinNum]
    PreviousBinEnergyUpperBound = TorsionRuleAnglesInfo["HistogramEnergyUpperBound"][PreviousBinNum]
    EnergyUpperBound =  BinEnergyUpperBound + (BinEnergyUpperBound - PreviousBinEnergyUpperBound)/10.0 * AngleOffset
    
    return (EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound)

def SetupStrainEnergyForRotatableBondByApproximateMethod(TorsionRuleAnglesInfo, TorsionAngle):
    """Setup strain energy for rotatable bond by approximate method."""

    EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound = ["Approximate", True, None, None, None]
    
    for AngleID in TorsionRuleAnglesInfo["IDs"]:
        Tolerance2 = TorsionRuleAnglesInfo["Tolerance2"][AngleID]
        Theta0 = TorsionRuleAnglesInfo["Theta0"][AngleID]
        
        AngleDiff = TorsionLibraryUtil.CalculateTorsionAngleDifference(TorsionAngle, Theta0)
        if abs(AngleDiff) <= Tolerance2:
            Beta1 = TorsionRuleAnglesInfo["Beta1"][AngleID]
            Beta2 = TorsionRuleAnglesInfo["Beta2"][AngleID]
            
            Energy = Beta1*(AngleDiff ** 2) + Beta2*(AngleDiff** 4)

            # Estimates of lower and upper bound are not available for
            # approximate method...
            EnergyLowerBound = Energy
            EnergyUpperBound = Energy
            
            AngleNotObserved = False
            
            break
    
    return (EnergyMethod, AngleNotObserved, Energy, EnergyLowerBound, EnergyUpperBound)

def MatchTorsionRuleToRotatableBond(Mol, RotBondAtomIndices, ElementNode):
    """Retrieve matched torsions for torsion rule matched to rotatable bond."""

    # Get torsion matches...
    TorsionMatches = GetMatchesForTorsionRule(Mol, ElementNode)
    if TorsionMatches is None or len(TorsionMatches) == 0:
        return (None, None)

    # Identify all torsion matches corresponding to central atoms in RotBondAtomIndices...
    RotBondAtomIndex1, RotBondAtomIndex2 = RotBondAtomIndices
    RotBondTorsionMatches, RotBondTorsionAngles = [None] * 2
    
    for TorsionMatch in TorsionMatches:
        CentralAtomIndex1 = TorsionMatch[1]
        CentralAtomIndex2 = TorsionMatch[2]
        
        if ((CentralAtomIndex1 == RotBondAtomIndex1 and CentralAtomIndex2 == RotBondAtomIndex2) or (CentralAtomIndex1 == RotBondAtomIndex2 and CentralAtomIndex2 == RotBondAtomIndex1)):
            TorsionAngle = CalculateTorsionAngle(Mol, TorsionMatch)
            if RotBondTorsionMatches is None:
                RotBondTorsionMatches = []
                RotBondTorsionAngles = []
            RotBondTorsionMatches.append(TorsionMatch)
            RotBondTorsionAngles.append(TorsionAngle)
            
    return (RotBondTorsionMatches, RotBondTorsionAngles)

def CalculateTorsionAngle(Mol, TorsionMatch):
    """Calculate torsion angle."""

    # Calculate torsion angle using torsion atom indices..
    MolConf = Mol.GetConformer(0)
    TorsionAngle = rdMolTransforms.GetDihedralDeg(MolConf, TorsionMatch[0], TorsionMatch[1], TorsionMatch[2], TorsionMatch[3])
    TorsionAngle = round(TorsionAngle, 2)
    
    return TorsionAngle

def GetMatchesForTorsionRule(Mol, ElementNode):
    """Get matches for torsion rule."""

    # Match torsions...
    TorsionMatches = GetSubstructureMatchesForTorsionRule(Mol, ElementNode)
    
    if TorsionMatches is None or len(TorsionMatches) == 0:
        return TorsionMatches

    # Filter torsion matches...
    FiltertedTorsionMatches = []
    for TorsionMatch in TorsionMatches:
        if len(TorsionMatch) != 4:
            continue

        # Ignore matches containing hydrogen atoms as first or last atom...
        if Mol.GetAtomWithIdx(TorsionMatch[0]).GetAtomicNum() == 1:
            continue
        if Mol.GetAtomWithIdx(TorsionMatch[3]).GetAtomicNum() == 1:
            continue
        
        FiltertedTorsionMatches.append(TorsionMatch)
    
    return FiltertedTorsionMatches

def GetSubstructureMatchesForTorsionRule(Mol, ElementNode):
    """Get substructure matches for a torsion rule."""
    
    # Setup torsion rule SMARTS pattern mol....
    TorsionRuleNodeID = ElementNode.get("NodeID")
    TorsionSMARTSPattern = ElementNode.get("smarts")
    TorsionPatternMol = TorsionLibraryUtil.SetupTorsionRuleElementPatternMol(TorsionLibraryInfo, ElementNode, TorsionRuleNodeID, TorsionSMARTSPattern)
    if TorsionPatternMol is None:
        return None
    
    # Match torsions...
    TorsionMatches = RDKitUtil.FilterSubstructureMatchesByAtomMapNumbers(Mol, TorsionPatternMol, Mol.GetSubstructMatches(TorsionPatternMol, useChirality = False))
    
    return TorsionMatches

def InitializeTorsionAlertsSummaryInfo():
    """Initialize torsion alerts summary."""

    if OptionsInfo["CountMode"]:
        return None
    
    if not OptionsInfo["TrackAlertsSummaryInfo"]:
        return None
    
    TorsionAlertsSummaryInfo = {}
    TorsionAlertsSummaryInfo["RuleIDs"] = []

    for DataLabel in ["SMARTSToRuleIDs", "RuleSMARTS", "HierarchyClassName", "HierarchySubClassName", "EnergyMethod", "MaxSingleEnergyAlertTypes", "MaxSingleEnergyAlertTypesMolCount"]:
        TorsionAlertsSummaryInfo[DataLabel] = {}
    
    return TorsionAlertsSummaryInfo

def TrackTorsionAlertsSummaryInfo(TorsionAlertsSummaryInfo, RotBondsAlertsInfo):
    """Track torsion alerts summary information for matched torsion rules in a
    molecule."""
    
    if OptionsInfo["CountMode"]:
        return
    
    if not OptionsInfo["TrackAlertsSummaryInfo"]:
        return

    if RotBondsAlertsInfo is None:
        return
    
    MolAlertsInfo = {}
    MolAlertsInfo["RuleIDs"] = []
    MolAlertsInfo["MaxSingleEnergyAlertTypes"] = {}

    for ID in RotBondsAlertsInfo["IDs"]:
        if not RotBondsAlertsInfo["MatchStatus"][ID]:
            continue
        
        if SkipRotatableBondAlertInfo(ID, RotBondsAlertsInfo):
            continue

        MaxSingleEnergyAlertType = SetupMaxSingleEnergyAlertStatusValue(RotBondsAlertsInfo["MaxSingleEnergyAlertStatus"][ID])
        
        TorsionRuleNodeID = RotBondsAlertsInfo["TorsionRuleNodeID"][ID]
        TorsionRuleSMARTS = RotBondsAlertsInfo["TorsionRuleSMARTS"][ID]
        
        # Track data for torsion alert summary information across molecules...
        if TorsionRuleNodeID not in TorsionAlertsSummaryInfo["RuleSMARTS"]:
            TorsionAlertsSummaryInfo["RuleIDs"].append(TorsionRuleNodeID)
            TorsionAlertsSummaryInfo["SMARTSToRuleIDs"][TorsionRuleSMARTS] = TorsionRuleNodeID
            
            TorsionAlertsSummaryInfo["RuleSMARTS"][TorsionRuleNodeID] = TorsionRuleSMARTS
            TorsionAlertsSummaryInfo["HierarchyClassName"][TorsionRuleNodeID] = RotBondsAlertsInfo["HierarchyClassName"][ID]
            TorsionAlertsSummaryInfo["HierarchySubClassName"][TorsionRuleNodeID] = RotBondsAlertsInfo["HierarchySubClassName"][ID]
            
            TorsionAlertsSummaryInfo["EnergyMethod"][TorsionRuleNodeID] = RotBondsAlertsInfo["EnergyMethod"][ID]
            
            # Initialize number of alert types across all molecules...
            TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID] = {}
            
            # Initialize number of molecules flagged by each alert type...
            TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypesMolCount"][TorsionRuleNodeID] = {}
        
        if MaxSingleEnergyAlertType not in TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID]:
            TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID][MaxSingleEnergyAlertType] = 0
            TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypesMolCount"][TorsionRuleNodeID][MaxSingleEnergyAlertType] = 0
        
        TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID][MaxSingleEnergyAlertType] += 1
        
        # Track data for torsion alert information in a molecule...
        if TorsionRuleNodeID not in MolAlertsInfo["MaxSingleEnergyAlertTypes"]:
            MolAlertsInfo["RuleIDs"].append(TorsionRuleNodeID)
            MolAlertsInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID] = {}
        
        if MaxSingleEnergyAlertType not in MolAlertsInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID]:
            MolAlertsInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID][MaxSingleEnergyAlertType] = 0
        MolAlertsInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID][MaxSingleEnergyAlertType] += 1
    
    # Track number of molecules flagged by a specific torsion alert...
    for TorsionRuleNodeID in MolAlertsInfo["RuleIDs"]:
        for MaxSingleEnergyAlertType in MolAlertsInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID]:
            if MolAlertsInfo["MaxSingleEnergyAlertTypes"][TorsionRuleNodeID][MaxSingleEnergyAlertType]:
                TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypesMolCount"][TorsionRuleNodeID][MaxSingleEnergyAlertType] += 1

def WriteTorsionAlertsSummaryInfo(Writer, TorsionAlertsSummaryInfo):
    """Write out torsion alerts summary informatio to a CSV file."""
    
    if OptionsInfo["CountMode"]:
        return
    
    if not OptionsInfo["OutfileSummaryMode"]:
        return

    if len(TorsionAlertsSummaryInfo["RuleIDs"]) == 0:
        return
    
    # Write headers...
    QuoteValues = True
    Values = ["TorsionRule", "HierarchyClass", "HierarchySubClass", "EnergyMethod", "MaxSingleEnergyTorsionAlertTypes", "MaxSingleEnergyTorsionAlertCount", "MaxSingleEnergyTorsionAlertMolCount"]
    Writer.write("%s\n" % MiscUtil.JoinWords(Values, ",", QuoteValues))
    
    SortedRuleIDs = GetSortedTorsionAlertsSummaryInfoRuleIDs(TorsionAlertsSummaryInfo)
    
    # Write alerts information...
    for ID in SortedRuleIDs:
        # Remove any double quotes in SMARTS...
        RuleSMARTS = TorsionAlertsSummaryInfo["RuleSMARTS"][ID]
        RuleSMARTS = re.sub("\"", "", RuleSMARTS, re.I)
        
        HierarchyClassName = TorsionAlertsSummaryInfo["HierarchyClassName"][ID]
        HierarchySubClassName = TorsionAlertsSummaryInfo["HierarchySubClassName"][ID]
        
        EnergyMethod = TorsionAlertsSummaryInfo["EnergyMethod"][ID]
        
        MaxSingleEnergyAlertTypes = []
        MaxSingleEnergyAlertTypesCount = []
        MaxSingleEnergyAlertTypesMolCount = []
        for MaxSingleEnergyAlertType in sorted(TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypes"][ID]):
            MaxSingleEnergyAlertTypes.append(MaxSingleEnergyAlertType)
            MaxSingleEnergyAlertTypesCount.append("%s" % TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypes"][ID][MaxSingleEnergyAlertType])
            MaxSingleEnergyAlertTypesMolCount.append("%s" % TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypesMolCount"][ID][MaxSingleEnergyAlertType])
        
        Values = [RuleSMARTS, HierarchyClassName, HierarchySubClassName, EnergyMethod, "%s" % MiscUtil.JoinWords(MaxSingleEnergyAlertTypes, ","), "%s" % (MiscUtil.JoinWords(MaxSingleEnergyAlertTypesCount, ",")), "%s" % (MiscUtil.JoinWords(MaxSingleEnergyAlertTypesMolCount, ","))]
        Writer.write("%s\n" % MiscUtil.JoinWords(Values, ",", QuoteValues))

def GetSortedTorsionAlertsSummaryInfoRuleIDs(TorsionAlertsSummaryInfo):
    """Sort torsion rule IDs by  alert types molecule count in descending order."""

    SortedRuleIDs = []
    
    RuleIDs = TorsionAlertsSummaryInfo["RuleIDs"]
    if len(RuleIDs) == 0:
        return SortedRuleIDs
    
    # Setup a map from AlertTypesMolCount to IDs for sorting alerts...
    RuleIDs = TorsionAlertsSummaryInfo["RuleIDs"]
    MolCountMap = {}
    for ID in RuleIDs:
        MolCount = 0
        for AlertType in sorted(TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypes"][ID]):
            MolCount += TorsionAlertsSummaryInfo["MaxSingleEnergyAlertTypesMolCount"][ID][AlertType]
        MolCountMap[ID] = MolCount

    SortedRuleIDs = sorted(RuleIDs, key = lambda ID: MolCountMap[ID], reverse = True)
    
    return SortedRuleIDs

def WriteTorsionAlertsFilteredByRulesInfo(TorsionAlertsSummaryInfo):
    """Write out torsion alerts SD files for individual torsion rules."""
    
    if OptionsInfo["CountMode"]:
        return
    
    if not OptionsInfo["OutfilesFilteredByRulesMode"]:
        return

    if len(TorsionAlertsSummaryInfo["RuleIDs"]) == 0:
        return

    # Setup a molecule reader for filtered molecules...
    FilteredMols  = RDKitUtil.ReadMolecules(OptionsInfo["OutfileFiltered"], **OptionsInfo["InfileParams"])

    # Get torsion rule IDs for writing out filtered SD files for individual torsion alert rules... 
    TorsionRuleIDs = GetTorsionAlertsFilteredByRuleFilesRuleIDs(TorsionAlertsSummaryInfo)

    # Setup writers...
    ByRuleOutfilesWriters = SetupByRuleOutfilesWriters(TorsionRuleIDs)
    
    for Mol in FilteredMols:
        # Retrieve torsion alerts info...
        TorsionAlertsInfo = RetrieveTorsionAlertsInfo(Mol, TorsionAlertsSummaryInfo)
        if TorsionAlertsInfo is None:
            continue
        
        for TorsionRuleID in TorsionRuleIDs:
            if TorsionRuleID not in TorsionAlertsInfo["RuleSMARTS"]:
                continue
            
            WriteMoleculeFilteredByRuleID(ByRuleOutfilesWriters[TorsionRuleID], Mol, TorsionRuleID, TorsionAlertsSummaryInfo, TorsionAlertsInfo)
        
    CloseByRuleOutfilesWriters(ByRuleOutfilesWriters)

def GetTorsionAlertsFilteredByRuleFilesRuleIDs(TorsionAlertsSummaryInfo):
    """Get torsion rule IDs for writing out individual SD files filtered by torsion alert rules."""
    
    # Get torsion rule IDs triggering torsion alerts sorted in the order from the most to
    # the least number of unique molecules...
    RuleIDs = GetSortedTorsionAlertsSummaryInfoRuleIDs(TorsionAlertsSummaryInfo)

    # Select torsion rule IDs for writing out SD files...
    if not OptionsInfo["OutfilesFilteredByRulesAllMode"]:
        MaxRuleIDs = OptionsInfo["OutfilesFilteredByRulesMaxCount"]
        if MaxRuleIDs < len(RuleIDs):
            RuleIDs = RuleIDs[0:MaxRuleIDs]
    
    return RuleIDs

def RetrieveTorsionAlertsInfo(Mol, TorsionAlertsSummaryInfo):
    """Parse torsion alerts data field value to retrieve alerts information for rotatable bonds."""
    
    TorsionAlertsLabel = OptionsInfo["SDFieldIDsToLabels"]["TorsionAlertsLabel"]
    TorsionAlerts = Mol.GetProp(TorsionAlertsLabel) if Mol.HasProp(TorsionAlertsLabel) else None
    
    if TorsionAlerts is None or len(TorsionAlerts) == 0:
        return None

    # Initialize for tracking by rule IDs...
    TorsionAlertsInfo = {}
    TorsionAlertsInfo["RuleIDs"] = []
    
    TorsionAlertsInfo["RuleSMARTS"] = {}
    TorsionAlertsInfo["HierarchyClassName"] = {}
    TorsionAlertsInfo["HierarchySubClassName"] = {}
    TorsionAlertsInfo["EnergyMethod"] = {}
    
    TorsionAlertsInfo["AtomIndices"] = {}
    TorsionAlertsInfo["TorsionAtomIndices"] = {}
    TorsionAlertsInfo["TorsionAngle"] = {}
    
    TorsionAlertsInfo["Energy"] = {}
    TorsionAlertsInfo["EnergyLowerBound"] = {}
    TorsionAlertsInfo["EnergyUpperBound"] = {}

    TorsionAlertsInfo["AngleNotObserved"] = {}
    TorsionAlertsInfo["MaxSingleEnergyAlertType"] = {}
    
    TorsionAlertsInfo["AnglesNotObservedCount"] = {}
    TorsionAlertsInfo["MaxSingleEnergyAlertsCount"] = {}
    
    ValuesDelimiter = OptionsInfo["IntraSetValuesDelim"]
    TorsionAlertsSetSize = 12
    
    TorsionAlertsWords = TorsionAlerts.split()
    if len(TorsionAlertsWords) % TorsionAlertsSetSize:
        MiscUtil.PrintError("The number of space delimited values, %s, for TorsionAlerts data field in filtered SD file must be a multiple of %s." % (len(TorsionAlertsWords), TorsionAlertsSetSize))

    ID = 0
    for Index in range(0, len(TorsionAlertsWords), TorsionAlertsSetSize):
        ID += 1
        
        RotBondIndices, TorsionIndices, TorsionAngle, Energy, EnergyLowerBound, EnergyUpperBound, HierarchyClass, HierarchySubClass, TorsionRule, EnergyMethod, AngleNotObserved, MaxSingleEnergyAlertType = TorsionAlertsWords[Index: Index + TorsionAlertsSetSize]
        RotBondIndices = RotBondIndices.split(ValuesDelimiter)
        TorsionIndices = TorsionIndices.split(ValuesDelimiter)

        if TorsionRule not in TorsionAlertsSummaryInfo["SMARTSToRuleIDs"]:
            MiscUtil.PrintWarning("The SMARTS pattern, %s, for TorsionAlerts data field in filtered SD file doesn't map to any torsion rule..." % TorsionRule)
            continue
        TorsionRuleNodeID = TorsionAlertsSummaryInfo["SMARTSToRuleIDs"][TorsionRule]
    
        # Track data for torsion alerts in a molecule...
        if TorsionRuleNodeID not in TorsionAlertsInfo["RuleSMARTS"]:
            TorsionAlertsInfo["RuleIDs"].append(TorsionRuleNodeID)

            TorsionAlertsInfo["RuleSMARTS"][TorsionRuleNodeID] = TorsionRule
            TorsionAlertsInfo["HierarchyClassName"][TorsionRuleNodeID] = HierarchyClass
            TorsionAlertsInfo["HierarchySubClassName"][TorsionRuleNodeID] = HierarchySubClass
            TorsionAlertsInfo["EnergyMethod"][TorsionRuleNodeID] = EnergyMethod

            TorsionAlertsInfo["AtomIndices"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["TorsionAtomIndices"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["TorsionAngle"][TorsionRuleNodeID] = []
    
            TorsionAlertsInfo["Energy"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["EnergyLowerBound"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["EnergyUpperBound"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["AngleNotObserved"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["MaxSingleEnergyAlertType"][TorsionRuleNodeID] = []
            
            TorsionAlertsInfo["AnglesNotObservedCount"][TorsionRuleNodeID] = 0
            TorsionAlertsInfo["MaxSingleEnergyAlertsCount"][TorsionRuleNodeID] = 0
            
        # Track multiple values for a rule ID...
        TorsionAlertsInfo["AtomIndices"][TorsionRuleNodeID].append(RotBondIndices)
        TorsionAlertsInfo["TorsionAtomIndices"][TorsionRuleNodeID].append(TorsionIndices)
        TorsionAlertsInfo["TorsionAngle"][TorsionRuleNodeID].append(TorsionAngle)
        
        TorsionAlertsInfo["Energy"][TorsionRuleNodeID].append(Energy)
        TorsionAlertsInfo["EnergyLowerBound"][TorsionRuleNodeID].append(EnergyLowerBound)
        TorsionAlertsInfo["EnergyUpperBound"][TorsionRuleNodeID].append(EnergyUpperBound)
        TorsionAlertsInfo["AngleNotObserved"][TorsionRuleNodeID].append(AngleNotObserved)
        
        TorsionAlertsInfo["MaxSingleEnergyAlertType"][TorsionRuleNodeID].append(MaxSingleEnergyAlertType)
        
        # Count angles not observer for a rule ID...
        if AngleNotObserved == 'Yes':
            TorsionAlertsInfo["AnglesNotObservedCount"][TorsionRuleNodeID] += 1

        # Count max single energy alert for a rule ID...
        if MaxSingleEnergyAlertType == 'Yes':
            TorsionAlertsInfo["MaxSingleEnergyAlertsCount"][TorsionRuleNodeID] += 1
    
    return TorsionAlertsInfo

def WriteMolecule(Writer, Mol, RotBondsAlertsInfo):
    """Write out molecule."""
    
    if OptionsInfo["CountMode"]:
        return True

    SetupMolPropertiesForAlertsInformation(Mol, RotBondsAlertsInfo)

    try:
        Writer.write(Mol)
    except Exception as ErrMsg:
        MiscUtil.PrintWarning("Failed to write molecule %s:\n%s\n" % (RDKitUtil.GetMolName(Mol), ErrMsg))
        return False
    
    return True

def SetupMolPropertiesForAlertsInformation(Mol, RotBondsAlertsInfo):
    """Setup molecule properties containing alerts information for rotatable bonds."""

    if not OptionsInfo["OutfileAlerts"]:
        return

    SDFieldIDsToLabels = OptionsInfo["SDFieldIDsToLabels"]
    Precision = OptionsInfo["Precision"]
    
    # Setup rotatable bonds count...
    RotBondsCount = 0
    if RotBondsAlertsInfo is not None:
        RotBondsCount =  len(RotBondsAlertsInfo["IDs"])
    Mol.SetProp(SDFieldIDsToLabels["RotBondsCountLabel"],  "%s" % RotBondsCount)

    if RotBondsAlertsInfo is not None:
        # Setup total energy along with lower and upper bounds...
        Mol.SetProp(SDFieldIDsToLabels["TotalEnergyLabel"],  "%s" % SetupEnergyValueForSDField(RotBondsAlertsInfo["TotalEnergy"], Precision))
        Mol.SetProp(SDFieldIDsToLabels["TotalEnergyLowerBoundCILabel"],  "%s" % SetupEnergyValueForSDField(RotBondsAlertsInfo["TotalEnergyLowerBound"], Precision))
        Mol.SetProp(SDFieldIDsToLabels["TotalEnergyUpperBoundCILabel"],  "%s" % SetupEnergyValueForSDField(RotBondsAlertsInfo["TotalEnergyUpperBound"], Precision))
        
        # Setup max single energy and alert count...
        if OptionsInfo["MaxSingleEnergyMode"] or OptionsInfo["TotalOrMaxSingleEnergyMode"]:
            Mol.SetProp(SDFieldIDsToLabels["MaxSingleEnergyLabel"],  "%s" % SetupEnergyValueForSDField(RotBondsAlertsInfo["MaxSingleEnergy"], Precision))
            Mol.SetProp(SDFieldIDsToLabels["MaxSingleEnergyAlertsCountLabel"],  "%s" % ("NA" if RotBondsAlertsInfo["MaxSingleEnergyAlertsCount"] is None else RotBondsAlertsInfo["MaxSingleEnergyAlertsCount"]))
            
        Mol.SetProp(SDFieldIDsToLabels["AnglesNotObservedCountLabel"],  "%s" % ("NA" if RotBondsAlertsInfo["AnglesNotObservedCount"] is None else RotBondsAlertsInfo["AnglesNotObservedCount"]))
    
    
    # Setup alert information for rotatable bonds...
    AlertsInfoValues = []

    # Delimiter for multiple values corresponding to specific set of information for
    # a rotatable bond. For example: TorsionAtomIndices
    ValuesDelim = OptionsInfo["IntraSetValuesDelim"]

    # Delimiter for various values for a rotatable bond...
    RotBondValuesDelim = OptionsInfo["InterSetValuesDelim"]
    
    # Delimiter for values corresponding to multiple rotatable bonds...
    AlertsInfoValuesDelim = OptionsInfo["InterSetValuesDelim"]
    
    if RotBondsAlertsInfo is not None:
        for ID in RotBondsAlertsInfo["IDs"]:
            if not RotBondsAlertsInfo["MatchStatus"][ID]:
                continue

            if SkipRotatableBondAlertInfo(ID, RotBondsAlertsInfo):
                continue
                
            RotBondValues = []
            
            # Bond atom indices...
            Values = ["%s" % Value for Value in RotBondsAlertsInfo["AtomIndices"][ID]]
            RotBondValues.append(ValuesDelim.join(Values))
    
            # Torsion atom indices...
            TorsionAtomIndices = SetupTorsionAtomIndicesValues(RotBondsAlertsInfo["TorsionAtomIndices"][ID], ValuesDelim)
            RotBondValues.append(TorsionAtomIndices)

            # Torsion angle...
            RotBondValues.append("%.2f" % RotBondsAlertsInfo["TorsionAngle"][ID])

            # Energy along with its lower and upper bound confidence interval...
            RotBondValues.append(SetupEnergyValueForSDField(RotBondsAlertsInfo["Energy"][ID], Precision))
            RotBondValues.append(SetupEnergyValueForSDField(RotBondsAlertsInfo["EnergyLowerBound"][ID], Precision))
            RotBondValues.append(SetupEnergyValueForSDField(RotBondsAlertsInfo["EnergyUpperBound"][ID], Precision))
            
            # Hierarchy class and subclass names...
            RotBondValues.append("%s" % RotBondsAlertsInfo["HierarchyClassName"][ID])
            RotBondValues.append("%s" % RotBondsAlertsInfo["HierarchySubClassName"][ID])
            
            # Torsion rule SMARTS...
            RotBondValues.append("%s" % RotBondsAlertsInfo["TorsionRuleSMARTS"][ID])
            
            # Energy method...
            RotBondValues.append("%s" % RotBondsAlertsInfo["EnergyMethod"][ID])

            # Angle not observed...
            RotBondValues.append("%s" % SetupAngleNotObservedValue(RotBondsAlertsInfo["AngleNotObserved"][ID]))
            
            # Max single energy alert status...
            RotBondValues.append("%s" % SetupMaxSingleEnergyAlertStatusValue(RotBondsAlertsInfo["MaxSingleEnergyAlertStatus"][ID]))
                
            # Track joined values for a rotatable bond...
            AlertsInfoValues.append("%s" % RotBondValuesDelim.join(RotBondValues))
    
    if len(AlertsInfoValues):
        Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionAlertsLabel"], "%s" % ("%s" % AlertsInfoValuesDelim.join(AlertsInfoValues)))

def WriteMoleculeFilteredByRuleID(Writer, Mol, TorsionRuleID, TorsionAlertsSummaryInfo, TorsionAlertsInfo):
    """Write out molecule."""
    
    if OptionsInfo["CountMode"]:
        return

    SetupMolPropertiesForFilteredByRuleIDAlertsInformation(Mol, TorsionRuleID, TorsionAlertsSummaryInfo, TorsionAlertsInfo)
        
    Writer.write(Mol)

def SetupMolPropertiesForFilteredByRuleIDAlertsInformation(Mol, TorsionRuleID, TorsionAlertsSummaryInfo, TorsionAlertsInfo):
    """Setup molecule properties containing alerts information for torsion alerts
    fileted by Rule IDs."""

    # Delete torsion alerts information for rotatable bonds...
    if Mol.HasProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionAlertsLabel"]):
        Mol.ClearProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionAlertsLabel"])

    # Delimiter for values...
    IntraSetValuesDelim = OptionsInfo["IntraSetValuesDelim"]
    InterSetValuesDelim = OptionsInfo["InterSetValuesDelim"]

    # Setup alert rule information...
    AlertRuleInfoValues = []
    
    AlertRuleInfoValues.append("%s" % TorsionAlertsInfo["HierarchyClassName"][TorsionRuleID])
    AlertRuleInfoValues.append("%s" % TorsionAlertsInfo["HierarchySubClassName"][TorsionRuleID])
    
    AlertRuleInfoValues.append("%s" % TorsionAlertsInfo["RuleSMARTS"][TorsionRuleID])
    AlertRuleInfoValues.append("%s" % TorsionAlertsInfo["EnergyMethod"][TorsionRuleID])
    
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleLabel"], "%s" % ("%s" % InterSetValuesDelim.join(AlertRuleInfoValues)))

    # Setup max single energy alert count for torsion rule...
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleMaxSingleEnergyAlertsCountLabel"],  "%s" % TorsionAlertsInfo["MaxSingleEnergyAlertsCount"][TorsionRuleID])
    
    # Setup angle not observed count for torsion rule...
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleAnglesNotObservedCountLabel"],  "%s" % TorsionAlertsInfo["AnglesNotObservedCount"][TorsionRuleID])
    
    # Setup torsion rule alerts...
    # "TorsionRuleAlertsLabel": "TorsionRuleAlerts (RotBondIndices TorsionIndices TorsionAngle Energy EnergyLowerBoundCI EnergyUpperBoundCI EnergyMethod AngleNotObserved MaxSingleEnergyAlert)
    AlertsInfoValues = []
    for Index in range(0, len(TorsionAlertsInfo["AtomIndices"][TorsionRuleID])):
        RotBondInfoValues = []
        
        # Bond atom indices...
        Values = ["%s" % Value for Value in TorsionAlertsInfo["AtomIndices"][TorsionRuleID][Index]]
        RotBondInfoValues.append(IntraSetValuesDelim.join(Values))
        
        # Torsion atom indices retrieved from the filtered SD file and stored as strings...
        Values = ["%s" % Value for Value in TorsionAlertsInfo["TorsionAtomIndices"][TorsionRuleID][Index]]
        RotBondInfoValues.append(IntraSetValuesDelim.join(Values))
        
        # Torsion angle...
        RotBondInfoValues.append(TorsionAlertsInfo["TorsionAngle"][TorsionRuleID][Index])

        # Energy and its bounds...
        RotBondInfoValues.append(TorsionAlertsInfo["Energy"][TorsionRuleID][Index])
        RotBondInfoValues.append(TorsionAlertsInfo["EnergyLowerBound"][TorsionRuleID][Index])
        RotBondInfoValues.append(TorsionAlertsInfo["EnergyUpperBound"][TorsionRuleID][Index])
        
        # Angle not observed......
        RotBondInfoValues.append(TorsionAlertsInfo["AngleNotObserved"][TorsionRuleID][Index])
        
        # Max single energy alert type...
        RotBondInfoValues.append(TorsionAlertsInfo["MaxSingleEnergyAlertType"][TorsionRuleID][Index])
        
        # Track alerts informaiton...
        AlertsInfoValues.append("%s" % InterSetValuesDelim.join(RotBondInfoValues))
        
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleAlertsLabel"],  "%s" % (InterSetValuesDelim.join(AlertsInfoValues)))
    
def SkipRotatableBondAlertInfo(ID, RotBondsAlertsInfo):
    """Skip rotatble bond alert info for a specific bond during writing to output files."""

    if not OptionsInfo["OutfileAlertsOnly"]:
        return False
    
    if RotBondsAlertsInfo["RotBondsAlertsStatus"] is None:
        return True

    Status = False
    if OptionsInfo["TotalEnergyMode"]:
        if not RotBondsAlertsInfo["RotBondsAlertsStatus"]:
            Status = True
    elif OptionsInfo["MaxSingleEnergyMode"]:
        if RotBondsAlertsInfo["MaxSingleEnergyAlertStatus"][ID] is None or not RotBondsAlertsInfo["MaxSingleEnergyAlertStatus"][ID]:
            Status = True
    elif OptionsInfo["TotalOrMaxSingleEnergyMode"]:
        if not RotBondsAlertsInfo["RotBondsAlertsStatus"]:
            Status = True
    
    return Status

def SetupEnergyValueForSDField(Value, Precision):
    """Setup energy value for SD field."""
    
    if Value is None or math.isnan(Value) or math.isinf(Value):
        return "NA"

    return "%.*f" % (Precision, Value)
    
def SetupAngleNotObservedValue(Value):
    """Setup angle not observed value."""
    
    if Value is None:
        return "NA"

    return "Yes" if Value else "No"

def SetupMaxSingleEnergyAlertStatusValue(Value):
    """Setup max single energy alert status value."""
    
    if Value is None:
        return "NA"

    return "Yes" if Value else "No"

def SetupTorsionAtomIndicesValues(TorsionAtomIndicesList, ValuesDelim):
    """Setup torsion atom indices value for output files."""

    Values = ["%s" % Value for Value in TorsionAtomIndicesList]
    
    return ValuesDelim.join(Values)
    
def SetupOutfilesWriters():
    """Setup molecule and summary writers."""

    OutfilesWriters = {"WriterRemaining": None, "WriterFiltered": None, "WriterAlertSummary": None}

    # Writers for SD files...
    WriterRemaining, WriterFiltered = SetupMoleculeWriters()
    OutfilesWriters["WriterRemaining"] = WriterRemaining
    OutfilesWriters["WriterFiltered"] = WriterFiltered
    
    # Writer for alert summary CSV file...
    WriterAlertSummary = SetupAlertSummaryWriter()
    OutfilesWriters["WriterAlertSummary"] = WriterAlertSummary

    return OutfilesWriters

def SetupMoleculeWriters():
    """Setup molecule writers."""
    
    Writer = None
    WriterFiltered = None

    if OptionsInfo["CountMode"]:
        return (Writer, WriterFiltered)

    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("\nGenerating file %s..." % OptionsInfo["Outfile"])
    
    if OptionsInfo["OutfileFilteredMode"]:
        WriterFiltered = RDKitUtil.MoleculesWriter(OptionsInfo["OutfileFiltered"], **OptionsInfo["OutfileParams"])
        if WriterFiltered is None:
            MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["OutfileFiltered"])
        MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["OutfileFiltered"])
    
    return (Writer, WriterFiltered)

def SetupAlertSummaryWriter():
    """Setup a alert summary writer."""
    
    Writer = None
    
    if OptionsInfo["CountMode"]:
        return Writer
        
    if not OptionsInfo["OutfileSummaryMode"]:
        return Writer
    
    Outfile = OptionsInfo["OutfileSummary"]
    Writer = open(Outfile, "w")
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % Outfile)
    
    MiscUtil.PrintInfo("Generating file %s..." % Outfile)
    
    return Writer
    
def CloseOutfilesWriters(OutfilesWriters):
    """Close outfile writers."""

    for WriterType, Writer in OutfilesWriters.items():
        if Writer is not None:
            Writer.close()

def SetupByRuleOutfilesWriters(RuleIDs):
    """Setup by rule outfiles writers."""

    # Initialize...
    OutfilesWriters = {}
    for RuleID in RuleIDs:
        OutfilesWriters[RuleID] = None
    
    if OptionsInfo["CountMode"]:
        return OutfilesWriters
        
    if not OptionsInfo["OutfilesFilteredByRulesMode"]:
        return OutfilesWriters
    
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    OutfilesRoot = "%s_Filtered_TopRule" % FileName
    OutfilesExt = "sdf"

    MsgTxt = "all" if OptionsInfo["OutfilesFilteredByRulesAllMode"] else "top %s" % OptionsInfo["OutfilesFilteredByRulesMaxCount"]
    MiscUtil.PrintInfo("\nGenerating output files %s*.%s for %s torsion rules triggering alerts..." % (OutfilesRoot, OutfilesExt, MsgTxt))
    
    # Delete any existing output files...
    Outfiles = glob.glob("%s*.%s" % (OutfilesRoot, OutfilesExt))
    if len(Outfiles):
        MiscUtil.PrintInfo("Deleting existing output files %s*.%s..." % (OutfilesRoot, OutfilesExt))
        for Outfile in Outfiles:
            try:
                os.remove(Outfile)
            except Exception as ErrMsg:
                MiscUtil.PrintWarning("Failed to delete file: %s" % ErrMsg)
    
    RuleIndex = 0
    for RuleID in RuleIDs:
        RuleIndex += 1
        Outfile = "%s%s.%s" % (OutfilesRoot, RuleIndex, OutfilesExt)
        Writer = RDKitUtil.MoleculesWriter(Outfile, **OptionsInfo["OutfileParams"])
        if Writer is None:
            MiscUtil.PrintError("Failed to setup a writer for output fie %s " % Outfile)
            
        OutfilesWriters[RuleID] = Writer

    return OutfilesWriters

def CloseByRuleOutfilesWriters(OutfilesWriters):
    """Close by rule outfile writers."""

    for RuleID, Writer in OutfilesWriters.items():
        if Writer is not None:
            Writer.close()

def ProcessTorsionLibraryInfo():
    """Process torsion library information."""

    RetrieveTorsionLibraryInfo()
    ListTorsionLibraryInfo()

def RetrieveTorsionLibraryInfo(Quiet = False):
    """Retrieve torsion library information."""
    
    TorsionLibraryFilePath = OptionsInfo["TorsionEnergyLibraryFile"]
    if TorsionLibraryFilePath is None:
        TorsionLibraryFile = "TorsionStrainEnergyLibrary.xml"
        MayaChemToolsDataDir = MiscUtil.GetMayaChemToolsLibDataPath()
        TorsionLibraryFilePath = os.path.join(MayaChemToolsDataDir, TorsionLibraryFile)
        if not Quiet:
            MiscUtil.PrintInfo("\nRetrieving data from default torsion strain energy library file %s..." % TorsionLibraryFile)
    else:
        TorsionLibraryFilePath = OptionsInfo["TorsionEnergyLibraryFile"]
        if not Quiet:
            MiscUtil.PrintInfo("\nRetrieving data from torsion strain energy library file %s..." % TorsionLibraryFilePath)
        
    TorsionLibraryInfo["TorsionLibElementTree"] = TorsionLibraryUtil.RetrieveTorsionLibraryInfo(TorsionLibraryFilePath)

def ListTorsionLibraryInfo():
    """List torsion library information."""
    
    TorsionLibraryUtil.ListTorsionLibraryInfo(TorsionLibraryInfo["TorsionLibElementTree"])

def SetupTorsionLibraryInfo(Quiet = False):
    """Setup torsion library information for matching rotatable bonds."""

    if not Quiet:
        MiscUtil.PrintInfo("\nSetting up torsion library information for matching rotatable bonds...")

    TorsionLibraryUtil.SetupTorsionLibraryInfoForMatchingRotatableBonds(TorsionLibraryInfo)
    
def ProcessRotatableBondsSMARTSMode():
    """"Process SMARTS pattern for rotatable bonds."""

    RotBondsMode = OptionsInfo["RotBondsSMARTSMode"]
    RotBondsSMARTSPattern = None
    if re.match("NonStrict", RotBondsMode, re.I):
        RotBondsSMARTSPattern = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    elif re.match("SemiStrict", RotBondsMode, re.I):
        RotBondsSMARTSPattern = "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
    elif re.match("Strict", RotBondsMode, re.I):
        RotBondsSMARTSPattern = "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
    elif re.match("Specify", RotBondsMode, re.I):
        RotBondsSMARTSPattern = OptionsInfo["RotBondsSMARTSPatternSpecified"]
        RotBondsSMARTSPattern = RotBondsSMARTSPattern.strip()
        if not len(RotBondsSMARTSPattern):
            MiscUtil.PrintError("Empty value specified for SMILES/SMARTS pattern in  \"--rotBondsSMARTSPattern\" option, %s." % RotBondsMode)
    else:
        MiscUtil.PrintError("The value, %s, specified for option \"-r, --rotBondsSMARTSMode\" is not valid. " % RotBondsMode)
        
    RotBondsPatternMol = Chem.MolFromSmarts(RotBondsSMARTSPattern)
    if RotBondsPatternMol is None:
        if re.match("Specify", RotBondsMode, re.I):
            MiscUtil.PrintError("Failed to create rotatable bonds pattern molecule. The rotatable bonds SMARTS pattern, \"%s\", specified using \"--rotBondsSMARTSPattern\" option is not valid." % (RotBondsSMARTSPattern))
        else:
            MiscUtil.PrintError("Failed to create rotatable bonds pattern molecule. The default rotatable bonds SMARTS pattern, \"%s\", used for, \"%s\", value of  \"-r, --rotBondsSMARTSMode\" option is not valid." % (RotBondsSMARTSPattern, RotBondsMode))
    
    OptionsInfo["RotBondsSMARTSPattern"] = RotBondsSMARTSPattern

def ProcessSDFieldLabelsOption():
    """Process SD data field label option."""

    ParamsOptionName = "--outfileSDFieldLabels"
    ParamsOptionValue = Options["--outfileSDFieldLabels"]

    ParamsIDsToLabels = {"RotBondsCountLabel": "RotBondsCount", "TotalEnergyLabel": "TotalEnergy", "TotalEnergyLowerBoundCILabel": "TotalEnergyLowerBoundCI", "TotalEnergyUpperBoundCILabel": "TotalEnergyUpperBoundCI", "MaxSingleEnergyLabel": "MaxSingleEnergy", "MaxSingleEnergyAlertsCountLabel": "MaxSingleEnergyAlertsCount", "AnglesNotObservedCountLabel": "AnglesNotObservedCount", "TorsionAlertsLabel": "TorsionAlerts(RotBondIndices TorsionIndices TorsionAngle Energy EnergyLowerBoundCI EnergyUpperBoundCI HierarchyClass HierarchySubClass TorsionRule EnergyMethod AngleNotObserved MaxSingleEnergyAlert)", "TorsionRuleLabel": "TorsionRule (HierarchyClass HierarchySubClass TorsionRule EnergyMethod)", "TorsionRuleMaxSingleEnergyAlertsCountLabel": "TorsionRuleMaxSingleEnergyAlertsCount", "TorsionRuleAnglesNotObservedCountLabel": "TorsionRuleAnglesNotObservedCount", "TorsionRuleAlertsLabel": "TorsionRuleAlerts (RotBondIndices TorsionIndices TorsionAngle Energy EnergyLowerBoundCI EnergyUpperBoundCI AngleNotObserved MaxSingleEnergyAlert)"}

    if re.match("^auto$", ParamsOptionValue, re.I):
        OptionsInfo["SDFieldIDsToLabels"] = ParamsIDsToLabels
        return
    
    # Setup a canonical paramater names...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(ParamsIDsToLabels):
        ValidParamNames.append(ParamName)
        CanonicalParamNamesMap[ParamName.lower()] = ParamName
    
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
        
        # Set value...
        ParamsIDsToLabels[ParamName] = ParamValue
    
    OptionsInfo["SDFieldIDsToLabels"] = ParamsIDsToLabels

def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()
    
    TotalEnergyMode, MaxSingleEnergyMode, TotalOrMaxSingleEnergyMode = [False] * 3
    AlertsMode = Options["--alertsMode"]
    if re.match("^TotalEnergy$", AlertsMode, re.I):
        TotalEnergyMode = True
    elif re.match("^MaxSingleEnergy$", AlertsMode, re.I):
        MaxSingleEnergyMode = True
    elif re.match("^TotalOrMaxSingleEnergy$", AlertsMode, re.I):
        TotalOrMaxSingleEnergyMode = True
    OptionsInfo["AlertsMode"] = AlertsMode
    OptionsInfo["TotalEnergyMode"] = TotalEnergyMode
    OptionsInfo["MaxSingleEnergyMode"] = MaxSingleEnergyMode
    OptionsInfo["TotalOrMaxSingleEnergyMode"] = TotalOrMaxSingleEnergyMode
    
    OptionsInfo["FilterTorsionsNotObserved"] = True if re.match("^yes$", Options["--filterTorsionsNotObserved"], re.I) else False
    
    OptionsInfo["MaxSingleEnergyCutoff"] = float(Options["--alertsMaxSingleEnergyCutoff"])
    OptionsInfo["TotalEnergyCutoff"] = float(Options["--alertsTotalEnergyCutoff"])
    
    OptionsInfo["Infile"] = Options["--infile"]
    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters("--infileParams", Options["--infileParams"], InfileName = Options["--infile"], ParamsDefaultInfo = ParamsDefaultInfoOverride)
    
    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters("--outfileParams", Options["--outfileParams"], Options["--infile"], Options["--outfile"])
    
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    OutfileFiltered = "%s_Filtered.%s" % (FileName, FileExt)
    OptionsInfo["OutfileFiltered"] = OutfileFiltered
    OptionsInfo["OutfileFilteredMode"] = True if re.match("^yes$", Options["--outfileFiltered"], re.I) else False

    OptionsInfo["OutfileSummary"] = "%s_AlertsSummary.csv" % (FileName)

    OutfileSummaryMode = Options["--outfileSummary"]
    if re.match("^auto$", OutfileSummaryMode, re.I):
        OutfileSummaryMode = 'yes' if re.match("^MaxSingleEnergy$", Options["--alertsMode"], re.I) else 'no'
    OptionsInfo["OutfileSummaryMode"] = True if re.match("^yes$", OutfileSummaryMode, re.I) else False
    
    if re.match("^yes$", Options["--outfileSummary"], re.I):
        if not re.match("^MaxSingleEnergy$", Options["--alertsMode"], re.I):
            MiscUtil.PrintError("The value \"%s\" specified for \"--outfileSummary\" option is not valid. The specified value is only allowed during \"MaxSingleEnergy\" value of \"-a, --alertsMode\" option." % (Options["--outfileSummary"]))
    
    OutfilesFilteredByRulesMode = Options["--outfilesFilteredByRules"]
    if re.match("^auto$", OutfilesFilteredByRulesMode, re.I):
        OutfilesFilteredByRulesMode = 'yes' if re.match("^MaxSingleEnergy$", Options["--alertsMode"], re.I) else 'no'
    OptionsInfo["OutfilesFilteredByRulesMode"] = True if re.match("^yes$", OutfilesFilteredByRulesMode, re.I) else False
    
    if re.match("^yes$", Options["--outfilesFilteredByRules"], re.I):
        if not re.match("^MaxSingleEnergy$", Options["--alertsMode"], re.I):
            MiscUtil.PrintError("The value \"%s\" specified for \"--outfilesFilteredByRules\" option is not valid. The specified value is only allowed during \"MaxSingleEnergy\" value of \"-a, --alertsMode\" option." % (Options["--outfileSummary"]))
    
    OptionsInfo["TrackAlertsSummaryInfo"] = True if (OptionsInfo["OutfileSummaryMode"] or OptionsInfo["OutfilesFilteredByRulesMode"]) else False
    
    OutfilesFilteredByRulesMaxCount = Options["--outfilesFilteredByRulesMaxCount"]
    if not re.match("^All$", OutfilesFilteredByRulesMaxCount, re.I):
        OutfilesFilteredByRulesMaxCount = int(OutfilesFilteredByRulesMaxCount)
    OptionsInfo["OutfilesFilteredByRulesMaxCount"] = OutfilesFilteredByRulesMaxCount
    OptionsInfo["OutfilesFilteredByRulesAllMode"] = True if re.match("^All$", Options["--outfilesFilteredByRulesMaxCount"], re.I) else False
    
    OptionsInfo["OutfileAlerts"] = True if re.match("^yes$", Options["--outfileAlerts"], re.I) else False
    
    if re.match("^yes$", Options["--outfilesFilteredByRules"], re.I):
        if not re.match("^yes$", Options["--outfileAlerts"], re.I):
            MiscUtil.PrintError("The value \"%s\" specified for \"--outfilesFilteredByRules\" option is not valid. The specified value is only allowed during \"yes\" value of \"--outfileAlerts\" option." % (Options["--outfilesFilteredByRules"]))
    
    OptionsInfo["OutfileAlertsMode"] = Options["--outfileAlertsMode"]
    OptionsInfo["OutfileAlertsOnly"] = True if re.match("^AlertsOnly$", Options["--outfileAlertsMode"], re.I) else False
    
    ProcessSDFieldLabelsOption()
    
    OptionsInfo["Overwrite"] = Options["--overwrite"]
    OptionsInfo["CountMode"] = True if re.match("^count$", Options["--mode"], re.I) else False

    OptionsInfo["Precision"] = int(Options["--precision"])
    
    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])
    
    OptionsInfo["RotBondsSMARTSMode"] = Options["--rotBondsSMARTSMode"]
    OptionsInfo["RotBondsSMARTSPatternSpecified"] = Options["--rotBondsSMARTSPattern"]
    ProcessRotatableBondsSMARTSMode()
    
    OptionsInfo["TorsionEnergyLibraryFile"] = None
    if not re.match("^auto$", Options["--torsionEnergyLibraryFile"], re.I):
        OptionsInfo["TorsionEnergyLibraryFile"] = Options["--torsionEnergyLibraryFile"]
    
    # Setup delimiter for writing out torsion alert information to output files...
    OptionsInfo["IntraSetValuesDelim"] = ","
    OptionsInfo["InterSetValuesDelim"] = " "
    
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
    
def ProcessListTorsionLibraryOption():
    """Process list torsion library information."""

    # Validate and process dataFile option for listing torsion library information...
    OptionsInfo["TorsionEnergyLibraryFile"] = None
    if not re.match("^auto$", Options["--torsionEnergyLibraryFile"], re.I):
        MiscUtil.ValidateOptionFilePath("-t, --torsionEnergyLibraryFile", Options["--torsionEnergyLibraryFile"])
        OptionsInfo["TorsionEnergyLibraryFile"] = Options["--torsionEnergyLibraryFile"]
    
    RetrieveTorsionLibraryInfo()
    ListTorsionLibraryInfo()
    
def ValidateOptions():
    """Validate option values."""

    MiscUtil.ValidateOptionTextValue("-a, --alertsMode", Options["--alertsMode"], "TotalEnergy MaxSingleEnergy TotalOrMaxSingleEnergy")
    
    MiscUtil.ValidateOptionFloatValue("--alertsMaxSingleEnergyCutoff", Options["--alertsMaxSingleEnergyCutoff"], {">": 0.0})
    MiscUtil.ValidateOptionFloatValue("--alertsTotalEnergyCutoff", Options["--alertsTotalEnergyCutoff"], {">": 0.0})
    
    MiscUtil.ValidateOptionTextValue("--filterTorsionsNotObserved", Options["--filterTorsionsNotObserved"], "yes no")
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol")
    
    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    if re.match("^filter$", Options["--mode"], re.I):
        MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
        MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])

    MiscUtil.ValidateOptionTextValue("--outfileFiltered", Options["--outfileFiltered"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--outfilesFilteredByRules", Options["--outfilesFilteredByRules"], "yes no auto")
    if not re.match("^All$", Options["--outfilesFilteredByRulesMaxCount"], re.I):
        MiscUtil.ValidateOptionIntegerValue("--outfilesFilteredByRulesMaxCount", Options["--outfilesFilteredByRulesMaxCount"], {">": 0})
    
    MiscUtil.ValidateOptionTextValue("--outfileSummary", Options["--outfileSummary"], "yes no auto")
    MiscUtil.ValidateOptionTextValue("--outfileAlerts", Options["--outfileAlerts"], "yes no")
    MiscUtil.ValidateOptionTextValue("--outfileAlertsMode", Options["--outfileAlertsMode"], "All AlertsOnly")
    
    MiscUtil.ValidateOptionTextValue("-m, --mode", Options["--mode"], "filter count")
    if re.match("^filter$", Options["--mode"], re.I):
        if not Options["--outfile"]:
            MiscUtil.PrintError("The outfile must be specified using \"-o, --outfile\" during \"filter\" value of \"-m, --mode\" option")
        
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    
    MiscUtil.ValidateOptionIntegerValue("-p, --precision", Options["--precision"], {">": 0})
    
    MiscUtil.ValidateOptionTextValue("-r, --rotBondsSMARTSMode", Options["--rotBondsSMARTSMode"], "NonStrict SemiStrict Strict Specify")
    if re.match("^Specify$", Options["--rotBondsSMARTSMode"], re.I):
        if not Options["--rotBondsSMARTSPattern"]:
            MiscUtil.PrintError("The SMARTS pattern must be specified using \"--rotBondsSMARTSPattern\" during \"Specify\" value of \"-r, --rotBondsSMARTS\" option")
    
    if not re.match("^auto$", Options["--torsionEnergyLibraryFile"], re.I):
        MiscUtil.ValidateOptionFilePath("-t, --torsionEnergyLibraryFile", Options["--torsionEnergyLibraryFile"])

# Setup a usage string for docopt...
_docoptUsage_ = """
RDKitFilterTorsionStrainEnergyAlerts.py - Filter torsion strain energy library alerts

Usage:
    RDKitFilterTorsionStrainEnergyAlerts.py [--alertsMode <TotalEnergy, MaxSingleEnergy, or TotalOrMaxSingleEnergy>]
                                            [--alertsMaxSingleEnergyCutoff <Number>] [--alertsTotalEnergyCutoff <Number>]
                                            [--filterTorsionsNotObserved <yes or no>] [--infileParams <Name,Value,...>] [--mode <filter or count>]
                                            [--mp <yes or no>] [--mpParams <Name,Value,...>] [--outfileAlerts <yes or no>]
                                            [--outfileAlertsMode <All or AlertsOnly>] [--outfileFiltered <yes or no>]
                                            [--outfilesFilteredByRules <yes or no>] [--outfilesFilteredByRulesMaxCount <All or number>]
                                            [--outfileSummary <yes or no>] [--outfileSDFieldLabels <Type,Label,...>] [--outfileParams <Name,Value,...>]
                                            [--overwrite] [--precision <number>] [ --rotBondsSMARTSMode <NonStrict, SemiStrict,...>]
                                            [--rotBondsSMARTSPattern <SMARTS>] [--torsionEnergyLibraryFile <FileName or auto>]
                                            [-w <dir>] -i <infile> -o <outfile>
    RDKitFilterTorsionStrainEnergyAlerts.py [--torsionEnergyLibraryFile <FileName or auto>] -l | --list
    RDKitFilterTorsionStrainEnergyAlerts.py -h | --help | -e | --examples

Description:
    Filter strained molecules from an input file for torsion strain energy library
    [ Ref 153 ] alerts by matching rotatable bonds against SMARTS patterns specified
    for torsion rules in a torsion energy library file and write out appropriate
    molecules to output files. The molecules must have 3D coordinates in input file.
    The default torsion strain energy library file, TorsionStrainEnergyLibrary.xml,
    is available under MAYACHEMTOOLS/lib/data directory.
    
    The data in torsion strain energy library file is organized in a hierarchical
    manner. It consists of one generic class and six specific classes at the highest
    level. Each class contains multiple subclasses corresponding to named functional
    groups or substructure patterns. The subclasses consist of torsion rules sorted
    from specific to generic torsion patterns. The torsion rule, in turn, contains a
    list of peak values for torsion angles and two tolerance values. A pair of tolerance
    values define torsion bins around a torsion peak value.
    
    A strain energy calculation method, 'exact' or 'approximate' [ Ref 153 ], is 
    associated with each torsion rule for calculating torsion strain energy. The 'exact'
    stain energy calculation relies on the energy bins available under the energy histogram
    consisting of 36 bins covering angles from -180 to 180. The width of each bin is 10
    degree. The energy bins are are defined at the right end points. The first and the
    last energy bins correspond to -170 and 180 respectively. The torsion angle is mapped
    to a energy bin. An angle offset is calculated for the torsion angle from the the right
    end point angle of the bin. The strain energy is estimated for the angle offset based
    on the energy difference between the current and previous bins. The torsion strain
    energy, in terms of torsion energy units (TEUs), corresponds to the sum of bin strain
    energy and the angle offset strain energy.
        
        Energy = BinEnergyDiff/10.0 * BinAngleOffset + BinEnergy[BinNum]
        
        Where:
        
        BinEnergyDiff = BinEnergy[BinNum] - BinEnergy[PreviousBinNum]
        BinAngleOffset = TorsionAngle - BinAngleRightSide
        
    The 'approximate' strain energy calculation relies on the angle difference between a
    torsion angle and the torsion peaks observed for the torsion rules in the torsion
    energy library. The torsion angle is matched to a torsion peak based on the value of
    torsion angle difference. It must be less than or equal to the value for the second
    tolerance 'tolerance2'. Otherwise, the torsion angle is not observed in the torsion
    energy library and a value of 'NA' is assigned for torsion energy along with the lower
    and upper bounds on energy at 95% confidence interval. The 'approximate' torsion
    energy (TEUs) for observed torsion angle is calculated using the following formula:
        
        Energy = beta_1 * (AngleDiff ** 2) + beta_2 * (AngleDiff ** 4)
        
    The coefficients 'beta_1' and 'beta_2' are available for the observed angles in 
    the torsion strain energy library. The 'AngleDiff' is the difference between the
    torsion angle and the matched torsion peak.
    
    For example:
         
        <library>
            <hierarchyClass id1="G" id2="G" name="GG">
            ...
            </hierarchyClass>
            <hierarchyClass id1="C" id2="O" name="CO">
                <hierarchySubClass name="Ester bond I" smarts="O=[C:2][O:3]">
                    <torsionRule method="exact" smarts=
                        "[O:1]=[C:2]!@[O:3]~[CH0:4]">
                        <angleList>
                            <angle score="56.52" tolerance1="20.00"
                            tolerance2="25.00" value="0.0"/>
                        </angleList>
                        <histogram>
                            <bin count="1"/>
                            ...
                        </histogram>
                        <histogram_shifted>
                            <bin count="0"/>
                            ...
                        </histogram_shifted>
                        <histogram_converted>
                            <bin energy="4.67... lower="2.14..." upper="Inf"/>
                            ...
                            <bin energy="1.86..." lower="1.58..." upper="2.40..."/>
                            ...
                           </histogram_converted>
                    </torsionRule>
                    <torsionRule method="approximate" smarts=
                        "[cH0:1][c:2]([cH0])!@[O:3][p:4]">
                        <angleList>
                        <angle beta_1="0.002..." beta_2="-7.843...e-07"
                            score="27.14" theta_0="-90.0" tolerance1="30.00"
                            tolerance2="45.00" value="-90.0"/>
                        ...
                        </angleList>
                        <histogram>
                            <bin count="0"/>
                             ...
                        </histogram>
                        <histogram_shifted>
                            <bin count="0"/>
                            ...
                        </histogram_shifted>
                    </torsionRule>
                ...
             ...
            </hierarchyClass>
             <hierarchyClass id1="N" id2="C" name="NC">
             ...
            </hierarchyClass>
            <hierarchyClass id1="S" id2="N" name="SN">
            ...
            </hierarchyClass>
            <hierarchyClass id1="C" id2="S" name="CS">
            ...
            </hierarchyClass>
            <hierarchyClass id1="C" id2="C" name="CC">
            ...
            </hierarchyClass>
            <hierarchyClass id1="S" id2="S" name="SS">
             ...
            </hierarchyClass>
        </library>
        
    The rotatable bonds in a 3D molecule are identified using a default SMARTS pattern.
    A custom SMARTS pattern may be optionally specified to detect rotatable bonds.
    Each rotatable bond is matched to a torsion rule in the torsion strain energy library.
    The strain energy is calculated for each rotatable bond using the calculation
    method, 'exact' or 'approximate', associated with the matched torsion rule.
    
    The total strain energy (TEUs) of a molecule corresponds to the sum of  'exact' and
    'approximate' strain energies calculated for all matched rotatable bonds in the
    molecule. The total strain energy is set to 'NA' for molecules containing a 'approximate'
    energy estimate for a torsion angle not observed in the torsion energy library. In
    addition, the lower and upper bounds on energy at 95% confidence interval are
    set to 'NA'.
    
    The following output files are generated after the filtering:
        
        <OutfileRoot>.sdf
        <OutfileRoot>_Filtered.sdf
        <OutfileRoot>_AlertsSummary.csv
        <OutfileRoot>_Filtered_TopRule*.sdf
        
    The last two set of outfile files, <OutfileRoot>_AlertsSummary.csv and
    <OutfileRoot>_<OutfileRoot>_AlertsSummary.csv, are only generated during filtering
    by 'MaxSingleEnergy'.
    
    The supported input file formats are: Mol (.mol), SD (.sdf, .sd)
    
    The supported output file formats are: SD (.sdf, .sd)

Options:
    -a, --alertsMode <TotalEnergy,...>  [default: TotalEnergy]
        Torsion strain energy library alert types to use for filtering molecules
        containing rotatable bonds based on the calculated values for the total
        torsion strain energy of a molecule and  the maximum single strain
        energy of a rotatable bond in a molecule.
        
        Possible values: TotalEnergy, MaxSingleEnergy, or TotalOrMaxSingleEnergy
        
        The strain energy cutoff values in terms of torsion energy units (TEUs) are
        used to filter molecules as shown below:
            
            AlertsMode                AlertsEnergyCutoffs (TEUs)
            
            TotalEnergy               >= TotalEnergyCutoff
            
            MaxSingleEnergy           >= MaxSingleEnergyCutoff
            
            TotalOrMaxSingleEnergy    >= TotalEnergyCutoff
                                      or >= MaxSingleEnergyCutoff
        
    --alertsMaxSingleEnergyCutoff <Number>  [default: 1.8]
        Maximum single strain energy (TEUs) cutoff [ Ref 153 ] for filtering molecules
        based on the maximum value of a single strain energy of a rotatable bond
        in  a molecule. This option is used during 'MaxSingleEnergy' or
        'TotalOrMaxSingleEnergy' values of '-a, --alertsMode' option.
        
        The maximum single strain energy must be greater than or equal to the
        specified cutoff value for filtering molecules.
    --alertsTotalEnergyCutoff <Number>  [default: 6.0]
        Total strain strain energy (TEUs) cutoff [ Ref 153 ] for filtering molecules
        based on total strain energy for all rotatable bonds in a molecule. This
        option is used during 'TotalEnergy' or 'TotalOrMaxSingleEnergy'
        values of '-a, --alertsMode' option.
        
        The total strain energy must be greater than or equal to the specified
        cutoff value for filtering molecules.
    --filterTorsionsNotObserved <yes or no>  [default: no]
        Filter molecules containing torsion angles not observed in torsion strain
        energy library. It's not possible to calculate torsion strain energies for
        these torsions during 'approximate' match to a specified torsion in the
        library.
        
        The 'approximate' strain energy calculation relies on the angle difference
        between a torsion angle and the torsion peaks observed for the torsion
        rules in the torsion energy library. The torsion angle is matched to a
        torsion peak based on the value of torsion angle difference. It must be
        less than or equal to the value for the second tolerance 'tolerance2'.
        Otherwise, the torsion angle is not observed in the torsion energy library
        and a value of 'NA' is assigned for torsion energy along with the lower and
        upper bounds on energy at 95% confidence interval.
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
        List torsion library information without performing any filtering.
    -m, --mode <filter or count>  [default: filter]
        Specify whether to filter molecules for torsion strain energy library [ Ref 153 ]
        alerts by matching rotatable bonds against SMARTS patterns specified for
        torsion rules to calculate torsion strain energies and write out the rest
        of the molecules to an outfile or simply count the number of matched
        molecules marked for filtering.
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
    --outfileAlerts <yes or no>  [default: yes]
        Write out alerts information to SD output files.
    --outfileAlertsMode <All or AlertsOnly>  [default: AlertsOnly]
        Write alerts information to SD output files for all alerts or only for alerts
        specified by '--AlertsMode' option. Possible values: All or AlertsOnly
        This option is only valid for 'Yes' value of '--outfileAlerts' option.
        
        The following alerts information is added to SD output files using
        'TorsionAlerts' data field:
            
            RotBondIndices TorsionIndices TorsionAngle
            Energy EnergyLowerBoundCI EnergyUpperBoundCI
            HierarchyClass HierarchySubClass TorsionRule
            EnergyMethod AngleNotObserved MaxSingleEnergyAlert
            
        The following data filelds are added to SD output files based on the value of
        '--AlertsMode' option:
            
            TotalEnergy
            TotalEnergyLowerBoundCI
            TotalEnergyUpperBoundCI
            
            MaxSingleEnergy
            MaxSingleEnergyAlertsCount
            
            AnglesNotObservedCount
            
        The 'RotBondsCount' is always added to SD output files containing both
        remaining and filtered molecules.
        
        Format:
            
            > <RotBondsCount>
            Number
            
            > <TotalEnergy>
            Number
            
            > <TotalEnergyLowerBoundCI>
            Number
            
            > <TotalEnergyUpperBoundCI>
            Number
            
            > <MaxSingleEnergy>
            Number
            
            > <MaxSingleEnergyAlertsCount>
            Number
            
            > <AnglesNotObservedCount>
            Number
            
            > <TorsionAlerts (RotBondIndices TorsionIndices TorsionAngle
                Energy EnergyLowerBoundCI EnergyUpperBoundCI
                HierarchyClass HierarchySubClass TorsionRule
                EnergyMethod AngleNotObserved MaxSingleEnergyAlert)>
            AtomIndex2,AtomIndex3  AtomIndex1,AtomIndex2,AtomIndex3,AtomIndex4
            Angle  Energy EnergyLowerBoundCI EnergyUpperBoundCI
            ClassName SubClassName SMARTS EnergyMethod Yes|No|NA Yes|No|NA
            ... ... ...
            ... ... ...
            
        A set of 12 values is written out as value of 'TorsionAlerts' data field for
        each torsion in a molecule. The space character is used as a delimiter
        to separate values with in a set and across set. The comma character
        is used to delimit multiple values for each value in a set.
        
        The 'RotBondIndices' and 'TorsionIndices' contain 2 and 4 comma delimited
        values representing atom indices for a rotatable bond and the matched
        torsion.
        
        The 'Energy' value is the estimated strain energy for the matched torsion.
        The 'EnergyLowerBoundCI' and 'EnergyUpperBoundCI' represent lower and
        bound energy estimates at 95% confidence interval. The 'EnergyMethod',
        exact or approximate, corresponds to the method employed to estimate
        torsion strain energy.
        
        The 'AngleNotObserved' is only valid for 'approximate' value of 'EnergyMethod'.
        It has three possible values: Yes, No, or NA. The 'Yes' value indicates that
        the 'TorsionAngle' is outside the 'tolerance2' of all peaks for the matched
        torsion rule in the torsion library.
        
        The 'MaxSingleEnergyAlert' is valid for the following values of '-a, --alertsMode'
        option: 'MaxSingleEnergy' or 'TotalOrMaxSingleEnergy'. It has three possible
        values: Yes, No, or NA. It's set to 'NA' for 'Yes' or 'NA' values of
        'AngleNotObserved'. The 'Yes' value indicates that the estimated torsion
        energy is greater than the specified value for '--alertsMaxSingleEnergyCutoff'
        option.
        
        For example:
            
            >  <RotBondsCount>  (1) 
            14
            
            >  <TotalEnergy>  (1) 
            6.8065
            
            >  <TotalEnergyLowerBoundCI>  (1) 
            5.9340
            
            >  <TotalEnergyUpperBoundCI>  (1) 
            NA
            
            >  <MaxSingleEnergy>  (1) 
            1.7108
            
            >  <MaxSingleEnergyAlertsCount>  (1) 
            0
            
            >  <AnglesNotObservedCount>  (1) 
            0
             
            >  <TorsionAlerts(RotBondIndices TorsionIndices TorsionAngle Energy
                EnergyLowerBoundCI EnergyUpperBoundCI HierarchyClass
                HierarchySubClass TorsionRule EnergyMethod AngleNotObserved
                MaxSingleEnergyAlert)>  (1) 
            2,1 48,2,1,0 61.90 0.0159 -0.0320 0.0674 CO Ether [O:1][CX4:2]!
            @[O:3][CX4:4] Exact NA No 2,3 1,2,3,4 109.12 1.5640 1.1175 NA CC
            None/[CX4][CX3] [O:1][CX4:2]!@[CX3:3]=[O:4] Exact NA No
            ... ... ...
            
    --outfileFiltered <yes or no>  [default: yes]
        Write out a file containing filtered molecules. Its name is automatically
        generated from the specified output file. Default: <OutfileRoot>_
        Filtered.<OutfileExt>.
    --outfilesFilteredByRules <yes or no>  [default: auto]
        Write out SD files containing filtered molecules for individual torsion
        rules triggering alerts in molecules. The name of SD files are automatically
        generated from the specified output file. Default file names: <OutfileRoot>_
        Filtered_TopRule*.sdf.
        
        Default value: 'yes' for 'MaxSingleEnergy' of '-a, --alertsMode' option';
        otherwise, 'no'.
        
        The output files are only generated for 'MaxSingleEnergy' of
        '-a, --alertsMode' option.
        
        The following alerts information is added to SD output files:
            
            > <RotBondsCount>
            Number
            
            > <TotalEnergy>
            Number
            
            > <TotalEnergyLowerBoundCI>
            Number
            
            > <TotalEnergyUpperBoundCI>
            Number
            
            > <MaxSingleEnergy>
            Number
            
            > <MaxSingleEnergyAlertsCount>
            Number
            
            > <AnglesNotObservedCount>
            Number
            
            >  <TorsionRule (HierarchyClass HierarchySubClass TorsionRule
                EnergyMethod)> 
            ClassName SubClassName EnergyMethod SMARTS
             ... ... ...
            
            > <TorsionRuleMaxSingleEnergyAlertsCount>
            Number
            
            > <TorsionRuleAnglesNotObservedCount>
            Number
            
            >  <TorsionRuleAlerts (RotBondIndices TorsionIndices TorsionAngle
                Energy EnergyLowerBoundCI EnergyUpperBoundCI
                AngleNotObserved MaxSingleEnergyAlert)>
            AtomIndex2,AtomIndex3  AtomIndex1,AtomIndex2,AtomIndex3,AtomIndex4
            Angle Energy EnergyLowerBoundCI EnergyUpperBoundCI EnergyMethod
            Yes|No|NA Yes|No|NA
             ... ... ...
            
        For example:
            
            >  <RotBondsCount>  (1) 
            8
            
            >  <TotalEnergy>  (1) 
            6.1889
            
            >  <TotalEnergyLowerBoundCI>  (1) 
            5.1940
            
            >  <TotalEnergyUpperBoundCI>  (1) 
            NA
            
            >  <MaxSingleEnergy>  (1) 
            1.9576
            
            >  <MaxSingleEnergyAlertsCount>  (1) 
            1
            
            >  <AnglesNotObservedCount>  (1) 
            0
            
            >  <TorsionRule (HierarchyClass HierarchySubClass TorsionRule
                EnergyMethod)>  (1) 
            CC None/[CX4:2][CX4:3] [!#1:1][CX4:2]!@[CX4:3][!#1:4] Exact
            
            >  <TorsionRuleMaxSingleEnergyAlertsCount>  (1) 
            0
            
            >  <TorsionRuleAnglesNotObservedCount>  (1) 
            0
            
            >  <TorsionRuleAlerts (RotBondIndices TorsionIndices TorsionAngle
                Energy EnergyLowerBoundCI EnergyUpperBoundCI AngleNotObserved
               MaxSingleEnergyAlert)>  (1) 
            1,3 0,1,3,4 72.63 0.8946 0.8756 0.9145 NA No
            
    --outfilesFilteredByRulesMaxCount <All or number>  [default: 10]
        Write out SD files containing filtered molecules for specified number of
        top N torsion rules triggering alerts for the largest number of molecules
        or for all torsion rules triggering alerts across all molecules.
        
        These output files are only generated for 'MaxSingleEnergy' value of
        '-a, --alertsMode' option.
    --outfileSummary <yes or no>  [default: auto]
        Write out a CVS text file containing summary of torsions rules responsible
        for triggering torsion alerts. Its name is automatically generated from the
        specified output file. Default: <OutfileRoot>_AlertsSummary.csv.
        
        Default value: 'yes' for 'MaxSingleEnergy' of '-a, --alertsMode' option';
        otherwise, 'no'.
        
        The summary output file is only generated for 'MaxSingleEnergy' of
        '-a, --alertsMode' option.
        
        The following alerts information is written to summary text file:
            
            TorsionRule, HierarchyClass, HierarchySubClass, EnergyMethod,
            MaxSingleEnergyTorsionAlertTypes, MaxSingleEnergyTorsionAlertCount,
            MaxSingleEnergyTorsionAlertMolCount
             
        The double quotes characters are removed from SMART patterns before
        before writing them to a CSV file. In addition, the torsion rules are sorted by
        TorsionAlertMolCount.
    --outfileSDFieldLabels <Type,Label,...>  [default: auto]
        A comma delimited list of SD data field type and label value pairs for writing
        torsion alerts information along with molecules to SD files.
        
        The supported SD data field label type along with their default values are
        shown below:
            
            For all SD files:
            
            RotBondsCountLabel, RotBondsCount,
            
            TotalEnergyLabel, TotalEnergy,
            TotalEnergyLowerBoundCILabel, TotalEnergyLowerBoundCI,
            TotalEnergyUpperBoundCILabel, TotalEnergyUpperBoundCI,
            
            MaxSingleEnergyLabel, MaxSingleEnergy,
            MaxSingleEnergyAlertsCountLabel,
                MaxSingleEnergyAlertsCount
            
            AnglesNotObservedCountLabel,
                AnglesNotObservedCount
            
            TorsionAlertsLabel, TorsionAlerts(RotBondIndices TorsionIndices
                TorsionAngle Energy EnergyLowerBoundCI EnergyUpperBoundCI
                HierarchyClass HierarchySubClass TorsionRule
                EnergyMethod AngleNotObserved)
            
            For individual SD files filtered by torsion rules:
            
            TorsionRuleLabel, TorsionRule (HierarchyClass HierarchySubClass
                EnergyMethod TorsionRule)
            TorsionRuleMaxSingleEnergyAlertsCountLabel,
                TorsionRuleMaxSingleEnergyAlertsCount,
            TorsionRuleAnglesNotObservedCountLabel,
                TorsionRuleAnglesNotObservedCount,
            TorsionRuleAlertsLabel, TorsionRuleAlerts (RotBondIndices
                TorsionIndices TorsionAngle Energy EnergyLowerBoundCI
                EnergyUpperBoundCI EnergyMethod AngleObserved)
            
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: kekulize,yes
            
    --overwrite
        Overwrite existing files.
    --precision <number>  [default: 4]
        Floating point precision for writing torsion strain energy values.
    -r, --rotBondsSMARTSMode <NonStrict, SemiStrict,...>  [default: SemiStrict]
        SMARTS pattern to use for identifying rotatable bonds in a molecule
        for matching against torsion rules in the torsion library. Possible values:
        NonStrict, SemiStrict, Strict or Specify. The rotatable bond SMARTS matches
        are filtered to ensure that each atom in the rotatable bond is attached to
        at least two heavy atoms.
        
        The following SMARTS patterns are used to identify rotatable bonds for
        different modes:
            
            NonStrict: [!$(*#*)&!D1]-&!@[!$(*#*)&!D1]
            
            SemiStrict:
            [!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)
            &!$(C([CH3])([CH3])[CH3])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)
            &!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]
            
            Strict:
            [!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)
            &!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])
            &!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])
            &!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)
            &!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]
            
        The 'NonStrict' and 'Strict' SMARTS patterns are available in RDKit. The 
        'NonStrict' SMARTS pattern corresponds to original Daylight SMARTS
        specification for rotatable bonds. The 'SemiStrict' SMARTS pattern is 
        derived from 'Strict' SMARTS patterns for its usage in this script.
        
        You may use any arbitrary SMARTS pattern to identify rotatable bonds by
        choosing 'Specify' value for '-r, --rotBondsSMARTSMode' option and providing its
        value via '--rotBondsSMARTSPattern' option.
    --rotBondsSMARTSPattern <SMARTS>
        SMARTS pattern for identifying rotatable bonds. This option is only valid
        for 'Specify' value of '-r, --rotBondsSMARTSMode' option.
    -t, --torsionEnergyLibraryFile <FileName or auto>  [default: auto]
        Specify a XML file name containing data for torsion starin energy library
        hierarchy or use default file, TorsionEnergyLibrary.xml, available in
        MAYACHEMTOOLS/lib/data directory.
        
        The format of data in local XML file must match format of the data in Torsion
        Library [ Ref 153 ] file available in MAYACHEMTOOLS data directory.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To filter molecules containing rotatable bonds with total strain energy value
    of >= 6.0 (TEUs) based on torsion rules in the torsion energy library and write
    write out SD files containing remaining and filtered molecules, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py -i Sample3D.sdf
          -o Sample3DOut.sdf

    To filter molecules containing any rotatable bonds with strain energy value of
    >= 1.8 (TEUs) based on torsion rules in the torsion energy library and write out
    SD files containing remaining and filtered molecules, and individual SD files for
    torsion rules triggering alerts along with appropriate torsion information for
    red alerts, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py -a MaxSingleEnergy
          -i Sample3D.sdf -o Sample3DOut.sdf

    To filter molecules containing rotatable bonds with total strain energy value
    of >= 6.0 (TEUs) or any single strain energy value of >= 1.8 (TEUs) and write out
    SD files containing remaining and filtered molecules, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py -a TotalOrMaxSingleEnergy
          -i Sample3D.sdf -o Sample3DOut.sdf

    To filter molecules containing rotatable bonds with specific cutoff values for
    total or single torsion strain energy and write out SD files containing
    remaining and filtered molecules, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py -a TotalOrMaxSingleEnergy
          -i Sample3D.sdf -o Sample3DOut.sdf --alertsTotalEnergyCutoff 6.0
          --alertsMaxSingleEnergyCutoff 1.8

    To run the first example for filtering molecules and writing out torsion
    information for all alert types to SD files, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py -i Sample3D.sdf
          -o Sample3DOut.sdf --outfileAlertsMode All

    To run the first example for filtering molecules in multiprocessing mode on
    all available CPUs without loading all data into memory and write out SD files,
    type:

        % RDKitFilterTorsionStrainEnergyAlerts.py --mp yes -i Sample3D.sdf
         -o Sample3DOut.sdf

    To run the first example for filtering molecules in multiprocessing mode on
    all available CPUs by loading all data into memory and write out a SD files,
    type:

        % RDKitFilterTorsionStrainEnergyAlerts.py  --mp yes --mpParams
          "inputDataMode, InMemory" -i Sample3D.sdf  -o Sample3DOut.sdf

    To run the first example for filtering molecules in multiprocessing mode on
    specific number of CPUs and chunksize without loading all data into memory
    and write out SD files, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py --mp yes --mpParams
          "inputDataMode,lazy,numProcesses,4,chunkSize,8"  -i Sample3D.sdf
          -o Sample3DOut.sdf

    To list information about default torsion library file without performing any
    filtering, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py -l

    To list information about a local torsion library XML file without performing
    any, filtering, type:

        % RDKitFilterTorsionStrainEnergyAlerts.py --torsionEnergyLibraryFile
          TorsionStrainEnergyLibrary.xml -l

Author:
    Manish Sud (msud@san.rr.com)

Collaborator:
    Pat Walters

See also:
    RDKitFilterChEMBLAlerts.py, RDKitFilterPAINS.py, RDKitFilterTorsionLibraryAlerts.py,
    RDKitConvertFileFormat.py, RDKitSearchSMARTS.py

Copyright:
    Copyright (C) 2022 Manish Sud. All rights reserved.

    This script uses the torsion strain energy library developed by Gu, S.;
    Smith, M. S.; Yang, Y.; Irwin, J. J.; Shoichet, B. K. [ Ref 153 ].

    The torsion strain enegy library is based on the Torsion Library jointly
    developed by the University of Hamburg, Center for Bioinformatics,
    Hamburg, Germany and F. Hoffmann-La-Roche Ltd., Basel, Switzerland.

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
