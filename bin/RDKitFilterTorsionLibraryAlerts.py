#!/usr/bin/env python
#
# File: RDKitFilterTorsionLibraryAlerts.py
# Author: Manish Sud <msud@san.rr.com>
#
# Collaborator: Pat Walters
#
# Acknowledgments: Wolfgang Guba, Patrick Penner, Levi Pierce
#
# Copyright (C) 2022 Manish Sud. All rights reserved.
#
# This script uses the Torsion Library jointly developed by the University
# of Hamburg, Center for Bioinformatics, Hamburg, Germany and
# F. Hoffmann-La-Roche Ltd., Basel, Switzerland.
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
import numpy as np

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
    """Filter molecules using SMARTS torsion rules in the torsion library file."""

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
    MiscUtil.PrintInfo("Encoding options info and rotatable bond pattern molecule...")
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

        if MatchInfo is None or len(MatchInfo) == 0:
            AlertType, TorsionAtomIndices, TorsionAngle, TorsionAngleViolation, HierarchyClassName, HierarchySubClassName, TorsionRuleNodeID, TorsionRulePeaks, TorsionRuleTolerances1, TorsionRuleTolerances2, TorsionRuleSMARTS = [None] * 11
        else:
            AlertType, TorsionAtomIndices, TorsionAngle, TorsionAngleViolation, HierarchyClassName, HierarchySubClassName, TorsionRuleNodeID, TorsionRulePeaks, TorsionRuleTolerances1, TorsionRuleTolerances2, TorsionRuleSMARTS = MatchInfo

        # Track alerts information...
        RotBondsAlertsInfo["IDs"].append(ID)
        RotBondsAlertsInfo["MatchStatus"][ID] = MatchStatus
        RotBondsAlertsInfo["AlertTypes"][ID] = AlertType
        RotBondsAlertsInfo["AtomIndices"][ID] = AtomIndices
        RotBondsAlertsInfo["TorsionAtomIndices"][ID] = TorsionAtomIndices
        RotBondsAlertsInfo["TorsionAngles"][ID] = TorsionAngle
        RotBondsAlertsInfo["TorsionAngleViolations"][ID] = TorsionAngleViolation
        RotBondsAlertsInfo["HierarchyClassNames"][ID] = HierarchyClassName
        RotBondsAlertsInfo["HierarchySubClassNames"][ID] = HierarchySubClassName
        RotBondsAlertsInfo["TorsionRuleNodeID"][ID] = TorsionRuleNodeID
        RotBondsAlertsInfo["TorsionRulePeaks"][ID] = TorsionRulePeaks
        RotBondsAlertsInfo["TorsionRuleTolerances1"][ID] = TorsionRuleTolerances1
        RotBondsAlertsInfo["TorsionRuleTolerances2"][ID] = TorsionRuleTolerances2
        RotBondsAlertsInfo["TorsionRuleSMARTS"][ID] = TorsionRuleSMARTS

        #  Count alert types...
        if AlertType is not None:
            if AlertType not in RotBondsAlertsInfo["Count"]:
                RotBondsAlertsInfo["Count"][AlertType] = 0
            RotBondsAlertsInfo["Count"][AlertType] += 1
    
    # Setup alert status for rotatable bonds...
    RotBondsAlertsStatus = False
    AlertsCount = 0
    for ID in RotBondsInfo["IDs"]:
        if RotBondsAlertsInfo["AlertTypes"][ID] in OptionsInfo["SpecifiedAlertsModeList"]:
            AlertsCount += 1
            if AlertsCount >= OptionsInfo["MinAlertsCount"]:
                RotBondsAlertsStatus = True
                break
    
    return (RotBondsAlertsStatus, RotBondsAlertsInfo)

def InitializeRotatableBondsAlertsInfo():
    """Initialize alerts information for rotatable bonds."""
    
    RotBondsAlertsInfo = {}
    RotBondsAlertsInfo["IDs"] = []

    for DataLabel in ["MatchStatus", "AlertTypes", "AtomIndices", "TorsionAtomIndices", "TorsionAngles", "TorsionAngleViolations", "HierarchyClassNames", "HierarchySubClassNames", "TorsionRuleNodeID", "TorsionRulePeaks", "TorsionRuleTolerances1", "TorsionRuleTolerances2", "TorsionRuleSMARTS", "Count"]:
        RotBondsAlertsInfo[DataLabel] = {}
        
    return RotBondsAlertsInfo
    
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
    
    #  Retrieve torsion matched to rotatable bond...
    TorsionAtomIndices, TorsionAngle = MatchTorsionRuleToRotatableBond(Mol, RotBondAtomIndices, ElementNode)
    if TorsionAtomIndices is None:
        return (False, None)
    
    # Setup torsion angles info for matched torsion rule...
    TorsionAnglesInfo = TorsionLibraryUtil.SetupTorsionRuleAnglesInfo(TorsionLibraryInfo, ElementNode)
    if TorsionAnglesInfo is None:
        return (False, None)
    
    #   Setup torsion alert type and angle violation...
    AlertType, TorsionAngleViolation = SetupTorsionAlertTypeForRotatableBond(TorsionAnglesInfo, TorsionAngle)

    # Setup hierarchy class and subclass names...
    HierarchyClassName, HierarchySubClassName = TorsionLibraryUtil.SetupHierarchyClassAndSubClassNamesForRotatableBond(TorsionLibraryInfo)
    
    # Setup rule node ID...
    TorsionRuleNodeID = ElementNode.get("NodeID")
    
    # Setup SMARTS...
    TorsionRuleSMARTS = ElementNode.get("smarts")
    if " " in TorsionRuleSMARTS:
        TorsionRuleSMARTS = TorsionRuleSMARTS.replace(" ", "")
    
    # Setup torsion peaks and tolerances...
    TorsionRulePeaks = TorsionAnglesInfo["ValuesList"]
    TorsionRuleTolerances1 = TorsionAnglesInfo["Tolerances1List"]
    TorsionRuleTolerances2 = TorsionAnglesInfo["Tolerances2List"]
    
    MatchInfo = [AlertType, TorsionAtomIndices, TorsionAngle, TorsionAngleViolation, HierarchyClassName, HierarchySubClassName, TorsionRuleNodeID, TorsionRulePeaks, TorsionRuleTolerances1, TorsionRuleTolerances2, TorsionRuleSMARTS]

    # Setup match status...
    MatchStatus = True
    
    return (MatchStatus, MatchInfo)

def MatchTorsionRuleToRotatableBond(Mol, RotBondAtomIndices, ElementNode):
    """Retrieve matched torsion for torsion rule matched to rotatable bond."""

    # Get torsion matches...
    TorsionMatches = GetMatchesForTorsionRule(Mol, ElementNode)
    if TorsionMatches is None or len(TorsionMatches) == 0:
        return (None, None)

    # Identify the first torsion match corresponding to central atoms in RotBondAtomIndices...
    RotBondAtomIndex1, RotBondAtomIndex2 = RotBondAtomIndices
    for TorsionMatch in TorsionMatches:
        CentralAtomIndex1 = TorsionMatch[1]
        CentralAtomIndex2 = TorsionMatch[2]
        
        if ((CentralAtomIndex1 == RotBondAtomIndex1 and CentralAtomIndex2 == RotBondAtomIndex2) or (CentralAtomIndex1 == RotBondAtomIndex2 and CentralAtomIndex2 == RotBondAtomIndex1)):
            TorsionAngle = CalculateTorsionAngle(Mol, TorsionMatch)
            
            return (TorsionMatch, TorsionAngle)
            
    return (None, None)

def CalculateTorsionAngle(Mol, TorsionMatch):
    """Calculate torsion angle."""

    if type(TorsionMatch[3]) is list:
        return CalculateTorsionAngleUsingNitrogenLonePairPosition(Mol, TorsionMatch)

    # Calculate torsion angle using torsion atom indices..
    MolConf = Mol.GetConformer(0)
    TorsionAngle = rdMolTransforms.GetDihedralDeg(MolConf, TorsionMatch[0], TorsionMatch[1], TorsionMatch[2], TorsionMatch[3])
    TorsionAngle = round(TorsionAngle, 2)
    
    return TorsionAngle

def CalculateTorsionAngleUsingNitrogenLonePairPosition(Mol, TorsionMatch):
    """Calculate torsion angle using nitrogen lone pair positon."""

    # Setup a carbon atom as position holder for lone pair position...
    TmpMol = Chem.RWMol(Mol)
    LonePairAtomIndex = TmpMol.AddAtom(Chem.Atom(6))
    
    TmpMolConf = TmpMol.GetConformer(0)
    TmpMolConf.SetAtomPosition(LonePairAtomIndex, TorsionMatch[3])

    TorsionAngle = rdMolTransforms.GetDihedralDeg(TmpMolConf, TorsionMatch[0], TorsionMatch[1], TorsionMatch[2], LonePairAtomIndex)
    TorsionAngle = round(TorsionAngle, 2)

    return TorsionAngle

def GetMatchesForTorsionRule(Mol, ElementNode):
    """Get matches for torsion rule."""

    # Match torsions...
    TorsionMatches = None
    if IsNitogenLonePairTorsionRule(ElementNode):
        TorsionMatches = GetSubstructureMatchesForNitrogenLonePairTorsionRule(Mol, ElementNode)
    else:
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
        if type(TorsionMatch[3]) is int:
            # May contains a list for type two nitrogen lone pair match...
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

def GetSubstructureMatchesForNitrogenLonePairTorsionRule(Mol, ElementNode):
    """Get substructure matches for a torsion rule containing N_lp."""

    if IsTypeOneNitogenLonePairTorsionRule(ElementNode):
        return GetSubstructureMatchesForTypeOneNitrogenLonePairTorsionRule(Mol, ElementNode)
    elif IsTypeTwoNitogenLonePairTorsionRule(ElementNode):
        return GetSubstructureMatchesForTypeTwoNitrogenLonePairTorsionRule(Mol, ElementNode)
    
    return None

def GetSubstructureMatchesForTypeOneNitrogenLonePairTorsionRule(Mol, ElementNode):
    """Get substructure matches for a torsion rule containing N_lp and four mapped atoms."""

    # For example:
    #    [CX4:1][CX4H2:2]!@[NX3;"N_lp":3][CX4:4]
    #    [C:1][CX4H2:2]!@[NX3;"N_lp":3][C:4]
    #    ... ... ...

    TorsionRuleNodeID = ElementNode.get("NodeID")
    TorsionPatternMol, LonePairMapNumber = SetupNitrogenLonePairTorsionRuleElementInfo(ElementNode, TorsionRuleNodeID)
    
    if TorsionPatternMol is None:
        return None
    
    if LonePairMapNumber is None:
        return None
    
    # Match torsions...
    TorsionMatches = RDKitUtil.FilterSubstructureMatchesByAtomMapNumbers(Mol, TorsionPatternMol, Mol.GetSubstructMatches(TorsionPatternMol, useChirality = False))

    # Filter matches...
    FiltertedTorsionMatches = []
    for TorsionMatch in TorsionMatches:
        if len(TorsionMatch) != 4:
            continue
        
        # Check for Nitogen atom at LonePairMapNumber...
        LonePairNitrogenAtom = Mol.GetAtomWithIdx(TorsionMatch[LonePairMapNumber - 1])
        if LonePairNitrogenAtom.GetSymbol() != "N":
            continue
        
        # Make sure LonePairNitrogenAtom is planar...
        #  test
        PlanarityStatus = IsLonePairNitrogenAtomPlanar(Mol, LonePairNitrogenAtom)
        if PlanarityStatus is None:
            continue
        
        if  not PlanarityStatus:
            continue
        
        FiltertedTorsionMatches.append(TorsionMatch)
        
    return FiltertedTorsionMatches
    
def GetSubstructureMatchesForTypeTwoNitrogenLonePairTorsionRule(Mol, ElementNode):
    """Get substructure matches for a torsion rule containing N_lp and three mapped atoms."""

    # For example:
    # [!#1:1][CX4:2]!@[NX3;"N_lp":3]
    # [!#1:1][$(S(=O)=O):2]!@["N_lp":3]@[Cr3]
    # [C:1][$(S(=O)=O):2]!@["N_lp":3]
    # [c:1][$(S(=O)=O):2]!@["N_lp":3]
    # [!#1:1][$(S(=O)=O):2]!@["N_lp":3]
    #    ... ... ...
    
    TorsionRuleNodeID = ElementNode.get("NodeID")
    TorsionPatternMol, LonePairMapNumber = SetupNitrogenLonePairTorsionRuleElementInfo(ElementNode, TorsionRuleNodeID)
    
    if TorsionPatternMol is None:
        return None
    
    if not IsValidTypeTwoNitrogenLonePairMapNumber(LonePairMapNumber):
        return None

    # Match torsions...
    TorsionMatches = RDKitUtil.FilterSubstructureMatchesByAtomMapNumbers(Mol, TorsionPatternMol, Mol.GetSubstructMatches(TorsionPatternMol, useChirality = False))

    # Filter matches...
    FiltertedTorsionMatches = []
    for TorsionMatch in TorsionMatches:
        if len(TorsionMatch) != 3:
            continue
        
        # Check for Nitogen atom at LonePairMapNumber...
        LonePairNitrogenAtom = Mol.GetAtomWithIdx(TorsionMatch[LonePairMapNumber - 1])
        if LonePairNitrogenAtom.GetSymbol() != "N":
            continue

        # Make sure LonePairNitrogenAtom is not planar...
        PlanarityStatus = IsLonePairNitrogenAtomPlanar(Mol, LonePairNitrogenAtom)
        
        if PlanarityStatus is None:
            continue
        
        if  PlanarityStatus:
            continue
        
        # Calculate lone pair coordinates for a non-planar nitrogen...
        LonePairPosition = CalculateLonePairCoordinatesForNitrogenAtom(Mol, LonePairNitrogenAtom)
        if LonePairPosition is None:
            continue

        # Append lone pair coodinate list to list of torsion match containing atom indices...
        TorsionMatch.append(LonePairPosition)

        # Track torsion matches...
        FiltertedTorsionMatches.append(TorsionMatch)
        
    return FiltertedTorsionMatches

def SetupNitrogenLonePairTorsionRuleElementInfo(ElementNode, TorsionRuleNodeID):
    """Setup pattern molecule and lone pair map number for type one and type
    two nitrogen lone pair rules."""

    TorsionPatternMol, LonePairMapNumber = [None] * 2
    
    if TorsionRuleNodeID in TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"]:
        TorsionPatternMol = TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"][TorsionRuleNodeID]
        LonePairMapNumber = TorsionLibraryInfo["DataCache"]["TorsionRuleLonePairMapNumber"][TorsionRuleNodeID]
    else:
        # Setup torsion pattern...
        TorsionSMARTSPattern = ElementNode.get("smarts")
        TorsionSMARTSPattern, LonePairMapNumber = ProcessSMARTSForNitrogenLonePairTorsionRule(TorsionSMARTSPattern)
        
        # Setup torsion pattern mol...
        TorsionPatternMol = Chem.MolFromSmarts(TorsionSMARTSPattern)
        if TorsionPatternMol is None:
            MiscUtil.PrintWarning("Ignoring torsion rule element containing invalid map atoms numbers in SMARTS pattern %s" % TorsionSMARTSPattern)

        # Cache data...
        TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"][TorsionRuleNodeID] = TorsionPatternMol
        TorsionLibraryInfo["DataCache"]["TorsionRuleLonePairMapNumber"][TorsionRuleNodeID] = LonePairMapNumber

    return (TorsionPatternMol, LonePairMapNumber)

def IsLonePairNitrogenAtomPlanar(Mol, NitrogenAtom):
    """Check for the planarity of nitrogen atom and its three neighbors."""

    AllowHydrogenNbrs = OptionsInfo["NitrogenLonePairParams"]["AllowHydrogenNbrs"]
    Tolerance = OptionsInfo["NitrogenLonePairParams"]["PlanarityTolerance"]
    
    # Get neighbors...
    if AllowHydrogenNbrs:
        AtomNeighbors = NitrogenAtom.GetNeighbors()
    else:
        AtomNeighbors = RDKitUtil.GetHeavyAtomNeighbors(NitrogenAtom)

    if len(AtomNeighbors) != 3:
        return None

    # Setup atom positions...
    AtomPositions = []
    MolAtomsPositions = RDKitUtil.GetAtomPositions(Mol)

    # Neighbor positions...
    for AtomNbr in AtomNeighbors:
        AtomNbrIndex = AtomNbr.GetIdx()
        AtomPositions.append(MolAtomsPositions[AtomNbrIndex])

    # Nitrogen position...
    NitrogenAtomIndex = NitrogenAtom.GetIdx()
    AtomPositions.append(MolAtomsPositions[NitrogenAtomIndex])
    
    Status =  AreFourPointsCoplanar(AtomPositions[0], AtomPositions[1], AtomPositions[2], AtomPositions[3], Tolerance)
    
    return Status

def AreFourPointsCoplanar(Point1, Point2, Point3, Point4, Tolerance = 1.0):
    """Check whether four points are coplanar with in the threshold of 1 degree."""

    # Setup  normalized direction vectors...
    VectorP2P1 = NormalizeVector(np.subtract(Point2, Point1))
    VectorP3P1 = NormalizeVector(np.subtract(Point3, Point1))
    VectorP1P4 = NormalizeVector(np.subtract(Point1, Point4))

    # Calculate angle between VectorP1P4 and normal to vectors VectorP2P1 and VectorP3P1...
    PlaneP1P2P3Normal = NormalizeVector(np.cross(VectorP2P1, VectorP3P1))
    PlanarityAngle = np.arccos(np.clip(np.dot(PlaneP1P2P3Normal, VectorP1P4), -1.0, 1.0))
    
    Status = math.isclose(PlanarityAngle, math.radians(90), abs_tol=math.radians(Tolerance))

    return Status

def NormalizeVector(Vector):
    """Normalize vector."""

    Norm = np.linalg.norm(Vector)
    
    return Vector if math.isclose(Norm, 0.0, abs_tol = 1e-08) else Vector/Norm

def CalculateLonePairCoordinatesForNitrogenAtom(Mol, NitrogenAtom):
    """Calculate approximate lone pair coordinates for non-plannar nitrogen atom."""

    AllowHydrogenNbrs = OptionsInfo["NitrogenLonePairParams"]["AllowHydrogenNbrs"]
    
    # Get neighbors...
    if AllowHydrogenNbrs:
        AtomNeighbors = NitrogenAtom.GetNeighbors()
    else:
        AtomNeighbors = RDKitUtil.GetHeavyAtomNeighbors(NitrogenAtom)
        
    if len(AtomNeighbors) != 3:
        return None
    
    # Setup positions for nitrogen and its neghbors...
    MolAtomsPositions = RDKitUtil.GetAtomPositions(Mol)

    NitrogenPosition = MolAtomsPositions[NitrogenAtom.GetIdx()]
    NbrPositions = []
    for AtomNbr in AtomNeighbors:
        NbrPositions.append(MolAtomsPositions[AtomNbr.GetIdx()])
    Nbr1Position, Nbr2Position, Nbr3Position = NbrPositions

    # Setup  normalized direction vectors...
    VectorP2P1 = NormalizeVector(np.subtract(Nbr2Position, Nbr1Position))
    VectorP3P1 = NormalizeVector(np.subtract(Nbr3Position, Nbr1Position))
    VectorP1P4 = NormalizeVector(np.subtract(Nbr1Position, NitrogenPosition))

    # Calculate angle between VectorP1P4 and normal to vectors VectorP2P1 and VectorP3P1...
    PlaneP1P2P3Normal = NormalizeVector(np.cross(VectorP2P1, VectorP3P1))
    PlanarityAngle = np.arccos(np.clip(np.dot(PlaneP1P2P3Normal, VectorP1P4), -1.0, 1.0))

    # Check for reversing the direction of the normal...
    if PlanarityAngle < math.radians(90):
        PlaneP1P2P3Normal = PlaneP1P2P3Normal * -1

    # Add normal to nitrogen cooridnates for the approximate coordinates of the
    # one pair. The exact VSEPR coordinates of the lone pair are not necessary to
    # calculate the torsion angle...
    LonePairPosition = NitrogenPosition + PlaneP1P2P3Normal
    
    return list(LonePairPosition)

def ProcessSMARTSForNitrogenLonePairTorsionRule(SMARTSPattern):
    """Process SMARTS pattern for a torion rule containing N_lp."""

    LonePairMapNumber = GetNitrogenLonePairMapNumber(SMARTSPattern)
    
    # Remove double quotes around N_lp..
    SMARTSPattern = re.sub("\"N_lp\"", "N_lp", SMARTSPattern, re.I)
    
    # Remove N_lp specification from SMARTS pattern for torsion rule...
    if re.search("\[N_lp", SMARTSPattern, re.I):
        # Handle missing NX3...
        SMARTSPattern = re.sub("\[N_lp", "[NX3", SMARTSPattern)
    else:
        SMARTSPattern = re.sub(";N_lp", "", SMARTSPattern)
        
    return (SMARTSPattern, LonePairMapNumber)

def GetNitrogenLonePairMapNumber(SMARTSPattern):
    """Get atom map number for nitrogen involved in N_lp."""
    
    LonePairMapNumber = None
    
    SMARTSPattern = re.sub("\"N_lp\"", "N_lp", SMARTSPattern, re.I)
    MatchedMappedAtoms = re.findall("N_lp:[0-9]", SMARTSPattern, re.I)
    
    if len(MatchedMappedAtoms) == 1:
        LonePairMapNumber = int(re.sub("N_lp:", "", MatchedMappedAtoms[0]))

    return LonePairMapNumber

def IsNitogenLonePairTorsionRule(ElementNode):
    """Check for the presence of N_lp in SMARTS pattern for a torsion rule."""

    if "N_lp" not  in ElementNode.get("smarts"):
        return False
    
    LonePairMatches = re.findall("N_lp", ElementNode.get("smarts"), re.I)
    
    return True if len(LonePairMatches) == 1 else False
    
def IsTypeOneNitogenLonePairTorsionRule(ElementNode):
    """Check for the presence four mapped atoms in a SMARTS pattern containing
    N_lp for a torsion rule."""

    # For example:
    #    [CX4:1][CX4H2:2]!@[NX3;"N_lp":3][CX4:4]
    #    [C:1][CX4H2:2]!@[NX3;"N_lp":3][C:4]
    #    ... ... ...
    
    MatchedMappedAtoms = re.findall(":[0-9]", ElementNode.get("smarts"), re.I)

    return True if len(MatchedMappedAtoms) == 4 else False
    
def IsTypeTwoNitogenLonePairTorsionRule(ElementNode):
    """Check for the presence three mapped atoms in a SMARTS pattern containing
    N_lp for a torsion rule."""

    # For example:
    # [!#1:1][CX4:2]!@[NX3;"N_lp":3]
    # [!#1:1][$(S(=O)=O):2]!@["N_lp":3]@[Cr3]
    # [C:1][$(S(=O)=O):2]!@["N_lp":3]
    # [c:1][$(S(=O)=O):2]!@["N_lp":3]
    # [!#1:1][$(S(=O)=O):2]!@["N_lp":3]
    #
    
    MatchedMappedAtoms = re.findall(":[0-9]", ElementNode.get("smarts"), re.I)

    return True if len(MatchedMappedAtoms) == 3 else False

def IsValidTypeTwoNitogenLonePairTorsionRule(ElementNode):
    """Validate atom map number for nitrogen involved in N_lp for type two nitrogen
    lone pair torsion rule."""

    LonePairMapNumber = GetNitrogenLonePairMapNumber(ElementNode.get("smarts"))
    
    return IsValidTypeTwoNitrogenLonePairMapNumber(LonePairMapNumber)

def IsValidTypeTwoNitrogenLonePairMapNumber(LonePairMapNumber):
    """Check that  the atom map number is 3."""

    return True if LonePairMapNumber is not None and LonePairMapNumber == 3 else False

def SetupTorsionAlertTypeForRotatableBond(TorsionAnglesInfo, TorsionAngle):
    """Setup torsion alert type and angle violation for a rotatable bond."""

    TorsionCategory, TorsionAngleViolation = [None, None]
    
    for ID in TorsionAnglesInfo["IDs"]:
        if IsTorsionAngleInWithinTolerance(TorsionAngle, TorsionAnglesInfo["Value"][ID], TorsionAnglesInfo["Tolerance1"][ID]):
            TorsionCategory = "Green"
            TorsionAngleViolation = 0.0
            break
        
        if IsTorsionAngleInWithinTolerance(TorsionAngle, TorsionAnglesInfo["Value"][ID], TorsionAnglesInfo["Tolerance2"][ID]):
            TorsionCategory = "Orange"
            TorsionAngleViolation = CalculateTorsionAngleViolation(TorsionAngle, TorsionAnglesInfo["ValuesIn360RangeList"], TorsionAnglesInfo["Tolerances1List"])
            break

    if TorsionCategory is None:
        TorsionCategory = "Red"
        TorsionAngleViolation = CalculateTorsionAngleViolation(TorsionAngle, TorsionAnglesInfo["ValuesIn360RangeList"], TorsionAnglesInfo["Tolerances2List"])
        
    return (TorsionCategory, TorsionAngleViolation)

def IsTorsionAngleInWithinTolerance(TorsionAngle, TorsionPeak, TorsionTolerance):
    """Check torsion angle against torsion tolerance."""

    TorsionAngleDiff = TorsionLibraryUtil.CalculateTorsionAngleDifference(TorsionPeak, TorsionAngle)
    
    return True if (abs(TorsionAngleDiff) <= TorsionTolerance) else False

def CalculateTorsionAngleViolation(TorsionAngle, TorsionPeaks, TorsionTolerances):
    """Calculate torsion angle violation."""

    TorsionAngleViolation = None

    # Map angle to 0 to 360 range. TorsionPeaks values must be in this range...
    if TorsionAngle < 0:
        TorsionAngle = TorsionAngle + 360

    # Identify the closest torsion peak index...
    if len(TorsionPeaks) == 1:
        NearestPeakIndex = 0
    else:
        NearestPeakIndex = min(range(len(TorsionPeaks)), key=lambda Index: abs(TorsionPeaks[Index] - TorsionAngle))

    # Calculate torsion angle violation from the nearest peak and its tolerance value...
    TorsionAngleDiff = TorsionLibraryUtil.CalculateTorsionAngleDifference(TorsionPeaks[NearestPeakIndex], TorsionAngle)
    TorsionAngleViolation = abs(abs(TorsionAngleDiff) - TorsionTolerances[NearestPeakIndex])
    
    return TorsionAngleViolation

def InitializeTorsionAlertsSummaryInfo():
    """Initialize torsion alerts summary."""

    if OptionsInfo["CountMode"]:
        return None
    
    if not OptionsInfo["TrackAlertsSummaryInfo"]:
        return None
    
    TorsionAlertsSummaryInfo = {}
    TorsionAlertsSummaryInfo["RuleIDs"] = []

    for DataLabel in ["SMARTSToRuleIDs", "RuleSMARTS", "HierarchyClassName", "HierarchySubClassName", "TorsionRulePeaks", "TorsionRuleTolerances1", "TorsionRuleTolerances2", "AlertTypes", "AlertTypesMolCount"]:
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
    MolAlertsInfo["AlertTypes"] = {}
    
    for ID in RotBondsAlertsInfo["IDs"]:
        if not RotBondsAlertsInfo["MatchStatus"][ID]:
            continue

        if OptionsInfo["OutfileAlertsOnly"]:
            if RotBondsAlertsInfo["AlertTypes"][ID] not in OptionsInfo["SpecifiedAlertsModeList"]:
                continue
        
        AlertType = RotBondsAlertsInfo["AlertTypes"][ID]
        TorsionRuleNodeID = RotBondsAlertsInfo["TorsionRuleNodeID"][ID]
        TorsionRuleSMARTS = RotBondsAlertsInfo["TorsionRuleSMARTS"][ID]

        # Track data for torsion alert summary information across molecules...
        if TorsionRuleNodeID not in TorsionAlertsSummaryInfo["RuleSMARTS"]:
            TorsionAlertsSummaryInfo["RuleIDs"].append(TorsionRuleNodeID)
            TorsionAlertsSummaryInfo["SMARTSToRuleIDs"][TorsionRuleSMARTS] = TorsionRuleNodeID
            
            TorsionAlertsSummaryInfo["RuleSMARTS"][TorsionRuleNodeID] = TorsionRuleSMARTS
            TorsionAlertsSummaryInfo["HierarchyClassName"][TorsionRuleNodeID] = RotBondsAlertsInfo["HierarchyClassNames"][ID]
            TorsionAlertsSummaryInfo["HierarchySubClassName"][TorsionRuleNodeID] = RotBondsAlertsInfo["HierarchySubClassNames"][ID]

            TorsionAlertsSummaryInfo["TorsionRulePeaks"][TorsionRuleNodeID] = RotBondsAlertsInfo["TorsionRulePeaks"][ID]
            TorsionAlertsSummaryInfo["TorsionRuleTolerances1"][TorsionRuleNodeID] = RotBondsAlertsInfo["TorsionRuleTolerances1"][ID]
            TorsionAlertsSummaryInfo["TorsionRuleTolerances2"][TorsionRuleNodeID] = RotBondsAlertsInfo["TorsionRuleTolerances2"][ID]
            
            # Initialize number of alert types across all molecules...
            TorsionAlertsSummaryInfo["AlertTypes"][TorsionRuleNodeID] = {}
            
            # Initialize number of molecules flagged by each alert type...
            TorsionAlertsSummaryInfo["AlertTypesMolCount"][TorsionRuleNodeID] = {}
        
        if AlertType not in TorsionAlertsSummaryInfo["AlertTypes"][TorsionRuleNodeID]:
            TorsionAlertsSummaryInfo["AlertTypes"][TorsionRuleNodeID][AlertType] = 0
            TorsionAlertsSummaryInfo["AlertTypesMolCount"][TorsionRuleNodeID][AlertType] = 0
        
        TorsionAlertsSummaryInfo["AlertTypes"][TorsionRuleNodeID][AlertType] += 1

        # Track data for torsion alert information in a molecule...
        if TorsionRuleNodeID not in MolAlertsInfo["AlertTypes"]:
            MolAlertsInfo["RuleIDs"].append(TorsionRuleNodeID)
            MolAlertsInfo["AlertTypes"][TorsionRuleNodeID] = {}
        
        if AlertType not in MolAlertsInfo["AlertTypes"][TorsionRuleNodeID]:
            MolAlertsInfo["AlertTypes"][TorsionRuleNodeID][AlertType] = 0
        MolAlertsInfo["AlertTypes"][TorsionRuleNodeID][AlertType] += 1

    # Track number of molecules flagged by a specific torsion alert...
    for TorsionRuleNodeID in MolAlertsInfo["RuleIDs"]:
        for AlertType in MolAlertsInfo["AlertTypes"][TorsionRuleNodeID]:
            if MolAlertsInfo["AlertTypes"][TorsionRuleNodeID][AlertType]:
                TorsionAlertsSummaryInfo["AlertTypesMolCount"][TorsionRuleNodeID][AlertType] += 1

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
    Values = ["TorsionRule", "TorsionPeaks", "Tolerances1", "Tolerances2", "HierarchyClass", "HierarchySubClass", "TorsionAlertTypes", "TorsionAlertCount", "TorsionAlertMolCount"]
    Writer.write("%s\n" % MiscUtil.JoinWords(Values, ",", QuoteValues))

    SortedRuleIDs = GetSortedTorsionAlertsSummaryInfoRuleIDs(TorsionAlertsSummaryInfo)

    # Write alerts information...
    for ID in SortedRuleIDs:
        # Remove any double quotes in SMARTS...
        RuleSMARTS = TorsionAlertsSummaryInfo["RuleSMARTS"][ID]
        RuleSMARTS = re.sub("\"", "", RuleSMARTS, re.I)
        
        HierarchyClassName = TorsionAlertsSummaryInfo["HierarchyClassName"][ID]
        HierarchySubClassName = TorsionAlertsSummaryInfo["HierarchySubClassName"][ID]

        TorsionPeaks = MiscUtil.JoinWords(["%s" % Value for Value in TorsionAlertsSummaryInfo["TorsionRulePeaks"][ID]], ",")
        TorsionRuleTolerances1 = MiscUtil.JoinWords(["%s" % Value for Value in TorsionAlertsSummaryInfo["TorsionRuleTolerances1"][ID]], ",")
        TorsionRuleTolerances2 = MiscUtil.JoinWords(["%s" % Value for Value in TorsionAlertsSummaryInfo["TorsionRuleTolerances2"][ID]], ",")
        
        AlertTypes = []
        AlertTypeCount = []
        AlertTypeMolCount = []
        for AlertType in sorted(TorsionAlertsSummaryInfo["AlertTypes"][ID]):
            AlertTypes.append(AlertType)
            AlertTypeCount.append("%s" % TorsionAlertsSummaryInfo["AlertTypes"][ID][AlertType])
            AlertTypeMolCount.append("%s" % TorsionAlertsSummaryInfo["AlertTypesMolCount"][ID][AlertType])
        
        Values = [RuleSMARTS, TorsionPeaks, TorsionRuleTolerances1, TorsionRuleTolerances2, HierarchyClassName, HierarchySubClassName, "%s" % MiscUtil.JoinWords(AlertTypes, ","), "%s" % (MiscUtil.JoinWords(AlertTypeCount, ",")), "%s" % (MiscUtil.JoinWords(AlertTypeMolCount, ","))]
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
        for AlertType in sorted(TorsionAlertsSummaryInfo["AlertTypes"][ID]):
            MolCount += TorsionAlertsSummaryInfo["AlertTypesMolCount"][ID][AlertType]
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
    
    for DataLabel in ["RuleSMARTS", "HierarchyClassName", "HierarchySubClassName", "TorsionRulePeaks", "TorsionRuleTolerances1", "TorsionRuleTolerances2", "AlertTypes", "AtomIndices", "TorsionAtomIndices", "TorsionAngles", "TorsionAngleViolations", "AlertTypesCount"]:
        TorsionAlertsInfo[DataLabel] = {}
        
    ValuesDelimiter = OptionsInfo["IntraSetValuesDelim"]
    TorsionAlertsSetSize = 11
    
    TorsionAlertsWords = TorsionAlerts.split()
    if len(TorsionAlertsWords) % TorsionAlertsSetSize:
        MiscUtil.PrintError("The number of space delimited values, %s, for TorsionAlerts data field in filtered SD file must be a multiple of %s." % (len(TorsionAlertsWords), TorsionAlertsSetSize))

    ID = 0
    for Index in range(0, len(TorsionAlertsWords), TorsionAlertsSetSize):
        ID += 1
        
        RotBondIndices, TorsionAlertType, TorsionIndices, TorsionAngle, TorsionAngleViolation, HierarchyClass, HierarchySubClass, TorsionPeaks, Tolerances1, Tolerances2, TorsionRule = TorsionAlertsWords[Index: Index + TorsionAlertsSetSize]
        RotBondIndices = RotBondIndices.split(ValuesDelimiter)
        TorsionIndices = TorsionIndices.split(ValuesDelimiter)
        TorsionPeaks = TorsionPeaks.split(ValuesDelimiter)
        Tolerances1 = Tolerances1.split(ValuesDelimiter)
        Tolerances2 = Tolerances2.split(ValuesDelimiter)

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
            TorsionAlertsInfo["TorsionRulePeaks"][TorsionRuleNodeID] = TorsionPeaks
            TorsionAlertsInfo["TorsionRuleTolerances1"][TorsionRuleNodeID] = Tolerances1
            TorsionAlertsInfo["TorsionRuleTolerances2"][TorsionRuleNodeID] = Tolerances2
            
            TorsionAlertsInfo["AlertTypes"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["AtomIndices"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["TorsionAtomIndices"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["TorsionAngles"][TorsionRuleNodeID] = []
            TorsionAlertsInfo["TorsionAngleViolations"][TorsionRuleNodeID] = []

            TorsionAlertsInfo["AlertTypesCount"][TorsionRuleNodeID] = {}
            
        # Track multiple values for a rule ID...
        TorsionAlertsInfo["AlertTypes"][TorsionRuleNodeID].append(TorsionAlertType)
        TorsionAlertsInfo["AtomIndices"][TorsionRuleNodeID].append(RotBondIndices)
        TorsionAlertsInfo["TorsionAtomIndices"][TorsionRuleNodeID].append(TorsionIndices)
        TorsionAlertsInfo["TorsionAngles"][TorsionRuleNodeID].append(TorsionAngle)
        TorsionAlertsInfo["TorsionAngleViolations"][TorsionRuleNodeID].append(TorsionAngleViolation)
        
        # Count alert type for a rule ID...
        if TorsionAlertType not in TorsionAlertsInfo["AlertTypesCount"][TorsionRuleNodeID]:
            TorsionAlertsInfo["AlertTypesCount"][TorsionRuleNodeID][TorsionAlertType] = 0
        TorsionAlertsInfo["AlertTypesCount"][TorsionRuleNodeID][TorsionAlertType] += 1
        
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
    
    # Setup rotatable bonds count...
    RotBondsCount = 0
    if RotBondsAlertsInfo is not None:
        RotBondsCount =  len(RotBondsAlertsInfo["IDs"])
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["RotBondsCountLabel"],  "%s" % RotBondsCount)
    
    # Setup alert counts for rotatable bonds...
    AlertsCount = []
    if RotBondsAlertsInfo is not None:
        for AlertType in ["Green", "Orange", "Red"]:
            if AlertType in RotBondsAlertsInfo["Count"]:
                AlertsCount.append("%s" % RotBondsAlertsInfo["Count"][AlertType])
            else:
                AlertsCount.append("0")
    
    if len(AlertsCount):
        Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionAlertsCountLabel"],  "%s" % MiscUtil.JoinWords(AlertsCount, " "))

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
            
            if OptionsInfo["OutfileAlertsOnly"]:
                if RotBondsAlertsInfo["AlertTypes"][ID] not in OptionsInfo["SpecifiedAlertsModeList"]:
                    continue
            
            RotBondValues = []
            
            # Bond atom indices...
            Values = ["%s" % Value for Value in RotBondsAlertsInfo["AtomIndices"][ID]]
            RotBondValues.append(ValuesDelim.join(Values))

            # Alert type...
            RotBondValues.append(RotBondsAlertsInfo["AlertTypes"][ID])

            # Torsion atom indices...
            TorsionAtomIndices = SetupTorsionAtomIndicesValues(RotBondsAlertsInfo["TorsionAtomIndices"][ID], ValuesDelim)
            RotBondValues.append(TorsionAtomIndices)

            # Torsion angle...
            RotBondValues.append("%.2f" % RotBondsAlertsInfo["TorsionAngles"][ID])

            # Torsion angle violation...
            RotBondValues.append("%.2f" % RotBondsAlertsInfo["TorsionAngleViolations"][ID])

            # Hierarchy class and subclass names...
            RotBondValues.append("%s" % RotBondsAlertsInfo["HierarchyClassNames"][ID])
            RotBondValues.append("%s" % RotBondsAlertsInfo["HierarchySubClassNames"][ID])

            # Torsion rule peaks...
            Values = ["%s" % Value for Value in RotBondsAlertsInfo["TorsionRulePeaks"][ID]]
            RotBondValues.append(ValuesDelim.join(Values))
            
            # Torsion rule tolerances...
            Values = ["%s" % Value for Value in RotBondsAlertsInfo["TorsionRuleTolerances1"][ID]]
            RotBondValues.append(ValuesDelim.join(Values))
            Values = ["%s" % Value for Value in RotBondsAlertsInfo["TorsionRuleTolerances2"][ID]]
            RotBondValues.append(ValuesDelim.join(Values))
            
            # Torsion rule SMARTS...
            RotBondValues.append("%s" % RotBondsAlertsInfo["TorsionRuleSMARTS"][ID])

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
    
    Values = ["%s" % Value for Value in TorsionAlertsInfo["TorsionRulePeaks"][TorsionRuleID]]
    AlertRuleInfoValues.append(IntraSetValuesDelim.join(Values))
    
    Values = ["%s" % Value for Value in TorsionAlertsInfo["TorsionRuleTolerances1"][TorsionRuleID]]
    AlertRuleInfoValues.append(IntraSetValuesDelim.join(Values))
    Values = ["%s" % Value for Value in TorsionAlertsInfo["TorsionRuleTolerances2"][TorsionRuleID]]
    AlertRuleInfoValues.append(IntraSetValuesDelim.join(Values))
    
    AlertRuleInfoValues.append("%s" % TorsionAlertsInfo["RuleSMARTS"][TorsionRuleID])
    
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleLabel"], "%s" % ("%s" % InterSetValuesDelim.join(AlertRuleInfoValues)))

    # Setup alerts count for torsion rule...
    AlertsCount = []
    for AlertType in ["Green", "Orange", "Red"]:
        if AlertType in TorsionAlertsInfo["AlertTypesCount"][TorsionRuleID]:
            AlertsCount.append("%s" % TorsionAlertsInfo["AlertTypesCount"][TorsionRuleID][AlertType])
        else:
            AlertsCount.append("0")
    
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleAlertsCountLabel"],  "%s" % (InterSetValuesDelim.join(AlertsCount)))
    
    # Setup torsion rule alerts...
    AlertsInfoValues = []
    for Index in range(0, len(TorsionAlertsInfo["AlertTypes"][TorsionRuleID])):
        RotBondInfoValues = []
        
        # Bond atom indices...
        Values = ["%s" % Value for Value in TorsionAlertsInfo["AtomIndices"][TorsionRuleID][Index]]
        RotBondInfoValues.append(IntraSetValuesDelim.join(Values))
        
        # Alert type...
        RotBondInfoValues.append(TorsionAlertsInfo["AlertTypes"][TorsionRuleID][Index])
        
        # Torsion atom indices retrieved from the filtered SD file and stored as strings...
        Values = ["%s" % Value for Value in TorsionAlertsInfo["TorsionAtomIndices"][TorsionRuleID][Index]]
        RotBondInfoValues.append(IntraSetValuesDelim.join(Values))
        
        # Torsion angle...
        RotBondInfoValues.append(TorsionAlertsInfo["TorsionAngles"][TorsionRuleID][Index])
        
        # Torsion angle violation...
        RotBondInfoValues.append(TorsionAlertsInfo["TorsionAngleViolations"][TorsionRuleID][Index])

        # Track alerts informaiton...
        AlertsInfoValues.append("%s" % InterSetValuesDelim.join(RotBondInfoValues))
    
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleAlertsLabel"],  "%s" % (InterSetValuesDelim.join(AlertsInfoValues)))
    
    # Setup torsion rule alert max angle violation...
    TorsionAngleViolations = [float(Angle) for Angle in TorsionAlertsInfo["TorsionAngleViolations"][TorsionRuleID]]
    Mol.SetProp(OptionsInfo["SDFieldIDsToLabels"]["TorsionRuleMaxAngleViolationLabel"],  "%.2f" % (max(TorsionAngleViolations)))

def SetupTorsionAtomIndicesValues(TorsionAtomIndicesList, ValuesDelim):
    """Setup torsion atom indices value for output files."""

    # Check for any list values in the list of torsion atom indices used as placeholders
    # for positions of lone pairs in torsion rules containing  N_lp...
    TorsionAtomsInfo = []
    for Value in TorsionAtomIndicesList:
        if type(Value) is list:
            TorsionAtomsInfo.append("N_lp")
        else:
            TorsionAtomsInfo.append(Value)
            
    Values = ["%s" % Value for Value in TorsionAtomsInfo]
    
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

    TorsionLibraryFilePath = OptionsInfo["TorsionLibraryFile"]
    if TorsionLibraryFilePath is None:
        TorsionLibraryFile = "TorsionLibrary.xml"
        MayaChemToolsDataDir = MiscUtil.GetMayaChemToolsLibDataPath()
        TorsionLibraryFilePath = os.path.join(MayaChemToolsDataDir, TorsionLibraryFile)
        if not Quiet:
            MiscUtil.PrintInfo("\nRetrieving data from default torsion library file %s..." % TorsionLibraryFile)
    else:
        TorsionLibraryFilePath = OptionsInfo["TorsionLibraryFile"]
        if not Quiet:
            MiscUtil.PrintInfo("\nRetrieving data from torsion library file %s..." % TorsionLibraryFilePath)
        
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
    
    ParamsIDsToLabels = {"RotBondsCountLabel": "RotBondsCount", "TorsionAlertsCountLabel": "TorsionAlertsCount (Green Orange Red)", "TorsionAlertsLabel": "TorsionAlerts (RotBondIndices TorsionAlert TorsionIndices TorsionAngle TorsionAngleViolation HierarchyClass HierarchySubClass TorsionPeaks Tolerances1 Tolerances2 TorsionRule)", "TorsionRuleLabel": "TorsionRule (HierarchyClass HierarchySubClass TorsionPeaks Tolerances1 Tolerances2 TorsionRule)", "TorsionRuleAlertsCountLabel": "TorsionRuleAlertsCount (Green Orange Red)", "TorsionRuleAlertsLabel": "TorsionRuleAlerts (RotBondIndices TorsionAlert TorsionIndices TorsionAngle TorsionAngleViolation)", "TorsionRuleMaxAngleViolationLabel": "TorsionRuleMaxAngleViolation"}
    
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

def ProcessOptionNitrogenLonePairParameters():
    """Process nitrogen lone pair parameters option."""

    ParamsOptionName = "--nitrogenLonePairParams"
    ParamsOptionValue = Options["--nitrogenLonePairParams"]
    
    ParamsInfo = {"AllowHydrogenNbrs": True, "PlanarityTolerance": 1.0,}
    
    if re.match("^auto$", ParamsOptionValue, re.I):
        OptionsInfo["NitrogenLonePairParams"] = ParamsInfo
        return
    
    # Setup a canonical paramater names...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(ParamsInfo):
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
        
        if re.match("^PlanarityTolerance$", ParamName, re.I):
            Value = float(Value)
            if Value < 0:
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: >= 0" % (Value, Name, ParamsOptionName))
            ParamValue = Value
        elif re.match("^AllowHydrogenNbrs$", ParamName, re.I):
            if not re.match("^(yes|no)$", Value, re.I):
                MiscUtil.PrintError("The parameter value, %s, specified for parameter name, %s, using \"%s\" option is not a valid value. Supported values: yes or no" % (Value, Name, ParamsOptionName))
            ParamValue = True if re.match("^yes$", Value, re.I) else False
            
        # Set value...
        ParamsInfo[ParamName] = ParamValue
    
    OptionsInfo["NitrogenLonePairParams"] = ParamsInfo
    
def ProcessOptions():
    """Process and validate command line arguments and options."""
    
    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()
    
    OptionsInfo["Infile"] = Options["--infile"]
    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters("--infileParams", Options["--infileParams"], InfileName = Options["--infile"], ParamsDefaultInfo = ParamsDefaultInfoOverride)
    
    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters("--outfileParams", Options["--outfileParams"], Options["--infile"], Options["--outfile"])
    
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    OutfileFiltered = "%s_Filtered.%s" % (FileName, FileExt)
    OptionsInfo["OutfileFiltered"] = OutfileFiltered
    OptionsInfo["OutfileFilteredMode"] = True if re.match("^yes$", Options["--outfileFiltered"], re.I) else False
    
    OutfileSummary = "%s_AlertsSummary.csv" % (FileName)
    OptionsInfo["OutfileSummary"] = OutfileSummary
    OptionsInfo["OutfileSummaryMode"] = True if re.match("^yes$", Options["--outfileSummary"], re.I) else False

    OptionsInfo["OutfilesFilteredByRulesMode"] = True if re.match("^yes$", Options["--outfilesFilteredByRules"], re.I) else False
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

    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])

    ProcessOptionNitrogenLonePairParameters()
    
    OptionsInfo["AlertsMode"] = Options["--alertsMode"]
    OptionsInfo["SpecifiedAlertsModeList"] = []
    if re.match("^Red$", Options["--alertsMode"], re.I):
        OptionsInfo["SpecifiedAlertsModeList"].append("Red")
    elif re.match("^RedAndOrange$", Options["--alertsMode"], re.I):
        OptionsInfo["SpecifiedAlertsModeList"].append("Red")
        OptionsInfo["SpecifiedAlertsModeList"].append("Orange")
    
    OptionsInfo["MinAlertsCount"] = int(Options["--alertsMinCount"])

    OptionsInfo["RotBondsSMARTSMode"] = Options["--rotBondsSMARTSMode"]
    OptionsInfo["RotBondsSMARTSPatternSpecified"] = Options["--rotBondsSMARTSPattern"]
    ProcessRotatableBondsSMARTSMode()

    OptionsInfo["TorsionLibraryFile"] = None
    if not re.match("^auto$", Options["--torsionLibraryFile"], re.I):
        OptionsInfo["TorsionLibraryFile"] = Options["--torsionLibraryFile"]

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
    OptionsInfo["TorsionLibraryFile"] = None
    if not re.match("^auto$", Options["--torsionLibraryFile"], re.I):
        MiscUtil.ValidateOptionFilePath("-t, --torsionLibraryFile", Options["--torsionLibraryFile"])
        OptionsInfo["TorsionLibraryFile"] = Options["--torsionLibraryFile"]
    
    RetrieveTorsionLibraryInfo()
    ListTorsionLibraryInfo()
    
def ValidateOptions():
    """Validate option values."""

    MiscUtil.ValidateOptionTextValue("-a, --alertsMode", Options["--alertsMode"], "Red RedAndOrange")
    MiscUtil.ValidateOptionIntegerValue("--alertsMinCount", Options["--alertsMinCount"], {">=": 1})
    
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol")
    
    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    if re.match("^filter$", Options["--mode"], re.I):
        MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])
        MiscUtil.ValidateOptionsDistinctFileNames("-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"])

    MiscUtil.ValidateOptionTextValue("--outfileFiltered", Options["--outfileFiltered"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("--outfilesFilteredByRules", Options["--outfilesFilteredByRules"], "yes no")
    if not re.match("^All$", Options["--outfilesFilteredByRulesMaxCount"], re.I):
        MiscUtil.ValidateOptionIntegerValue("--outfilesFilteredByRulesMaxCount", Options["--outfilesFilteredByRulesMaxCount"], {">": 0})
    
    MiscUtil.ValidateOptionTextValue("--outfileSummary", Options["--outfileSummary"], "yes no")
    MiscUtil.ValidateOptionTextValue("--outfileAlerts", Options["--outfileAlerts"], "yes no")
    MiscUtil.ValidateOptionTextValue("--outfileAlertsMode", Options["--outfileAlertsMode"], "All AlertsOnly")
    
    MiscUtil.ValidateOptionTextValue("-m, --mode", Options["--mode"], "filter count")
    if re.match("^filter$", Options["--mode"], re.I):
        if not Options["--outfile"]:
            MiscUtil.PrintError("The outfile must be specified using \"-o, --outfile\" during \"filter\" value of \"-m, --mode\" option")
        
    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    
    MiscUtil.ValidateOptionTextValue("-r, --rotBondsSMARTSMode", Options["--rotBondsSMARTSMode"], "NonStrict SemiStrict Strict Specify")
    if re.match("^Specify$", Options["--rotBondsSMARTSMode"], re.I):
        if not Options["--rotBondsSMARTSPattern"]:
            MiscUtil.PrintError("The SMARTS pattern must be specified using \"--rotBondsSMARTSPattern\" during \"Specify\" value of \"-r, --rotBondsSMARTS\" option")
    
    if not re.match("^auto$", Options["--torsionLibraryFile"], re.I):
        MiscUtil.ValidateOptionFilePath("-t, --torsionLibraryFile", Options["--torsionLibraryFile"])

# Setup a usage string for docopt...
_docoptUsage_ = """
RDKitFilterTorsionLibraryAlerts.py - Filter torsion library alerts

Usage:
    RDKitFilterTorsionLibraryAlerts.py  [--alertsMode <Red, RedAndOrange>] [--alertsMinCount <Number>]
                                        [--infileParams <Name,Value,...>] [--mode <filter or count>] [--mp <yes or no>] [--mpParams <Name,Value,...>]
                                        [--nitrogenLonePairParams <Name,Value,...>] [--outfileAlerts <yes or no>]
                                        [--outfileAlertsMode <All or AlertsOnly>] [--outfileFiltered <yes or no>]
                                        [--outfilesFilteredByRules <yes or no>] [--outfilesFilteredByRulesMaxCount <All or number>]
                                        [--outfileSummary <yes or no>] [--outfileSDFieldLabels <Type,Label,...>]
                                        [--outfileParams <Name,Value,...>] [--overwrite] [ --rotBondsSMARTSMode <NonStrict, SemiStrict,...>]
                                        [--rotBondsSMARTSPattern <SMARTS>] [--torsionLibraryFile <FileName or auto>] [-w <dir>] -i <infile> -o <outfile>
    RDKitFilterTorsionLibraryAlerts.py [--torsionLibraryFile <FileName or auto>] -l | --list
    RDKitFilterTorsionLibraryAlerts.py -h | --help | -e | --examples

Description:
    Filter strained molecules from an input file for torsion library [ Ref 146, 152, 159 ]
    alerts by matching rotatable bonds against SMARTS patterns specified for torsion
    rules in a torsion library file and write out appropriate molecules to output
    files. The molecules must have 3D coordinates in input file. The default torsion
    library file, TorsionLibrary.xml, is available under MAYACHEMTOOLS/lib/data
    directory.
    
    The data in torsion library file is organized in a hierarchical manner. It consists
    of one generic class and six specific classes at the highest level. Each class
    contains multiple subclasses corresponding to named functional groups or
    substructure patterns. The subclasses consist of torsion rules sorted from
    specific to generic torsion patterns. The torsion rule, in turn, contains a list
    of peak values for torsion angles and two tolerance values. A pair of tolerance
    values define torsion bins around a torsion peak value. For example:
         
        <library>
            <hierarchyClass name="GG" id1="G" id2="G">
            ...
            </hierarchyClass>
            <hierarchyClass name="CO" id1="C" id2="O">
                <hierarchySubClass name="Ester bond I" smarts="O=[C:2][O:3]">
                    <torsionRule smarts="[O:1]=[C:2]!@[O:3]~[CH0:4]">
                        <angleList>
                            <angle value="0.0" tolerance1="20.00"
                             tolerance2="25.00" score="56.52"/>
                        </angleList>
                    </torsionRule>
                    ...
                ...
             ...
            </hierarchyClass>
            <hierarchyClass name="NC" id1="N" id2="C">
             ...
            </hierarchyClass>
            <hierarchyClass name="SN" id1="S" id2="N">
            ...
            </hierarchyClass>
            <hierarchyClass name="CS" id1="C" id2="S">
            ...
            </hierarchyClass>
            <hierarchyClass name="CC" id1="C" id2="C">
            ...
            </hierarchyClass>
            <hierarchyClass name="SS" id1="S" id2="S">
             ...
            </hierarchyClass>
        </library>
        
    The rotatable bonds in a 3D molecule are identified using a default SMARTS pattern.
    A custom SMARTS pattern may be optionally specified to detect rotatable bonds.
    Each rotatable bond is matched to a torsion rule in the torsion library and
    assigned one of the following three alert categories: Green, Orange or Red. The 
    rotatable bond is marked Green or Orange for the measured angle of the torsion
    pattern within the first or second tolerance bins around a torsion peak.
    Otherwise, it's marked Red implying that the measured angle is not observed in
    the structure databases employed to generate the torsion library.

    The following output files are generated after the filtering:
        
        <OutfileRoot>.sdf
        <OutfileRoot>_Filtered.sdf
        <OutfileRoot>_AlertsSummary.csv
        <OutfileRoot>_Filtered_TopRule*.sdf
        
    The supported input file formats are: Mol (.mol), SD (.sdf, .sd)

    The supported output file formats are: SD (.sdf, .sd)

Options:
    -a, --alertsMode <Red, RedAndOrange>  [default: Red]
        Torsion library alert types to use for filtering molecules containing
        rotatable bonds marked with Green, Orange, or Red alerts. Possible
        values: Red or RedAndOrange.
    --alertsMinCount <Number>  [default: 1]
        Minimum number of rotatable bond alerts in a molecule for filtering the
        molecule.
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
        Specify whether to filter molecules for torsion library [ Ref 146, 152, 159 ] alerts
        by matching rotatable bonds against SMARTS patterns specified for torsion
        rules and write out the rest of the molecules to an outfile or simply count
        the number of matched molecules marked for filtering.
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
    -n, --nitrogenLonePairParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs to match
        torsion SMARTS patterns containing non-standard construct 'N_lp'
        corresponding to nitrogen lone pair.
        
        The supported parameter names along with their default and possible
        values are shown below:
        
            allowHydrogenNbrs, yes   [ Possible values: yes or no ]
            planarityTolerance, 1  [Possible values: >=0] 
            
        These parameters are used during the matching of torsion rules containing
        'N_lp' in their SMARTS patterns. The 'allowHydrogensNbrs' allows the use
        hydrogen neighbors attached to nitrogen during the determination of its
        planarity. The 'planarityTolerance' in degrees represents the tolerance
        allowed for nitrogen to be considered coplanar with its three neighbors.
        
        The torsion rules containing 'N_lp' in their SMARTS patterns are categorized
        into the following two types of rules:
         
            TypeOne:  
            
            [CX4:1][CX4H2:2]!@[NX3;"N_lp":3][CX4:4]
            [C:1][CX4H2:2]!@[NX3;"N_lp":3][C:4]
            ... ... ...
         
            TypeTwo:  
            
            [!#1:1][CX4:2]!@[NX3;"N_lp":3]
            [C:1][$(S(=O)=O):2]!@["N_lp":3]
            ... ... ...
            
        The torsions are matched to torsion rules containing 'N_lp' using specified
        SMARTS patterns without the 'N_lp' along with additional constraints using
        the following methodology:
            
            TypeOne:  
            
            . SMARTS pattern must contain four mapped atoms and the third
                mapped atom must be a nitrogen matched with 'NX3:3'
            . Nitrogen atom must have 3 neighbors. The 'allowHydrogens'
                parameter controls inclusion of hydrogens as its neighbors.
            . Nitrogen atom and its 3 neighbors must be coplanar.
                'planarityTolerance' parameter provides tolerance in degrees
                for nitrogen to be considered coplanar with its 3 neighbors.
            
            TypeTwo:  
            
            . SMARTS pattern must contain three mapped atoms and the third
                mapped atom must be a nitrogen matched with 'NX3:3'. The 
                third mapped atom may contain only 'N_lp:3' The missing 'NX3'
                is automatically detected.
            . Nitrogen atom must have 3 neighbors. 'allowHydrogens'
                parameter controls inclusion of hydrogens as neighbors.
            . Nitrogen atom and its 3 neighbors must not be coplanar.
                'planarityTolerance' parameter provides tolerance in degrees
                for nitrogen to be considered coplanar with its 3 neighbors.
            . Nitrogen lone pair position equivalent to VSEPR theory is
                determined based on the position of nitrogen and its neighbors.
                A vector normal to 3 nitrogen neighbors is calculated and added
                to the coordinates of nitrogen atom to determine the approximate
                position of the lone pair. It is used as the fourth position to
                calculate the torsion angle.
            
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
            
            RotBondIndices TorsionAlert TorsionIndices TorsionAngle
            TorsionAngleViolation HierarchyClass HierarchySubClass
            TorsionRule TorsionPeaks Tolerances1 Tolerances2
            
        The 'RotBondsCount' and 'TorsionAlertsCount' data fields are always added
        to SD output files containing both remaining and filtered molecules.
        
        Format:
            
            > <RotBondsCount>
            Number
            
            > <TorsionAlertsCount (Green Orange Red)>
            Number Number Number
            
            > <TorsionAlerts (RotBondIndices TorsionAlert TorsionIndices
                TorsionAngle TorsionAngleViolation HierarchyClass
                HierarchySubClass TorsionPeaks Tolerances1 Tolerances2
                TorsionRule)>
            AtomIndex2,AtomIndex3  AlertType AtomIndex1,AtomIndex2,AtomIndex3,
            AtomIndex4 Angle AngleViolation ClassName SubClassName
            CommaDelimPeakValues CommaDelimTol1Values CommDelimTol2Values
            SMARTS ... ... ...
             ... ... ...
            
        A set of 11 values is written out as value of 'TorsionAlerts' data field for
        each torsion in a molecule. The space character is used as a delimiter
        to separate values with in a set and across set. The comma character
        is used to delimit multiple values for each value in a set.
        
        The 'RotBondIndices' and 'TorsionIndices' contain 2 and 4 comma delimited
        values representing atom indices for a rotatable bond and matched torsion.
        The 'TorsionPeaks',  'Tolerances1', and 'Tolerances2' contain same number
        of comma delimited values corresponding to  torsion angle peaks and
        tolerance intervals specified in torsion library. For example:
            
            ... ... ...
            >  <RotBondsCount>  (1) 
            7
            
            >  <TorsionAlertsCount (Green Orange Red)>  (1) 
            3 2 2
            
            >  <TorsionAlerts (RotBondIndices TorsionAlert TorsionIndices
                TorsionAngle TorsionAngleViolation HierarchyClass
                HierarchySubClass TorsionPeaks Tolerances1 Tolerances2
                TorsionRule)>
            1,2 Red 32,2,1,0 0.13 149.87 NC Anilines 180.0 10.0 30.0 [cH0:1][c:2]
            ([cH,nX2H0])!@[NX3H1:3][CX4:4] 8,9 Red 10,9,8,28 -0.85 GG
            None -90.0,90.0 30.0,30.0 60.0,60.0 [cH1:1][a:2]([cH1])!@[a:3]
            ([cH0])[cH0:4]
            ... ... ...
            
    --outfileFiltered <yes or no>  [default: yes]
        Write out a file containing filtered molecules. Its name is automatically
        generated from the specified output file. Default: <OutfileRoot>_
        Filtered.<OutfileExt>.
    --outfilesFilteredByRules <yes or no>  [default: yes]
        Write out SD files containing filtered molecules for individual torsion
        rules triggering alerts in molecules. The name of SD files are automatically
        generated from the specified output file. Default file names: <OutfileRoot>_
        Filtered_TopRule*.sdf
                
        The following alerts information is added to SD output files:
            
            > <RotBondsCount>
            Number
            
            >  <TorsionAlertsCount (Green Orange Red)> 
            Number Number Number
            
            >  <TorsionRule (HierarchyClass HierarchySubClass TorsionPeaks
                Tolerances1 Tolerances2 TorsionRule)> 
            ClassName SubClassName CommaDelimPeakValues CommaDelimTol1Values
            CommDelimTol2Values SMARTS ... ... ...
             ... ... ...
            
            > <TorsionRuleAlertsCount (Green Orange Red)>
            Number Number Number
            
            >  <TorsionRuleAlerts (RotBondIndices TorsionAlert TorsionIndices
                TorsionAngle TorsionAngleViolation)>
            AtomIndex2,AtomIndex3  AlertType AtomIndex1,AtomIndex2,AtomIndex3,
            AtomIndex4 Angle AngleViolation ... ... ...
            
            >  <TorsionRuleMaxAngleViolation>
            Number
             ... ... ...
            
        For example:
            
            ... ... ...
            >  <RotBondsCount>  (1) 
            7
             
            >  <TorsionAlertsCount (Green Orange Red)>  (1) 
            3 2 2
            
            >  <TorsionRule (HierarchyClass HierarchySubClass TorsionPeaks
                Tolerances1 Tolerances2 TorsionRule)>  (1) 
            NC Anilines 180.0 10.0 30.0 [cH0:1][c:2]([cH,nX2H0])!@[NX3H1:3][CX4:4]
            
            >  <TorsionRuleAlertsCount (Green Orange Red)>  (1) 
            0 0 1
            
            >  <TorsionRuleAlerts (RotBondIndices TorsionAlert TorsionIndices
                TorsionAngle TorsionAngleViolation)>  (1) 
            1,2 Red 32,2,1,0 0.13 149.87
            
            >  <TorsionRuleMaxAngleViolation>  (1) 
            149.87
            ... ... ...
            
    --outfilesFilteredByRulesMaxCount <All or number>  [default: 10]
        Write out SD files containing filtered molecules for specified number of
        top N torsion rules triggering alerts for the largest number of molecules
        or for all torsion rules triggering alerts across all molecules.
    --outfileSummary <yes or no>  [default: yes] 
        Write out a CVS text file containing summary of torsions rules responsible
        for triggering torsion alerts. Its name is automatically generated from the
        specified output file. Default: <OutfileRoot>_AlertsSummary.csv.
        
        The following alerts information is written to summary text file:
            
            TorsionRule, TorsionPeaks, Tolerances1, Tolerances2,
            HierarchyClass, HierarchySubClass, TorsionAlertType,
            TorsionAlertCount, TorsionAlertMolCount
             
        The double quotes characters are removed from SMART patterns before
        before writing them to a CSV file. In addition, the torsion rules are sorted by
        TorsionAlertMolCount. For example:
            
            "TorsionRule","TorsionPeaks","Tolerances1","Tolerances2",
                "HierarchyClass","HierarchySubClass","TorsionAlertTypes",
                "TorsionAlertCount","TorsionAlertMolCount"
            "[!#1:1][CX4H2:2]!@[CX4H2:3][!#1:4]","-60.0,60.0,180.0",
                "20.0,20.0,20.0","30.0,30.0,30.0","CC","None/[CX4:2][CX4:3]",
                "Red","16","11"
            ... ... ...
            
    --outfileSDFieldLabels <Type,Label,...>  [default: auto]
        A comma delimited list of SD data field type and label value pairs for writing
        torsion alerts information along with molecules to SD files.
        
        The supported SD data field label type along with their default values are
        shown below:
            
            For all SD files:
            
            RotBondsCountLabel, RotBondsCount
            TorsionAlertsCountLabel, TorsionAlertsCount (Green Orange Red)
            TorsionAlertsLabel, TorsionAlerts (RotBondIndices TorsionAlert
                TorsionIndices TorsionAngle TorsionAngleViolation
                HierarchyClass HierarchySubClass TorsionPeaks Tolerances1
                Tolerances2 TorsionRule)
            
            For individual SD files filtered by torsion rules:
            
            TorsionRuleLabel, TorsionRule (HierarchyClass HierarchySubClass
                TorsionPeaks Tolerances1 Tolerances2 TorsionRule)
            TorsionRuleAlertsCountLabel, TorsionRuleAlertsCount (Green Orange
                Red)
            TorsionRuleAlertsLabel, TorsionRuleAlerts (RotBondIndices
                TorsionAlert TorsionIndices TorsionAngle TorsionAngleViolation)
            TorsionRuleMaxAngleViolationLabel, TorsionRuleMaxAngleViolation
            
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: kekulize,yes
            
    --overwrite
        Overwrite existing files.
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
    -t, --torsionLibraryFile <FileName or auto>  [default: auto]
        Specify a XML file name containing data for torsion library hierarchy or use
        default file, TorsionLibrary.xml, available in MAYACHEMTOOLS/lib/data
        directory.
        
        The format of data in local XML file must match format of the data in Torsion
        Library [ Ref 146, 152, 159 ] file available in MAYACHEMTOOLS data directory.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To filter molecules containing any rotatable bonds marked with Red alerts
    based on torsion rules in the torsion library and write out SD files containing
    remaining and filtered molecules, and individual SD files for torsion rules
    triggering alerts along with appropriate torsion information for red alerts,
    type:

        % RDKitFilterTorsionLibraryAlerts.py -i Sample3D.sdf -o Sample3DOut.sdf

    To run the first example for only counting number of alerts without writing
    out any SD files, type:

        % RDKitFilterTorsionLibraryAlerts.py -m count -i Sample3D.sdf -o
          Sample3DOut.sdf
    
    To run the first example for filtertering molecules marked with Orange or
    Red alerts and write out SD files, tye:

        % RDKitFilterTorsionLibraryAlerts.py -m Filter --alertsMode RedAndOrange
          -i Sample3D.sdf -o Sample3DOut.sdf
    
    To run the first example for filtering molecules and writing out torsion
    information for all alert types to SD files, type:

        % RDKitFilterTorsionLibraryAlerts.py --outfileAlertsMode All
          -i Sample3D.sdf -o Sample3DOut.sdf

    To run the first example for filtering molecules in multiprocessing mode on
    all available CPUs without loading all data into memory and write out SD files,
    type:

        % RDKitFilterTorsionLibraryAlerts.py --mp yes -i Sample3D.sdf
         -o Sample3DOut.sdf

    To run the first example for filtering molecules in multiprocessing mode on
    all available CPUs by loading all data into memory and write out a SD files,
    type:

        % RDKitFilterTorsionLibraryAlerts.py  --mp yes --mpParams
          "inputDataMode, InMemory" -i Sample3D.sdf  -o Sample3DOut.sdf

    To run the first example for filtering molecules in multiprocessing mode on
    specific number of CPUs and chunksize without loading all data into memory
    and write out SD files, type:

        % RDKitFilterTorsionLibraryAlerts.py --mp yes --mpParams
          "inputDataMode,lazy,numProcesses,4,chunkSize,8"  -i Sample3D.sdf
          -o Sample3DOut.sdf

    To list information about default torsion library file without performing any
    filtering, type:

        % RDKitFilterTorsionLibraryAlerts.py -l

    To list information about a local torsion library XML file without performing
    any, filtering, type:

        % RDKitFilterTorsionLibraryAlerts.py --torsionLibraryFile
          TorsionLibrary.xml -l

Author:
    Manish Sud (msud@san.rr.com)

Collaborator:
    Pat Walters

Acknowledgments:
    Wolfgang Guba, Patrick Penner, Levi Pierce

See also:
    RDKitFilterChEMBLAlerts.py, RDKitFilterPAINS.py, RDKitFilterTorsionStrainEnergyAlerts.py,
    RDKitConvertFileFormat.py, RDKitSearchSMARTS.py

Copyright:
    Copyright (C) 2022 Manish Sud. All rights reserved.

    This script uses the Torsion Library jointly developed by the University
    of Hamburg, Center for Bioinformatics, Hamburg, Germany and
    F. Hoffmann-La-Roche Ltd., Basel, Switzerland.

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
