#
# File: TorsionsUtil.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2022 Manish Sud. All rights reserved.
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
import xml.etree.ElementTree as ET

from rdkit import Chem

import RDKitUtil
import MiscUtil

__all__ = ["CalculateTorsionAngleDifference", "DoesSMARTSContainsMappedAtoms", "DoesSMARTSContainValidSubClassMappedAtoms", "DoesSMARTSContainValidTorsionRuleMappedAtoms", "GetGenericHierarchyClassElementNode", "IdentifyRotatableBondsForTorsionLibraryMatch", "IsSpecificHierarchyClass", "ListTorsionLibraryInfo", "RemoveLastHierarchyClassElementNodeFromTracking", "RemoveLastHierarchySubClassElementNodeFromTracking", "RetrieveTorsionLibraryInfo", "SetupHierarchyClassAndSubClassNamesForRotatableBond", "SetupHierarchySubClassElementPatternMol", "SetupTorsionRuleElementPatternMol", "SetupTorsionLibraryInfoForMatchingRotatableBonds", "TrackHierarchyClassElementNode", "TrackHierarchySubClassElementNode"]

def RetrieveTorsionLibraryInfo(TorsionLibraryFilePath, Quiet = True):
    """Retrieve torsion library information.
    
    Arguments:
        TorsionLibraryFilePath (str):  Torsion library XML file path.

    Returns:
        object: An object returned by xml.etree.ElementTree.parse function.

    Notes:
        The XML file is parsed using xml.etree.ElementTree.parse function and
        object created by the parse function is simply returned.

    """
    
    if not Quiet:
        MiscUtil.PrintInfo("\nRetrieving data from torsion library file %s..." % TorsionLibraryFilePath)

    try:
        TorsionLibElementTree = ET.parse(TorsionLibraryFilePath)
    except Exception as ErrMsg:
        MiscUtil.PrintError("Failed to parse torsion library file: %s" % ErrMsg)

    return TorsionLibElementTree

def ListTorsionLibraryInfo(TorsionLibElementTree):
    """List torsion library information using XML tree object. The following
    information is listed:
    
        Summary:
        
            Total number of HierarchyClass nodes: <Number>
            Total number of HierarchyClassSubClass nodes: <Number
            Total number of TorsionRule nodes: <Number
    
        Details:
        
            HierarchyClass: <Name>; HierarchySubClass nodes: <Number>;
                TorsionRule nodes: <SMARTS>
             ... ... ...
        
    Arguments:
        TorsionLibElementTree (object): XML tree object.

    Returns:
        Nothing.

    """
    HierarchyClassesInfo = {}
    HierarchyClassesInfo["HierarchyClassNames"] = []
    HierarchyClassesInfo["HierarchySubClassCount"] = {}
    HierarchyClassesInfo["TorsionRuleCount"] = {}

    HierarchyClassCount, HierarchySubClassCount, TorsionRuleCount = [0] * 3

    for HierarchyClassNode in TorsionLibElementTree.findall("hierarchyClass"):
        HierarchyClassCount += 1
        HierarchyClassName = HierarchyClassNode.get("name")
        if HierarchyClassName in HierarchyClassesInfo["HierarchyClassNames"]:
            MiscUtil.PrintWarning("Hierarchy class name, %s, already exists..." % HierarchyClassName)
        HierarchyClassesInfo["HierarchyClassNames"].append(HierarchyClassName)

        SubClassCount = 0
        for HierarchySubClassNode in HierarchyClassNode.iter("hierarchySubClass"):
            SubClassCount += 1
        HierarchyClassesInfo["HierarchySubClassCount"][HierarchyClassName] = SubClassCount
        HierarchySubClassCount += SubClassCount

        RuleCount = 0
        for TorsionRuleNode in HierarchyClassNode.iter("torsionRule"):
            RuleCount += 1
        
        HierarchyClassesInfo["TorsionRuleCount"][HierarchyClassName] = RuleCount
        TorsionRuleCount += RuleCount

    MiscUtil.PrintInfo("\nTotal number of HierarchyClass nodes: %s" % HierarchyClassCount)
    MiscUtil.PrintInfo("Total number of HierarchyClassSubClass nodes: %s" % HierarchySubClassCount)
    MiscUtil.PrintInfo("Total number of TorsionRule nodes: %s" % TorsionRuleCount)

    # List info for each hierarchyClass...
    MiscUtil.PrintInfo("")
    
    # Generic class first...
    GenericClassName = "GG"
    if GenericClassName in HierarchyClassesInfo["HierarchyClassNames"]:
        MiscUtil.PrintInfo("HierarchyClass: %s; HierarchySubClass nodes: %s; TorsionRule nodes: %s" % (GenericClassName, HierarchyClassesInfo["HierarchySubClassCount"][GenericClassName], HierarchyClassesInfo["TorsionRuleCount"][GenericClassName]))
    
    for HierarchyClassName in sorted(HierarchyClassesInfo["HierarchyClassNames"]):
        if HierarchyClassName == GenericClassName:
            continue
        MiscUtil.PrintInfo("HierarchyClass: %s; HierarchySubClass nodes: %s; TorsionRule nodes: %s" % (HierarchyClassName, HierarchyClassesInfo["HierarchySubClassCount"][HierarchyClassName], HierarchyClassesInfo["TorsionRuleCount"][HierarchyClassName]))

def SetupTorsionLibraryInfoForMatchingRotatableBonds(TorsionLibraryInfo):
    """Setup torsion  library information for matching rotatable bonds. The
    following information is initialized and updated in torsion library
    dictionary for matching rotatable bonds:
        
            TorsionLibraryInfo["GenericClass"] = None
            TorsionLibraryInfo["GenericClassElementNode"] = None
            
            TorsionLibraryInfo["SpecificClasses"] = {}
            TorsionLibraryInfo["SpecificClasses"]["Names"] = []
            TorsionLibraryInfo["SpecificClasses"]["ElementNode"] = {}
            
            TorsionLibraryInfo["HierarchyClassNodes"] = []
            TorsionLibraryInfo["HierarchySubClassNodes"] = []
            
            TorsionLibraryInfo["DataCache"] = {}
            TorsionLibraryInfo["DataCache"]["SubClassPatternMol"] = {}
            
            TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"] = {}
            TorsionLibraryInfo["DataCache"]["TorsionRuleAnglesInfo"] = {}
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing root node for
            torsion library element tree.

    Returns:
        Nonthing. The torsion library information dictionary is updated.

    """
    _SetupTorsionLibraryHierarchyClassesInfoForMatchingRotatableBonds(TorsionLibraryInfo)
    _SetupTorsionLibraryDataCacheInfoForMatchingRotatableBonds(TorsionLibraryInfo)
    
def _SetupTorsionLibraryHierarchyClassesInfoForMatchingRotatableBonds(TorsionLibraryInfo):
    """Setup  hierarchy classes information for generic and specific classes."""
    
    RootElementNode = TorsionLibraryInfo["TorsionLibElementTree"]
    
    TorsionLibraryInfo["GenericClass"] = None
    TorsionLibraryInfo["GenericClassElementNode"] = None
    
    TorsionLibraryInfo["SpecificClasses"] = {}
    TorsionLibraryInfo["SpecificClasses"]["Names"] = []
    TorsionLibraryInfo["SpecificClasses"]["ElementNode"] = {}

    # Class name stacks for tracking names during processing of torsion rules..
    TorsionLibraryInfo["HierarchyClassNodes"] = []
    TorsionLibraryInfo["HierarchySubClassNodes"] = []

    ElementNames = []
    for ElementNode in RootElementNode.findall("hierarchyClass"):
        ElementName = ElementNode.get("name")
        if ElementName in ElementNames:
            MiscUtil.PrintWarning("Hierarchy class name, %s, already exists. Ignoring duplicate name..." % ElementName)
            continue

        ElementNames.append(ElementName)
        
        if re.match("^GG$", ElementName, re.I):
            TorsionLibraryInfo["GenericClass"] = ElementName
            TorsionLibraryInfo["GenericClassElementNode"] = ElementNode
        else:
            TorsionLibraryInfo["SpecificClasses"]["Names"].append(ElementName)
            TorsionLibraryInfo["SpecificClasses"]["ElementNode"][ElementName] = ElementNode

def _SetupTorsionLibraryDataCacheInfoForMatchingRotatableBonds(TorsionLibraryInfo):
    """Setup information for caching molecules for hierarchy subclass and torsion rule patterns."""
    
    TorsionLibElementTree = TorsionLibraryInfo["TorsionLibElementTree"]

    # Initialize data cache for pattern molecules corresponding to SMARTS patterns for
    # hierarchy subclasses and torsion rules. The pattern mols are generated and cached
    # later.
    TorsionLibraryInfo["DataCache"] = {}
    TorsionLibraryInfo["DataCache"]["SubClassPatternMol"] = {}
    
    TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"] = {}
    TorsionLibraryInfo["DataCache"]["TorsionRuleLonePairMapNumber"] = {}
    TorsionLibraryInfo["DataCache"]["TorsionRuleAnglesInfo"] = {}
    
    HierarchyClassID, HierarchySubClassID, TorsionRuleID = [0] * 3
    
    for HierarchyClassNode in TorsionLibElementTree.findall("hierarchyClass"):
        HierarchyClassID += 1
        
        for HierarchySubClassNode in HierarchyClassNode.iter("hierarchySubClass"):
            HierarchySubClassID += 1
            # Add unique ID to node...
            HierarchySubClassNode.set("NodeID", HierarchySubClassID)
        
        for TorsionRuleNode in HierarchyClassNode.iter("torsionRule"):
            TorsionRuleID += 1
            # Add unique ID to node...
            TorsionRuleNode.set("NodeID", TorsionRuleID)

def IdentifyRotatableBondsForTorsionLibraryMatch(TorsionLibraryInfo, Mol, RotBondsPatternMol):
    """Identify rotatable bonds in a molecule for torsion library match.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.
        Mol (object): RDKit molecule object.
        RotBondsPatternMol (object): RDKit molecule object for SMARTS pattern
            corresponding to rotatable bonds.

    Returns:
        bool: True - Rotatable bonds present in molecule; Otherwise, false.
        None or dict: None - For no rotatable bonds in molecule; otherwise, a
            dictionary containing the following informations for rotatable bonds
            matched to RotBondsPatternMol:
                
                RotBondsInfo["IDs"] = []
                RotBondsInfo["AtomIndices"] = {}
                RotBondsInfo["HierarchyClass"] = {}

    """
    # Match rotatable bonds...
    RotBondsMatches = RDKitUtil.FilterSubstructureMatchesByAtomMapNumbers(Mol, RotBondsPatternMol, Mol.GetSubstructMatches(RotBondsPatternMol, useChirality = False))

    #  Check and filter rotatable bond matches...
    RotBondsMatches = _FilterRotatableBondMatches(Mol, RotBondsMatches)

    if not len(RotBondsMatches):
        return False, None

    # Initialize rotatable bonds info...
    RotBondsInfo = {}
    RotBondsInfo["IDs"] = []
    RotBondsInfo["AtomIndices"] = {}
    RotBondsInfo["HierarchyClass"] = {}
    
    # Setup rotatable bonds info...
    ID = 0
    for RotBondAtomIndices in RotBondsMatches:
        ID += 1

        RotBondAtoms = [Mol.GetAtomWithIdx(RotBondAtomIndices[0]), Mol.GetAtomWithIdx(RotBondAtomIndices[1])]
        RotBondAtomSymbols = [RotBondAtoms[0].GetSymbol(), RotBondAtoms[1].GetSymbol()]
        
        ClassID = "%s%s" % (RotBondAtomSymbols[0], RotBondAtomSymbols[1])
        if ClassID not in TorsionLibraryInfo["SpecificClasses"]["Names"]:
            ReverseClassID = "%s%s" % (RotBondAtomSymbols[1], RotBondAtomSymbols[0])
            if ReverseClassID in TorsionLibraryInfo["SpecificClasses"]["Names"]:
                ClassID = ReverseClassID
                # Reverse atom indices and related information...
                RotBondAtomIndices = list(reversed(RotBondAtomIndices))
                RotBondAtoms = list(reversed(RotBondAtoms))
                RotBondAtomSymbols = list(reversed(RotBondAtomSymbols))
            
        # Track information...
        RotBondsInfo["IDs"].append(ID)
        RotBondsInfo["AtomIndices"][ID] = RotBondAtomIndices
        RotBondsInfo["HierarchyClass"][ID] = ClassID
    
    return True, RotBondsInfo

def _FilterRotatableBondMatches(Mol, RotBondsMatches):
    """Filter rotatable bond matches to ensure that each rotatable bond atom
    is attached to at least two heavy atoms. Otherwise, the torsion rules might match
    hydrogens."""
    
    FilteredRotBondMatches = []
    
    # Go over rotatable bonds...
    for RotBondMatch in RotBondsMatches:
        SkipRotBondMatch = False
        for AtomIndex in RotBondMatch:
            Atom = Mol.GetAtomWithIdx(AtomIndex)
            HeavyAtomNeighborCount = RDKitUtil.GetNumHeavyAtomNeighbors(Atom)
            
            if HeavyAtomNeighborCount <= 1:
                SkipRotBondMatch = True
                break
            
        if not SkipRotBondMatch:
            FilteredRotBondMatches.append(RotBondMatch)

    return FilteredRotBondMatches

def SetupHierarchySubClassElementPatternMol(TorsionLibraryInfo, ElementNode):
    """Setup pattern molecule for SMARTS pattern in hierarchy subclass element.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.
        ElementNode (object): A hierarchy sub class element node being matched
           in torsion library XML tree.

    Returns:
        object: RDKit molecule object corresponding to SMARTS pattern for
            hierarchy sub class element node.

    """
    # Check data cache...
    SubClassNodeID = ElementNode.get("NodeID")
    if SubClassNodeID in TorsionLibraryInfo["DataCache"]["SubClassPatternMol"]:
        return(TorsionLibraryInfo["DataCache"]["SubClassPatternMol"][SubClassNodeID])

    # Setup and track pattern mol...
    SubClassSMARTSPattern = ElementNode.get("smarts")
    SubClassPatternMol = Chem.MolFromSmarts(SubClassSMARTSPattern)
        
    if SubClassPatternMol is None:
        MiscUtil.PrintWarning("Ignoring hierachical subclass, %s, containing invalid SMARTS pattern %s" % (ElementNode.get("name"), SubClassSMARTSPattern))
    
    if not DoesSMARTSContainValidSubClassMappedAtoms(SubClassSMARTSPattern):
        SubClassPatternMol = None
        MiscUtil.PrintWarning("Ignoring hierachical subclass, %s, containing invalid map atom numbers in SMARTS pattern %s" % (ElementNode.get("name"), SubClassSMARTSPattern))
    
    TorsionLibraryInfo["DataCache"]["SubClassPatternMol"][SubClassNodeID] = SubClassPatternMol
    
    return SubClassPatternMol

def SetupTorsionRuleElementPatternMol(TorsionLibraryInfo, ElementNode, TorsionRuleNodeID, TorsionSMARTSPattern):
    """Setup pattern molecule for SMARTS pattern in torsion rule element.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.
        ElementNode (object): A torsion rule element node being matched in
           torsion library XML tree.
        TorsionRuleNodeID (int): Torsion rule element node ID.
        TorsionSMARTSPattern (str): SMARTS pattern for torsion rule element node.

    Returns:
        object: RDKit molecule object corresponding to SMARTS pattern for
            torsion rule element node.

    """

    if TorsionRuleNodeID in TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"]:
        return (TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"][TorsionRuleNodeID])
    
    TorsionPatternMol = Chem.MolFromSmarts(TorsionSMARTSPattern)
    if TorsionPatternMol is None:
        MiscUtil.PrintWarning("Ignoring torsion rule element containing invalid SMARTS pattern %s" % TorsionSMARTSPattern)
        
    if not DoesSMARTSContainValidTorsionRuleMappedAtoms(TorsionSMARTSPattern):
        TorsionPatternMol = None
        MiscUtil.PrintWarning("Ignoring torsion rule element containing invalid map atoms numbers in SMARTS pattern %s" % TorsionSMARTSPattern)
    
    TorsionLibraryInfo["DataCache"]["TorsionRulePatternMol"][TorsionRuleNodeID] = TorsionPatternMol
    
    return TorsionPatternMol

def SetupHierarchyClassAndSubClassNamesForRotatableBond(TorsionLibraryInfo):
    """ Setup hierarchy class and subclass names for a rotatable bond matched to
    a torsion rule element node.

    Returns:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.

    Returns:
        str: A back slash delimited string containing hierarchy class names at
            the level of torsion rule element node.
        str: A back slash delimited string containing hierarchy sub class names
          at the level of torsion rule element node.

    """
    HierarchyClassName, HierarchyClassSubName = ["None"] * 2

    # Setup hierarchy class name...
    if len(TorsionLibraryInfo["HierarchyClassNodes"]):
        HierarchyClassElementNode = TorsionLibraryInfo["HierarchyClassNodes"][-1]
        HierarchyClassName = HierarchyClassElementNode.get("name")
        if len(HierarchyClassName) == 0:
            HierarchyClassName = "None"
    
    # Setup hierarchy class name...
    if len(TorsionLibraryInfo["HierarchySubClassNodes"]):
        HierarchySubClassNames = []
        for ElementNode in TorsionLibraryInfo["HierarchySubClassNodes"]:
            Name = ElementNode.get("name")
            if len(Name) == 0:
                Name = "None"
            HierarchySubClassNames.append(Name)
        
        HierarchyClassSubName = "/".join(HierarchySubClassNames)
    
    # Replace spaces by underscores in class and subclass names...
    if HierarchyClassName is not None:
        if " " in HierarchyClassName:
            HierarchyClassName = HierarchyClassName.replace(" ", "_")
    
    if HierarchyClassSubName is not None:
        if " " in HierarchyClassSubName:
            HierarchyClassSubName = HierarchyClassSubName.replace(" ", "_")
        
    return (HierarchyClassName, HierarchyClassSubName)

def SetupTorsionRuleAnglesInfo(TorsionLibraryInfo, TorsionRuleElementNode):
    """Setup torsion angles and energy info for matching a torsion rule.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.
        TorsionRuleElementNode (object): A torsion rule element node being
           matched in torsion library XML tree.

    Returns:
        dict: A dictionary containing the following information for torsion rule
            being matched to a rotatable bond:
                
            RuleAnglesInfo = {}
            
            RuleAnglesInfo["IDs"] = []
            RuleAnglesInfo["Value"] = {}
            RuleAnglesInfo["Score"] = {}
            RuleAnglesInfo["Tolerance1"] = {}
            RuleAnglesInfo["Tolerance2"] = {}
            
            RuleAnglesInfo["ValuesList"] = []
            RuleAnglesInfo["ValuesIn360RangeList"] = []
            RuleAnglesInfo["Tolerances1List"] = []
            RuleAnglesInfo["Tolerances2List"] = []
             
            # Strain energy calculations...
            RuleAnglesInfo["EnergyMethod"] = None
            RuleAnglesInfo["EnergyMethodExact"] = None
            RuleAnglesInfo["EnergyMethodApproximate"] = None
            
            # For approximate strain energy calculation...
            RuleAnglesInfo["Beta1"] = {}
            RuleAnglesInfo["Beta2"] = {}
            RuleAnglesInfo["Theta0"] = {}
            
            # For exact strain energy calculation...
            RuleAnglesInfo["HistogramEnergy"] = []
            RuleAnglesInfo["HistogramEnergyLowerBound"] = []
            RuleAnglesInfo["HistogramEnergyUpperBound"] = []
                
    """
    # Check data cache...
    TorsionRuleNodeID = TorsionRuleElementNode.get("NodeID")
    if TorsionRuleNodeID in TorsionLibraryInfo["DataCache"]["TorsionRuleAnglesInfo"]:
        return TorsionLibraryInfo["DataCache"]["TorsionRuleAnglesInfo"][TorsionRuleNodeID]
    
    # Initialize rule angles info...
    RuleAnglesInfo = {}
    
    RuleAnglesInfo["IDs"] = []
    RuleAnglesInfo["Value"] = {}
    RuleAnglesInfo["Score"] = {}
    RuleAnglesInfo["Tolerance1"] = {}
    RuleAnglesInfo["Tolerance2"] = {}
    
    RuleAnglesInfo["ValuesList"] = []
    RuleAnglesInfo["ValuesIn360RangeList"] = []
    RuleAnglesInfo["Tolerances1List"] = []
    RuleAnglesInfo["Tolerances2List"] = []

    # Strain energy calculations...
    RuleAnglesInfo["EnergyMethod"] = None
    RuleAnglesInfo["EnergyMethodExact"] = None
    RuleAnglesInfo["EnergyMethodApproximate"] = None

    # For approximate strain energy calculation....
    RuleAnglesInfo["Beta1"] = {}
    RuleAnglesInfo["Beta2"] = {}
    RuleAnglesInfo["Theta0"] = {}
    
    # For exact strain energy calculation...
    RuleAnglesInfo["HistogramEnergy"] = []
    RuleAnglesInfo["HistogramEnergyLowerBound"] = []
    RuleAnglesInfo["HistogramEnergyUpperBound"] = []

    # Setup strain energy calculation information...
    EnergyMethod, EnergyMethodExact, EnergyMethodApproximate = [None] * 3
    EnergyMethod = TorsionRuleElementNode.get("method")
    if EnergyMethod is not None:
        EnergyMethodExact = True if re.match("^exact$", EnergyMethod, re.I) else False
        EnergyMethodApproximate = True if re.match("^approximate$", EnergyMethod, re.I) else False
    
    RuleAnglesInfo["EnergyMethod"] = EnergyMethod
    RuleAnglesInfo["EnergyMethodExact"] = EnergyMethodExact
    RuleAnglesInfo["EnergyMethodApproximate"] = EnergyMethodApproximate
    
    # Setup angles information....
    AngleID = 0
    for AngleListElementNode in TorsionRuleElementNode.findall("angleList"):
        for AngleNode in AngleListElementNode.iter("angle"):
            AngleID += 1
            Value = float(AngleNode.get("value"))
            Tolerance1 = float(AngleNode.get("tolerance1"))
            Tolerance2 = float(AngleNode.get("tolerance2"))
            Score = float(AngleNode.get("score"))

            # Track values...
            RuleAnglesInfo["IDs"].append(AngleID)
            RuleAnglesInfo["Value"][AngleID] = Value
            RuleAnglesInfo["Score"][AngleID] = Score
            RuleAnglesInfo["Tolerance1"][AngleID] = Tolerance1
            RuleAnglesInfo["Tolerance2"][AngleID] = Tolerance2

            RuleAnglesInfo["ValuesList"].append(Value)
            RuleAnglesInfo["Tolerances1List"].append(Tolerance1)
            RuleAnglesInfo["Tolerances2List"].append(Tolerance2)

            # Map value to 0 to 360 range...
            MappedValue = Value + 360 if Value < 0 else Value
            RuleAnglesInfo["ValuesIn360RangeList"].append(MappedValue)

            # Approximate strain energy calculation information...
            if EnergyMethodApproximate:
                Beta1 = float(AngleNode.get("beta_1"))
                Beta2 = float(AngleNode.get("beta_2"))
                Theta0 = float(AngleNode.get("theta_0"))
                
                RuleAnglesInfo["Beta1"][AngleID] = Beta1
                RuleAnglesInfo["Beta2"][AngleID] = Beta2
                RuleAnglesInfo["Theta0"][AngleID] = Theta0
    
    #  Exact energy method  information...
    if EnergyMethodExact:
        for HistogramBinNode in TorsionRuleElementNode.find("histogram_converted").iter("bin"):
            Energy = float(HistogramBinNode.get("energy"))
            Lower = float(HistogramBinNode.get("lower"))
            Upper = float(HistogramBinNode.get("upper"))
            
            RuleAnglesInfo["HistogramEnergy"].append(Energy)
            RuleAnglesInfo["HistogramEnergyLowerBound"].append(Lower)
            RuleAnglesInfo["HistogramEnergyUpperBound"].append(Upper)
    
    if len(RuleAnglesInfo["IDs"]) == 0:
        RuleInfo = None
    
    # Cache data...
    TorsionLibraryInfo["DataCache"]["TorsionRuleAnglesInfo"][TorsionRuleNodeID] = RuleAnglesInfo

    return RuleAnglesInfo

def DoesSMARTSContainValidSubClassMappedAtoms(SMARTS):
    """Check for the presence of two central mapped atoms in SMARTS pattern.
    A valid SMARTS pattern must contain only two mapped atoms corresponding
    to map atom numbers ':2' and ':3'.
    
    Arguments:
        SMARTS (str): SMARTS pattern for sub class in torsion library XML tree.

    Returns:
        bool: True - A valid pattern; Otherwise, false.

    """
    MatchedMappedAtoms = re.findall(":[0-9]", SMARTS, re.I)
    if len(MatchedMappedAtoms) < 2 or len(MatchedMappedAtoms) > 4:
        return False

    # Check for the presence of two central atom map numbers for a torsion...
    for MapAtomNum in [":2", ":3"]:
        if MapAtomNum not in MatchedMappedAtoms:
            return False
    
    return True

def DoesSMARTSContainValidTorsionRuleMappedAtoms(SMARTS):
    """Check for the presence of four mapped atoms in a SMARTS pattern.
    A valid SMARTS pattern must contain only four mapped atoms corresponding
    to map atom numbers ':1', ':2', ':3' and ':4'.
    
    Arguments:
        SMARTS (str): SMARTS pattern for torsion rule in torsion library XML
            tree.

    Returns:
        bool: True - A valid pattern; Otherwise, false.

    """
    MatchedMappedAtoms = re.findall(":[0-9]", SMARTS, re.I)
    if len(MatchedMappedAtoms) != 4:
        return False

    # Check for the presence of four atom map numbers for a torsion...
    for MapAtomNum in [":1", ":2", ":3", ":4"]:
        if MapAtomNum not in MatchedMappedAtoms:
            return False
    
    return True

def DoesSMARTSContainsMappedAtoms(SMARTS, MappedAtomNumsList):
    """Check for the presence of specified mapped atoms in SMARTS pattern.
    The mapped atom numbers in the list are specified as ':1', ':2', ':3' etc.
    
    Arguments:
        SMARTS (str): SMARTS pattern in torsion library XML tree.
        MappedAtoms (list): Mapped atom numbers as ":1", ":2" etc.

    Returns:
        bool: True - All mapped atoms present in pattern; Otherwise, false.

    """
    MatchedMappedAtoms = re.findall(":[0-9]", SMARTS, re.I)
    if len(MatchedMappedAtoms) == 0:
        return False

    # Check for the presence of specified mapped atoms in pattern...
    for MapAtomNum in MappedAtomNumsList:
        if MapAtomNum not in MatchedMappedAtoms:
            return False
    
    return True

def IsSpecificHierarchyClass(TorsionLibraryInfo, HierarchyClass):
    """Check whether it's a specific hierarchy class.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.
        HierarchyClass (str): Hierarchy class name.

    Returns:
        bool: True - A valid hierarchy class name; Otherwise, false.

    """
    return True if HierarchyClass in TorsionLibraryInfo["SpecificClasses"]["ElementNode"] else False

def GetGenericHierarchyClassElementNode(TorsionLibraryInfo):
    """Get generic hierarchy class element node.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.

    Returns:
        object: Generic hierarchy class element node in torsion library XML
            tree.

    """
    return TorsionLibraryInfo["GenericClassElementNode"]

def TrackHierarchyClassElementNode(TorsionLibraryInfo, ElementNode):
    """Track hierarchy class element node using a stack.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.
        ElementNode (object): Hierarchy class element node in torsion library
            XML tree. 

    Returns:
        Nothing. The torsion library info is updated.

    """
    TorsionLibraryInfo["HierarchyClassNodes"].append(ElementNode)

def RemoveLastHierarchyClassElementNodeFromTracking(TorsionLibraryInfo):
    """Remove last hierarchy class element node from tracking by removing it
    from a stack.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.

    Returns:
        Nothing. The torsion library info is updated.

    """
    TorsionLibraryInfo["HierarchyClassNodes"].pop()

def TrackHierarchySubClassElementNode(TorsionLibraryInfo, ElementNode):
    """Track hierarchy sub class element node using a stack.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.
        ElementNode (object): Hierarchy sub class element node in torsion
            library XML tree. 

    Returns:
        Nothing. The torsion library info is updated.

    """
    TorsionLibraryInfo["HierarchySubClassNodes"].append(ElementNode)

def RemoveLastHierarchySubClassElementNodeFromTracking(TorsionLibraryInfo):
    """Remove last hierarchy sub class element node from tracking by removing it
    from a stack.
    
    Arguments:
        TorsionLibraryInfo (dict): A dictionary containing information for
            matching rotatable bonds.

    Returns:
        Nothing. The torsion library info is updated.

    """
    TorsionLibraryInfo["HierarchySubClassNodes"].pop()

def CalculateTorsionAngleDifference(TorsionAngle1, TorsionAngle2):
    """Calculate torsion angle difference in the range from 0 to 180.
    
    Arguments:
        TorsionAngle1 (float): First torsion angle.
        TorsionAngle2 (float): Second torsion angle.

    Returns:
        float: Difference between first and second torsion angle.

    """

    # Map angles to 0 to 360 range...
    if TorsionAngle1 < 0:
        TorsionAngle1 = TorsionAngle1 + 360
    if TorsionAngle2 < 0:
        TorsionAngle2 = TorsionAngle2 + 360

    # Calculate and map angle difference in the range from 0 to 180 range...
    TorsionAngleDiff = abs(TorsionAngle1 - TorsionAngle2)
    if TorsionAngleDiff > 180.0:
        TorsionAngleDiff = abs(TorsionAngleDiff - 360)

    return TorsionAngleDiff
