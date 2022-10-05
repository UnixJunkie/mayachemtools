#
# File: RDKitUtil.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2022 Manish Sud. All rights reserved.
#
# The functionality available in this file is implemented using RDKit, an
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

import os
import sys
import re
import base64
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import MiscUtil

__all__ = ["AreAtomIndicesSequentiallyConnected", "AreAtomMapNumbersPresentInMol", "AreHydrogensMissingInMolecule", "ClearAtomMapNumbers", "ConstrainAndEmbed", "FilterSubstructureMatchByAtomMapNumbers", "FilterSubstructureMatchesByAtomMapNumbers", "GetAtomIndices", "GetAtomMapIndices", "GetAtomMapIndicesAndMapNumbers", "GetAtomSymbols", "GetAtomPositions", "GetFormalCharge", "GetHeavyAtomNeighbors", "GetInlineSVGForMolecule", "GetInlineSVGForMolecules", "GetMolName", "GetNumFragments", "GetNumHeavyAtomNeighbors", "GetSpinMultiplicity", "GetSVGForMolecule", "GetSVGForMolecules", "GetPsi4XYZFormatString", "GenerateBase64EncodedMolStrings", "GenerateBase64EncodedMolStringWithConfIDs", "IsAtomSymbolPresentInMol", "IsMolEmpty", "IsValidElementSymbol", "IsValidAtomIndex", "MolFromBase64EncodedMolString", "GenerateBase64EncodedMolStringsWithIDs", "MolToBase64EncodedMolString", "MolFromSubstructureMatch", "MolsFromSubstructureMatches", "ReadMolecules", "ReadAndValidateMolecules", "ReadMoleculesFromSDFile", "ReadMoleculesFromMolFile", "ReadMoleculesFromMol2File", "ReadMoleculesFromPDBFile", "ReadMoleculesFromSMILESFile", "ReorderAtomIndicesInSequentiallyConnectedManner", "SetAtomPositions", "SetWriterMolProps", "ValidateElementSymbols", "WriteMolecules"]

def GetMolName(Mol, MolNum = None):
    """Get molecule name.
    
    Arguments:
        Mol (object): RDKit molecule object.
        MolNum (int or None): Molecule number in input file.

    Returns:
        str : Molname corresponding to _Name property of a molecule, generated
            from specieid MolNum using the format "Mol%d" % MolNum, or an
            empty string.

    """
    
    MolName = ''
    if Mol.HasProp("_Name"):
        MolName = Mol.GetProp("_Name")

    if not len(MolName):
        if MolNum is not None:
            MolName = "Mol%d" % MolNum
    
    return MolName

def GetInlineSVGForMolecule(Mol, Width, Height, Legend = None, AtomListToHighlight = None, BondListToHighlight = None, BoldText = True, Base64Encoded = True):
    """Get SVG image text for a molecule suitable for inline embedding into a HTML page.
    
    Arguments:
        Mol (object): RDKit molecule object.
        Width (int): Width of a molecule image in pixels.
        Height (int): Height of a molecule image in pixels.
        Legend (str): Text to display under the image.
        AtomListToHighlight (list): List of atoms to highlight.
        BondListToHighlight (list): List of bonds to highlight.
        BoldText (bool): Flag to make text bold in the image of molecule. 
        Base64Encoded (bool): Flag to return base64 encoded string. 

    Returns:
        str : SVG image text for inline embedding into a HTML page using "img"
            tag: <img src="data:image/svg+xml;charset=UTF-8,SVGImageText> or
            tag: <img src="data:image/svg+xml;base64,SVGImageText>

    """

    SVGText = GetSVGForMolecule(Mol, Width, Height, Legend, AtomListToHighlight, BondListToHighlight, BoldText)
    return _ModifySVGForInlineEmbedding(SVGText, Base64Encoded)
    
def GetInlineSVGForMolecules(Mols, MolsPerRow, MolWidth, MolHeight, Legends = None, AtomListsToHighlight = None, BondListsToHighLight = None, BoldText = True, Base64Encoded = True):
    """Get SVG image text for  molecules suitable for inline embedding into a HTML page.
    
    Arguments:
        Mols (list): List of RDKit molecule objects.
        MolsPerRow (int): Number of molecules per row.
        Width (int): Width of a molecule image in pixels.
        Height (int): Height of a molecule image in pixels.
        Legends (list): List containing strings to display under images.
        AtomListsToHighlight (list): List of lists containing atoms to highlight
            for molecules.
        BondListsToHighlight (list): List of lists containing bonds to highlight
            for molecules
        BoldText (bool): Flag to make text bold in the image of molecules. 
        Base64Encoded (bool): Flag to return base64 encoded string. 

    Returns:
        str : SVG image text for inline embedding into a HTML page using "img"
            tag: <img src="data:image/svg+xml;charset=UTF-8,SVGImageText> or
            tag: <img src="data:image/svg+xml;base64,SVGImageText>

    """
    
    SVGText = GetSVGForMolecules(Mols, MolsPerRow, MolWidth, MolHeight, Legends, AtomListsToHighlight, BondListsToHighLight, BoldText)
    return _ModifySVGForInlineEmbedding(SVGText, Base64Encoded)

def _ModifySVGForInlineEmbedding(SVGText, Base64Encoded):
    """Modify SVG for inline embedding into a HTML page using "img" tag
    along with performing base64 encoding.
    """
    
    # Take out all tags till the start of '<svg' tag...
    Pattern = re.compile("^.*<svg", re.I | re.S)
    SVGText = Pattern.sub("<svg", SVGText)
    
    # Add an extra space before the "width=..." tag. Otherwise, inline embedding may
    # cause the following XML error on some browsers due to start of the "width=..."
    # at the begining of the line in <svg ...> tag:
    #
    #  XML5607: Whitespace expected.
    #
    SVGText = re.sub("width='", " width='", SVGText, flags = re.I)
    
    # Take out trailing new line...
    SVGText = SVGText.strip()

    # Perform base64 encoding by turning text into byte stream using string
    # encode and transform byte stream returned by b64encode into a string
    # by string decode...
    #
    if Base64Encoded:
        SVGText = base64.b64encode(SVGText.encode()).decode()

    return SVGText

def GetSVGForMolecule(Mol, Width, Height, Legend = None, AtomListToHighlight = None, BondListToHighlight = None, BoldText = True):
    """Get SVG image text for a molecule suitable for viewing in a browser.
    
    Arguments:
        Mol (object): RDKit molecule object.
        Width (int): Width of a molecule image in pixels.
        Height (int): Height of a molecule image in pixels.
        Legend (str): Text to display under the image.
        AtomListToHighlight (list): List of atoms to highlight.
        BondListToHighlight (list): List of bonds to highlight.
        BoldText (bool): Flag to make text bold in the image of molecule. 

    Returns:
        str : SVG image text for writing to a SVG file for viewing in a browser.

    """
    
    Mols = [Mol]
    
    MolsPerRow = 1
    MolWidth = Width
    MolHeight = Height
    
    Legends = [Legend] if Legend is not None else None
    AtomListsToHighlight = [AtomListToHighlight] if AtomListToHighlight is not None else None
    BondListsToHighLight = [BondListsToHighLight] if BondListToHighlight is not None else None
    
    return GetSVGForMolecules(Mols, MolsPerRow, MolWidth, MolHeight, Legends, AtomListsToHighlight, BondListsToHighLight, BoldText)

def GetSVGForMolecules(Mols, MolsPerRow, MolWidth, MolHeight, Legends = None, AtomListsToHighlight = None, BondListsToHighlight = None, BoldText = True):
    """Get SVG image text for molecules suitable for viewing in a browser.
    
    Arguments:
        Mols (list): List of RDKit molecule objects.
        MolsPerRow (int): Number of molecules per row.
        Width (int): Width of a molecule image in pixels.
        Height (int): Height of a molecule image in pixels.
        Legends (list): List containing strings to display under images.
        AtomListsToHighlight (list): List of lists containing atoms to highlight
            for molecules.
        BondListsToHighlight (list): List of lists containing bonds to highlight
            for molecules
        BoldText (bool): Flag to make text bold in the image of molecules. 

    Returns:
        str : SVG image text for writing to a SVG file for viewing in a browser.

    """
    
    SVGText = Draw.MolsToGridImage(Mols, molsPerRow = MolsPerRow, subImgSize = (MolWidth,MolHeight), legends = Legends, highlightAtomLists = AtomListsToHighlight, highlightBondLists = BondListsToHighlight, useSVG = True)
    
    return _ModifySVGForBrowserViewing(SVGText, BoldText)

def _ModifySVGForBrowserViewing(SVGText, BoldText = True):
    """Modify SVG for loading into a browser."""
    
    # It appears that the string 'xmlns:svg' needs to be replaced with 'xmlns' in the
    # SVG image string generated by older versions of RDKit. Otherwise, the image
    # doesn't load in web browsers.
    #
    if re.search("xmlns:svg", SVGText, re.I):
        SVGText = re.sub("xmlns:svg", "xmlns", SVGText, flags = re.I)
    
    # Make text bold...
    if BoldText:
        SVGText = re.sub("font-weight:normal;", "font-weight:bold;", SVGText, flags = re.I)
    
    return SVGText

def IsMolEmpty(Mol):
    """Check for the presence of atoms in a molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        bool : True - No atoms in molecule; Otherwise, false. 

    """

    Status = False if Mol.GetNumAtoms() else True
    
    return Status

def IsAtomSymbolPresentInMol(Mol, AtomSymbol, IgnoreCase = True):
    """ Check for the presence of an atom symbol in a molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.
        AtomSymbol (str): Atom symbol.

    Returns:
        bool : True - Atom symbol in molecule; Otherwise, false. 

    """
    
    for Atom in Mol.GetAtoms():
        Symbol = Atom.GetSymbol()
        if IgnoreCase:
            if re.match("^%s$" % AtomSymbol, Symbol, re.I):
                return True
        else:
            if re.match("^%s$" % AtomSymbol, Symbol):
                return True
    
    return False

def ValidateElementSymbols(ElementSymbols):
    """Validate element symbols.
    
    Arguments:
        ElementSymbols (list): List of element symbols to validate.

    Returns:
        bool : True - All element symbols are valid; Otherwise, false. 

    """
    for ElementSymbol in ElementSymbols:
        if not IsValidElementSymbol(ElementSymbol):
            return False
    
    return True

def GetAtomPositions(Mol, ConfID = -1):
    """Retrieve a list of lists containing coordinates of all atoms in a
    molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.
        ConfID (int): Conformer number.

    Returns:
        list : List of lists containing atom positions.

    Examples:

        for AtomPosition in RDKitUtil.GetAtomPositions(Mol):
            print("X: %s; Y: %s; Z: %s" % (AtomPosition[0], AtomPosition[1], AtomPosition[2]))

    """

    return Mol.GetConformer(id = ConfID).GetPositions().tolist()

def SetAtomPositions(Mol, AtomPositions, ConfID = -1):
    """Set atom positions of all atoms in a molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.
        AtomPositions (object): List of lists containing atom positions.
        ConfID (int): Conformer number.

    Returns:
        object : RDKit molecule object.

    """
    
    MolConf = Mol.GetConformer(ConfID)

    for Index in range(len(AtomPositions)):
            MolConf.SetAtomPosition(Index, tuple(AtomPositions[Index]))
    
    return Mol

def GetAtomSymbols(Mol):
    """Retrieve a list containing atom symbols of all atoms a molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        list : List of atom symbols.

    """

    return [Atom.GetSymbol() for Atom in Mol.GetAtoms()]

def GetAtomIndices(Mol):
    """Retrieve a list containing atom indices of all atoms a molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        list : List of atom indices.

    """

    return [Atom.GetIdx() for Atom in Mol.GetAtoms()]

def GetFormalCharge(Mol, CheckMolProp = True):
    """Get formal charge of a molecule. The formal charge is either retrieved
    from 'FormalCharge' molecule property or calculated using RDKit function
    Chem.GetFormalCharge(Mol).

    Arguments:
        Mol (object): RDKit molecule object.
        CheckMolProp (bool): Check 'FormalCharge' molecule property to
            retrieve formal charge.

    Returns:
        int : Formal charge.

    """
    
    Name = 'FormalCharge'
    if (CheckMolProp and Mol.HasProp(Name)):
        FormalCharge = int(float(Mol.GetProp(Name)))
    else:
        FormalCharge =  Chem.GetFormalCharge(Mol)
    
    return FormalCharge;

def GetSpinMultiplicity(Mol, CheckMolProp = True):
    """Get spin multiplicity of a molecule. The spin multiplicity is either
    retrieved from 'SpinMultiplicity' molecule property or calculated from
    from the number of free radical electrons using Hund's rule of maximum
    multiplicity defined as 2S + 1 where S is the total electron spin. The
    total spin is 1/2 the number of free radical electrons in a molecule.

    Arguments:
        Mol (object): RDKit molecule object.
        CheckMolProp (bool): Check 'SpinMultiplicity' molecule property to
            retrieve spin multiplicity.

    Returns:
        int : Spin multiplicity.

    """
    
    Name = 'SpinMultiplicity'
    if (CheckMolProp and Mol.HasProp(Name)):
        return int(float(Mol.GetProp(Name)))

    # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
    NumRadicalElectrons = 0
    for Atom in Mol.GetAtoms():
        NumRadicalElectrons += Atom.GetNumRadicalElectrons()

    TotalElectronicSpin = NumRadicalElectrons/2
    SpinMultiplicity = 2 * TotalElectronicSpin + 1
    
    return int(SpinMultiplicity)

def GetPsi4XYZFormatString(Mol, ConfID = -1, FormalCharge = "auto", SpinMultiplicity = "auto", Symmetry = "auto", NoCom = False, NoReorient = False, CheckFragments = False):
    """Retrieve geometry string of a molecule in Psi4ish XYZ format to perform
    Psi4 quantum chemistry calculations.

    Arguments:
        Mol (object): RDKit molecule object.
        ConfID (int): Conformer number.
        FormalCharge (str): Specified formal charge or 'auto' to calculate
           its value
        SpinMultiplicity (str): Specified spin multiplicity or 'auto' to calculate
           its value.
        Symmetry (str): Specified symmetry or 'auto' to calculate its value.
        NoCom (bool): Flag to disable recentering of a molecule by Psi4.
        NoReorient (bool): Flag to disable reorientation of a molecule by Psi4.
        CheckFragments (bool): Check for fragments and setup geometry string
           using  -- separator between fragments.

    Returns:
        str : Geometry string of a molecule in Psi4ish XYZ format.

    """

    # Setup formal charge for molecule...
    if re.match("^auto$", FormalCharge, re.I):
        MolFormalCharge = GetFormalCharge(Mol)
    else:
        MolFormalCharge = int(FormalCharge)
        
    # Setup spin multiplicity for molecule...
    if re.match("^auto$", SpinMultiplicity, re.I):
        MolSpinMultiplicity = GetSpinMultiplicity(Mol)
    else:
        MolSpinMultiplicity = int(SpinMultiplicity)
    
    # Check for fragments...
    Mols = [Mol]
    if CheckFragments:
        Fragments = list(Chem.rdmolops.GetMolFrags(Mol, asMols = True))
        if len(Fragments) > 1:
            Mols = Fragments

    # Setup geometry string for Ps4...
    GeometryList = []
    FragMolCount = 0
    
    for FragMol in Mols:
        FragMolCount += 1
        if FragMolCount > 1:
            GeometryList.append("--")
            
        AtomSymbols = GetAtomSymbols(FragMol)
        AtomPositions = GetAtomPositions(FragMol, ConfID)
        
        for AtomSymbol, AtomPosition in zip(AtomSymbols, AtomPositions):
            GeometryList.append("%s %s %s %s" % (AtomSymbol, AtomPosition[0], AtomPosition[1], AtomPosition[2]))

        GeometryList.append("%s %s" % (MolFormalCharge, MolSpinMultiplicity))
        
    GeometryList.append("units angstrom")
    
    if not re.match("^auto$", Symmetry, re.I):
        Name = 'Symmetry'
        if (Mol.HasProp(Name)):
            Symmetry =  Mol.GetProp(Name)
        GeometryList.append("symmetry %s" % Symmetry)
    
    if NoCom:
        GeometryList.append("no_com")
        
    if NoReorient:
        GeometryList.append("no_reorient")

    Geometry = "\n".join(GeometryList)

    return Geometry

def GetNumFragments(Mol):
    """Get number of fragment in a molecule.

    Arguments:
        Atom (object): RDKit molecule object.

    Returns:
        int : Number of fragments.

    """
    
    Fragments = Chem.rdmolops.GetMolFrags(Mol, asMols = False)
    
    return len(Fragments) if Fragments is not None else 0

def GetNumHeavyAtomNeighbors(Atom):
    """Get number of heavy atom neighbors.

    Arguments:
        Atom (object): RDKit atom object.

    Returns:
        int : Number of neighbors.

    """
    
    NbrCount = 0
    for AtomNbr in Atom.GetNeighbors():
        if AtomNbr.GetAtomicNum() > 1:
            NbrCount += 1
    
    return NbrCount

def GetHeavyAtomNeighbors(Atom):
    """Get a list of heavy atom neighbors.

    Arguments:
        Atom (object): RDKit atom object.

    Returns:
        list : List of heavy atom neighbors.

    """
    
    AtomNeighbors = []
    for AtomNbr in Atom.GetNeighbors():
        if AtomNbr.GetAtomicNum() > 1:
            AtomNeighbors.append(AtomNbr)
    
    return AtomNeighbors

def IsValidElementSymbol(ElementSymbol):
    """Validate element symbol.
    
    Arguments:
        ElementSymbol (str): Element symbol

    Returns:
        bool : True - Valid element symbol; Otherwise, false. 

    """

    try:
        AtomicNumber = Chem.GetPeriodicTable().GetAtomicNumber(ElementSymbol)
        Status = True if AtomicNumber > 0  else False
    except Exception as ErrMsg:
        Status = False
    
    return Status

def IsValidAtomIndex(Mol, AtomIndex):
    """Validate presence  atom index in a molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.
        AtomIndex (int): Atom index.

    Returns:
        bool : True - Valid atom index; Otherwise, false. 

    """
    for Atom in Mol.GetAtoms():
        if AtomIndex == Atom.GetIdx():
            return True
    
    return False

def AreHydrogensMissingInMolecule(Mol):
    """Check for any missing hydrogens in  in a molecue.

    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        bool : True - Missing hydrogens; Otherwise, false. 

    """

    for Atom in Mol.GetAtoms():
        NumExplicitAndImplicitHs = Atom.GetNumExplicitHs() + Atom.GetNumImplicitHs()
        if NumExplicitAndImplicitHs > 0:
            return True

    return False

def AreAtomIndicesSequentiallyConnected(Mol, AtomIndices):
    """Check for the presence bonds between sequential pairs of atoms in a
    molecule.
    
    Arguments:
        Mol (object): RDKit molecule object.
        AtomIndices (list): List of atom indices.

    Returns:
        bool : True - Sequentially connected; Otherwise, false. 

    """

    for Index in range(0, (len(AtomIndices) -1)):
        Bond = Mol.GetBondBetweenAtoms(AtomIndices[Index], AtomIndices[Index + 1])
        if Bond is None:
            return False
        
        if Bond.GetIdx() is None:
            return False
    
    return True
    
def ReorderAtomIndicesInSequentiallyConnectedManner(Mol, AtomIndices):
    """Check for the presence of sequentially connected list of atoms in an
    arbitray list of atoms in molecule.
   
    Arguments:
        Mol (object): RDKit molecule object.
        AtomIndices (list): List of atom indices.

    Returns:
        bool : True - Sequentially connected list found; Otherwise, false. 
        list : List of seqeuntially connected atoms or None.

    """
    
    # Count the number of neighbors for specified atom indices ensuring
    # that the neighbors are also part of atom indices...
    AtomNbrsCount = {}
    for AtomIndex in AtomIndices:
        Atom = Mol.GetAtomWithIdx(AtomIndex)
        
        AtomNbrsCount[AtomIndex] = 0
        for AtomNbr in Atom.GetNeighbors():
            AtomNbrIndex = AtomNbr.GetIdx()
            if AtomNbrIndex not in AtomIndices:
                continue
            AtomNbrsCount[AtomIndex] += 1
    
    # Number of neighbors for each specified atom indices must be 1 or 2
    # for sequentially connected list of atom indices...
    AtomsWithOneNbr = []
    for AtomIndex, NbrsCount  in AtomNbrsCount.items():
        if not (NbrsCount == 1 or NbrsCount ==2):
            return (False, None)
        
        if NbrsCount == 1:
            AtomsWithOneNbr.append(AtomIndex)

    # A sequentially connected list of indices must have two atom indices with
    # exactly # one neighbor...
    if len(AtomsWithOneNbr) != 2:
            return (False, None)

    # Setup a reordered list of sequentially connected atoms...
    ReorderedAtomIndices = []
    
    AtomIndex1, AtomIndex2 = AtomsWithOneNbr
    AtomIndex = AtomIndex1 if AtomIndex1 < AtomIndex2 else AtomIndex2
    ReorderedAtomIndices.append(AtomIndex)

    while (len(ReorderedAtomIndices) < len(AtomIndices)):
        Atom = Mol.GetAtomWithIdx(AtomIndex)
        
        for AtomNbr in Atom.GetNeighbors():
            AtomNbrIndex = AtomNbr.GetIdx()
            if AtomNbrIndex not in AtomIndices:
                continue
            
            if AtomNbrIndex in ReorderedAtomIndices:
                continue
            
            # Treat neighbor as next connected atom...
            AtomIndex = AtomNbrIndex
            ReorderedAtomIndices.append(AtomIndex)
            break

    # Check reorderd list size...
    if (len(ReorderedAtomIndices) != len(AtomIndices)):
        return (False, None)

    # A final check to validate reorderd list...
    if not AreAtomIndicesSequentiallyConnected(Mol, ReorderedAtomIndices):
        return (False, None)
    
    return (True, ReorderedAtomIndices)

def MolToBase64EncodedMolString(Mol, PropertyPickleFlags = Chem.PropertyPickleOptions.AllProps):
    """Encode RDkit molecule object into a base64 encoded string. The properties
    can be optionally excluded.
    
    The molecule is pickled using RDKit Mol.ToBinary() function before
    their encoding.
   
    Arguments:
        Mol (object): RDKit molecule object.
        PropertyPickleFlags: RDKit property pickle options.

    Returns:
        str : Base64 encode molecule string or None.

    Notes:
        The following property pickle flags are currently available in RDKit:
            
            Chem.PropertyPickleOptions.NoProps
            Chem.PropertyPickleOptions.MolProps
            Chem.PropertyPickleOptions.AtomProps
            Chem.PropertyPickleOptions.BondProps
            Chem.PropertyPickleOptions.PrivateProps
            Chem.PropertyPickleOptions.AllProps

    """

    return None if Mol is None else base64.b64encode(Mol.ToBinary(PropertyPickleFlags)).decode()

def MolFromBase64EncodedMolString(EncodedMol):
    """Generate a RDKit molecule object from a base64 encoded string.
    
    Arguments:
        str: Base64 encoded molecule string.

    Returns:
        object : RDKit molecule object or None.

    """

    return None if EncodedMol is None else Chem.Mol(base64.b64decode(EncodedMol))

def GenerateBase64EncodedMolStrings(Mols, PropertyPickleFlags = Chem.PropertyPickleOptions.AllProps):
    """Setup an iterator for generating base64 encoded molecule string
    from a RDKit molecule iterator. The iterator returns a list containing
    a molecule index and encoded molecule string or None.
    
    The molecules are pickled using RDKit Mol.ToBinary() function
    before their encoding.
    
    Arguments:
        iterator: RDKit molecules iterator.
        PropertyFlags: RDKit property pickle options.

    Returns:
        object : Base64 endcoded molecules iterator. The iterator returns a
            list containing a molecule index and an encoded molecule string
            or None.

    Notes:
        The following property pickle flags are currently available in RDKit:
            
            Chem.PropertyPickleOptions.NoProps
            Chem.PropertyPickleOptions.MolProps
            Chem.PropertyPickleOptions.AtomProps
            Chem.PropertyPickleOptions.BondProps
            Chem.PropertyPickleOptions.PrivateProps
            Chem.PropertyPickleOptions.AllProps

    Examples:

        EncodedMolsInfo = GenerateBase64EncodedMolStrings(Mols)
        for MolIndex, EncodedMol in EncodedMolsInfo:
            if EncodeMol is not None:
                Mol = MolFromBase64EncodedMolString(EncodedMol)

    """
    for MolIndex, Mol in enumerate(Mols):
        yield [MolIndex, None] if Mol is None else [MolIndex, MolToBase64EncodedMolString(Mol, PropertyPickleFlags)]

def GenerateBase64EncodedMolStringsWithIDs(Mols, MolIDs, PropertyPickleFlags = Chem.PropertyPickleOptions.AllProps):
    """Setup an iterator for generating base64 encoded molecule string
    from a RDKit molecule iterator. The iterator returns a list containing
    a molecule ID and encoded molecule string or None.
    
    The molecules are pickled using RDKit Mol.ToBinary() function
    before their encoding.
    
    Arguments:
        iterator: RDKit molecules iterator.
        MolIDs (list): Molecule IDs.
        PropertyFlags: RDKit property pickle options.

    Returns:
        object : Base64 endcoded molecules iterator. The iterator returns a
            list containing a molecule ID and an encoded molecule string
            or None.

    Notes:
        The following property pickle flags are currently available in RDKit:
            
            Chem.PropertyPickleOptions.NoProps
            Chem.PropertyPickleOptions.MolProps
            Chem.PropertyPickleOptions.AtomProps
            Chem.PropertyPickleOptions.BondProps
            Chem.PropertyPickleOptions.PrivateProps
            Chem.PropertyPickleOptions.AllProps

    Examples:

        EncodedMolsInfo = GenerateBase64EncodedMolStringsWithIDs(Mols)
        for MolID, EncodedMol in EncodedMolsInfo:
            if EncodeMol is not None:
                Mol = MolFromBase64EncodedMolString(EncodedMol)

    """
    for MolIndex, Mol in enumerate(Mols):
        yield [MolIDs[MolIndex], None] if Mol is None else [MolIDs[MolIndex], MolToBase64EncodedMolString(Mol, PropertyPickleFlags)]

def GenerateBase64EncodedMolStringWithConfIDs(Mol, MolIndex, ConfIDs, PropertyPickleFlags = Chem.PropertyPickleOptions.AllProps):
    """Setup an iterator generating base64 encoded molecule string for a 
    molecule. The iterator returns a list containing a molecule index, an encoded
    molecule string, and conf ID.
    
    The molecules are pickled using RDKit Mol.ToBinary() function
    before their encoding.
    
    Arguments:
        Mol (object): RDKit molecule object.
        MolIndex (int): Molecule index.
        ConfIDs (list): Conformer IDs.
        PropertyFlags: RDKit property pickle options.

    Returns:
        object : Base64 endcoded molecules iterator. The iterator returns a
            list containing a molecule index, an encoded molecule string, and
            conf ID.

    Notes:
        The following property pickle flags are currently available in RDKit:
            
            Chem.PropertyPickleOptions.NoProps
            Chem.PropertyPickleOptions.MolProps
            Chem.PropertyPickleOptions.AtomProps
            Chem.PropertyPickleOptions.BondProps
            Chem.PropertyPickleOptions.PrivateProps
            Chem.PropertyPickleOptions.AllProps

    Examples:

        EncodedMolsInfo = GenerateBase64EncodedMolStringWithConfIDs(Mol, MolIndex, ConfIDs)
        for MolIndex, EncodedMol, ConfID in EncodedMolsInfo:
            if EncodeMol is not None:
                Mol = MolFromBase64EncodedMolString(EncodedMol)

    """
    for ConfID in ConfIDs:
        yield [MolIndex, None, ConfID] if Mol is None else [MolIndex, MolToBase64EncodedMolString(Mol, PropertyPickleFlags), ConfID]

def AreAtomMapNumbersPresentInMol(Mol):
    """Check for the presence of atom map numbers in a molecue.
    
    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        bool : True - Atom map numbers present; Otherwise, false. 

    """

    return False if _GetAtomMapIndices(Mol) is None else True

def ClearAtomMapNumbers(Mol, AllowImplicitValence = True, ClearRadicalElectrons = True):
    """Check and clear atom map numbers in a molecule. In addition, allow implicit
    valence and clear radical electrons for atoms with associated map numbers.
    
    For example, the following atomic properties are assigned by RDKit to atom
    map number 1 in a molecule corresponding to SMILES C[C:1](C)C:
    
    NoImplicit: True; ImplicitValence: 0; ExplicitValence: 3; NumExplicitHs: 0;
    NumImplicitHs: 0; NumRadicalElectrons: 1
    
    This function clears atoms map numbers in the molecule leading to SMILES 
    CC(C)C, along with optionally updating atomic properties as shown below:
    
    NoImplicit: False; ImplicitValence: 1; ExplicitValence: 3; NumExplicitHs: 0;
    NumImplicitHs: 1; NumRadicalElectrons: 0
    
    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        Mol (object): RDKit molecule object.

    """
    
    AtomMapIndices = GetAtomMapIndices(Mol)
    
    if AtomMapIndices is None:
        return Mol
    
    for AtomMapIndex in AtomMapIndices:
        Atom = Mol.GetAtomWithIdx(AtomMapIndex)
        
        # Clear map number property 'molAtomMapNumber'...
        Atom.SetAtomMapNum(0)
        
        # Allow implit valence...
        if AllowImplicitValence:
            Atom.SetNoImplicit(False)
        
        # Set number of electrons to 0...
        if ClearRadicalElectrons:
            Atom.SetNumRadicalElectrons(0)
        
        Atom.UpdatePropertyCache()
    
    Mol.UpdatePropertyCache()

    return Mol

def GetAtomMapIndices(Mol):
    """Get a list of available atom indices corresponding to atom map numbers
    present in a SMILES/SMARTS pattern used for creating a molecule. The list of
    atom indices is sorted in ascending order by atom map numbers.
    
    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        list : List of atom indices sorted in the ascending order of atom map
            numbers or None.

    """
    
    return _GetAtomMapIndices(Mol)

def GetAtomMapIndicesAndMapNumbers(Mol):
    """Get lists of available atom indices and atom map numbers present in a
    SMILES/SMARTS pattern used for creating a molecule. Both lists are sorted
    in ascending order by atom map numbers.
    
    Arguments:
        Mol (object): RDKit molecule object.

    Returns:
        list : List of atom indices sorted in the ascending order of atom map
            numbers or None.
        list : List of atom map numbers sorted in the ascending order or None.

    """
    
    return (_GetAtomMapIndicesAndMapNumbers(Mol))

def MolFromSubstructureMatch(Mol, PatternMol, AtomIndices, FilterByAtomMapNums = False):
    """Generate a RDKit molecule object for a list of matched atom indices
    present in a pattern molecule. The list of atom indices correspond to a
    list retrieved by RDKit function GetSubstructureMatches using SMILES/SMARTS
    pattern. The atom indices are optionally filtered by mapping atom numbers
    to appropriate atom indices during the generation of the molecule. For
    example: [O:1]=[S:2](=[O])[C:3][C:4].
    
    Arguments:
        Mol (object): RDKit molecule object.
        PatternMol (object): RDKit molecule object for a SMILES/SMARTS pattern.
        AtomIndices (list): Atom indices.
        FilterByAtomMapNums (bool): Filter matches by atom map numbers.

    Returns:
        object : RDKit molecule object or None.

    """

    AtomMapIndices = _GetAtomMapIndices(PatternMol) if FilterByAtomMapNums else None

    return (_MolFromSubstructureMatch(Mol, PatternMol, AtomIndices, AtomMapIndices))

def MolsFromSubstructureMatches(Mol, PatternMol, AtomIndicesList, FilterByAtomMapNums = False):
    """Generate  a list of RDKit molecule objects for a list containing lists of
    matched atom indices present in a pattern molecule. The list of atom indices
    correspond to a list retrieved by RDKit function GetSubstructureMatches using
    SMILES/SMARTS pattern. The atom indices are optionally filtered by mapping
    atom numbers to appropriate atom indices during the generation of the molecule. For
    example: [O:1]=[S:2](=[O])[C:3][C:4].
     
    Arguments:
        Mol (object): RDKit molecule object.
        PatternMol (object): RDKit molecule object for a SMILES/SMARTS pattern.
        AtomIndicesList (list): A list of lists containing atom indices.
        FilterByAtomMapNums (bool): Filter matches by atom map numbers.

    Returns:
        list : A list of lists containg RDKit molecule objects or None.

    """

    AtomMapIndices = _GetAtomMapIndices(PatternMol) if FilterByAtomMapNums else None

    Mols = []
    for AtomIndices in AtomIndicesList:
        Mols.append(_MolFromSubstructureMatch(Mol, PatternMol, AtomIndices, AtomMapIndices))
    
    return Mols if len(Mols) else None

def FilterSubstructureMatchByAtomMapNumbers(Mol, PatternMol, AtomIndices):
    """Filter a list of matched atom indices by map atom numbers present in a
    pattern molecule. The list of atom indices correspond to a list retrieved by
    RDKit function GetSubstructureMatches using SMILES/SMARTS pattern. The
    atom map numbers are mapped to appropriate atom indices during the generation
    of molecules. For example: [O:1]=[S:2](=[O])[C:3][C:4].
    
    Arguments:
        Mol (object): RDKit molecule object.
        PatternMol (object): RDKit molecule object for a SMILES/SMARTS pattern.
        AtomIndices (list): Atom indices.

    Returns:
        list : A list of filtered atom indices.

    """
    AtomMapIndices = _GetAtomMapIndices(PatternMol)

    return _FilterSubstructureMatchByAtomMapNumbers(Mol, PatternMol, AtomIndices, AtomMapIndices)

def FilterSubstructureMatchesByAtomMapNumbers(Mol, PatternMol, AtomIndicesList):
    """Filter a list of lists containing matched atom indices by map atom numbers
    present in a pattern molecule. The list of atom indices correspond to a list retrieved by
    RDKit function GetSubstructureMatches using SMILES/SMARTS pattern. The
    atom map numbers are mapped to appropriate atom indices during the generation
    of molecules. For example: [O:1]=[S:2](=[O])[C:3][C:4].
     
    Arguments:
        Mol (object): RDKit molecule object.
        PatternMol (object): RDKit molecule object for a SMILES/SMARTS pattern.
        AtomIndicesList (list): A list of lists containing atom indices.

    Returns:
        list : A list of lists containing filtered atom indices.

    """
    AtomMapIndices = _GetAtomMapIndices(PatternMol)

    MatchedAtomIndicesList = []
    for AtomIndices in AtomIndicesList:
        MatchedAtomIndicesList.append(_FilterSubstructureMatchByAtomMapNumbers(Mol, PatternMol, AtomIndices, AtomMapIndices))
    
    return MatchedAtomIndicesList

def _MolFromSubstructureMatch(Mol, PatternMol, AtomIndices, AtomMapIndices):
    """Generate a RDKit molecule object for a list of matched atom indices and available
   atom map indices.
    """

    if AtomMapIndices is not None:
        MatchedAtomIndices = [AtomIndices[Index] for Index in AtomMapIndices]
    else:
        MatchedAtomIndices = list(AtomIndices)

    return _GetMolFromAtomIndices(Mol, MatchedAtomIndices)

def _GetAtomMapIndices(Mol):
    """Get a list of available atom indices corresponding to sorted atom map
    numbers present in a SMILES/SMARTS pattern used for creating a molecule.
    """
    
    AtomMapIndices, AtomMapNumbers = _GetAtomMapIndicesAndMapNumbers(Mol)
    
    return AtomMapIndices
    
def _GetAtomMapIndicesAndMapNumbers(Mol):
    """Get a list of available atom indices and atom map numbers present
    in  a SMILES/SMARTS pattern used for creating a molecule. Both lists
    are sorted in ascending order by atom map numbers.
    """

    # Setup a atom map number to atom indices map..
    AtomMapNumToIndices = {}
    for Atom in Mol.GetAtoms():
        AtomMapNum = Atom.GetAtomMapNum()
        
        if AtomMapNum:
            AtomMapNumToIndices[AtomMapNum] = Atom.GetIdx()
    
    # Setup atom indices corresponding to sorted atom map numbers...
    AtomMapIndices = None
    AtomMapNumbers = None
    if len(AtomMapNumToIndices):
        AtomMapNumbers = sorted(AtomMapNumToIndices)
        AtomMapIndices = [AtomMapNumToIndices[AtomMapNum] for AtomMapNum in AtomMapNumbers]

    return (AtomMapIndices, AtomMapNumbers)

def _FilterSubstructureMatchByAtomMapNumbers(Mol, PatternMol, AtomIndices, AtomMapIndices):
    """Filter substructure match atom indices by atom map indices corresponding to
    atom map numbers.
    """
    
    if AtomMapIndices is None:
        return list(AtomIndices)
                                               
    return [AtomIndices[Index] for Index in AtomMapIndices]

def _GetMolFromAtomIndices(Mol, AtomIndices):
    """Generate a RDKit molecule object from atom indices returned by
   substructure search.
    """

    BondIndices = []
    for AtomIndex in AtomIndices:
        Atom = Mol.GetAtomWithIdx(AtomIndex)
        
        for AtomNbr in Atom.GetNeighbors():
            AtomNbrIndex = AtomNbr.GetIdx()
            if AtomNbrIndex not in AtomIndices:
                continue
            
            BondIndex = Mol.GetBondBetweenAtoms(AtomIndex, AtomNbrIndex).GetIdx()
            if BondIndex in BondIndices:
                continue
                
            BondIndices.append(BondIndex)
            
    MatchedMol = Chem.PathToSubmol(Mol, BondIndices) if len(BondIndices) else None
    
    return MatchedMol

def ConstrainAndEmbed(mol, core, coreMatchesMol=None, useTethers=True, coreConfId=-1, randomseed=2342, getForceField=AllChem.UFFGetMoleculeForceField, **kwargs):
    """
    The function is a local copy of RDKit fucntion AllChem.ConstrainedEmbed().
    It has been enhanced to support an explicit list of core matches corresponding
    to the matched atom indices in the molecule. The number of matched atom indices
    must be equal to the number of atoms in core molecule.

    Arguments:
        mol (object): RDKit molecule object to embed.
        core (object): RDKit molecule to use as a source of constraints.
        coreMatchesMol (list): A list matches atom indices in mol.
        useTethers: (bool) if True, the final conformation will be optimized
            subject to a series of extra forces that pull the matching atoms to
            the positions of the core atoms. Otherwise simple distance
            constraints based on the core atoms will be used in the
            optimization.
        coreConfId (int): ID of the core conformation to use.
        randomSeed (int): Seed for the random number generator

    Returns:
        mol (object): RDKit molecule object.

    """
    if coreMatchesMol is None:
        match = mol.GetSubstructMatch(core)
        if not match:
            raise ValueError("Molecule doesn't match the core.")
    else:
        if core.GetNumAtoms() != len(coreMatchesMol):
            raise ValueError("Number of atoms, %s, in core molecule must match number of atom indices, %s, specified in the list coreMatchesMol." % (core.GetNumAtoms(), len(coreMatchesMol)))
        # Check specified matched atom indices in  coreMatchesMol and use the match atom
        # indices returned by GetSubstructMatches() for embedding...
        coreMatch = None
        matches = mol.GetSubstructMatches(core)
        for match in matches:
            if len(match) != len(coreMatchesMol):
                continue
            matchFound = True
            for atomIndex in match:
                if atomIndex not in coreMatchesMol:
                    matchFound = False
                    break
            if matchFound:
                coreMatch = match
                break
        if coreMatch is None:
            raise ValueError("Molecule doesn't match the atom indices specified in the list coreMatchesMol.")
        match = coreMatch
    
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI
    
    ci = AllChem.EmbedMolecule(mol, coordMap=coordMap, randomSeed=randomseed, **kwargs)
    if ci < 0:
        raise ValueError('Could not embed molecule.')
    
    algMap = [(j, i) for i, j in enumerate(match)]
    
    if not useTethers:
        # clean up the conformation
        ff = getForceField(mol, confId=0)
        for i, idxI in enumerate(match):
            for j in range(i + 1, len(match)):
                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.)
        ff.Initialize()
        n = 4
        more = ff.Minimize()
        while more and n:
            more = ff.Minimize()
            n -= 1
        # rotate the embedded conformation onto the core:
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)
        ff = getForceField(mol, confId=0)
        conf = core.GetConformer()
        for i in range(core.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AllChem.AlignMol(mol, core, atomMap=algMap)

    mol.SetProp('EmbedRMS', str(rms))
    return mol

def ReadAndValidateMolecules(FileName, **KeyWordArgs):
    """Read molecules from an input file, validate all molecule objects, and return
    a list of valid and non-valid molecule objects along with their counts.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        **KeyWordArgs (dictionary) : Parameter name and value pairs for reading and
            processing molecules.

    Returns:
        list : List of valid RDKit molecule objects.
        int : Number of total molecules in input file. 
        int : Number of valid molecules in input file. 

    Notes:
        The file extension is used to determine type of the file and set up an appropriate
        file reader.

    """

    AllowEmptyMols = True
    if "AllowEmptyMols" in KeyWordArgs:
        AllowEmptyMols = KeyWordArgs["AllowEmptyMols"]
    
    Mols = ReadMolecules(FileName, **KeyWordArgs)

    if AllowEmptyMols:
        ValidMols = [Mol for Mol in Mols if Mol is not None]
    else:
        ValidMols = []
        MolCount = 0
        for Mol in Mols:
            MolCount += 1
            if Mol is None:
                continue
            
            if IsMolEmpty(Mol):
                MolName = GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Ignoring empty molecule: %s" % MolName)
                continue
            
            ValidMols.append(Mol)
            
    MolCount = len(Mols)
    ValidMolCount = len(ValidMols)

    return (ValidMols, MolCount, ValidMolCount)

def ReadMolecules(FileName, **KeyWordArgs):
    """Read molecules from an input file without performing any validation
    and creation of molecule objects.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        **KeyWordArgs (dictionary) : Parameter name and value pairs for reading and
            processing molecules.

    Returns:
        list : List of RDKit molecule objects.

    Notes:
        The file extension is used to determine type of the file and set up an appropriate
        file reader.

    """

    # Set default values for possible arguments...
    ReaderArgs = {"Sanitize": True, "RemoveHydrogens": True, "StrictParsing": True,  "SMILESDelimiter" : ' ', "SMILESColumn": 1, "SMILESNameColumn": 2, "SMILESTitleLine": True }

    # Set specified values for possible arguments...
    for Arg in ReaderArgs:
        if Arg in KeyWordArgs:
            ReaderArgs[Arg] = KeyWordArgs[Arg]

    # Modify specific valeus for SMILES...
    if MiscUtil.CheckFileExt(FileName, "smi csv tsv txt"):
        Args = ["Sanitize", "SMILESTitleLine"]
        for Arg in Args:
            if ReaderArgs[Arg] is True:
                ReaderArgs[Arg] = 1
            else:
                ReaderArgs[Arg] = 0
    
    Mols = []
    if MiscUtil.CheckFileExt(FileName, "sdf sd"):
        return ReadMoleculesFromSDFile(FileName, ReaderArgs["Sanitize"], ReaderArgs["RemoveHydrogens"], ReaderArgs['StrictParsing'])
    elif MiscUtil.CheckFileExt(FileName, "mol"):
        return ReadMoleculesFromMolFile(FileName, ReaderArgs["Sanitize"], ReaderArgs["RemoveHydrogens"], ReaderArgs['StrictParsing'])
    elif MiscUtil.CheckFileExt(FileName, "mol2"):
        return ReadMoleculesFromMol2File(FileName, ReaderArgs["Sanitize"], ReaderArgs["RemoveHydrogens"])
    elif MiscUtil.CheckFileExt(FileName, "pdb"):
        return ReadMoleculesFromPDBFile(FileName, ReaderArgs["Sanitize"], ReaderArgs["RemoveHydrogens"])
    elif MiscUtil.CheckFileExt(FileName, "smi txt csv tsv"):
        SMILESColumnIndex = ReaderArgs["SMILESColumn"] - 1
        SMILESNameColumnIndex = ReaderArgs["SMILESNameColumn"] - 1
        return ReadMoleculesFromSMILESFile(FileName, ReaderArgs["SMILESDelimiter"], SMILESColumnIndex, SMILESNameColumnIndex, ReaderArgs["SMILESTitleLine"], ReaderArgs["Sanitize"])
    else:
        MiscUtil.PrintWarning("RDKitUtil.ReadMolecules: Non supported file type: %s" % FileName)
    
    return Mols

def ReadMoleculesFromSDFile(FileName, Sanitize = True, RemoveHydrogens = True, StrictParsing = True):
    """Read molecules from a SD file.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        Sanitize (bool): Sanitize molecules.
        RemoveHydrogens (bool): Remove hydrogens from molecules.
        StrictParsing (bool): Perform strict parsing.

    Returns:
        list : List of RDKit molecule objects.

    """
    return  Chem.SDMolSupplier(FileName, sanitize = Sanitize, removeHs = RemoveHydrogens, strictParsing = StrictParsing)

def ReadMoleculesFromMolFile(FileName, Sanitize = True, RemoveHydrogens = True, StrictParsing = True):
    """Read molecule from a MDL Mol file.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        Sanitize (bool): Sanitize molecules.
        RemoveHydrogens (bool): Remove hydrogens from molecules.
        StrictParsing (bool): Perform strict parsing.

    Returns:
        list : List of RDKit molecule objects.

    """
    
    Mols = []
    Mols.append(Chem.MolFromMolFile(FileName, sanitize = Sanitize, removeHs = RemoveHydrogens, strictParsing = StrictParsing))
    return Mols

def ReadMoleculesFromMol2File(FileName, Sanitize = True, RemoveHydrogens = True):
    """Read molecule from a Tripos Mol2  file.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        Sanitize (bool): Sanitize molecules.
        RemoveHydrogens (bool): Remove hydrogens from molecules.

    Returns:
        list : List of RDKit molecule objects.

    """
    
    Mols = []
    Mols.append(Chem.MolFromMol2File(FileName,  sanitize = Sanitize, removeHs = RemoveHydrogens))
    return Mols

def ReadMoleculesFromPDBFile(FileName, Sanitize = True, RemoveHydrogens = True):
    """Read molecule from a PDB  file.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        Sanitize (bool): Sanitize molecules.
        RemoveHydrogens (bool): Remove hydrogens from molecules.

    Returns:
        list : List of RDKit molecule objects.

    """
    
    Mols = []
    Mols.append(Chem.MolFromPDBFile(FileName,  sanitize = Sanitize, removeHs = RemoveHydrogens))
    return Mols

def ReadMoleculesFromSMILESFile(FileName, SMILESDelimiter = ' ', SMILESColIndex = 0, SMILESNameColIndex = 1, SMILESTitleLine = 1, Sanitize = 1):
    """Read molecules from a SMILES file.
    
    Arguments:
        SMILESDelimiter (str): Delimiter for parsing SMILES line
        SMILESColIndex (int): Column index containing SMILES string.
        SMILESNameColIndex (int): Column index containing molecule name.
        SMILESTitleLine (int): Flag to indicate presence of title line.
        Sanitize (int): Sanitize molecules.

    Returns:
        list : List of RDKit molecule objects.

    """
    
    return  Chem.SmilesMolSupplier(FileName, delimiter = SMILESDelimiter, smilesColumn = SMILESColIndex, nameColumn = SMILESNameColIndex, titleLine = SMILESTitleLine, sanitize = Sanitize)

def MoleculesWriter(FileName, **KeyWordArgs):
    """Set up a molecule writer.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        **KeyWordArgs (dictionary) : Parameter name and value pairs for writing and
            processing molecules.

    Returns:
        RDKit object : Molecule writer.

    Notes:
        The file extension is used to determine type of the file and set up an appropriate
        file writer.

    """
    
    # Set default values for possible arguments...
    WriterArgs = {"Compute2DCoords" : False, "Kekulize": True, "SMILESKekulize": False, "SMILESDelimiter" : ' ', "SMILESIsomeric": True, "SMILESTitleLine": True, "SMILESMolName": True}

    # Set specified values for possible arguments...
    for Arg in WriterArgs:
        if Arg in KeyWordArgs:
            WriterArgs[Arg] = KeyWordArgs[Arg]
    
    Writer = None
    if MiscUtil.CheckFileExt(FileName, "sdf sd"):
        Writer = Chem.SDWriter(FileName)
        Writer.SetKekulize(WriterArgs["Kekulize"])
    elif MiscUtil.CheckFileExt(FileName, "pdb"):
        Writer = Chem.PDBWriter(FileName)
    elif MiscUtil.CheckFileExt(FileName, "smi"):
        # Text for the name column in the title line. Blank indicates not to include name column
        # in the output file...
        NameHeader = 'Name' if WriterArgs["SMILESMolName"] else ''
        Writer = Chem.SmilesWriter(FileName, delimiter = WriterArgs["SMILESDelimiter"], nameHeader = NameHeader, includeHeader = WriterArgs["SMILESTitleLine"],  isomericSmiles = WriterArgs["SMILESIsomeric"], kekuleSmiles = WriterArgs["SMILESKekulize"])
    else:
        MiscUtil.PrintWarning("RDKitUtil.WriteMolecules: Non supported file type: %s" % FileName)
    
    return Writer
    
def WriteMolecules(FileName, Mols, **KeyWordArgs):
    """Write molecules to an output file.
    
    Arguments:
        FileName (str): Name of a file with complete path.
        Mols (list): List of RDKit molecule objects. 
        **KeyWordArgs (dictionary) : Parameter name and value pairs for writing and
            processing molecules.

    Returns:
        int : Number of total molecules.
        int : Number of processed molecules written to output file.

    Notes:
        The file extension is used to determine type of the file and set up an appropriate
        file writer.

    """
    
    Compute2DCoords = False
    if "Compute2DCoords" in KeyWordArgs:
        Compute2DCoords = KeyWordArgs["Compute2DCoords"]
    
    SetSMILESMolProps = KeyWordArgs["SetSMILESMolProps"] if "SetSMILESMolProps" in KeyWordArgs else False
        
    MolCount = len(Mols)
    ProcessedMolCount = 0
    
    Writer = MoleculesWriter(FileName, **KeyWordArgs)
    
    if Writer is None:
        return (MolCount, ProcessedMolCount)
    
    FirstMol = True
    for Mol in Mols:
        if Mol is None:
            continue

        if FirstMol:
            FirstMol = False
            if SetSMILESMolProps:
                SetWriterMolProps(Writer, Mol)
                
        ProcessedMolCount += 1
        if Compute2DCoords:
            AllChem.Compute2DCoords(Mol)
        
        Writer.write(Mol)
    
    Writer.close()
    
    return (MolCount, ProcessedMolCount)

def SetWriterMolProps(Writer, Mol):
    """Setup molecule properties for a writer to output.
    
    Arguments:
        Writer (object): RDKit writer object.
        Mol (object): RDKit molecule object.

    Returns:
        object : Writer object.

    """
    PropNames = list(Mol.GetPropNames())
    if len(PropNames):
        Writer.SetProps(PropNames)
        
    return Writer
    
