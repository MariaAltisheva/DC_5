import rdkit
import streamlit as st

from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit import DataStructs

from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Fragments
from rdkit.Chem import rdMolDescriptors


st.title('Построение предсказательной модели регрессии для прогнозирования ZOI')

smiles = st.text_input('Введите SMILES', 'SMILES')
st.write('Вы ввели', smiles)

m = Chem.MolFromSmiles(smiles)
def getMolDescriptors(mol, missingVal=None):  # Рассчет дескрипторов для одной молекулы
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            import traceback
            traceback.print_exc()
            val = missingVal
        res[nm] = val
    return res


st.write('FpDensityMorgan1 =', getMolDescriptors(m)['FpDensityMorgan1'])
st.write('EState_VSA7', getMolDescriptors(m)['EState_VSA7'])
st.write('LabuteASA', getMolDescriptors(m)['LabuteASA'])
if st.button('Вывести все дескрипторы для данного SMILES'):
    st.write(getMolDescriptors(m))






genus = st.number_input('Insert genus')
st.write('The current genus is ', genus)

NP_Synthesis = st.number_input('Insert NP_Synthesis')
st.write('The current NP_Synthesis is ', NP_Synthesis)

NP_Size_min = st.number_input('Insert NP_Size_min')
st.write('The current NP_Size_min is ', NP_Size_min)

NP_Size_avg = st.number_input('Insert NP_Size_avg')
st.write('The current NP_Size_avg is ', NP_Size_avg)

avg_incup_period = st.number_input('Insert avg_incup_period')
st.write('The current avg_incup_period is ', avg_incup_period)

growth_temp = st.number_input('Insert growth_temp, C')
st.write('The current growth_temp is ', growth_temp)

ZOI_drug = st.number_input('Insert ZOI_drug')
st.write('The current ZOI_drug is ', ZOI_drug)

ZOI_NP = st.number_input('Insert ZOI_NP')
st.write('The current ZOI_NP is ', ZOI_NP)

# if smiles and genus and NP_Synthesis and NP_Size_min and NP_Size_avg and avg_incup_period and growth_temp and ZOI_drug and ZOI_NP:

if st.button('Предсказать значение Zol_drag_np'):
    st.write('Прогнозируемое значение Zol_drag_np:')