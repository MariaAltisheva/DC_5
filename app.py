import streamlit as st

st.title('Построение предсказательной модели регрессии для прогнозирования ZOI')

smiles = st.text_input('Введите SMILES', 'SMILES')
st.write('Вы ввели', smiles)

genus = st.number_input('Insert genus')
st.write('The current number is ', genus)

NP_Synthesis = st.number_input('Insert NP_Synthesis')
st.write('The current number is ', NP_Synthesis)

NP_Size_min = st.number_input('Insert NP_Size_min')
st.write('The current number is ', NP_Size_min)

NP_Size_avg = st.number_input('Insert NP_Size_avg')
st.write('The current number is ', NP_Size_avg)

avg_incup_period = st.number_input('Insert avg_incup_period')
st.write('The current number is ', avg_incup_period)

growth_temp = st.number_input('Insert growth_temp, C')
st.write('The current number is ', growth_temp)

ZOI_drug = st.number_input('Insert ZOI_drug')
st.write('The current number is ', ZOI_drug)

ZOI_NP = st.number_input('Insert ZOI_NP')
st.write('The current number is ', ZOI_NP)