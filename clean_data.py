import pandas as pd
df = pd.read_csv('/home/rhji/CO2_solubility/SMILESDH.csv')
new_df = df[df['Pressure(mPa)']>0]
newest_df = new_df[new_df['xCO2(mole fraction)']>0]
#new_df = df.dropna(subset=['Pressure(mPa)','xCO2(mole fraction)'])
newest_df.to_csv('NEWSMILESDH.csv',index=False,encoding='utf-8')