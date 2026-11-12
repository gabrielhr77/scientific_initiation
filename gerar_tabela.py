import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill

# 1. Lê o CSV original
df = pd.read_csv('tabela.xlsx')

# 2. Garante que tenha 40 linhas (adiciona vazias se precisar)
num_linhas = 40
if len(df) < num_linhas:
    linhas_vazias = pd.DataFrame([[""] * len(df.columns)] * (num_linhas - len(df)), columns=df.columns)
    df = pd.concat([df, linhas_vazias], ignore_index=True)

# 3. Cria o Excel com formatação
wb = Workbook()
ws = wb.active

# escreve cabeçalhos
ws.append(list(df.columns))

# escreve os dados
for r in df.itertuples(index=False):
    ws.append(r)

# 4. Aplica cores alternadas (cinza e branco)
fill_cinza = PatternFill(start_color="00E9E9E9", end_color="00E9E9E9", fill_type="solid")

for i in range(2, num_linhas + 2):  # +2 porque cabeçalho ocupa a primeira linha
    if i % 2 == 0:  # linhas pares cinzas
        for cell in ws[i]:
            cell.fill = fill_cinza

# 5. Salva o arquivo
wb.save('tabela_expandida.xlsx')
print("Arquivo gerado: tabela_expandida.xlsx")
