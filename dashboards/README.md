# dashboards/

Este diretório abriga o relatório `blink_analysis.pbix`.

O `.pbix` é um artefato binário do Power BI Desktop e precisa ser gerado/atualizado
na ferramenta (não é gerado pelo pipeline Python). Roteiro de montagem:

1. Abra o **Power BI Desktop** → **Obter Dados → Pasta** → selecione `data/processed/`
   (ou **Texto/CSV** para uma única sessão) → **Combinar e Transformar**.
2. Renomeie a consulta para `blink_telemetry` e ajuste os tipos:
   - `session_id`, `timestamp` → Texto
   - `frame_index`, `is_blink`, `accumulated_blinks`, `has_blue_filter` → Número Inteiro
   - `ear_value` → Número Decimal
3. Adicione a coluna calculada `Segundos` e as medidas DAX documentadas na seção
   "5. Power BI" do [README principal](../README.md).
4. Monte os visuais sugeridos (cartões de KPI, linha de `ear_value` com linha constante
   em `τ`, área de `accumulated_blinks`, comparativo por `has_blue_filter`).
5. Salve como `dashboards/blink_analysis.pbix`.
