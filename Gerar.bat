
cd "C:\Users\jnrcr\OneDrive\Lacem\Interface Predicao"

pyinstaller --workpath ./temp --windowed --icon=Lacem.ico --add-data "Lacem.ico;." --add-data "ENXOFRE MERCAPTIDICO.xlsx;." --add-data "TAN.xlsx;." --add-data "SCC.xlsx;." --add-data "BIODEGRADACAO APPI.xlsx;." --add-data "BIODEGRADACAO ESI.xlsx;." --add-data "EVOLUCAO APPI.xlsx;." --add-data "EVOLUCAO ESI.xlsx;." --add-data "ORIGEM APPI.xlsx;." --add-data "ORIGEM ESI.xlsx;." --add-data "UFG.png;." --add-data "Lacem.png;." --onefile interface_de_predicao_110.py

echo Pronto
pause