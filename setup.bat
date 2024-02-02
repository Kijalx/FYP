pip install -r requirements.txt
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
start "" "%ProgramFiles%\WinRAR\winrar.exe" x -ibck kagglecatsanddogs_5340.zip *.* ./
timeout /t 5 /nobreak
rm -f ./CDLA-Permissive-2.0.pdf
rm -f ./kagglecatsanddogs_5340.zip

msg * "Setup complete!"