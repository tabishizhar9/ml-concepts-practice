@echo off
if exist assignment1_submission.zip del /F /Q assignment1_submission.zip
tar -a -c -f assignment1_submission.zip code/*.ipynb checkpoints/logistic_regression.pkl