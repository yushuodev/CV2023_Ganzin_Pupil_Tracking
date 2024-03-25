1. 環境建置.txt中的版本內容安裝完成
2. 檔案集分為public跟private，放到dataset的資料夾裡面，然後public中放的是S1~S4的資料，private中放的是S5~S8的資料
3. 執行split_train_val.py，把檔案分成train_eyes跟val_eyes的檔案
4. 執行pupil_training.py，訓練完的model會在experiment的資料夾裡面
5. 把裡面的model取出來之後，放到checkpoint的資料夾裡面 （已經先把上傳時訓練的model放入m11107004檔案裡面，如果需要執行則需放入checkpoint）
6. 執行submit.py，最終就是我們的預測結果
* 請注意在跑submit的時候，請確保資料夾裡面沒有solution的資料夾，如果已經存在的話，請先刪除後，再跑

# 檔案擺放格式
# algorithm.py
# config.py
# dataset.py
# model.py
# pupil_training.py
# submit.py
# utils.py
# split_train_val.py
# experiment
# checkpoint/
# • model_best.pth（執行完步驟4後，步驟5放入此處）
# dataset/（步驟2的處理）
# • private/
# •• S5 
# •• S6 
# •• S7 
# •• S8 
# • public/
# •• S1 
# •• S2 
# •• S3 
# •• S4 
