import os
from skimage.io import imread, imshow
import cv2
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

dir = '20_Validation'
img_ls = []

for fname in sorted(os.listdir(dir)):
    img = imread(os.path.join(dir, fname))
    if (img.ndim < 3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_ls.append(img)

df = pd.DataFrame()
df["Pictures"] = img_ls
df.head()

df.shape

labels = list(pd.read_csv("labels.csv")["labels"])
labels.index('goat')
ground_truth = np.zeros(len(df))

# penguin: 1-46
ground_truth[0:46] = labels.index("penguin")
# zebra: 47-78
ground_truth[46:78] = labels.index("zebra")
# goat: 79-114
ground_truth[78:114] = labels.index("goat")
# triceratops: 115-145
ground_truth[114:145] = labels.index("triceratops")
# porcupine: 146-167
ground_truth[145:167] = labels.index("porcupine")
# teddy-bear: 168-179
ground_truth[167:179] = labels.index("teddy-bear")
# comet: 180-198
ground_truth[179:198] = labels.index("comet")
# leopards: 199-229
ground_truth[198:229] = labels.index("leopards")
# unicorn: 230-250
ground_truth[229:250] = labels.index("unicorn")
# llama : 251-286
ground_truth[250:286] = labels.index("llama")
# gorilla: 287-290
ground_truth[286:290] = labels.index("gorilla")
# blimp: 291-293
ground_truth[290:293] = labels.index("blimp")
# airplanes: 294-304
ground_truth[293:304] = labels.index("airplanes")
# blimp: 305-310
ground_truth[304:310] = labels.index("blimp")
# unicorn: 311
ground_truth[310:311] = labels.index("unicorn")
# leopards: 312-330
ground_truth[311:330] = labels.index("leopards")
# kangaroo: 331-356
ground_truth[330:356] = labels.index("kangaroo")
# unicorn: 357-360
ground_truth[356:360] = labels.index("unicorn")
# comet: 361
ground_truth[360:361] = labels.index("comet")
# porcupine: 362-371
ground_truth[361:371] = labels.index("porcupine")
# teddy-bear: 372-388
ground_truth[371:388] = labels.index("teddy-bear")
# comet: 398-402
ground_truth[388:402] = labels.index("comet")
# penguin: 403
ground_truth[402:403] = labels.index("penguin")
# dolphin: 404-409
ground_truth[403:409] = labels.index("dolphin")
# giraffe: 410-419
ground_truth[409:419] = labels.index("giraffe")
# bear: 420-433
ground_truth[419:433] = labels.index("bear")
# killer-whale: 434-445
ground_truth[433:445] = labels.index("killer-whale")
# penguin: 446-448
ground_truth[445:448] = labels.index("penguin")
# gorilla: 449-515
ground_truth[448:515] = labels.index("gorilla")
# crab: 516-543
ground_truth[515:543] = labels.index("crab")
# blimp: 544-563
ground_truth[543:563] = labels.index("blimp")
# airplanes: 564-597
ground_truth[563:597] = labels.index("airplanes")
# dog: 598-631
ground_truth[597:631] = labels.index("dog")
# dolphin: 632-660
ground_truth[631:660] = labels.index("dolphin")
# giraffe: 661-678
ground_truth[660:678] = labels.index("giraffe")
# bear: 679-698
ground_truth[678:698] = labels.index("bear")
# killer-whale: 699-716
ground_truth[698:716] = labels.index("killer-whale")

# Generate prediction with SVM
data = pd.read_pickle("train_with_feature.pkl")
y = data[["Encodings"]]
X = data.drop(["Encodings", "Pictures"], axis=1)

clf = OneVsRestClassifier(SVC(gamma="auto", probability=False, C=400))
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
X_scaled = scaling.transform(X)
print("CV =", np.mean(cross_val_score(clf, X_scaled, y.values.ravel(), cv=5)))


def accuracy(pred, actual):
    # Calculate the accuracy percentage of the predicted values
    return sum(pred == actual) / len(actual)


clf = OneVsRestClassifier(SVC(gamma="auto", probability=False, C=400))
clf.fit(X_scaled, y.values.ravel())
X_test = pd.read_pickle("test_with_feature.pkl").drop(["Pictures"], axis=1)
X_test_scaled = scaling.transform(X_test)
pred = clf.predict(X_test_scaled)
test_acc = accuracy(pred, ground_truth)
print(test_acc)
pred

# Random Forest
clf = RandomForestClassifier(n_estimators=1000, max_depth=18, random_state=42)
# print("CV =", np.mean(cross_val_score(clf, X, y.values.ravel(), cv=5)))
clf.fit(X, y.values.ravel())
pred = clf.predict(X_test)
test_acc = accuracy(pred, ground_truth)
print(test_acc)
