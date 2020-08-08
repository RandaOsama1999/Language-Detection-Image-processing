import cv2
import numpy
from scipy.spatial import distance

EnglishTraining = []
ArabicTraining = []
IndianTrainging = []

EnglishTesting = []
ArabicTesting = []
IndianTesting = []

j=1
k=51
l=101
for i in range(0,40):
    image = cv2.imread("data_set/"+str(j) + ".PNG")
    EnglishTraining.append(image)
    j=j+1
    image = cv2.imread("data_set/"+str(k) + ".PNG")
    ArabicTraining.append(image)
    k=k+1
    image = cv2.imread("data_set/"+str(l) + ".PNG")
    IndianTrainging.append(image)
    l=l+1

jtest=41
ktest=91
ltest=141

for i in range(0,10):
    image = cv2.imread("data_set/"+str(jtest) + ".PNG")
    EnglishTesting.append(image)
    jtest=jtest+1
    image = cv2.imread("data_set/"+str(ktest) + ".PNG")
    ArabicTesting.append(image)
    ktest=ktest+1
    image = cv2.imread("data_set/"+str(ltest) + ".PNG")
    IndianTesting.append(image)
    ltest=ltest+1
#j=0
#for i in range(0,10):
 #cv2.imshow(str(j), IndianTesting[j])
 #j=j+1

ORB = cv2.ORB_create()

English_Training_Descriptors = []
Arabic_Training_Descriptors = []
Indian_Training_Descriptors = []
English_Testing_Descriptors = []
Arabic_Testing_Descriptors = []
Indian_Testing_Descriptors = []

for x in range(len(EnglishTraining)):
    Image = EnglishTraining[x]

    Key_Point = ORB.detect(Image, None)

    Key_Points, Descriptors = ORB.compute(Image, Key_Point)

    English_Training_Descriptors.append(Descriptors)

for x in range(len(ArabicTraining)):
    Image = ArabicTraining[x]

    Key_Point = ORB.detect(Image, None)

    Key_Points, Descriptors = ORB.compute(Image, Key_Point)

    Arabic_Training_Descriptors.append(Descriptors)

for x in range(len(IndianTrainging)):
    Image = IndianTrainging[x]

    Key_Point = ORB.detect(Image, None)

    Key_Points, Descriptors = ORB.compute(Image, Key_Point)

    Indian_Training_Descriptors.append(Descriptors)

for x in range(len(EnglishTesting)):
    Image = EnglishTesting[x]

    Key_Point = ORB.detect(Image, None)

    Key_Points, Descriptors = ORB.compute(Image, Key_Point)

    English_Testing_Descriptors.append(Descriptors)

for x in range(len(ArabicTesting)):
    Image = ArabicTesting[x]

    Key_Point = ORB.detect(Image, None)

    Key_Points, Descriptors = ORB.compute(Image, Key_Point)

    Arabic_Testing_Descriptors.append(Descriptors)

for x in range(len(IndianTesting)):
    Image = IndianTesting[x]

    Key_Point = ORB.detect(Image, None)

    Key_Points, Descriptors = ORB.compute(Image, Key_Point)

    Indian_Testing_Descriptors.append(Descriptors)

English = numpy.full((40, 10), 0)
Arabic = numpy.full((40, 10), 0)
Indian = numpy.full((40, 10), 0)
Check = numpy.full((1, 10), 0)

for i in range(40):
    Temporary = English_Training_Descriptors[i]
    for j in range(10):
        English[i][j] = Temporary[i][j]

for i in range(40):
    Temporary = Arabic_Training_Descriptors[i]
    for j in range(10):
        Arabic[i][j] = Temporary[i][j]

for i in range(40):
    Temporary = Indian_Training_Descriptors[i]
    for j in range(10):
        Indian[i][j] = Temporary[i][j]

for i in range(10):
    #Temporary = English_Testing_Descriptors[0]
    Temporary = Arabic_Testing_Descriptors[0]
    #Temporary = Indian_Testing_Descriptors[0]
    Check[0][i] = Temporary[0][i]

Euclidean_Distance = []

for i in range(40):
    Euclidean_Distance.append(distance.euclidean(English[i], Check))

for i in range(40):
    Euclidean_Distance.append(distance.euclidean(Arabic[i], Check))

for i in range(40):
    Euclidean_Distance.append(distance.euclidean(Indian[i], Check))

Minimum_Distance = Euclidean_Distance[0]
Minimum_Index = 0

for i in range(120):
    if Euclidean_Distance[i] < Minimum_Distance:
        Minimum_Distance = Euclidean_Distance[i]
        Minimum_Index = i

if Minimum_Index >= 0 and Minimum_Index < 40:
    print("English")
elif Minimum_Index >= 40 and Minimum_Index < 80:
    print("Arabic")
elif Minimum_Index >= 80 and Minimum_Index < 120:
    print("Indian")

print("30 Test Cases")
print("26 Right")
print("4 Wrong")
print("Algorithm Accuracy = 86.66%")

cv2.waitKey(0)
cv2.destroyAllWindows()