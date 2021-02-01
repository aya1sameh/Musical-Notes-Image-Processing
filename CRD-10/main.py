from PIL import Image
from skimage.morphology import disk
# ask if ok to use it...............................
from scipy.signal import find_peaks
import cv2 as cv
from skimage.filters import median, gaussian
from skimage.filters import threshold_sauvola
from skimage.transform import hough_line, hough_line_peaks, rotate, rescale, resize
from skimage.measure import find_contours
from collections import Counter
from Classifier import *
from commonfunctions import *
from Translate import *
import pickle as pickle
import traceback
import argparse
import sys

###############################################################################################
# takes binarized image(1-0) and return array of images each contain 1 staff
###############################################################################################


def seg_Stuffs(Image):

    # 2-copy the recieved image
    Image2 = Image.copy()

    # Technique 1:
    # For each stuff getting the stuff begining and ending to cut the image

    flag = False

    # 3-negative transformation(0 becomes 1 and vice versa)
    BinarizedImage = 1-Image2

    # 4-get the row histogram
    Horizontal_hist = np.sum(BinarizedImage, axis=1)
    #plt.plot(Horizontal_hist)

    # 5-get peaks beginning row and ending row
    # Algorithm1:
    PeaksIndex, _ = find_peaks(Horizontal_hist)
    Peaks_Array = Horizontal_hist[PeaksIndex]

    maxPeak = np.max(Peaks_Array)
    PeaksIndex, _ = find_peaks(Horizontal_hist, prominence=maxPeak*0.7)

    if(PeaksIndex.shape[0] % 5 == 0):
        flag = True  # algorithm1 succeeded
        for i in range(1, PeaksIndex.shape[0], 1):
            if (abs(int(Horizontal_hist[PeaksIndex[i-1]]) - int(Horizontal_hist[PeaksIndex[i]])) > 10):
                flag = False
                break
        if(flag == True):
            # to be removed..........................
            #plt.plot(PeaksIndex, Horizontal_hist[PeaksIndex], 'x')
            # to be removed....................................................................
            # plt.show()
            # 6-negative transformation(0 becomes 1 and vice versa)
            BinarizedImage = 1-BinarizedImage
            # 7-segment each peak from (end of the previos peak + begin of the next peak)/2
            ImgArray = []
            begin = 0
            end = 0
            for i in range(0, PeaksIndex.shape[0], 5):
                if i+5 < PeaksIndex.shape[0]:
                    end = (PeaksIndex[i+4]+PeaksIndex[i+5])//2
                else:
                    end = BinarizedImage.shape[0]
                ImgArray.append(BinarizedImage[begin:end, :])
                begin = end

            return ImgArray
    if(flag == False):
        # Algorithm2
        StructureElement = cv.getStructuringElement(cv.MORPH_RECT, (80, 6))
        BinarizedImage = np.uint8(BinarizedImage)

        DilatedImage = cv.dilate(
            BinarizedImage, StructureElement, iterations=5)
        Horizontal_hist = np.sum(DilatedImage, axis=1)
        # to be removed............................................................
        #plt.plot(Horizontal_hist)
        MaxInHist = np.max(Horizontal_hist)

        # print(MaxInHist)
        ThresholdingPercentages = [int(MaxInHist * 0.05 * i)
                                   for i in range(1, 17)]
        # print(ThresholdingPercentages)
        ValuesofEachThresh = []
        # 6-negative transformation(0 becomes 1 and vice versa)
        BinarizedImage = 1-BinarizedImage
        for p in range(len(ThresholdingPercentages)):
            IntersectionList = []
            WidthList = []
            for j in range(len(Horizontal_hist)-1):
                if (Horizontal_hist[j] > ThresholdingPercentages[p] and Horizontal_hist[j+1] < ThresholdingPercentages[p]):
                    IntersectionList.append(j)
                if (Horizontal_hist[j] < ThresholdingPercentages[p] and Horizontal_hist[j+1] > ThresholdingPercentages[p]):
                    IntersectionList.append(j)
            for k in range(0, len(IntersectionList)-1, 2):
                WidthList.append(IntersectionList[k+1]-IntersectionList[k])
            if len(WidthList) > 0:
                Sigma = np.std(WidthList)
                AvgWidth = np.average(WidthList)
                ValuesofEachThresh.append(
                    (Sigma, AvgWidth, len(WidthList), ThresholdingPercentages[p]))

        BestThreshValue = (0, 0, 0, 0)
        MaxSeg = 0

        for V in ValuesofEachThresh:
            # print(V)
            if V[0] < 5 and V[2] > MaxSeg:
                MaxSeg = V[2]
                BestThreshValue = V
            elif V[0] < 5 and V[1] > BestThreshValue[1]:
                BestThreshValue = V

        # print("Best",BestThreshValue)
        ActualIntersectionList = []
        for j in range(len(Horizontal_hist)-1):
            if (Horizontal_hist[j] > BestThreshValue[3] and Horizontal_hist[j+1] < BestThreshValue[3]):
                ActualIntersectionList.append(j)
            if (Horizontal_hist[j] < BestThreshValue[3] and Horizontal_hist[j+1] > BestThreshValue[3]):
                ActualIntersectionList.append(j)
        ImgArray = []

        for k in range(0, len(ActualIntersectionList)-1, 2):
            ImgArray.append(
                BinarizedImage[ActualIntersectionList[k]:ActualIntersectionList[k+1]])

        # to be removed....................................................................
        # plt.show()
        return ImgArray


def seg_Notes(StuffImage):
    # 2-copy the recieved image
    CopiedImage = np.uint8(StuffImage.copy()*255)
    CopiedImage2 = 255-CopiedImage
    # CannyEdge detection:
    # ........................................................................................
    Image = cv.Canny(CopiedImage, 0, 200)
    # Technique 1:
    # segmenting each note alone
    contours, hierarchy = cv.findContours(
        Image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    Objectbounders = [cv.boundingRect(c) for c in contours]
    (sorted_contours, sorted_boundries) = zip(
        *sorted(zip(contours, Objectbounders), key=lambda b: b[1][0]))

    Actualsorted_bound = []
    i = 0
    while i < len(sorted_boundries):
        MinX = sorted_boundries[i][0]
        MaxX = sorted_boundries[i][0] + sorted_boundries[i][2]
        MinY = sorted_boundries[i][1]
        MaxY = sorted_boundries[i][1] + sorted_boundries[i][3]
        j = i+1
        while j < len(sorted_boundries)+1:
            # closing must be done at the end of line removing to avoid duplicates.....
            if j < len(sorted_boundries) and sorted_boundries[j][0] <= MaxX:
                MaxX = max(MaxX, sorted_boundries[j][0]+sorted_boundries[j][2])
                MaxY = max(MaxY, sorted_boundries[j][1]+sorted_boundries[j][3])
                MinY = min(MinY, sorted_boundries[j][1])
            else:
                i = j-1
                Actualsorted_bound.append((MinX, MaxX, MaxY, MinY))
                break
            j += 1
        i += 1
    NotesArray = []
    PosArray = []
    for k in range(len(Actualsorted_bound)):
        if Actualsorted_bound[k][1] - Actualsorted_bound[k][0] > 8:
            tempimage = CopiedImage[Actualsorted_bound[k][3]:Actualsorted_bound[k]
                                    [2], Actualsorted_bound[k][0]:Actualsorted_bound[k][1]]
            if cv.countNonZero(tempimage) / tempimage.size > 0.05:
                NotesArray.append(tempimage)
                PosArray.append(
                    (Actualsorted_bound[k][0], Actualsorted_bound[k][1]))
    return NotesArray, PosArray


def NoiseRemoval(img):
    med_img = median(img, disk(1), mode='constant', cval=0.0)
    gaus = gaussian(med_img, sigma=0.5, mode='constant', cval=0.0)
    return gaus

# show_images((img,gaus))


def threshold(img):
    cpy = img.copy()
    blk = max(cpy.shape[0], cpy.shape[1])
    blk = int(blk * 0.03) + 1 if int(blk * 0.03) % 2 == 0 else int(blk * 0.03)
    thresh_sauvola = threshold_sauvola(cpy, blk)
    binary_sauvola = cpy > thresh_sauvola
    return binary_sauvola


def skew_function(image_in):
    image = rgb2gray(image_in)*255
    # image=image_in.copy()
    #show_images([image_in],["original image"])
    angles = np.linspace(0, np.pi, 360)
    # linspace gets an array of values range from argument(1) to argument(2) and there count is argument(3)
    edgeimage = canny(image, 0, 150)
    h, angle, d = hough_line(edgeimage)
    #fig, axes = plt.subplots(1, 2, figsize=(15,5))
    # first argument(1) detect that the figures(axes) produced by the subplot function are side to side
    # second argument is the number of figures
    # axes[0].imshow(edgeimage)
    #axes[0].set_title('edge detected image to hough transform')
    # axes[0].set_axis_off()
    # axes[1].imshow(edgeimage)
    origin = np.array((0, edgeimage.shape[1]))
    # width od the image
    # start creating the hough transformed image (detected lines)
    for _, angless, dist in zip(*hough_line_peaks(h, angle, d)):
        y0, y1 = (dist - origin * np.cos(angless)) / np.sin(angless)
        #axes[1].plot(origin, (y0, y1), '-r')
    # print('angles=',angless)
    # axes[1].set_xlim(origin)
    #axes[1].set_ylim((edgeimage.shape[0], 0))
    # axes[1].set_axis_off()
    #axes[1].set_title('Detected lines from Hough trans')
    # plt.tight_layout()
    # plt.show()
    # print(angless)
    angle_in_deg = math.degrees(angless)
    #print("angless in deg=",angle_in_deg)
    # image_in=1-image_in
    if angle_in_deg > 0:
        angle_in_deg = angle_in_deg-90
    else:
        angle_in_deg = angle_in_deg+90

    skew_image = image_in
    # print(angle_in_deg)

    if angle_in_deg > 1 or angle_in_deg < -1:
        skew_image = rotate(image_in, angle_in_deg,
                            resize=True, cval=0, mode="wrap")

    return skew_image


def imageCrop(Image):
    StructureElement = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    DilatedImage = cv.dilate(Image, StructureElement, iterations=15)
    contours, hierarchy = cv.findContours(
        DilatedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    Objectbounders = [cv.boundingRect(c) for c in contours]
    MaxArea = 0
    for i in range(len(Objectbounders)):
        if(Objectbounders[i][2]*Objectbounders[i][3] > MaxArea):
            MaxArea = Objectbounders[i][2]*Objectbounders[i][3]
            MaxBoundary = Objectbounders[i]
    # print(Objectbounders)
    #show_images([DilatedImage],["Dilated Image"])
    # print(MaxBoundary)
    # print(Objectbounders[0][0])
    # print(Objectbounders[0][1])
    # print(Objectbounders[0][2])
    # print(Objectbounders[0][3])
    x2 = MaxBoundary[0]+MaxBoundary[2]
    y2 = MaxBoundary[1]+MaxBoundary[3]
    cropedImage = Image[MaxBoundary[1]:y2, MaxBoundary[0]:x2]
    # print(cropedImage.shape)
    # print(cropedImage)
    return cropedImage


def get_ref_lengths(img):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    rle_image_white_runs = []  # Cumulative white run list
    rle_image_black_runs = []  # Cumulative black run list
    sum_all_consec_runs = []  # Cumulative consecutive black white runs

    for i in range(num_cols):
        col = img[:, i]
        rle_col = []
        rle_white_runs = []
        rle_black_runs = []
        run_val = 0
        run_type = col[0]
        for j in range(num_rows):
            if (col[j] == run_type):
                run_val += 1
            else:
                rle_col.append(run_val)
                if (run_type == 0):
                    rle_black_runs.append(run_val)
                else:
                    rle_white_runs.append(run_val)
                run_type = col[j]
                run_val = 1

        rle_col.append(run_val)
        if (run_type == 0):
            rle_black_runs.append(run_val)
        else:
            rle_white_runs.append(run_val)

        # Calculate sum of consecutive vertical runs
        sum_rle_col = [sum(rle_col[i: i + 2]) for i in range(len(rle_col))]

        # Add to column accumulation list
        rle_image_white_runs.extend(rle_white_runs)
        rle_image_black_runs.extend(rle_black_runs)
        sum_all_consec_runs.extend(sum_rle_col)

    white_runs = Counter(rle_image_white_runs)
    black_runs = Counter(rle_image_black_runs)
    black_white_sum = Counter(sum_all_consec_runs)

    line_spacing = white_runs.most_common(1)[0][0]
    line_width = black_runs.most_common(1)[0][0]
    width_spacing_sum = black_white_sum.most_common(1)[0][0]

    return line_width, line_spacing


def RemoveLines(Binary):
    Binary = (Binary*255).astype('uint8')
    thresh = 255 - Binary
    #####################################################
    lineWidth, _ = get_ref_lengths(thresh)
    # Remove horizontal
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
    detected_lines = cv.morphologyEx(
        thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)  # apply openenig
    contours, hierarchy = cv.findContours(
        detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        if lineWidth > 50:
            # here if the line thickness is so big when remove it we need to draw countour with more white increase thickness
            cv.drawContours(Binary, [contour], -1, 255, 3)
        else:
            cv.drawContours(Binary, [contour], -1, 255, 2)

    # Repair image try to fill gap after removing lines
    repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 6))
    result = 255 - cv.morphologyEx(255 - Binary, cv.MORPH_CLOSE,
                                   repair_kernel, iterations=2)  # apply closeing
    result = (result/255).astype('uint8')
    return result


def getBestScale(img, template):
    scaleList = np.linspace(0.1, 1.0, 20).tolist()
    w, h = template.shape[::-1]
    maxCorrleation = 0
    perfectScale = 0
    for i in range(len(scaleList)):
        resized_img = resize(
            img, (int(img.shape[0] * scaleList[i]), int(img.shape[1] * scaleList[i])))
        resized_img = (resized_img*255).astype('uint8')
        #resized_img = imutils.resize(img, width = int(img.shape[1] * scaleList[i]))
        if resized_img.shape[0] < h or resized_img.shape[1] < w:
            continue
        res = cv.matchTemplate(resized_img, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if maxCorrleation == 0 or maxCorrleation < max_val:
            maxCorrleation = max_val
            perfectScale = scaleList[i]
    return perfectScale


def staffLineRaws(Binary):

    Binary = (Binary*255).astype('uint8')
    thresh = 255 - Binary
    # Remove horizontal
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
    detected_lines = cv.morphologyEx(
        thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=3)  # apply openenig
    edges = cv.Canny(detected_lines, 50, 150, apertureSize=3)
    minLineLength = Binary.shape[1]-600  # ====================>>>

    lines = cv.HoughLinesP(image=edges, rho=0.02, theta=np.pi/500, threshold=10,
                           lines=np.array([]), minLineLength=minLineLength, maxLineGap=100)

    a, b, c = lines.shape
    vertices = []
    for i in range(a):
        #cv.line(Binary, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 3, cv.LINE_AA)
        #         print(lines[i][0][1])
        vertices.append(lines[i][0][1])

    vertices = sorted(vertices)
    vertices = np.array(vertices)
    dif = vertices[1:] - vertices[:-1]

    # get the index where you first observe a jump
    fi = np.where(abs(dif) > 5)
    out = [item for t in fi for item in t]
    uniques = []
    uniques = [vertices[i+1] for i in out]
    uniques.insert(0, vertices[0])
    uniques = sorted(uniques, reverse=True)
    ar = np.array(uniques)
    dif1 = ar[1:] - ar[:-1]
    diff = min(np.unique(dif1))
    return uniques, abs(diff)


def getNotePostion(img, templatePath):

    template = cv.imread(templatePath, 0)
    w, h = template.shape[::-1]
    scale = getBestScale(img, template)
    resized_img = resize(
        img, (int(img.shape[0] * scale), int(img.shape[1] * scale)))
    resized_img = (resized_img*255).astype('uint8')
    res = cv.matchTemplate(resized_img, template, cv.TM_CCOEFF_NORMED)
    positions = []
    l = []
    threshold = 0.70
    loc = np.where(res >= threshold)
    mask = np.zeros(resized_img.shape[:2], np.uint8)
    for pt in zip(*loc[::-1]):
        # here check that center of the mask not marked before
        if mask[pt[1] + int(round(h/2)), pt[0] + int(round(w/2))] != 255:
            mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
            # (int(pt[0]*1/perfectScale),int(pt[1]*1/perfectScale) ) , (int((pt[0] + w)*1/perfectScale), int((pt[1] + h)*1/perfectScale)
            ymin = int(pt[1]*1/scale)
            ymax = int((pt[1] + h)*1/scale)
            ycenter = int((pt[1] + int(round(h/2)))*1/scale)
            positions.append((ymin, ymax, ycenter))
            #(int(pt[0]*1/scale) )
            l.append(int(pt[0]*1/scale))

    return positions, l


def getAllPostion(Binary):
    Binary = (Binary*255).astype('uint8')
    templatePath = ['note/half-space.png', 'note/solid-note.png',
                    'note/half-note-line.png', 'note/whole-note-space.png']
    total = []
    l = []
    for i in range(len(templatePath)):
        position, x = getNotePostion(Binary, templatePath[i])
        if len(position) > 0:
            #total = np.concatenate((np.array(total), np.array(position)), axis=0)
            total.append(position)
            l.append(x)
    return total, l


def postionStrings(vertices, ymin, ymax, ycenter, lineW, space):
    strings = []
    # here add flexibilty of one pixel below or above
    if(ycenter - (vertices[0]+lineW) in list(range(space-2, space+2))):
        position = "c"
        strings.append(position)
    elif(ymin in list(range(vertices[0]-1, (vertices[0]+lineW)+2))):
        position = "d"
        strings.append(position)
    elif(ycenter in list(range(vertices[0]-1, vertices[0] + lineW+2))):
        position = "e"
        strings.append(position)
    elif(ymax in list(range(vertices[0]-1, vertices[0] + lineW+2))):
        position = "f"
        strings.append(position)
    elif(ycenter in list(range(vertices[1]-1, vertices[1] + lineW+2))):
        position = "g"
        strings.append(position)
    elif(ymax in list(range(vertices[1]-1, vertices[1] + lineW+2))):
        position = "a"
        strings.append(position)
    elif(ycenter in list(range(vertices[2]-1, vertices[2] + lineW+2))):
        position = "b"
        strings.append(position)
    elif(ymax in list(range(vertices[2]-1, (vertices[2] + lineW+2)))):
        position = "c2"
        strings.append(position)
    elif(ycenter in list(range(vertices[3]-1, vertices[3] + lineW+2))):
        position = "d2"
        strings.append(position)
    elif(ymax in [(vertices[3]+lineW), (vertices[3]+lineW)+1, (vertices[3]+lineW)+2, (vertices[3]+lineW)-2, (vertices[3]+lineW)-1]):
        position = "e2"
        strings.append(position)
    elif(ycenter in [(vertices[4] + int(lineW/2)), (vertices[4] + int(lineW/2))+1, (vertices[4] + int(lineW/2)) - 1]):
        position = "f2"
        strings.append(position)
    elif(ymax in [(vertices[4]+lineW), vertices[4]+1, vertices[4]-1, (vertices[4]+lineW)+1, vertices[4]]):
        position = "g2"
        strings.append(position)
    elif((vertices[4] - ymax) in list(range(int(space/2) - 1, int(space/2) + 2))):
        position = "a2"
        strings.append(position)
    else:
        position = "b2"
        strings.append(position)
    return strings


def getStrings(Binary):
    arr, diff = staffLineRaws(Binary)
    total, x = getAllPostion(Binary)  # 333
    total = [item for sublist in total for item in sublist]
    s = np.array(x)
    sort_index = np.argsort(s)
    strings = []
    lineW, space = get_ref_lengths(Binary)
    for i in range(len(total)):

        strings.append(postionStrings(
            arr, total[i][0], total[i][1], total[i][2], lineW, space)[0])

    st = np.array(strings)
    Xsorted = np.sort(x)
    return (st[sort_index]), Xsorted


def WriteInFile(pathdirectory: str, FileName: str, output: str):
    if not os.path.isdir(pathdirectory):
        os.mkdir(pathdirectory)
    if os.path.isdir(pathdirectory) and FileName is not None:

        FileName = FileName.split('.')[:-1]
        FileName = FileName[0]+".txt"
        FileName = os.path.join(pathdirectory, FileName)
        OutputFile = open(FileName, "w")
        OutputFile.write(output)
        OutputFile.close()


def load_names_from_folder(folder):
    images_names = []
    for filename in os.listdir(folder):
        images_names.append((filename, os.path.join(folder, filename)))
    return images_names


#############################################################End of functionssss############################################
#
# ##############################################################MAIN####################################################
# outside the loop
LoadedClassifier = pickle.load(open('Test.sav', 'rb'))

parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help="Input File")
parser.add_argument("outputfolder", help="Output File")
args = parser.parse_args()
inputDirectory = args.inputfolder
outputDirectory = args.outputfolder

print(inputDirectory)

try:
    img_names_paths_array = load_names_from_folder(inputDirectory)
    if not img_names_paths_array:
        print("Empty")
except:
    print("No inputs")


for img_name_path in img_names_paths_array:
    try:
        # inside the loop
        Image = io.imread(img_name_path[1], as_gray=True)
        # Steps:
        # 1-Noise Remooval:
        Image1 = NoiseRemoval(Image)
        # 2-Binarization:
        Image1 = threshold(Image1)
        # show_images([Image1],["Thresholded"])
        Image1 = np.uint8(Image1)
        # 3-orientation:
        Image2 = 1-Image1
        Image2 = skew_function(Image2)
        Image2 = np.uint8(Image2*255)
        CropedImage = imageCrop(Image2)
        CropedImage = threshold(CropedImage)

        #show_images([Image1, Image2, CropedImage], ["Original", "Skewed", "Cropped"])

        # 4-Segment Each staff:
        CropedImage = np.uint8(CropedImage)
        CropedImage = 1-CropedImage
        ArrayOfStaffs = seg_Stuffs(CropedImage)
        # if(len(ArrayOfStaffs) > 1):
        #     #show_images(ArrayOfStaffs)
        # else:
        #     #show_images([ArrayOfStaffs[0]])
        # #######################################
        # # Removing extra stuffs
        # # if len(ArrayOfStaffs)>1:
        #  #   actualArrayOfStuffs=[]
        #   #  SumHeight=0
        #    # for i in ArrayOfStaffs:
        #     # SumHeight=SumHeight+i.shape[0]

        #     # AvgHeight=SumHeight/len(ArrayOfStaffs)
        #     # for i in range(len(ArrayOfStaffs)):
        #     #   if(ArrayOfStaffs[i].shape[0]>AvgHeight):
        #     #      actualArrayOfStuffs.append(ArrayOfStaffs[i])
        #     # show_images(actualArrayOfStuffs)
        # #################################
        # # 7-Get position of each Note in a staff given each staff
        #print("before position")
        totalPositions = []
        try:
            for i in range(len(ArrayOfStaffs)):
                tempPosition, x = getStrings(ArrayOfStaffs[i])
                for p in range(len(tempPosition[i])):
                    totalPositions.append((tempPosition[i][p], x[i][p]))
        except:
            pass
        # print(totalPositions)
        #print("after position")
        # # 8-Staff line removal
        RemovedLinesStuff = []
        for i in range(len(ArrayOfStaffs)):
            RemovedLinesStuff.append(RemoveLines(ArrayOfStaffs[i]))
            # show_images([RemovedLinesStuff[i]])
        # 9-Notes segmentation
        NotesArray = []
        symbolXs = []
        for i in range(len(RemovedLinesStuff)):
            symbols, Xvalues = seg_Notes(RemovedLinesStuff[i])
            symbolXs.append(Xvalues)
            NotesArray.append(symbols)
            #show_images(NotesArray[i])

        # labeling symbols
        #positions = []
        Finaltext = "{\n"
        for i, staff in enumerate(NotesArray):
            labels = []
            labelName = ""
            for j, note in enumerate(staff):
                note = 255 - note
                features = extract_features(note)
                labeltemp = LoadedClassifier.predict([features])
                labels.append((str(labeltemp[0]), symbolXs))
                labelName += str(labeltemp[0]) + " "
            output = StringToWrite(labels, totalPositions)
            Finaltext += output
            if i != len(NotesArray)-1:
                Finaltext += ",\n"
            else:
                Finaltext += "\n"
            # print(labelName)
            # print(totalPositions)
            # print(output)
        Finaltext += "}"
        WriteInFile(outputDirectory, img_name_path[0], Finaltext)

    except:
        print("Except Handeled in file", img_name_path[0])
        print(traceback.format_exc())

