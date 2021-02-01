NameShapes = {
    "clef": "",
    "a_4": "/4",
    "a_1": "/1",
    "a_2": "/2",
    "a_8": "/8",
    "b_8": "/8",
    "a_16": "/16",
    "b_16": "/16",
    "a_32": "/32",
    "b_32": "/32",
    "sharp": "#",
    "natural": "",
    "flat": "&",
    "double_sharp": "##",
    "double_flat": "&&",
    "dot": ".",
    "barline": "",
    "chord": "",
    "t_2_2": "\meter<\"2/2\">",
    "t_4_4": "\meter<\"4/4\">"
}

####TODO Change the input and positions to what will come 
# Shapes = [ { (label),((start,end)) }  ]
# positions = [ { (label),((x)) }  ]

def StringToWrite(Shapes,positions):
    output = "[ "
    for i,shape in enumerate(Shapes):
        if shape[0][0] == "a":
            start = shape[1][0][i][0]
            end = shape[1][0][i][1]
            #print(start,end)
            flag = False
            for pos in positions:
                #print(pos[0])
                if start <= pos[1] <= end:
                    temp=pos[0]
                    if len(pos[0]) == 1:
                        temp= pos[0]+"1"
                    output += temp+ NameShapes[shape[0]] + " "
                    flag = True
                    break
            if flag == False:
                output += "a1" + NameShapes[shape[0]] + " "

        elif shape[0][0] == "b" and shape[0][1] == "_" :
            start = shape[1][0][i][0]
            end = shape[1][0][i][1]
            #print(start,end)
            counter = 0
            for pos in positions:
                #print(pos[0])
                if start <= pos[1] <= end:
                    temp=pos[0]
                    if len(pos[0]) == 1:
                        temp= pos[0]+"1"
                    output += temp + NameShapes[shape[0]] + " "
                    counter += 1
            
            remain = 0
            if "8" in shape[0]:
                remain = 2 - counter
            elif "16" in shape[0] or "32" in shape[0]:
                remain = 4 - counter
            for i in range(remain):
                output += "b1" + NameShapes[shape[0]] + " "

        elif shape[0] == "chord":
            start = shape[1][0][i][0]
            end = shape[1][0][i][1]
        
            array = []
            for pos in positions:
                if start <= pos[1] <= end:
                    array.append(pos[0])

            array = sorted(array)

            output += "{ "
            if len(array) != 0:
                for symb in array:
                    output += symb + ","
            else:
                output += "b1" + NameShapes[shape[0]] + ","
                output += "b1" + NameShapes[shape[0]]
                
            output += "} "
        
        elif shape[0] == "dot":
            output = output.strip() + NameShapes[shape[0]] + " "

        else:
            output += NameShapes[shape[0]] + " "

    output += "]"

    wordslist = output.strip().split(' ')

    for i, word in enumerate(wordslist):
        if "#" in word or "&" in word:
            if i != len(wordslist) - 1 and "/" in wordslist[i + 1] and "<" not in wordslist[i + 1]:
                    out = wordslist[i + 1][0] + word + wordslist[i + 1][1:]
                    wordslist[i+1] = out

    output = ""
    for word in wordslist:
        if word != '' and word[0] != "#"  and word[0] != "&" :
            output += word + " "

    # ####Testing
    # teststring = "\n"
    # for shape in Shapes:
    #     teststring += shape[0]
    # teststring += "\n"
    # output = output +teststring

    return output
