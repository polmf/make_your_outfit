import csv
images = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        images.append(row)
outfits = []
with open('datathon/datathon/dataset/outfit_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        outfits.append(row)

combined = []
maxlen = 0
for ima in images:
    aa = []
    for ou in outfits:
        if ou[1] == ima[0]:
            aa.append(ou[0])
    combined.append(aa)


print(combined)
with open('datathon/datathon/dataset/combined_data.csv', 'w', newline='') as f:
    write = csv.writer(f,  )
    for row in combined:
        write.writerow(row)

        