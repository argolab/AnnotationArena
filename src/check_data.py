import json
with open("gaussian_train_10_new.json", "r") as file:
    dev_data = json.load(file)["data"]
with open("gaussian_train_10_new.json", "r") as file:
    train_data = json.load(file)["data"]
data_to_remove = []
count = 0
for index, entry in enumerate(dev_data):
    equal = False
    for index1, entry1 in enumerate(train_data):
        if entry["known_questions"] == entry1["known_questions"]:
            equal = True
            for i in range(5):
                if not entry["input"][i] == entry1["input"][i]:
                    equal = False
                    break
            if equal:
                break
    if equal:
        data_to_remove.append(entry)
        count += 1
print(count)