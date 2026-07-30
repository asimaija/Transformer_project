import requests
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text=requests.get(url).text
print(len(text))



def get_unique_characters(text):
    chars =  sorted(list(set(text)))
    return chars
unique_characters = get_unique_characters(text)
print(unique_characters)
print(len(unique_characters))


def create_mapping(unique_chracters):
    stoi = {ch:i for i,ch in enumerate(unique_chracters)}
    itos = {i:ch for i,ch in enumerate(unique_chracters)}
    return stoi,itos
stoi,itos = create_mapping(unique_characters)

print("\n character -> integer")
for ch,i in stoi.items():
    print(f"{ch}->{i}")
print("\n Integer -> character")
for i,ch in itos.items():
    print(f"{i}->{ch}")




def encode(text):
    return[stoi[ch] for ch in text]
encoded_text = encode(text)
print(encoded_text[:10])


def split_data(encoded_text,train_ratio=0.9):
    split_text = int(len(encoded_text)*train_ratio)
    train_data = encoded_text[:split_text]
    val_data = encoded_text[split_text:]
    return train_data,val_data
train_data,val_data = split_data(encoded_text)
print("Total data",len(encoded_text))
print("Training data" ,len(train_data))
print("Validate data" , len(val_data))

def create_batch(data,block_size):
    x=data[:block_size]
    y=data[1:block_size+1]
    return x,y
block_size = 4
x,y = create_batch(train_data,block_size)
print("input" ,x)
print("output" ,y)








    




